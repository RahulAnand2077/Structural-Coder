"""
Python File Chunker
===================
Parses Python source files into structured chunks (classes, methods, functions)
and integrates them as nodes/edges into the CsvGraph for retrieval.

Usage:
    from src.graph_rag.python_chunker import PythonChunker
    
    chunker = PythonChunker(graph)
    chunks = chunker.chunk_file("/path/to/model.py")
    chunker.integrate_chunks(chunks)  # Adds to the live graph
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.integration_pipeline.graph_loader import CsvGraph, Edge, Node


@dataclass
class CodeChunk:
    """A single parsed chunk from a Python file."""
    name: str                   # e.g. "MyModel" or "MyModel.forward" or "train_loop"
    chunk_type: str             # "class", "method", "function"
    source: str                 # The actual source code
    file_path: str              # Original file path
    start_line: int             # Line number in the original file
    end_line: int               # End line in the original file
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent class name if this is a method
    decorators: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)  # Parameter names


class PythonChunker:
    """Parses Python files into class/method/function chunks and
    integrates them into the knowledge graph as nodes + edges."""

    def __init__(self, graph: CsvGraph) -> None:
        self.graph = graph
        # Track the next available node ID (after existing CSV nodes)
        self._next_id = max(graph.nodes.keys(), default=-1) + 1

    def chunk_file(self, file_path: str | Path) -> List[CodeChunk]:
        """Parse a Python file into structured chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Warning: Cannot parse {path}: {e}")
            return []

        lines = source.splitlines()
        chunks: List[CodeChunk] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._extract_class(node, lines, str(path)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._extract_function(node, lines, str(path)))

        return chunks

    def chunk_directory(self, dir_path: str | Path, pattern: str = "**/*.py") -> List[CodeChunk]:
        """Parse all Python files in a directory."""
        all_chunks: List[CodeChunk] = []
        for py_file in sorted(Path(dir_path).glob(pattern)):
            if "__pycache__" in str(py_file):
                continue
            try:
                chunks = self.chunk_file(py_file)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Skipping {py_file}: {e}")
        return all_chunks

    def integrate_chunks(self, chunks: List[CodeChunk]) -> dict:
        """Add chunks as nodes + edges into the live CsvGraph.
        
        Returns summary of what was added.
        """
        added_nodes = 0
        added_edges = 0
        chunk_node_ids: dict[str, int] = {}  # chunk.name -> node_id

        for chunk in chunks:
            nid = self._next_id
            self._next_id += 1

            # Map chunk_type to graph labels
            label_map = {
                "class": "API_Class",
                "method": "API_Method",
                "function": "API_Function",
            }
            label = label_map.get(chunk.chunk_type, "CodeSnippet")

            node = Node(
                id=nid,
                label=label,
                name=chunk.name,
                url=f"file://{chunk.file_path}#L{chunk.start_line}-L{chunk.end_line}",
            )
            self.graph.nodes[nid] = node
            chunk_node_ids[chunk.name] = nid
            added_nodes += 1

            # If this is a method, create CONTAINS edge from parent class
            if chunk.parent and chunk.parent in chunk_node_ids:
                parent_id = chunk_node_ids[chunk.parent]
                edge = Edge(source=parent_id, target=nid, relation="CONTAINS")
                self.graph.edges.append(edge)
                self.graph.adj[parent_id].append(edge)
                self.graph.rev_adj[nid].append(edge)
                added_edges += 1

            # Create CALLS edges for any torch.* references in the source
            self._link_torch_references(nid, chunk.source)
            added_edges += self._count_new_edges_for(nid)

        return {
            "chunks_processed": len(chunks),
            "nodes_added": added_nodes,
            "edges_added": added_edges,
        }

    # ────────────────────────────────────────────────────────────────
    #  PRIVATE: AST Extraction
    # ────────────────────────────────────────────────────────────────

    def _extract_class(self, node: ast.ClassDef, lines: list[str], file_path: str) -> List[CodeChunk]:
        """Extract a class and all its methods as separate chunks."""
        chunks: List[CodeChunk] = []

        # The class itself
        class_source = self._get_source(node, lines)
        class_chunk = CodeChunk(
            name=node.name,
            chunk_type="class",
            source=class_source,
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            decorators=[self._decorator_name(d) for d in node.decorator_list],
        )
        chunks.append(class_chunk)

        # Each method in the class
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_source = self._get_source(item, lines)
                method_chunk = CodeChunk(
                    name=f"{node.name}.{item.name}",
                    chunk_type="method",
                    source=method_source,
                    file_path=file_path,
                    start_line=item.lineno,
                    end_line=item.end_lineno or item.lineno,
                    docstring=ast.get_docstring(item),
                    parent=node.name,
                    decorators=[self._decorator_name(d) for d in item.decorator_list],
                    args=[arg.arg for arg in item.args.args if arg.arg != "self"],
                )
                chunks.append(method_chunk)

        return chunks

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str], file_path: str) -> CodeChunk:
        """Extract a standalone function."""
        return CodeChunk(
            name=node.name,
            chunk_type="function",
            source=self._get_source(node, lines),
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            decorators=[self._decorator_name(d) for d in node.decorator_list],
            args=[arg.arg for arg in node.args.args],
        )

    @staticmethod
    def _get_source(node: ast.AST, lines: list[str]) -> str:
        """Extract source code lines for an AST node."""
        start = node.lineno - 1  # 0-indexed
        end = getattr(node, "end_lineno", node.lineno)
        return "\n".join(lines[start:end])

    @staticmethod
    def _decorator_name(dec: ast.expr) -> str:
        """Get decorator name as string."""
        if isinstance(dec, ast.Name):
            return dec.id
        elif isinstance(dec, ast.Attribute):
            return ast.dump(dec)
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                return dec.func.id
            elif isinstance(dec.func, ast.Attribute):
                return ast.dump(dec.func)
        return "unknown"

    def _link_torch_references(self, nid: int, source: str) -> None:
        """Find torch.xxx references in source and create CALLS edges to matching graph nodes."""
        import re
        # Find all torch.something.something patterns
        torch_refs = set(re.findall(r'(torch\.\w+(?:\.\w+)*)', source))
        
        for ref in torch_refs:
            ref_lower = ref.lower()
            # Search existing graph nodes for a match
            for existing_id, existing_node in self.graph.nodes.items():
                if existing_id == nid:
                    continue
                node_name = existing_node.name.lower().replace("_", ".")
                if ref_lower == node_name or ref_lower.endswith(node_name.split(".")[-1]):
                    edge = Edge(source=nid, target=existing_id, relation="CALLS")
                    self.graph.edges.append(edge)
                    self.graph.adj[nid].append(edge)
                    self.graph.rev_adj[existing_id].append(edge)
                    break  # One match per reference is enough

    def _count_new_edges_for(self, nid: int) -> int:
        """Count edges that were just added for this node."""
        return len(self.graph.adj.get(nid, []))

    # ────────────────────────────────────────────────────────────────
    #  CONVENIENCE: Pretty Print
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def summarize_chunks(chunks: List[CodeChunk]) -> str:
        """Return a human-readable summary of parsed chunks."""
        lines = [f"Parsed {len(chunks)} chunks:"]
        for c in chunks:
            prefix = "  📦" if c.chunk_type == "class" else "  🔧" if c.chunk_type == "function" else "    ↳"
            doc_hint = f' — "{c.docstring[:50]}..."' if c.docstring else ""
            lines.append(f"{prefix} [{c.chunk_type}] {c.name} (L{c.start_line}-{c.end_line}){doc_hint}")
        return "\n".join(lines)
