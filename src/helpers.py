"""
helpers.py
==========
Combined helper utilities:
  - GraphRetriever: lexical keyword-based graph search (ablation baseline)
  - PythonChunker: parse .py files into graph nodes
  - Ollama caller: send prompts to local LLM
  - Prompt builder: construct advisory prompts from retrieved nodes
  - Scoring: token hit, grounding, validity scoring
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.config import MODEL_NAME
from src.graph_loader import CsvGraph, Edge, Node
from src.validator import ActiveValidator


# ─────────────────────────────────────────────────────────────────────
#  LEXICAL RETRIEVER (ablation baseline)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalContext:
    nodes: List[Node]
    edges: List[Edge]
    mode: str
    selected_node: str
    ranking_profile: str
    top_k: int


class GraphRetriever:
    """Keyword-based graph search (lexical baseline for ablation comparison)."""
    SYMBOL_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]+")
    LEGACY_MARKERS = ("variable", "upsample", "torch 1.")
    RELATION_WEIGHTS = {
        "CALLS": 1.6, "IMPLEMENTS": 1.4, "HAS_PARAM": 1.2,
        "EXPLAINS": 1.1, "RELATED_TO": 1.0, "REFERENCES": 1.0, "CONTAINS": 0.9,
    }

    def __init__(self, graph: CsvGraph) -> None:
        self.graph = graph

    def retrieve(self, query: str, mode: str = "detective", top_k: int = 80, ranking_profile: str = "balanced") -> RetrievalContext:
        mode = mode.lower().strip()
        selected = self._select_seed(query)
        if mode == "microscope":
            nodes, edges = self.graph.one_hop(selected.id)
        elif mode == "detective":
            nodes, edges = self.graph.multi_hop(selected.id, depth=2, max_nodes=600)
        elif mode == "telescope":
            nodes, edges = self.graph.highest_degree(30), []
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        nodes = [n for n in nodes if not any(m in f"{n.label} {n.name} {n.url}".lower() for m in self.LEGACY_MARKERS)]
        ranked = self._rank_nodes(query, selected.id, nodes, edges, ranking_profile)
        return RetrievalContext(nodes=ranked[:max(1, top_k)], edges=edges, mode=mode,
                                selected_node=selected.name, ranking_profile=ranking_profile, top_k=max(1, top_k))

    def _select_seed(self, query: str) -> Node:
        lowered = query.lower()
        tokens = {t.lower() for t in self.SYMBOL_PATTERN.findall(query)}
        best, best_score = None, -1
        for node in self.graph.nodes.values():
            if not node.name.strip():
                continue
            text = f"{node.label} {node.name} {node.url}".lower()
            score = 5 if node.name.lower() in lowered else 0
            score += sum(1 for t in tokens if t in text)
            score += 1 if "torch" in text else 0
            score += 3 if "compile" in lowered and "compile" in text else 0
            score += min(3, len(self.graph.adj[node.id]) + len(self.graph.rev_adj[node.id]))
            if score > best_score:
                best, best_score = node, score
        if best is None:
            named = [n for n in self.graph.nodes.values() if n.name.strip()]
            return sorted(named, key=lambda n: n.id)[0] if named else self.graph.nodes[min(self.graph.nodes.keys())]
        return best

    def _rank_nodes(self, query, seed_id, nodes, edges, ranking_profile):
        w = {"balanced": {"token": 2.0, "distance": 1.5, "degree": 0.3, "relation": 0.5, "exact": 4.0},
             "precision": {"token": 2.8, "distance": 2.1, "degree": 0.2, "relation": 0.4, "exact": 5.0},
             "coverage": {"token": 1.8, "distance": 0.8, "degree": 0.8, "relation": 0.7, "exact": 3.5},
        }.get(ranking_profile, {"token": 2.0, "distance": 1.5, "degree": 0.3, "relation": 0.5, "exact": 4.0})
        lowered = query.lower()
        tokens = {t.lower() for t in self.SYMBOL_PATTERN.findall(query)}
        scored = []
        for node in nodes:
            text = f"{node.label} {node.name} {node.url}".lower()
            hits = sum(1 for t in tokens if t in text)
            exact = 1.0 if node.name.strip() and node.name.lower() in lowered else 0.0
            deg = len(self.graph.adj[node.id]) + len(self.graph.rev_adj[node.id])
            s = w["token"] * hits + w["exact"] * exact + w["degree"] * min(10, deg)
            scored.append((s, node))
        scored.sort(key=lambda x: (x[0], -x[1].id), reverse=True)
        return [n for _, n in scored]


# ─────────────────────────────────────────────────────────────────────
#  OLLAMA CALLER
# ─────────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = MODEL_NAME) -> tuple[str, str]:
    """Send a prompt to local Ollama server, return (text, source)."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url="http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
        text = json.loads(body).get("response", "").strip()
        return (text or "Ollama returned an empty response."), "ollama"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Ollama unavailable: {type(exc).__name__}: {exc}", "fallback"


# ─────────────────────────────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_gnn_prompt(query: str, nodes: list[Node]) -> str:
    """Build advisory LLM prompt from GNN-predicted nodes."""
    lines = [
        "You are a PyTorch 2.x coding assistant.",
        "Below is a list of PyTorch resources retrieved from official documentation.",
        "ADVISORY RULE: ONLY use the provided APIs if they fit naturally.",
        "A simple, clean, runnable solution is better than forcing obscure APIs!",
        "If the tools seem irrelevant, IGNORE THEM and rely on your own knowledge.",
        "WARNING: Items marked as concepts/tutorials are NOT callable modules.",
        "Always write complete, runnable Python code with proper standard imports.",
        "",
        f"User query: {query}",
        ""
    ]
    api_lines, concept_lines = [], []
    for i, node in enumerate(nodes[:5], 1):
        name = node.name.strip()
        if not name and node.url:
            match = re.search(r'/([^/]+?)\.html$', node.url)
            if match:
                name = match.group(1)
        if not name:
            continue
        url_hint = f"  (docs: {node.url})" if node.url else ""
        label = node.label.replace('API_', '').replace('_', ' ').title() if node.label else ''
        entry = f"  {i}. {name} [{label}]{url_hint}"
        if node.label and node.label.startswith("API_"):
            api_lines.append(entry)
        else:
            concept_lines.append(entry)
    if api_lines:
        lines.append("💻 Valid PyTorch APIs (Advisory - ONLY use if relevant):")
        lines.extend(api_lines)
        lines.append("")
    if concept_lines:
        lines.append("📚 Conceptual Context (READ-ONLY. Do not import as code!):")
        lines.extend(concept_lines)
        lines.append("")
    lines.append("Write complete working Python code. Include all necessary imports.")
    lines.append("Use ```python code blocks for your code.")
    return "\n".join(lines)


def build_standalone_prompt(query: str) -> str:
    """Build zero-context prompt for standalone LLM."""
    return (
        "You are a PyTorch 2.x coding assistant.\n"
        f"User query: {query}\n\n"
        "Write complete, runnable Python code with proper imports.\n"
        "Use ```python code blocks for your code."
    )


# ─────────────────────────────────────────────────────────────────────
#  SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def token_hit_score(query: str, nodes: list[Node]) -> float:
    """What fraction of query keywords appear in the retrieved node text."""
    tokens = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]+', query.lower()))
    tokens -= {"a", "an", "the", "for", "with", "and", "or", "to", "in", "of", "is", "how", "do", "show"}
    if not tokens:
        return 0.0
    node_text = " ".join(f"{n.label} {n.name} {n.url}" for n in nodes).lower()
    hits = sum(1 for t in tokens if t in node_text)
    return hits / len(tokens)


def grounding_score(answer: str, nodes: list[Node], query: str = "") -> float:
    """What fraction of retrieved API names appear in the generated code."""
    api_nodes = [n for n in nodes if n.label and n.label.startswith("API_") and n.name.strip()]
    if not api_nodes:
        return 0.0
    code = answer.lower()
    hits = sum(1 for n in api_nodes if n.name.lower().split(".")[-1] in code)
    return hits / len(api_nodes)


def validity_score(answer: str, target_hardware: str = "H100") -> float:
    """Run C0-C5 checks and return fraction passed."""
    validator = ActiveValidator()
    code = _extract_code(answer)
    if not code.strip():
        return 0.0
    report = validator.validate(code, target_hardware=target_hardware)
    score = 0.0
    for c in report.checks:
        if c.status == "pass":
            score += 1.0
        elif c.status == "skipped":
            score += 0.5
    return score / max(1, len(report.checks))


def _extract_code(text: str) -> str:
    """Extract code from markdown fenced blocks."""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text


def compute_final_score(retrieval: float, grounding: float, validity: float) -> float:
    """Final = 0.4 × Retrieval + 0.6 × (0.5 × Grounding + 0.5 × Validity)"""
    return 0.4 * retrieval + 0.6 * (0.5 * grounding + 0.5 * validity)
