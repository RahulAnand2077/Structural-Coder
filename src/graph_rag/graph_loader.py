from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class Node:
    id: int
    label: str
    name: str
    url: str


@dataclass
class Edge:
    source: int
    target: int
    relation: str


class CsvGraph:
    """Read-only graph view over his exported nodes/edges CSV files."""

    def __init__(self, nodes_csv: str | Path, edges_csv: str | Path) -> None:
        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.adj: Dict[int, List[Edge]] = defaultdict(list)
        self.rev_adj: Dict[int, List[Edge]] = defaultdict(list)

    def load(self) -> None:
        with self.nodes_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = Node(
                    id=int(row["Id"]),
                    label=row.get("Label", ""),
                    name=row.get("Name", ""),
                    url=row.get("URL", ""),
                )
                self.nodes[node.id] = node

        with self.edges_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                edge = Edge(
                    source=int(row["Source"]),
                    target=int(row["Target"]),
                    relation=row.get("Type", "RELATED_TO"),
                )
                self.edges.append(edge)
                self.adj[edge.source].append(edge)
                self.rev_adj[edge.target].append(edge)

    def one_hop(self, node_id: int) -> Tuple[List[Node], List[Edge]]:
        if node_id not in self.nodes:
            return [], []
        nset: Dict[int, Node] = {node_id: self.nodes[node_id]}
        eset: Dict[Tuple[int, int, str], Edge] = {}
        for edge in self.adj[node_id] + self.rev_adj[node_id]:
            eset[(edge.source, edge.target, edge.relation)] = edge
            if edge.source in self.nodes:
                nset[edge.source] = self.nodes[edge.source]
            if edge.target in self.nodes:
                nset[edge.target] = self.nodes[edge.target]
        return list(nset.values()), list(eset.values())

    def multi_hop(self, node_id: int, depth: int = 2, max_nodes: int = 800) -> Tuple[List[Node], List[Edge]]:
        if node_id not in self.nodes:
            return [], []
        seen: Set[int] = {node_id}
        q: deque[Tuple[int, int]] = deque([(node_id, 0)])
        eset: Dict[Tuple[int, int, str], Edge] = {}

        while q:
            cur, d = q.popleft()
            if d >= depth:
                continue
            for edge in self.adj[cur] + self.rev_adj[cur]:
                eset[(edge.source, edge.target, edge.relation)] = edge
                nxt = edge.target if edge.source == cur else edge.source
                if nxt not in seen and nxt in self.nodes:
                    seen.add(nxt)
                    if len(seen) >= max_nodes:
                        break
                    q.append((nxt, d + 1))
            if len(seen) >= max_nodes:
                break

        return [self.nodes[n] for n in seen], list(eset.values())

    def highest_degree(self, top_k: int = 25) -> List[Node]:
        scored = []
        for node_id, node in self.nodes.items():
            deg = len(self.adj[node_id]) + len(self.rev_adj[node_id])
            scored.append((deg, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]
