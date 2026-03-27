from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .graph_loader import CsvGraph, Edge, Node


@dataclass
class RetrievalContext:
    nodes: List[Node]
    edges: List[Edge]
    mode: str
    selected_node: str
    ranking_profile: str
    top_k: int


class GraphRetriever:
    SYMBOL_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]+")
    LEGACY_MARKERS = ("variable", "upsample", "torch 1.")
    RELATION_WEIGHTS = {
        "CALLS": 1.6,
        "IMPLEMENTS": 1.4,
        "HAS_PARAM": 1.2,
        "EXPLAINS": 1.1,
        "RELATED_TO": 1.0,
        "REFERENCES": 1.0,
        "CONTAINS": 0.9,
    }

    def __init__(self, graph: CsvGraph) -> None:
        self.graph = graph

    def retrieve(
        self,
        query: str,
        mode: str = "detective",
        top_k: int = 80,
        ranking_profile: str = "balanced",
    ) -> RetrievalContext:
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

        nodes = self._filter_legacy(nodes)
        ranked_nodes = self._rank_nodes(
            query=query,
            seed_id=selected.id,
            nodes=nodes,
            edges=edges,
            ranking_profile=ranking_profile,
        )
        top_nodes = ranked_nodes[:max(1, top_k)]
        return RetrievalContext(
            nodes=top_nodes,
            edges=edges,
            mode=mode,
            selected_node=selected.name,
            ranking_profile=ranking_profile,
            top_k=max(1, top_k),
        )

    def _select_seed(self, query: str) -> Node:
        lowered = query.lower()
        tokens = {t.lower() for t in self.SYMBOL_PATTERN.findall(query)}

        best = None
        best_score = -1
        for node in self.graph.nodes.values():
            if not node.name.strip():
                continue
            name_l = node.name.lower()
            text = f"{node.label} {node.name} {node.url}".lower()
            score = 0
            if name_l and name_l in lowered:
                score += 5
            for tok in tokens:
                if tok and tok in text:
                    score += 1
            if "torch" in text:
                score += 1
            if "compile" in lowered and "compile" in text:
                score += 3
            score += min(3, len(self.graph.adj[node.id]) + len(self.graph.rev_adj[node.id]))
            if score > best_score:
                best = node
                best_score = score

        if best is None:
            # fallback deterministic named node
            named = [n for n in self.graph.nodes.values() if n.name.strip()]
            if named:
                return sorted(named, key=lambda n: n.id)[0]
            return self.graph.nodes[min(self.graph.nodes.keys())]
        return best

    def _filter_legacy(self, nodes: List[Node]) -> List[Node]:
        out = []
        for node in nodes:
            text = f"{node.label} {node.name} {node.url}".lower()
            if any(marker in text for marker in self.LEGACY_MARKERS):
                continue
            out.append(node)
        return out

    def _rank_nodes(
        self,
        query: str,
        seed_id: int,
        nodes: List[Node],
        edges: List[Edge],
        ranking_profile: str,
    ) -> List[Node]:
        profile = ranking_profile.lower().strip()
        profile_weights = {
            "balanced": {"token": 2.0, "distance": 1.5, "degree": 0.3, "relation": 0.5, "exact": 4.0},
            "precision": {"token": 2.8, "distance": 2.1, "degree": 0.2, "relation": 0.4, "exact": 5.0},
            "coverage": {"token": 1.8, "distance": 0.8, "degree": 0.8, "relation": 0.7, "exact": 3.5},
        }
        if profile not in profile_weights:
            profile = "balanced"
        w = profile_weights[profile]

        lowered = query.lower()
        query_tokens = {t.lower() for t in self.SYMBOL_PATTERN.findall(query)}
        node_ids = {n.id for n in nodes}
        distances = self._distance_map(seed_id, node_ids)
        relation_strength = self._relation_strength(node_ids, edges)

        scored = []
        for node in nodes:
            text = f"{node.label} {node.name} {node.url}".lower()
            token_hits = sum(1 for tok in query_tokens if tok in text)
            exact_bonus = 1.0 if node.name.strip() and node.name.lower() in lowered else 0.0
            degree = len(self.graph.adj[node.id]) + len(self.graph.rev_adj[node.id])
            dist_penalty = float(distances.get(node.id, 4))

            score = 0.0
            score += w["token"] * token_hits
            score += w["exact"] * exact_bonus
            score += w["degree"] * min(10, degree)
            score += w["relation"] * relation_strength.get(node.id, 0.0)
            score -= w["distance"] * dist_penalty

            if "compile" in lowered and "compile" in text:
                score += 2.0
            if "torch" in text:
                score += 0.4

            scored.append((score, node))

        scored.sort(key=lambda x: (x[0], -x[1].id), reverse=True)
        return [n for _, n in scored]

    def _distance_map(self, seed_id: int, allowed: set[int]) -> Dict[int, int]:
        if seed_id not in self.graph.nodes:
            return {}
        distance: Dict[int, int] = {seed_id: 0}
        queue = [seed_id]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            cur_dist = distance[current]
            if cur_dist >= 3:
                continue
            neighbors = []
            for edge in self.graph.adj[current]:
                neighbors.append(edge.target)
            for edge in self.graph.rev_adj[current]:
                neighbors.append(edge.source)
            for nxt in neighbors:
                if nxt not in allowed or nxt in distance:
                    continue
                distance[nxt] = cur_dist + 1
                queue.append(nxt)
        return distance

    def _relation_strength(self, allowed: set[int], edges: List[Edge]) -> Dict[int, float]:
        strength: Dict[int, float] = {n: 0.0 for n in allowed}
        for edge in edges:
            if edge.source not in allowed or edge.target not in allowed:
                continue
            weight = self.RELATION_WEIGHTS.get(edge.relation.upper(), 1.0)
            strength[edge.source] += weight
            strength[edge.target] += weight
        return strength
