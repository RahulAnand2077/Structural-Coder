from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from src.integration_pipeline.graph_loader import CsvGraph, Edge, Node


@dataclass
class RetrievedContext:
    query: str
    seed_nodes: list[str]
    nodes: list[Node]
    edges: list[Edge]


class GraphRAGRetriever:
    """Hybrid retriever: GNN embedding similarity + graph expansion + lexical rerank."""

    def __init__(
        self,
        graph: CsvGraph,
        node_ids: list[int],
        embeddings: torch.Tensor,
    ) -> None:
        self.graph = graph
        self.node_ids = node_ids
        self.embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.id_to_index = {nid: i for i, nid in enumerate(node_ids)}

    def retrieve(self, query: str, top_k: int = 20, seed_k: int = 4, expansion_hops: int = 1) -> RetrievedContext:
        q = self._query_embedding(query, self.embeddings.size(1))
        scores = torch.mv(self.embeddings, q)
        top_seed_idx = torch.topk(scores, k=min(max(20, seed_k * 8), scores.numel())).indices.tolist()
        seed_ids = self._select_seed_ids(top_seed_idx, seed_k)
        if not seed_ids:
            seed_ids = self._global_named_fallback_ids(query, seed_k)

        collected_node_ids: set[int] = set(seed_ids)
        collected_edges: dict[tuple[int, int, str], Edge] = {}

        frontier = set(seed_ids)
        for _ in range(max(1, expansion_hops)):
            next_frontier: set[int] = set()
            for nid in list(frontier):
                for edge in self.graph.adj.get(nid, []):
                    collected_edges[(edge.source, edge.target, edge.relation)] = edge
                    collected_node_ids.add(edge.target)
                    next_frontier.add(edge.target)
                for edge in self.graph.rev_adj.get(nid, []):
                    collected_edges[(edge.source, edge.target, edge.relation)] = edge
                    collected_node_ids.add(edge.source)
                    next_frontier.add(edge.source)
            frontier = next_frontier

        ranked = self._hybrid_rank(query, collected_node_ids, scores)
        selected_ids = [nid for nid, _ in ranked[: max(1, top_k)]]
        selected_nodes = [self.graph.nodes[nid] for nid in selected_ids if nid in self.graph.nodes]
        seed_names = [self.graph.nodes[nid].name for nid in seed_ids if nid in self.graph.nodes and self.graph.nodes[nid].name.strip()]

        # Ensure context contains meaningful symbols; fallback to top ranked named nodes.
        if not any(n.name.strip() for n in selected_nodes):
            ranked_named = [nid for nid, _ in ranked if nid in self.graph.nodes and self.graph.nodes[nid].name.strip()]
            selected_ids = ranked_named[: max(1, top_k)]
            selected_nodes = [self.graph.nodes[nid] for nid in selected_ids]

        if not selected_nodes:
            fallback_ids = self._global_named_fallback_ids(query, max(1, top_k))
            selected_nodes = [self.graph.nodes[nid] for nid in fallback_ids if nid in self.graph.nodes]

        if not seed_names:
            seed_names = [n.name for n in selected_nodes if n.name.strip()][:seed_k]

        return RetrievedContext(
            query=query,
            seed_nodes=seed_names,
            nodes=selected_nodes,
            edges=list(collected_edges.values()),
        )

    def _hybrid_rank(self, query: str, node_ids: set[int], gnn_scores: torch.Tensor) -> list[tuple[int, float]]:
        query_tokens = {t.lower() for t in query.split() if t.strip()}
        ranked: list[tuple[int, float]] = []
        for nid in node_ids:
            if nid not in self.id_to_index or nid not in self.graph.nodes:
                continue
            idx = self.id_to_index[nid]
            node = self.graph.nodes[nid]
            text = f"{node.label} {node.name} {node.url}".lower()
            lexical = sum(1.0 for t in query_tokens if t in text)
            degree = float(len(self.graph.adj.get(nid, [])) + len(self.graph.rev_adj.get(nid, [])))
            score = 2.5 * float(gnn_scores[idx]) + 0.7 * lexical + 0.05 * min(20.0, degree)
            if node.name.strip():
                score += 0.8
            if self._is_api_like(node.label):
                score += 0.6
            ranked.append((nid, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _select_seed_ids(self, candidate_indices: list[int], seed_k: int) -> list[int]:
        preferred: list[int] = []
        fallback: list[int] = []

        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.node_ids):
                continue
            nid = self.node_ids[idx]
            node = self.graph.nodes.get(nid)
            if node is None:
                continue
            if not node.name.strip():
                fallback.append(nid)
                continue
            if self._is_api_like(node.label):
                preferred.append(nid)
            else:
                fallback.append(nid)

            if len(preferred) >= seed_k:
                break

        out = preferred[:seed_k]
        for nid in fallback:
            if len(out) >= seed_k:
                break
            if nid not in out:
                out.append(nid)

        if not out and self.node_ids:
            out = [self.node_ids[0]]
        return out

    def _global_named_fallback_ids(self, query: str, k: int) -> list[int]:
        query_tokens = {t.lower() for t in query.split() if t.strip()}
        scored: list[tuple[float, int]] = []
        for nid, node in self.graph.nodes.items():
            name = node.name.strip()
            if not name:
                continue
            text = f"{node.label} {node.name} {node.url}".lower()
            lexical = sum(1.0 for t in query_tokens if t in text)
            degree = float(len(self.graph.adj.get(nid, [])) + len(self.graph.rev_adj.get(nid, [])))
            api_bonus = 1.0 if self._is_api_like(node.label) else 0.0
            score = 1.6 * lexical + 0.04 * min(25.0, degree) + api_bonus
            scored.append((score, nid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[: max(1, k)]]

    @staticmethod
    def _is_api_like(label: str) -> bool:
        label_l = (label or "").lower()
        return label_l.startswith("api_") or "pytorchconcept" in label_l

    @staticmethod
    def _query_embedding(query: str, dim: int) -> torch.Tensor:
        vec = torch.zeros(dim, dtype=torch.float32)
        for tok in query.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = torch.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
