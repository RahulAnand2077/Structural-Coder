"""
Graph-RAG Retriever — Core Retrieval Logic
============================================

This is the heart of the Graph-RAG retriever. Given a query, it:

    1. SEED SELECTION:   Find starting nodes whose features match the query
    2. GRAPH EXPANSION:  Walk outward through edges to discover neighbors
    3. HYBRID RANKING:   Score every candidate and return the best ones

How it uses two kinds of embeddings:
    - input_features (hash-bag vectors) → for seed selection
      (same space as query, so cosine similarity is meaningful)
    - embeddings (GNN-learned vectors) → for re-ranking
      (captures graph structure, so structurally important nodes rank higher)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .graph_loader import CsvGraph, Edge, Node


# ──────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """Raw retrieval output before it becomes the final JSON result."""
    query: str
    seed_nodes: list[str]   # names of the starting nodes
    nodes: list[Node]       # all retrieved nodes
    edges: list[Edge]       # all retrieved edges


@dataclass
class RankingWeights:
    """Configurable weights for hybrid ranking.

    You can tune these to change how retrieval works:
        - Higher gnn    → prefer structurally important nodes
        - Higher lexical → prefer nodes whose text matches the query
        - Higher degree  → prefer highly-connected nodes
    """
    gnn: float = 2.5           # weight for GNN embedding similarity
    lexical: float = 0.7       # weight per keyword match
    degree: float = 0.05       # weight per graph connection
    degree_cap: float = 20.0   # max degree contribution
    named_bonus: float = 0.8   # bonus for nodes with a name
    api_bonus: float = 0.6     # bonus for API-like nodes


# ──────────────────────────────────────────────────────────────────
# Main retriever class
# ──────────────────────────────────────────────────────────────────

class GraphRAGRetriever:
    """Hybrid retriever: input-feature seed selection + GNN re-ranking + graph expansion."""

    def __init__(
        self,
        graph: CsvGraph,
        node_ids: list[int],
        embeddings: torch.Tensor,
        input_features: torch.Tensor | None = None,
        ranking_weights: RankingWeights | None = None,
    ) -> None:
        self.graph = graph
        self.node_ids = node_ids
        self.embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Input features are in the same hash-bag space as query vectors.
        # If not provided (legacy cache), fall back to GNN embeddings.
        if input_features is not None:
            self.input_features = torch.nn.functional.normalize(input_features, p=2, dim=1)
        else:
            self.input_features = self.embeddings

        self.id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        self.weights = ranking_weights or RankingWeights()

    # ──────────────────────────────────────────────────────────────
    # Main retrieve method
    # ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        seed_k: int = 4,
        expansion_hops: int = 1,
    ) -> RetrievedContext:
        """Run the full retrieval pipeline for a query.

        Step 1: Find seed nodes (input features, same space as query)
        Step 2: Expand seeds through graph edges
        Step 3: Re-rank all candidates (GNN + lexical + degree)
        Step 4: Return the top-k results
        """

        # ── Step 1: Seed Selection ──────────────────────────────
        # Convert query to a hash-bag vector and find similar nodes.
        q = self._query_embedding(query, self.input_features.size(1))
        seed_scores = torch.mv(self.input_features, q)
        top_seed_idx = torch.topk(
            seed_scores,
            k=min(max(20, seed_k * 8), seed_scores.numel()),
        ).indices.tolist()

        seed_ids = self._select_seed_ids(top_seed_idx, seed_k)
        if not seed_ids:
            seed_ids = self._global_named_fallback_ids(query, seed_k)

        # ── Step 2: Graph Expansion ─────────────────────────────
        # Walk outward from seeds through graph edges.
        collected_node_ids: set[int] = set(seed_ids)
        collected_edges: dict[tuple[int, int, str], Edge] = {}

        frontier = set(seed_ids)
        for _ in range(max(1, expansion_hops)):
            next_frontier: set[int] = set()
            for nid in list(frontier):
                # Follow outgoing edges.
                for edge in self.graph.adj.get(nid, []):
                    collected_edges[(edge.source, edge.target, edge.relation)] = edge
                    collected_node_ids.add(edge.target)
                    next_frontier.add(edge.target)
                # Follow incoming edges.
                for edge in self.graph.rev_adj.get(nid, []):
                    collected_edges[(edge.source, edge.target, edge.relation)] = edge
                    collected_node_ids.add(edge.source)
                    next_frontier.add(edge.source)
            frontier = next_frontier

        # ── Step 3: Hybrid Re-ranking ───────────────────────────
        # Score every candidate using GNN + lexical + degree.
        gnn_scores = torch.mv(
            self.embeddings,
            self._query_embedding(query, self.embeddings.size(1)),
        )
        ranked = self._hybrid_rank(query, collected_node_ids, gnn_scores)

        # ── Step 4: Build results ───────────────────────────────
        selected_ids = [nid for nid, _ in ranked[: max(1, top_k)]]
        selected_nodes = [self.graph.nodes[nid] for nid in selected_ids if nid in self.graph.nodes]
        seed_names = [
            self.graph.nodes[nid].name
            for nid in seed_ids
            if nid in self.graph.nodes and self.graph.nodes[nid].name.strip()
        ]

        # Fallback: if no named nodes were selected, try harder.
        if not any(n.name.strip() for n in selected_nodes):
            ranked_named = [
                nid for nid, _ in ranked
                if nid in self.graph.nodes and self.graph.nodes[nid].name.strip()
            ]
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

    # ──────────────────────────────────────────────────────────────
    # Hybrid ranking
    # ──────────────────────────────────────────────────────────────

    def _hybrid_rank(
        self,
        query: str,
        node_ids: set[int],
        gnn_scores: torch.Tensor,
    ) -> list[tuple[int, float]]:
        """Score each node using a weighted combination of signals."""
        w = self.weights
        query_tokens = {t.lower() for t in query.split() if t.strip()}
        ranked: list[tuple[int, float]] = []

        for nid in node_ids:
            if nid not in self.id_to_index or nid not in self.graph.nodes:
                continue

            idx = self.id_to_index[nid]
            node = self.graph.nodes[nid]
            text = f"{node.label} {node.name} {node.url}".lower()

            # Count how many query words appear in the node's text.
            lexical = sum(1.0 for t in query_tokens if t in text)

            # Count how connected this node is.
            degree = float(len(self.graph.adj.get(nid, [])) + len(self.graph.rev_adj.get(nid, [])))

            # Combined score.
            score = (
                w.gnn * float(gnn_scores[idx])
                + w.lexical * lexical
                + w.degree * min(w.degree_cap, degree)
            )
            if node.name.strip():
                score += w.named_bonus
            if self._is_api_like(node.label):
                score += w.api_bonus

            ranked.append((nid, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ──────────────────────────────────────────────────────────────
    # Seed selection helpers
    # ──────────────────────────────────────────────────────────────

    def _select_seed_ids(self, candidate_indices: list[int], seed_k: int) -> list[int]:
        """Pick the best seed nodes, preferring API-like nodes."""
        preferred: list[int] = []   # API-like nodes (best seeds)
        fallback: list[int] = []    # other named nodes

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

        # Take API nodes first, fill remaining slots with fallback.
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
        """Last-resort: scan ALL nodes and pick the best matches by text."""
        query_tokens = {t.lower() for t in query.split() if t.strip()}
        scored: list[tuple[float, int]] = []
        for nid, node in self.graph.nodes.items():
            if not node.name.strip():
                continue
            text = f"{node.label} {node.name} {node.url}".lower()
            lexical = sum(1.0 for t in query_tokens if t in text)
            degree = float(len(self.graph.adj.get(nid, [])) + len(self.graph.rev_adj.get(nid, [])))
            api_bonus = 1.0 if self._is_api_like(node.label) else 0.0
            score = 1.6 * lexical + 0.04 * min(25.0, degree) + api_bonus
            scored.append((score, nid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[: max(1, k)]]

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_api_like(label: str) -> bool:
        """Check if a node label looks like a PyTorch API entry."""
        label_l = (label or "").lower()
        return label_l.startswith("api_") or "pytorchconcept" in label_l

    @staticmethod
    def _query_embedding(query: str, dim: int) -> torch.Tensor:
        """Convert a query string to a hash-bag vector.

        Same algorithm as _hashed_text_features in gnn_encoder.py,
        so queries and node features live in the same vector space.
        """
        vec = torch.zeros(dim, dtype=torch.float32)
        for tok in query.lower().split():
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = torch.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
