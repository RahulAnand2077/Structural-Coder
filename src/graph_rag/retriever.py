from __future__ import annotations

import hashlib
import re
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
    """Hybrid retriever: GNN embedding similarity + global lexical search + graph expansion."""

    # Words too generic to discriminate between 24K nodes
    STOPWORDS = {
        "torch", "pytorch", "python", "write", "create", "implement", "using",
        "with", "from", "script", "code", "how", "the", "for", "that", "this",
        "show", "use", "model", "module", "function", "class", "method",
        "import", "setup", "define", "build", "make", "get", "set", "run",
        "example", "simple", "basic", "custom", "specific", "apply", "add",
    }

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

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                         #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 20, seed_k: int = 4, expansion_hops: int = 1) -> RetrievedContext:
        query_tokens = self._clean_tokens(query)

        # --- Phase 1: Dual-source seed selection ---
        # Source A: GNN cosine pool (may be noisy due to MD5 hashing)
        q_vec = self._query_embedding(query, self.embeddings.size(1))
        gnn_scores = torch.mv(self.embeddings, q_vec)
        gnn_top_idx = torch.topk(gnn_scores, k=min(max(40, seed_k * 10), gnn_scores.numel())).indices.tolist()
        gnn_seeds = self._pick_lexical_seeds(gnn_top_idx, seed_k, query_tokens)

        # Source B: Global lexical scan (brute-force but precise)
        global_seeds = self._global_lexical_search(query_tokens, seed_k)

        # Merge: prefer global lexical (precise) over GNN (noisy)
        seen: set[int] = set()
        seed_ids: list[int] = []
        for nid in global_seeds + gnn_seeds:
            if nid not in seen:
                seen.add(nid)
                seed_ids.append(nid)
        seed_ids = seed_ids[:seed_k]

        if not seed_ids:
            seed_ids = [self.node_ids[0]] if self.node_ids else []

        # --- Phase 2: 1-hop graph expansion ---
        collected_nids: set[int] = set(seed_ids)
        collected_edges: dict[tuple[int, int, str], Edge] = {}

        frontier = set(seed_ids)
        for _ in range(max(1, expansion_hops)):
            next_frontier: set[int] = set()
            for nid in list(frontier):
                for edge in self.graph.adj.get(nid, []):
                    collected_edges[(edge.source, edge.target, edge.relation)] = edge
                    collected_nids.add(edge.target)
                    next_frontier.add(edge.target)
                for edge in self.graph.rev_adj.get(nid, []):
                    collected_edges[(edge.source, edge.target, edge.relation)] = edge
                    collected_nids.add(edge.source)
                    next_frontier.add(edge.source)
            frontier = next_frontier

        # --- Phase 3: Also inject top-K from global lexical scan ---
        # This guarantees relevant nodes appear even if graph expansion
        # went through dead-end URL-only nodes with zero edges.
        global_top = self._global_lexical_search(query_tokens, top_k)
        for nid in global_top:
            collected_nids.add(nid)

        # --- Phase 4: Hybrid rank + select ---
        ranked = self._hybrid_rank(query_tokens, collected_nids, gnn_scores)
        selected_ids = [nid for nid, _ in ranked[:max(1, top_k)]]
        selected_nodes = [self.graph.nodes[nid] for nid in selected_ids if nid in self.graph.nodes]

        # Build seed display names (extract from URL if name is empty)
        seed_names = []
        for nid in seed_ids:
            node = self.graph.nodes.get(nid)
            if node is None:
                continue
            display = node.name.strip() or self._name_from_url(node.url)
            if display:
                seed_names.append(display)

        if not seed_names:
            seed_names = [self._node_display_name(n) for n in selected_nodes if self._node_display_name(n)][:seed_k]

        return RetrievedContext(
            query=query,
            seed_nodes=seed_names,
            nodes=selected_nodes,
            edges=list(collected_edges.values()),
        )

    # ------------------------------------------------------------------ #
    #  HYBRID RANKING                                                     #
    # ------------------------------------------------------------------ #

    def _hybrid_rank(self, query_tokens: set[str], node_ids: set[int], gnn_scores: torch.Tensor) -> list[tuple[int, float]]:
        ranked: list[tuple[int, float]] = []
        for nid in node_ids:
            if nid not in self.graph.nodes:
                continue
            node = self.graph.nodes[nid]
            searchable = self._searchable_text(node)
            lexical = sum(1.0 for t in query_tokens if t in searchable) if query_tokens else 0.0
            degree = float(len(self.graph.adj.get(nid, [])) + len(self.graph.rev_adj.get(nid, [])))

            # GNN score (may be 0 for nodes not in the embedding index)
            gnn_val = 0.0
            if nid in self.id_to_index:
                gnn_val = float(gnn_scores[self.id_to_index[nid]])

            # Lexical-dominant: 1.5x lexical, 0.8x GNN, plus bonuses
            score = 0.8 * gnn_val + 1.5 * lexical + 0.05 * min(20.0, degree)
            if node.name.strip():
                score += 0.5
            if self._is_api_like(node.label):
                score += 0.4
            ranked.append((nid, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ------------------------------------------------------------------ #
    #  SEED SELECTION                                                     #
    # ------------------------------------------------------------------ #

    def _pick_lexical_seeds(self, candidate_indices: list[int], seed_k: int, query_tokens: set[str]) -> list[int]:
        """Pick seeds from the GNN candidate pool that have lexical overlap."""
        hits: list[int] = []
        others: list[int] = []

        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.node_ids):
                continue
            nid = self.node_ids[idx]
            node = self.graph.nodes.get(nid)
            if node is None:
                continue

            searchable = self._searchable_text(node)
            has_match = any(tok in searchable for tok in query_tokens) if query_tokens else False

            if has_match:
                hits.append(nid)
            elif node.name.strip() and self._is_api_like(node.label):
                others.append(nid)

            if len(hits) >= seed_k:
                break

        out = hits[:seed_k]
        for nid in others:
            if len(out) >= seed_k:
                break
            if nid not in out:
                out.append(nid)
        return out

    def _global_lexical_search(self, query_tokens: set[str], k: int) -> list[int]:
        """Brute-force search ALL 24K nodes by token overlap. Fast (~20ms)."""
        if not query_tokens:
            return []
        scored: list[tuple[float, int]] = []
        for nid, node in self.graph.nodes.items():
            searchable = self._searchable_text(node)
            hits = sum(1.0 for t in query_tokens if t in searchable)
            if hits == 0:
                continue
            degree = float(len(self.graph.adj.get(nid, [])) + len(self.graph.rev_adj.get(nid, [])))
            name_bonus = 0.5 if node.name.strip() else 0.0
            api_bonus = 0.5 if self._is_api_like(node.label) else 0.0
            score = 2.0 * hits + 0.04 * min(25.0, degree) + name_bonus + api_bonus
            scored.append((score, nid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[:max(1, k)]]

    # ------------------------------------------------------------------ #
    #  UTILITIES                                                          #
    # ------------------------------------------------------------------ #

    def _clean_tokens(self, text: str) -> set[str]:
        return {
            t.lower() for t in text.replace('.', ' ').replace('_', ' ').split()
            if len(t.strip()) > 2 and t.lower() not in self.STOPWORDS
        }

    @staticmethod
    def _searchable_text(node: Node) -> str:
        """Combine all node fields into a lowercase, normalized search string."""
        return f"{node.label} {node.name} {node.url}".lower().replace('.', ' ').replace('_', ' ').replace('/', ' ')

    @staticmethod
    def _name_from_url(url: str) -> str:
        """Extract a human-readable name from a PyTorch docs URL."""
        if not url:
            return ""
        # e.g. ".../torch.compiler_dynamic_shapes.html" -> "torch.compiler_dynamic_shapes"
        match = re.search(r'/([^/]+?)\.html$', url)
        if match:
            raw = match.group(1)
            # Clean up: "torch.nn.MultiheadAttention" or "torch.compiler_dynamic_shapes"
            if raw.startswith("torch"):
                return raw.replace('_', '.')
            return raw
        return ""

    @staticmethod
    def _node_display_name(node: Node) -> str:
        name = node.name.strip()
        if name:
            return name
        if node.url:
            match = re.search(r'/([^/]+?)\.html$', node.url)
            if match:
                return match.group(1)
        return ""

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
