"""
Graph-RAG Pipeline — Orchestrator
===================================

This file is the brain of the retriever. It connects everything together:

    nodes.csv + edges.csv
           │
           ▼
    ┌─────────────┐
    │ CsvGraph     │  (from graph_loader.py)
    │ loads nodes  │
    │ and edges    │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ GNN Encoder  │  (from gnn_encoder.py)
    │ trains or    │
    │ loads node   │
    │ embeddings   │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ Retriever    │  (from retriever.py)
    │ finds best   │
    │ nodes for a  │
    │ query        │
    └──────┬──────┘
           │
           ▼
    JSON output with retrieved symbols

Classes:
    StructuralCoderGraphRAG — main pipeline class
    GraphRAGRetrieveResult  — result dataclass
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .graph_loader import CsvGraph

from .gnn_encoder import (
    _hashed_text_features,
    build_graph_tensor_data,
    compute_graph_hash,
    load_embeddings,
    save_embeddings,
    train_gnn_embeddings,
)
from .retriever import GraphRAGRetriever


# ──────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────

@dataclass
class GraphRAGRetrieveResult:
    """What the pipeline returns after a retrieval run."""
    query: str                      # the original query
    seed_nodes: list[str]           # starting nodes chosen by embedding similarity
    retrieved_symbols: list[str]    # final ranked API symbol names
    retrieved_nodes: int            # total nodes in the subgraph
    retrieved_edges: int            # total edges in the subgraph


# ──────────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────────

class StructuralCoderGraphRAG:
    """End-to-end Graph-RAG retrieval pipeline.

    Usage:
        pipeline = StructuralCoderGraphRAG("nodes.csv", "edges.csv")
        pipeline.prepare_embeddings(...)   # or load_external_embeddings(...)
        result = pipeline.run(query="compile-safe transformer block")
        print(pipeline.to_json(result))
    """

    def __init__(self, nodes_csv: str | Path, edges_csv: str | Path) -> None:
        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)

        # Load the knowledge graph from CSV files.
        self.graph = CsvGraph(self.nodes_csv, self.edges_csv)
        self.graph.load()

        # These get filled in by prepare_embeddings() or load_external_embeddings().
        self._node_ids: list[int] | None = None
        self._embeddings = None           # GNN-learned vectors (for re-ranking)
        self._input_features = None       # hash-bag vectors  (for seed selection)

    # ──────────────────────────────────────────────────────────────
    # Option A: Train or load local GNN embeddings
    # ──────────────────────────────────────────────────────────────

    def prepare_embeddings(
        self,
        embedding_cache: str | Path,
        retrain: bool = False,
        epochs: int = 20,
        hidden_dim: int = 128,
        out_dim: int = 96,
        feature_dim: int = 128,
        allow_csv_training: bool = False,
    ) -> None:
        """Train GNN embeddings from CSV graph, or load from cache.

        The cache includes a hash of the CSV files. If the graph data has
        changed since the cache was created, it automatically retrains.
        """
        cache_path = Path(embedding_cache)
        current_hash = compute_graph_hash(self.nodes_csv, self.edges_csv)

        # Try loading from cache first.
        if not retrain:
            try:
                node_ids, embeddings, input_features, cached_hash = load_embeddings(cache_path)
            except FileNotFoundError:
                node_ids, embeddings, input_features, cached_hash = None, None, None, ""

            if node_ids is not None:
                if cached_hash and cached_hash != current_hash:
                    print(
                        f"[graph-rag] Cache is stale "
                        f"(cached={cached_hash}, current={current_hash}). Retraining."
                    )
                else:
                    # Cache is valid — use it.
                    self._node_ids = node_ids
                    self._embeddings = embeddings
                    self._input_features = input_features
                    return

        # No valid cache — train from scratch.
        data = build_graph_tensor_data(
            self.graph,
            feature_dim=feature_dim,
            allow_csv_training=allow_csv_training,
        )
        _, embeddings = train_gnn_embeddings(
            data,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            epochs=epochs,
        )

        # Save to cache for next time.
        save_embeddings(
            cache_path,
            data.node_ids,
            embeddings,
            input_features=data.x,
            graph_hash=current_hash,
        )

        self._node_ids = data.node_ids
        self._embeddings = embeddings
        self._input_features = data.x

    # ──────────────────────────────────────────────────────────────
    # Option B: Load Mohit's pre-trained GNN embeddings
    # ──────────────────────────────────────────────────────────────

    def load_external_embeddings(
        self,
        external_path: str | Path,
        feature_dim: int = 128,
    ) -> None:
        """Load pre-trained embeddings from Mohit's GNN encoder.

        Mohit's embeddings capture graph structure better (trained with
        torch_geometric on Neo4j). We still build our own hash-bag features
        for seed selection so queries are compared in the right vector space.
        """
        ext = Path(external_path)
        if not ext.exists():
            raise FileNotFoundError(f"External embeddings not found: {ext}")

        # Load Mohit's embeddings (JSON with "node_ids" + "embeddings").
        raw = json.loads(ext.read_text(encoding="utf-8"))
        node_ids = [int(n) for n in raw["node_ids"]]

        import torch
        embeddings = torch.tensor(raw["embeddings"], dtype=torch.float32)

        # Build our own hash-bag input features for seed selection.
        # (Mohit's embeddings are in GNN space; queries are in hash-bag space.
        #  We need hash-bag features so seed selection compares like-with-like.)
        input_features = torch.zeros((len(node_ids), feature_dim), dtype=torch.float32)
        for i, nid in enumerate(node_ids):
            node = self.graph.nodes.get(nid)
            if node is not None:
                text = f"{node.label} {node.name} {node.url}"
                input_features[i] = _hashed_text_features(text, feature_dim)

        self._node_ids = node_ids
        self._embeddings = embeddings
        self._input_features = input_features
        print(
            f"[graph-rag] Loaded external embeddings: "
            f"{len(node_ids)} nodes × {embeddings.size(1)} dims from {ext.name}"
        )

    # ──────────────────────────────────────────────────────────────
    # Run retrieval
    # ──────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        top_k: int = 20,
        seed_k: int = 4,
        expansion_hops: int = 1,
    ) -> GraphRAGRetrieveResult:
        """Run the full retrieval pipeline for a query.

        Returns a GraphRAGRetrieveResult with the ranked symbols.
        """
        if self._node_ids is None or self._embeddings is None:
            raise RuntimeError("Embeddings not ready. Call prepare_embeddings() or load_external_embeddings() first.")

        # Create a retriever with our graph + embeddings.
        retriever = GraphRAGRetriever(
            graph=self.graph,
            node_ids=self._node_ids,
            embeddings=self._embeddings,
            input_features=self._input_features,
        )

        # Retrieve relevant context.
        context = retriever.retrieve(
            query=query,
            top_k=top_k,
            seed_k=seed_k,
            expansion_hops=expansion_hops,
        )

        # Extract unique symbol names from retrieved nodes.
        symbol_names: list[str] = []
        seen: set[str] = set()
        for node in context.nodes:
            name = node.name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            symbol_names.append(name)

        return GraphRAGRetrieveResult(
            query=query,
            seed_nodes=context.seed_nodes,
            retrieved_symbols=symbol_names[: max(1, top_k)],
            retrieved_nodes=len(context.nodes),
            retrieved_edges=len(context.edges),
        )

    # ──────────────────────────────────────────────────────────────
    # JSON output
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def to_json(result: GraphRAGRetrieveResult) -> str:
        """Convert result to a JSON string."""
        return json.dumps(asdict(result), indent=2)
