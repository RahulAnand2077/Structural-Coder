"""
Graph-RAG Retriever — CLI Entry Point
======================================

This is the main script you run from the terminal.
It wires together: graph loading → embeddings → retrieval → JSON output.

Usage examples:

    # With local GNN training:
    python3 run_graph_rag_pipeline.py \\
        --allow-csv-gnn-training \\
        --query "compile-safe transformer block" \\
        --output-json outputs/result.json

    # With Mohit's pre-trained embeddings:
    python3 run_graph_rag_pipeline.py \\
        --external-embeddings "../Structural-Coder-mohit/artifacts/research_gnn_embeddings.json" \\
        --query "compile-safe transformer block" \\
        --output-json outputs/result.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.graph_rag import StructuralCoderGraphRAG


def build_parser() -> argparse.ArgumentParser:
    """Define all CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Graph-RAG retriever with GNN embeddings")

    # --- Input files ---
    parser.add_argument("--nodes", default="nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="edges.csv", help="Path to edges.csv")
    parser.add_argument("--query", required=True, help="Natural language query for retrieval")

    # --- Retrieval tuning ---
    parser.add_argument("--top-k", type=int, default=20, help="How many nodes to return")
    parser.add_argument("--seed-k", type=int, default=4, help="How many seed nodes to start from")
    parser.add_argument("--expansion-hops", type=int, default=1, help="How many hops to expand from seeds")

    # --- Embedding source ---
    parser.add_argument("--embedding-cache", default="artifacts/gnn_embeddings.json",
                        help="Where to save/load local GNN embeddings")
    parser.add_argument("--external-embeddings", default=None,
                        help="Path to Mohit's pre-trained GNN embeddings (skips local training)")

    # --- Local GNN training options (only used without --external-embeddings) ---
    parser.add_argument("--retrain-gnn", action="store_true", help="Force retrain even if cache exists")
    parser.add_argument("--allow-csv-gnn-training", action="store_true",
                        help="Required safety flag to allow training from CSV snapshots")
    parser.add_argument("--gnn-epochs", type=int, default=20)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-out-dim", type=int, default=96)
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Input feature vector dimension")

    # --- Output ---
    parser.add_argument("--output-json", default=None, help="Save full JSON result to this file")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Step 1: Load the knowledge graph from CSVs.
    pipeline = StructuralCoderGraphRAG(args.nodes, args.edges)

    # Step 2: Load embeddings (either external or local).
    if args.external_embeddings:
        # Use Mohit's pre-trained GNN embeddings.
        pipeline.load_external_embeddings(
            external_path=args.external_embeddings,
            feature_dim=args.feature_dim,
        )
    else:
        # Train or load local GNN embeddings.
        pipeline.prepare_embeddings(
            embedding_cache=args.embedding_cache,
            retrain=args.retrain_gnn,
            epochs=args.gnn_epochs,
            hidden_dim=args.gnn_hidden_dim,
            out_dim=args.gnn_out_dim,
            feature_dim=args.feature_dim,
            allow_csv_training=args.allow_csv_gnn_training,
        )

    # Step 3: Run retrieval.
    result = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        seed_k=args.seed_k,
        expansion_hops=args.expansion_hops,
    )

    # Step 4: Output results.
    json_out = pipeline.to_json(result)
    print(json_out)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_out, encoding="utf-8")


if __name__ == "__main__":
    main()
