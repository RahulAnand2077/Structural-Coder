from __future__ import annotations

import argparse
from pathlib import Path

from src.graph_rag import StructuralCoderGraphRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end Graph-RAG pipeline with GNN encoder + generator integration")
    parser.add_argument("--nodes", default="nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="edges.csv", help="Path to edges.csv")
    parser.add_argument("--query", required=True, help="Query for retrieval and generation")

    parser.add_argument("--top-k", type=int, default=20, help="Retrieved context node count")
    parser.add_argument("--seed-k", type=int, default=4, help="Top seed nodes from embedding similarity")
    parser.add_argument("--expansion-hops", type=int, default=1, help="Neighbor expansion hops from seed set")

    parser.add_argument("--embedding-cache", default="artifacts/gnn_embeddings.json")
    parser.add_argument("--retrain-gnn", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=20)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-out-dim", type=int, default=96)

    parser.add_argument("--target-hardware", default="H100")
    parser.add_argument("--no-validate", action="store_true")

    parser.add_argument("--output-json", default=None, help="Optional file path to save full JSON result")
    parser.add_argument("--output-code", default=None, help="Optional file path to save generated code")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pipeline = StructuralCoderGraphRAG(args.nodes, args.edges)
    pipeline.prepare_embeddings(
        embedding_cache=args.embedding_cache,
        retrain=args.retrain_gnn,
        epochs=args.gnn_epochs,
        hidden_dim=args.gnn_hidden_dim,
        out_dim=args.gnn_out_dim,
    )

    result = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        seed_k=args.seed_k,
        expansion_hops=args.expansion_hops,
        target_hardware=args.target_hardware,
        validate=(not args.no_validate),
    )

    json_out = pipeline.to_json(result)
    print(json_out)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_out, encoding="utf-8")

    if args.output_code:
        code_path = Path(args.output_code)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(result.generated_code, encoding="utf-8")


if __name__ == "__main__":
    main()
