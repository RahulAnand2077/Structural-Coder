from __future__ import annotations

import argparse
from pathlib import Path

from src.integration_pipeline import CombinedPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combined pipeline (his code + our code)")
    parser.add_argument("--nodes", default="nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="edges.csv", help="Path to edges.csv")
    parser.add_argument("--query", required=True, help="Query for retrieval")
    parser.add_argument("--mode", default="detective", choices=["microscope", "detective", "telescope"])
    parser.add_argument("--top-k", type=int, default=80, help="Top ranked nodes to keep")
    parser.add_argument(
        "--ranking-profile",
        default="balanced",
        choices=["balanced", "precision", "coverage"],
        help="Ranking weights profile",
    )
    parser.add_argument("--code-file", default=None, help="Optional generated code file for active validation")
    parser.add_argument("--target-hardware", default="H100")
    parser.add_argument("--no-self-heal", action="store_true")
    parser.add_argument("--max-repair-iterations", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    code = None
    if args.code_file:
        code = Path(args.code_file).read_text(encoding="utf-8")

    pipeline = CombinedPipeline(args.nodes, args.edges)
    result = pipeline.run(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        ranking_profile=args.ranking_profile,
        generated_code=code,
        target_hardware=args.target_hardware,
        enable_self_heal=(not args.no_self_heal),
        max_repair_iterations=args.max_repair_iterations,
    )
    print(pipeline.to_json(result))


if __name__ == "__main__":
    main()
