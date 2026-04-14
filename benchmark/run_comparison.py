"""
run_comparison.py
=================
Batch benchmark: System A (Standalone LLM) vs System B (GNN + LLM)
Runs all queries from queries.json and produces comparison report.

Usage:
  python benchmark/run_comparison.py --model llama3.1:8b
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_NAME
from src.pipeline import Pipeline

SEPARATOR = "═" * 80


def main() -> None:
    parser = argparse.ArgumentParser(description="System A vs System B Batch Benchmark")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model")
    parser.add_argument("--queries-file", default="benchmark/queries/queries.json")
    parser.add_argument("--output-json", default="benchmark/outputs/comparison.json")
    parser.add_argument("--output-report", default="benchmark/outputs/comparison_report.md")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))

    print(SEPARATOR)
    print("  System A (Standalone LLM) vs System B (GNN + LLM)")
    print(f"  Model: {args.model} | Queries: {len(queries)}")
    print(SEPARATOR)

    pipe = Pipeline()
    results = []

    for qi, query in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"  [{qi}/{len(queries)}] \"{query}\"")
        print(f"{'─'*80}")

        result = pipe.compare(query, model=args.model)
        a = result["system_a"]
        b = result["system_b"]

        print(f"  [A] Val: {a['validity']*100:.0f}% | Final: {a['final_score']*100:.0f}% ({a['latency_sec']}s)")
        print(f"  [B] Ret: {b['retrieval']*100:.0f}% | Grnd: {b['grounding']*100:.0f}% | Val: {b['validity']*100:.0f}% | Final: {b['final_score']*100:.0f}% ({b['latency_sec']}s)")
        print(f"      GNN Path: {', '.join(b.get('gnn_path', [])[:3])}")
        print(f"      Winner: {result['winner']}")

        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────
    a_scores = [r["system_a"]["final_score"] for r in results]
    b_scores = [r["system_b"]["final_score"] for r in results]
    avg = lambda v: sum(v)/len(v) if v else 0.0

    wins_b = sum(1 for r in results if r["winner"] == "B")
    wins_a = sum(1 for r in results if r["winner"] == "A")
    ties = sum(1 for r in results if r["winner"] == "TIE")

    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(f"{SEPARATOR}")
    print(f"  Avg Final Score — A: {avg(a_scores)*100:.0f}% | B: {avg(b_scores)*100:.0f}%")
    print(f"  Wins — B: {wins_b} | A: {wins_a} | Ties: {ties}")
    print(SEPARATOR)

    # ── Save outputs ─────────────────────────────────────────────────
    # Strip verbose answers for JSON
    save_results = []
    for r in results:
        sr = {"query": r["query"], "winner": r["winner"]}
        for key in ("system_a", "system_b"):
            sr[key] = {k: v for k, v in r[key].items() if k != "answer"}
        save_results.append(sr)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(save_results, indent=2))

    out_md = Path(args.output_report)
    out_md.write_text(_markdown_report(results, args.model, avg, wins_a, wins_b, ties))

    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_md}\n")


def _markdown_report(results, model, avg, wins_a, wins_b, ties) -> str:
    a_scores = [r["system_a"]["final_score"] for r in results]
    b_scores = [r["system_b"]["final_score"] for r in results]
    a_val = [r["system_a"]["validity"] for r in results]
    b_val = [r["system_b"]["validity"] for r in results]
    b_ret = [r["system_b"]["retrieval"] for r in results]
    b_grnd = [r["system_b"]["grounding"] for r in results]

    lines = [
        "# Standalone LLM vs GNN + LLM — Benchmark Report", "",
        "## Setup", "",
        f"- **Model**: {model}",
        f"- **Queries**: {len(results)}",
        f"- **Graph**: 24,485 nodes, 47,958 edges", "",
        "## Average Scores", "",
        "| System | Retrieval | Grounding | Validity | Final |",
        "|--------|-----------|-----------|----------|-------|",
        f"| A: Standalone | — | — | {avg(a_val)*100:.0f}% | {avg(a_scores)*100:.0f}% |",
        f"| B: GNN + LLM | {avg(b_ret)*100:.0f}% | {avg(b_grnd)*100:.0f}% | {avg(b_val)*100:.0f}% | {avg(b_scores)*100:.0f}% |",
        "",
        f"**Wins**: B={wins_b} | A={wins_a} | Ties={ties}", "",
        "## Per-Query Results", "",
    ]

    for r in results:
        a = r["system_a"]
        b = r["system_b"]
        delta = b["final_score"] - a["final_score"]
        emoji = "🏆 B" if r["winner"] == "B" else ("🏅 A" if r["winner"] == "A" else "🤝 Tie")
        lines.append(f"### `{r['query']}`")
        lines.append(f"- **A**: Final={a['final_score']*100:.0f}% | Val={a['validity']*100:.0f}%")
        lines.append(f"- **B**: Final={b['final_score']*100:.0f}% | Ret={b['retrieval']*100:.0f}% | Grnd={b['grounding']*100:.0f}% | Val={b['validity']*100:.0f}%")
        if b.get("gnn_path"):
            lines.append(f"  - GNN Path: {', '.join(b['gnn_path'])}")
        lines.append(f"  - **{emoji}** (Δ={delta:+.0%})")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
