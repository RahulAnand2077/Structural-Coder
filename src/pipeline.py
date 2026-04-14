"""
pipeline.py — Global Pipeline
==============================
Single entry point for generating and comparing code.

System A: Standalone LLM  → query → Ollama (no context) → code
System B: GNN + LLM       → query → GNN predicts API path → Ollama (with context) → code

Usage:
    python -m src.pipeline "How to use torch.compile with dynamic shapes?"
    python -m src.pipeline --query "FSDP fully sharded data parallel" --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.config import MODEL_NAME
from src.graph_loader import CsvGraph
from src.query_encoder import GNNQueryPredictor
from src.helpers import (
    build_gnn_prompt,
    build_standalone_prompt,
    call_ollama,
    compute_final_score,
    grounding_score,
    token_hit_score,
    validity_score,
)


# ─────────────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────────────

class Pipeline:
    """Unified pipeline for System A vs System B code generation."""

    def __init__(
        self,
        nodes_csv: str = "data/nodes.csv",
        edges_csv: str = "data/edges.csv",
        model_path: str = "outputs/best_model.pt",
        embeddings_pt: str = "outputs/gnn_embeddings.pt",
    ) -> None:
        print("🔧 Loading knowledge graph...")
        self.graph = CsvGraph(nodes_csv, edges_csv)
        self.graph.load()
        print(f"   {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        print("🧠 Loading GNN model weights...")
        self.predictor = GNNQueryPredictor(
            model_path=model_path,
            embeddings_pt=embeddings_pt,
            nodes_csv=nodes_csv,
        )
        print("✅ Ready\n")

    def system_a(self, query: str, model: str = MODEL_NAME) -> dict:
        """System A: Standalone LLM — zero context."""
        prompt = build_standalone_prompt(query)
        answer, source = call_ollama(prompt, model)
        val = validity_score(answer)
        return {
            "system": "A_Standalone",
            "query": query,
            "model": model,
            "answer": answer,
            "source": source,
            "retrieval": 0.0,
            "grounding": 0.0,
            "validity": val,
            "final_score": 0.3 * val,
        }

    def system_b(self, query: str, model: str = MODEL_NAME, top_k: int = 10) -> dict:
        """System B: GNN predicts API path → LLM generates with context."""
        # GNN prediction
        path = self.predictor.predict_path(query, top_k=top_k, graph=self.graph)
        path_nodes = [self.graph.nodes[nid] for nid, _ in path.nodes if nid in self.graph.nodes]

        # LLM generation with GNN context
        prompt = build_gnn_prompt(query, path_nodes)
        answer, source = call_ollama(prompt, model)

        # Score
        ret = token_hit_score(query, path_nodes)
        grnd = grounding_score(answer, path_nodes, query=query)
        val = validity_score(answer)
        final = compute_final_score(ret, grnd, val)

        return {
            "system": "B_GNN_LLM",
            "query": query,
            "model": model,
            "answer": answer,
            "source": source,
            "gnn_path": [n.name for n in path_nodes[:5]],
            "gnn_scores": [(nid, f"{s:.3f}") for nid, s in path.nodes[:5]],
            "retrieval": ret,
            "grounding": grnd,
            "validity": val,
            "final_score": final,
        }

    def compare(self, query: str, model: str = MODEL_NAME) -> dict:
        """Run both systems and return comparison."""
        t0 = time.time()
        a = self.system_a(query, model)
        t1 = time.time()
        b = self.system_b(query, model)
        t2 = time.time()

        a["latency_sec"] = round(t1 - t0, 1)
        b["latency_sec"] = round(t2 - t1, 1)

        winner = "B" if b["final_score"] > a["final_score"] + 0.01 else \
                 "A" if a["final_score"] > b["final_score"] + 0.01 else "TIE"

        return {
            "query": query,
            "system_a": a,
            "system_b": b,
            "winner": winner,
        }


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    a = result["system_a"]
    b = result["system_b"]

    print(f"\n{'═'*80}")
    print(f" Query: \"{result['query']}\"")
    print(f"{'═'*80}")

    # System A
    print(f"\n{'─'*80}")
    print(f" ❌ System A: Standalone LLM")
    print(f"    Val: {a['validity']*100:.0f}% │ Final: {a['final_score']*100:.0f}% │ {a['latency_sec']}s")
    print(f"{'─'*80}")
    print(a["answer"][:500])

    # System B
    print(f"\n{'─'*80}")
    print(f" 🏆 System B: GNN + LLM")
    print(f"    Ret: {b['retrieval']*100:.0f}% │ Grnd: {b['grounding']*100:.0f}% │ Val: {b['validity']*100:.0f}% │ Final: {b['final_score']*100:.0f}% │ {b['latency_sec']}s")
    print(f"    GNN Path: {', '.join(b.get('gnn_path', []))}")
    print(f"{'─'*80}")
    print(b["answer"][:500])

    # Winner
    print(f"\n{'═'*80}")
    w = result["winner"]
    label = {"A": "System A (Standalone)", "B": "System B (GNN+LLM)", "TIE": "TIE"}[w]
    print(f" 🏁 Winner: {label}   (A={a['final_score']*100:.0f}% vs B={b['final_score']*100:.0f}%)")
    print(f"{'═'*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Structural-Coder: System A vs System B")
    parser.add_argument("query", nargs="?", help="PyTorch coding query")
    parser.add_argument("--query", "-q", dest="query_flag", help="Alternative query flag")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--nodes", default="data/nodes.csv")
    parser.add_argument("--edges", default="data/edges.csv")
    parser.add_argument("--model-path", default="outputs/best_model.pt")
    parser.add_argument("--embeddings", default="outputs/gnn_embeddings.pt")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        parser.error("Provide a query as positional argument or --query")

    pipe = Pipeline(
        nodes_csv=args.nodes,
        edges_csv=args.edges,
        model_path=args.model_path,
        embeddings_pt=args.embeddings,
    )

    result = pipe.compare(query, model=args.model)

    if args.json:
        # Remove verbose answer text for JSON output
        for key in ("system_a", "system_b"):
            result[key].pop("answer", None)
        print(json.dumps(result, indent=2))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
