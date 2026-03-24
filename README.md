# Structural-Coder

Graph-RAG code generation for PyTorch 2.x with active validation.

This repository reduces version-hallucination by grounding generation in a
PyTorch knowledge graph and validating generated code before use.

## What This Project Does

1. Loads graph data from `nodes.csv` and `edges.csv`.
2. Builds or loads GNN-based node embeddings.
3. Retrieves graph context for a query.
4. Generates PyTorch 2.x style code from retrieved context.
5. Runs active checks (C0, C1, C2, C3, C5).

## Clean Folder Layout

```text
Structural-Coder/
  src/
    graph_rag/                # Graph-RAG pipeline (embed, retrieve, generate)
    integration_pipeline/     # Validator + combined retrieval/repair pipeline
  benchmarks_downloads/       # Benchmark source folders and smoke runner
  datasets/                   # Query lists and benchmark-derived query files
  outputs/                    # Generated code, run JSONs, benchmark summaries
  artifacts/                  # Embedding cache and training artifacts
  docs/                       # Focused documentation
  run_graph_rag_pipeline.py   # Main Graph-RAG CLI
  run_combined_pipeline.py    # Retrieval + validation/repair CLI
  nodes.csv                   # Graph nodes snapshot
  edges.csv                   # Graph edges snapshot
```

## Core Files

- `src/graph_rag/gnn_encoder.py`: Graph tensor building + GraphSAGE-style training.
- `src/graph_rag/retriever.py`: Hybrid retrieval from graph structure + embeddings.
- `src/graph_rag/generator.py`: Retrieval-grounded code generation bridge.
- `src/graph_rag/pipeline.py`: End-to-end Graph-RAG orchestration.
- `src/integration_pipeline/validator.py`: Active validation checks and C5 handling.
- `benchmarks_downloads/run_benchmark_smoke.py`: Smoke benchmark runner + summary output.

## Quick Start

```bash
cd "/Volumes/Extreme SSD/tdl project/Structural-Coder"
source ../.venv/bin/activate
```

Run Graph-RAG:

```bash
python3 run_graph_rag_pipeline.py \
  --query "compile-safe transformer block" \
  --output-json outputs/graph_rag_run.json \
  --output-code outputs/graph_rag_run.py
```

Run smoke benchmarks:

```bash
python3 benchmarks_downloads/run_benchmark_smoke.py --smoke-n 5
```

Check smoke summary:

```bash
cat outputs/benchmarks_smoke/summary.json
```

## Validation Levels

- `C0`: syntax/import/static symbol checks.
- `C1`: CUDA guard and CPU fallback checks.
- `C2`: target hardware path alignment checks.
- `C3`: runtime torch availability check.
- `C5`: compile-safety probe (reports pass/fail/skipped status).

The smoke summary reports both:

- `smoke_validation_passed_with_skips`
- `smoke_validation_passed_strict`

## Notes

- Some benchmark families may remain unavailable in `downloaded` depending on access.
- Generated files in `outputs/` are run artifacts and can be regenerated anytime.
