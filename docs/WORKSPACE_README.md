# TDL Project Workspace

This document mirrors the workspace-level README from the parent folder so it is versioned in this repository.

## Start Here

1. Main project folder: `Structural-Coder`
2. Main project documentation: `README.md`

## Workspace Layout

- `Structural-Coder`: production Graph-RAG + validation pipeline.
- `src`: older prototype code from early iterations in the parent workspace.
- `data`: data assets used by earlier prototype runs in the parent workspace.

## Data Flow and Architecture (Child to Pro)

### Level 1 (Child View)

Think of this project like a smart homework helper:

1. You ask a question.
2. It looks inside a big PyTorch knowledge map.
3. It writes code using what it found.
4. It checks if the code looks safe and correct.
5. It saves the result.

### Level 2 (School View)

The system has five simple stages:

1. Input: query text (for example, "compile-safe transformer block").
2. Retrieval: find relevant API nodes and edges from graph data (`nodes.csv`, `edges.csv`).
3. Generation: create PyTorch code based on retrieved context.
4. Validation: run C0/C1/C2/C3/C5 checks.
5. Output: JSON report + generated `.py` file in `outputs/`.

### Level 3 (Engineering View)

End-to-end Graph-RAG path:

1. Graph loading:
   - `src/integration_pipeline/graph_loader.py`
2. Embedding preparation (GraphSAGE-style):
   - `src/graph_rag/gnn_encoder.py`
3. Retrieval over graph + embeddings:
   - `src/graph_rag/retriever.py`
4. Generation bridge:
   - `src/graph_rag/generator.py`
5. Validation and reporting:
   - `src/integration_pipeline/validator.py`
6. Orchestration CLI:
   - `run_graph_rag_pipeline.py`

### Level 4 (Pro/System Design View)

Architecture layers:

1. Data Layer:
   - Documentation-derived graph snapshots (`nodes.csv`, `edges.csv`).
2. Retrieval Layer:
   - Hybrid ranking from graph structure + learned embeddings.
3. Generation Layer:
   - Retrieval-conditioned code synthesis.
4. Assurance Layer:
   - Validation pyramid:
     - C0: syntax/import/static symbol checks
     - C1: guarded CUDA fallback checks
     - C2: target hardware path alignment
     - C3: runtime torch availability
     - C5: compile-safety probe status
5. Evaluation Layer:
   - Smoke benchmark runner and summary in `benchmarks_downloads/run_benchmark_smoke.py`

Operational outputs:

1. Per-run payloads in `outputs/`
2. Smoke summary in `outputs/benchmarks_smoke/summary.json`
3. Embedding and other artifacts in `artifacts/`

## What To Run

```bash
cd "/Volumes/Extreme SSD/tdl project/Structural-Coder"
source ../.venv/bin/activate
python3 run_graph_rag_pipeline.py --query "compile-safe transformer block" --output-json outputs/run.json --output-code outputs/run.py
python3 benchmarks_downloads/run_benchmark_smoke.py --smoke-n 5
```

## Notes

- If you are reviewing implementation status, use `README.md` in this repository.
- The parent workspace root contains older prototype folders not used by the main runtime path.
