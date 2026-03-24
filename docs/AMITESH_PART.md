# Amitesh Part: Quick Guide

This document focuses only on the implemented Graph-RAG path.

## Core idea

Use graph structure + learned node embeddings to retrieve PyTorch 2.x context,
then generate and validate code with guardrails.

## Files you should present

- `src/graph_rag/gnn_encoder.py`
- `src/graph_rag/retriever.py`
- `src/graph_rag/generator.py`
- `src/graph_rag/pipeline.py`
- `run_graph_rag_pipeline.py`

## Minimal demo command

```bash
python run_graph_rag_pipeline.py \
  --query "compile-safe transformer block" \
  --output-json outputs/graph_rag_demo.json \
  --output-code outputs/graph_rag_demo.py
```

## What each step does

1. Load `nodes.csv` + `edges.csv`
2. Build/train (or load cached) GNN node embeddings
3. Retrieve graph context with hybrid ranking
4. Generate PyTorch 2.x code from retrieved context
5. Run validation checks (C0/C1-C3/C5)

## Known caveat

On some macOS setups, C5 compile probe can fail due to local toolchain/runtime,
which is environment-specific and not necessarily a graph-break in generated logic.
