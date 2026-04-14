# 📊 Benchmark Guide

**Path:** `docs/benchmark_guide.md`

---

## What Gets Benchmarked

We compare **two systems** head-to-head on the same PyTorch coding queries:

| System | How It Works |
|--------|-------------|
| **Structural-Coder** (Ours) | Graph-RAG retrieval → top 5 nodes → advisory LLM prompt → code validation |
| **Standalone LLM** | Same Ollama model, zero context, pure training-data memory |

---

## How to Run

### Interactive (one query at a time)
```bash
../.venv/bin/python benchmark/interactive_comparison.py
```
Type any PyTorch question. See both outputs side-by-side with scores.

### Batch (all 10 queries, full report)
```bash
../.venv/bin/python benchmark/run_comparison.py
```
Produces `benchmark/outputs/comparison_report.md` and `comparison.json`.

---

## Scoring Formula

```
Final = 0.4 × Retrieval + 0.6 × (0.5 × Grounding + 0.5 × Validity)
```

| Metric | Range | What It Measures |
|--------|-------|-----------------|
| **Retrieval** | 0.0–1.0 | Fraction of query keywords covered by retrieved nodes |
| **Grounding** | 0.0–1.0 | Fraction of retrieved API names that appear in the generated code |
| **Validity** | 0.0–1.0 | Fraction of C0–C5 checks that pass (see [validation_checks.md](validation_checks.md)) |

### Standalone scoring caveat
The standalone LLM gets `Retrieval = 0.0` and `Grounding = 0.0` because it never sees retrieved context. Its score is purely `0.6 × (0.5 × 0.0 + 0.5 × Validity) = 0.3 × Validity`. This is intentionally asymmetric — the standalone cannot do retrieval by design.

---

## Query File

**Path:** `benchmark/queries/queries.json`

A JSON array of 10 test prompts covering diverse PyTorch topics:
- Dynamic shapes, torch.compile
- Flash attention, gradient checkpointing
- Custom autograd functions
- FSDP data parallel
- Mixed precision training

---

## Files

| File | Path | Purpose |
|------|------|---------|
| Interactive tester | `benchmark/interactive_comparison.py` | Live side-by-side terminal UI |
| Batch runner | `benchmark/run_comparison.py` | Runs all queries, writes report |
| Query bank | `benchmark/queries/queries.json` | 10 test prompts |
| Results | `benchmark/outputs/` | Generated reports (gitignored) |
