# 🗑️ Deprecated Code

**Path:** `docs/deprecated_code.md`

---

## What Was Moved to `_deprecated/`

These files were identified as **dead-end code** — defined but never imported or called by any active part of the system.

### `_deprecated/combined_pipeline.py`
**Original path:** `src/integration_pipeline/pipeline.py`

**What it was:** `CombinedPipeline` — an alternative orchestrator that combined retrieval + validation + self-healing into one class.

**Why it's dead:** The active system uses `CsvFirstResearchPipeline` (in `src/research_pipeline/pipeline.py`) instead. `CombinedPipeline` was never imported by any benchmark, script, or other module. It was only exported via `src/integration_pipeline/__init__.py`, but nothing ever consumed that export.

**Dependency analysis:**
- `CombinedPipeline` imports: `graph_loader.CsvGraph`, `retriever.GraphRetriever`, `validator.ActiveValidator` — all still active (used elsewhere)
- `CombinedPipeline` is imported by: **nothing** (0 callers)

---

### `_deprecated/scripts_README.md`
**Original path:** `scripts/README.md`

**Why it's dead:** The `scripts/` folder contained only this README — no actual scripts. The README described a CLI runner that was never implemented.

---

### `_deprecated/test_graph_rag.py`
**Original path:** `tests/test_graph_rag.py`

**Why it's dead:** The test file imports `RankingWeights` from `src.graph_rag.retriever` and `CsvGraph` from `src.graph_rag.graph_loader` — neither of which exist in the codebase. The test would crash on import. It was written for an earlier version of the retriever that had a different API.

---

## Files That Were NOT Moved

| File | Why It Stays |
|------|-------------|
| `src/integration_pipeline/retriever.py` (`GraphRetriever`) | Imported by `research_pipeline/pipeline.py` for ablation study baseline |
| `src/integration_pipeline/validator.py` (`ActiveValidator`) | Imported by `research_pipeline/pipeline.py` for C0–C5 scoring |
| `src/integration_pipeline/graph_loader.py` (`CsvGraph`) | Core dependency — imported by 4 modules |
| `src/graph_rag/python_chunker.py` | New feature — usable standalone |
