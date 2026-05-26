# 🧪 `tests/` — Automated Tests

This folder contains automated tests that verify each component works correctly.

---

## Running the Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

---

## What Gets Tested

| Test File | What It Checks |
|-----------|---------------|
| `test_graph_rag.py` | GNN encoder, retriever, pipeline end-to-end |

The tests use a small 5-node fake graph (so they run fast) to verify:
- The GNN can train without crashing
- The retriever returns the expected number of nodes
- Retrieval context contains valid Node objects


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
