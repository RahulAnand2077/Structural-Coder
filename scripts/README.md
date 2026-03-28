# ▶️ `scripts/` — Entry Point Runner Scripts

These are the **main scripts** you run to use the system. Think of them as the
"start buttons" for different parts of the project.

---

## Scripts



### `run_research_pipeline.py` — Full Research Experiment

Runs the complete research workflow: load data → train GNN → retrieve → integrate
with optional LLM → run ablation study.

```bash
python scripts/run_research_pipeline.py \
  --nodes data/nodes.csv \
  --edges data/edges.csv \
  --query "compile-safe transformer block"

# Compare GNN vs lexical retrieval experiment:
python scripts/run_research_pipeline.py \
  --nodes data/nodes.csv \
  --edges data/edges.csv \
  --run-experiment
```

---

## ⚙️ Common Options

| Flag | Default | What It Does |
|------|---------|-------------|
| `--nodes` | `nodes.csv` | Path to the nodes CSV |
| `--edges` | `edges.csv` | Path to the edges CSV |
| `--query` | `"compile-safe transformer block"` | Your coding question |
| `--top-k` | 20 | How many results to retrieve |
| `--epochs` | 20 | GNN training epochs |
| `--ollama-model` | `llama3.2` | Which Ollama model to use |


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
