# 📓 `notebooks/` — Jupyter Notebooks (Experiments & Exploration)

These notebooks were used during research and development. They are great for
understanding how different parts of the system were built.

---

## Notebooks

| Notebook | What It Explores |
|----------|-----------------|
| `creating_embed.ipynb` | How we create embeddings from the knowledge graph |
| `gnn_encoder.ipynb` | Experiments with training the Graph Neural Network |
| `web_crawler.ipynb` | How the original PyTorch docs were scraped to build nodes/edges |

---

## How to Open a Notebook

```bash
# Install Jupyter if needed
pip install jupyter

# Launch
jupyter notebook notebooks/
```

> **Note**: These notebooks are exploratory. The production code lives in `src/`.


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
