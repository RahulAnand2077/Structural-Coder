# 🧪 `src/research_pipeline/` — The Research Manager

This module is the **experiment controller**. It strings together all the steps
from data loading to code generation to benchmarking.

---

## Files in This Folder

| File | What It Does (Plain English) |
|------|------------------------------|
| `pipeline.py` | Main class `CsvFirstResearchPipeline` — orchestrates all steps |

---

## 🔬 The 5 Steps It Runs

```
Step 1: Data Scrape        → Load nodes.csv and edges.csv
Step 2: Graph Construction → Analyse graph structure (counts, relations)
Step 3: GNN Training       → Train Graph Neural Network, save embeddings
Step 4: Ollama Integration → Generate answer using Graph-RAG + LLM (optional)
Step 5: Ablation Study     → Compare GNN retrieval vs Lexical retrieval
```

---

## 🧪 Ablation Study

An **ablation study** is a science experiment where you test each part alone to see
how much it contributes.

Here we compare:
- **GNN Retriever** — uses learned embeddings (our full system)
- **Lexical Retriever** — uses keyword matching only (simpler baseline)

Metrics collected per query:
- **Retrieval Score** — how well the retrieved nodes match the query tokens
- **Grounding Score** — how many retrieved symbols actually appear in the generated code
- **Validity Score** — how many (C0–C5) code checks pass
- **Final Score** = `0.4 × retrieval + 0.6 × (0.5 × grounding + 0.5 × validity)`


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
