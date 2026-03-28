# 🔍 `src/graph_rag/` — The Smart Searcher (Graph-RAG Engine)

This module is the **brain** of the system. It uses a **Heterogeneous Graph Neural Network (HGTConv)** to search the 24K-node PyTorch knowledge graph by meaning — not just keywords.

---

## Files in This Folder

| File | What It Does |
|------|--------------|
| `graph_loader.py` | Reads `nodes.csv` + `edges.csv` and builds the graph in memory |
| `gnn_encoder.py` | Trains the Heterogeneous GNN (HGTConv) to give every node a smart embedding |
| `retriever.py` | Uses those embeddings to find nodes closest to your question |
| `pipeline.py` | Runs all steps in the correct order |
| `generator.py` | Converts retrieved nodes into a code answer |

---

## 🧠 GNN Architecture (HeteroGraphEncoder)

The GNN is based on **HGTConv (Heterogeneous Graph Transformer)** and processes 9 distinct node types:
`API_Class`, `API_Method`, `API_Function`, `API_Parameter`, `API_Endpoint`, `CodeSnippet`, `Concept`, `PyTorchConcept`, `DeprecatedAPI`

```
Raw CSV Text Features (hashed → 128-dim)
         │
         ▼
[Input Projection]  ← Per-node-type Linear + LayerNorm
         │
         ▼
[HGTConv Layer 1]  ← 4-head attention across typed edges
         │
[HGTConv Layer 2]  ← Residual passthrough for isolated types
         │
[HGTConv Layer 3]
         │
         ▼
[Jumping Knowledge (JK)]  ← Concat all layer outputs
         │
         ▼
[Final Linear]  → 96-dim L2-normalized embedding per node
```

### Training: What Makes It Smarter

| Upgrade | What It Does |
|---------|--------------|
| **Supervision Isolation** | Only 6 semantic relations used for loss (`IMPLEMENTS`, `CONTAINS`, `HAS_PARAM`, `CALLS`, `RELATED_TO`, `REPLACES`). Structural edges (`EXPLAINS`, `REFERENCES`) used only for message-passing. |
| **Hard Negative Sampling** | Samples degree-weighted negative edges filtered to exclude true positives — prevents contradictory training signal. |
| **Gradient Clipping** | `max_norm=2.0` stabilizes HGT attention during backprop. |
| **LowRankBilinearDecoder** | Factored scorer `(z@U)·(z@V)` with rank=32 — avoids over-parameterising on rare relations. |

---

## 🗺️ Retrieval Strategy

1. **Seed Selection** — Find the 4 best starting nodes using GNN cosine similarity
2. **Graph Expansion** — Follow 1-hop edges to gather related neighbours
3. **Hybrid Re-Ranking** — Combine GNN score + keyword overlap + graph degree → final top-K list

---

## 💾 Embedding Cache

After the first run, embeddings are saved to `artifacts/research_gnn_embeddings.json`. On all subsequent runs, this file is loaded instantly — no retraining needed.

To force a full retrain, simply delete the cache file:
```bash
rm artifacts/research_gnn_embeddings.json
```

---

### 💡 Architectural Note: Decoupled GNN & Pipeline
The **GNN Encoder** and the **Integration Pipeline** are strictly decoupled. Any changes to the GNN (dimensions, training algorithm, node attributes) will **not** break the pipeline. The pipeline simply reads the resulting `research_gnn_embeddings.json` artifact.
