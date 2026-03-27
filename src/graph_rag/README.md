# 🔍 `src/graph_rag/` — The Smart Searcher (Graph-RAG Engine)

This module is the **brain** of the system. It knows how to search our PyTorch
knowledge graph like a detective — starting from the most relevant entry point,
then following connections to discover related concepts.

---

## Files in This Folder

| File | What It Does (Plain English) |
|------|------------------------------|
| `graph_loader.py` | Reads `nodes.csv` and `edges.csv` and builds the graph in memory |
| `gnn_encoder.py` | Trains a Graph Neural Network to give every node a smart "fingerprint" |
| `retriever.py` | Uses those fingerprints to find nodes closest to your question |
| `pipeline.py` | Runs all the above steps in the right order |
| `generator.py` | Converts retrieved nodes into a code answer |

---

## 🧠 How the Graph Neural Network (GNN) Works

Imagine every PyTorch function (`torch.compile`, `nn.TransformerEncoder`, etc.) as
a person. People who work together closely become more similar over time.

The GNN does exactly that:
1. Starts with a rough description (a "hash bag") for each node
2. Passes messages along edges (e.g., "TransformerEncoder CALLS TransformerEncoderLayer")
3. After training, nodes that are semantically connected have similar vector embeddings

This allows us to search by **meaning**, not just keywords.

---

## 🗺️ Retrieval Strategy

1. **Seed Selection** — find the 4 best starting nodes using GNN cosine similarity
2. **Graph Expansion** — follow 1-hop edges to gather related neighbours
3. **Hybrid Re-Ranking** — combine GNN score + keyword hits + graph degree to produce final top-K list


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
