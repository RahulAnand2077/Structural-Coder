# 📦 `src/` — All Source Code

This folder contains the three main Python modules that power the system.
Think of them as three workers in a factory:

---

## 🏭 The Three Workers

### 1. `graph_rag/` — The Smart Searcher
Finds the most relevant PyTorch API symbols for your question using:
- **GNN Encoder** — trains a Graph Neural Network on the knowledge graph
- **Retriever** — searches the graph using GNN embeddings + keyword matching
- **Generator** — formats retrieved results into a prompt for code generation

→ See [`src/graph_rag/README.md`](graph_rag/README.md)

---

### 2. `integration_pipeline/` — The Quality Inspector
Validates and self-heals the generated code:
- **Validator** — runs 6 checks (C0: imports, C1: syntax, C2: API existence, C3: compile probe, C4: device guard, C5: dtype safety)
- **Retriever** — an alternative keyword-based graph search
- **Graph Loader** — loads `nodes.csv` and `edges.csv` into memory

→ See [`src/integration_pipeline/README.md`](integration_pipeline/README.md)

---

### 3. `research_pipeline/` — The Manager
Orchestrates the full experiment:
- Loads data
- Trains GNN
- Calls the retriever
- Runs ablation experiments
- Compares GNN vs lexical retrieval

→ See [`src/research_pipeline/README.md`](research_pipeline/README.md)


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
