# 📊 `data/` — The PyTorch Knowledge Graph

This folder contains the **core database** that the whole system is built on.

---

## Files

| File | Size | What It Contains |
|------|------|-----------------|
| `nodes.csv` | ~660 KB | 24,485 PyTorch API concepts (functions, classes, parameters) |
| `edges.csv` | ~960 KB | 47,958 relationships between those concepts |

---

## 🕸️ What Is a Knowledge Graph?

Think of it like Wikipedia, but for PyTorch code:
- Every **node** is one concept (e.g. `torch.compile`, `nn.TransformerEncoder`)
- Every **edge** is a relationship (e.g. `TransformerEncoder CALLS TransformerEncoderLayer`)

### Node Columns (`nodes.csv`)

| Column | Example | Meaning |
|--------|---------|---------|
| `Id` | 1042 | Unique ID |
| `Label` | `API_Function` | Type of concept |
| `Name` | `torch.compile` | The PyTorch symbol name |
| `URL` | `https://pytorch.org/...` | Link to official docs |

### Edge Columns (`edges.csv`)

| Column | Example | Meaning |
|--------|---------|---------|
| `Source` | 1042 | ID of the starting node |
| `Target` | 1055 | ID of the ending node |
| `Relation` | `CALLS` | How they're connected |

---

## 📈 Relation Type Breakdown

| Relation | Count | Meaning |
|----------|-------|---------|
| `EXPLAINS` | 27,090 | One concept explains another |
| `REFERENCES` | 16,710 | One concept points to another |
| `IMPLEMENTS` | 2,343 | One implements another |
| `HAS_PARAM` | 1,119 | A function has a specific parameter |
| `CONTAINS` | 637 | A module contains a sub-concept |
| `CALLS` | 48 | One function calls another |


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
