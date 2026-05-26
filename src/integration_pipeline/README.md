# 🛡️ `src/integration_pipeline/` — The Quality Inspector

Once code has been retrieved or generated, this module **checks it for errors** and
can automatically **fix** problems it finds.

---

## Files in This Folder

| File | What It Does (Plain English) |
|------|------------------------------|
| `graph_loader.py` | Loads the graph from CSV files |
| `retriever.py` | Alternative keyword-based graph search (the "lexical" retriever) |
| `validator.py` | Runs 6 safety checks on any generated code |
| `pipeline.py` | Combines retrieval + validation + self-healing |

---

## 🔬 The 6 Validation Checks (C0 to C5)

Think of this as a car inspection — we check different parts of the code one by one:

| Check | Name | What We Test |
|-------|------|-------------|
| **C0** | Import Check | Does the code import `torch`? |
| **C1** | Syntax Check | Can Python parse the code (no typos)? |
| **C2** | API Existence | Do the PyTorch functions called actually exist? |
| **C3** | Compile Probe | Can `torch.compile()` accept this without crashing? |
| **C4** | Device Guard | Is CUDA used safely (with `is_available()` checks)? |
| **C5** | Dtype Safety | Are data types mixed safely (no float16/bfloat16 mistakes)? |

If a check **fails**, the self-healing loop modifies the code and re-checks (up to 3 times).

---

## 🔄 Lexical Retriever

The `retriever.py` here uses three search modes:

- **Microscope** — deep 1-hop neighbourhood of one node
- **Detective** ✅ (default) — 2-hop exploration of up to 600 nodes
- **Telescope** — returns the 30 most connected nodes regardless of query


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
