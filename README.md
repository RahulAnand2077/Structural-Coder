# 🧠 Structural-Coder-Amitesh
### A Smart PyTorch Code Generator powered by a Knowledge Graph + AI

---

## 🤔 What is this project? (For a complete beginner)

Imagine you want to write PyTorch code (a popular AI framework) but you don't know which
exact function names, parameters, or class names to use. Normally you would:
1. Search Google/Stack Overflow
2. Read 10 different documentation pages
3. Try-and-fail many times

**This project does all of that FOR YOU — automatically.**

It uses a massive **knowledge graph** (a database of 24,485 PyTorch API concepts and 47,958
connections between them) to find the most relevant code building blocks for any question
you ask, and then assembles them into working, validated code.

---

## 🆚 How Are We Better Than a Regular AI Chatbot (e.g. llama3.2)?

| Feature | Structural-Coder (Ours) | Standalone AI (e.g. llama3.2) |
|---------|------------------------|-------------------------------|
| Uses a structured PyTorch knowledge graph | ✅ Yes — 24K nodes, 47K edges | ❌ No |
| Grounded in real API symbols | ✅ Always | ❌ Often hallucinates |
| Validates generated code (C0–C5 checks) | ✅ Automatic | ❌ Never |
| Graph Neural Network (GNN) for smart search | ✅ Yes | ❌ No |
| **Benchmark Result (avg final score)** | **🏆 0.75** | **0.24** |

We **won all 10 out of 10** benchmark queries against llama3.2.

---

## 📁 Folder Structure at a Glance

```
Structural-Coder-Amitesh/
│
├── README.md                   ← You are here
├── requirements.txt            ← Python packages needed
│
├── data/                       ← The PyTorch knowledge graph database
│   ├── nodes.csv               ← 24,485 PyTorch API concepts
│   └── edges.csv               ← 47,958 connections between concepts
│
├── src/                        ← All Python source code
│   ├── graph_rag/              ← Core Graph-RAG retrieval engine
│   ├── integration_pipeline/   ← Validation + self-healing
│   └── research_pipeline/      ← Orchestrator (runs everything together)
│
├── benchmark/                  ← Proof that our system beats LLMs
│   ├── run_comparison.py       ← Script to run the benchmark
│   ├── queries/                ← 10 test questions
│   └── outputs/                ← Live comparison results
│
├── scripts/                    ← Entry-point runner scripts
│   ├── run_graph_rag_pipeline.py
│   └── run_research_pipeline.py
│
├── notebooks/                  ← Jupyter notebooks (experiments / exploration)
├── artifacts/                  ← Saved GNN embeddings (model weights)
├── configs/                    ← Configuration files
└── tests/                      ← Automated tests
```

---

## 🚀 How to Run (Step by Step)

### Step 1 — Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Run the full pipeline (retrieve + generate code)
```bash
python scripts/run_research_pipeline.py \
  --nodes data/nodes.csv \
  --edges data/edges.csv \
  --query "compile-safe transformer block"
```

### Step 3 — Run the Interactive Side-by-Side Tester
```bash
# First make sure Ollama is running in the background:  ollama serve
# Then run the interactive comparison tool:
python benchmark/interactive_comparison.py
```

### Step 4 — Run the Automated Batch Benchmark (Full Report)
```bash
python benchmark/run_comparison.py --models llama3.2
```

---

## 🔬 How It Works (10 Second Explanation)

```
Your Question
    │
    ▼
[GNN Retriever] — searches 24K PyTorch concepts using a Graph Neural Network
    │
    ▼
[Knowledge Graph Expansion] — finds related concepts by following edges
    │
    ▼
[Code Scaffolding] — assembles retrieved symbols into working code pattern
    │
    ▼
[Active Validator] — runs 6 checks (C0–C5) to verify the code is correct
    │
    ▼
✅ Valid, Grounded PyTorch Code
```

---

## 📊 Benchmark Results (Real Run vs llama3.2)

| System | Avg Grounding | Avg Validity | **Avg Score** |
|--------|-------------|------------|------------|
| 🏆 **Structural-Coder (Ours)** | 0.60 | 0.80 | **0.75** |
| llama3.2 (Standalone) | 0.03 | 0.78 | 0.24 |

See `benchmark/outputs/comparison_report.md` for full per-query details.

---

## 👤 Author

**Amitesh Sinha** — Project Structural-Coder-Amitesh


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
