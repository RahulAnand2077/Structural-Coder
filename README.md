# 🧠 Structural-Coder
### A Smart PyTorch Code Generator powered by a Knowledge Graph + Heterogeneous GNN

---

## 🤔 What is this project?

Imagine you want to write PyTorch code but don't know which exact function names, parameters, or class names to use. This project does that **FOR YOU — automatically**.

It uses a massive **knowledge graph** (24,485 PyTorch API concepts, 47,958 connections) and a **Heterogeneous Graph Neural Network (HGTConv)** to find and assemble the most relevant code for any question you ask.

---

## 🆚 How Are We Better Than a Regular AI Chatbot?

| Feature | Structural-Coder (Ours) | Standalone AI (e.g. llama3.2) |
|---------|------------------------|-------------------------------|
| PyTorch knowledge graph (24K nodes, 47K edges) | ✅ | ❌ |
| Heterogeneous GNN (9 distinct node types) | ✅ HGTConv | ❌ |
| Grounded in real API symbols | ✅ Always | ❌ Often hallucinates |
| Code validation (C0–C5 checks) | ✅ Automatic | ❌ Never |
| **Avg Final Score (benchmark)** | **🏆 0.75** | **0.24** |

We won **all 10 out of 10** benchmark queries against llama3.2.

---

## 📁 Folder Structure

```
Structural-Coder/
│
├── README.md                   ← You are here
├── requirements.txt
│
├── data/
│   ├── nodes.csv               ← 24,485 PyTorch API concepts
│   └── edges.csv               ← 47,958 connections
│
├── src/
│   ├── graph_rag/              ← Heterogeneous GNN retrieval engine
│   ├── integration_pipeline/   ← Validation + self-healing
│   └── research_pipeline/      ← Orchestrator
│
├── benchmark/
│   ├── interactive_comparison.py  ← Live side-by-side tester
│   ├── run_comparison.py          ← Full batch benchmark
│   ├── queries/queries.json       ← 10 test questions
│   └── outputs/                   ← Live results
│
├── notebooks/
│   └── gnn_encoder_improved.ipynb ← Source of the HGTConv architecture
│
├── artifacts/                  ← Cached GNN embeddings
├── configs/
└── tests/
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start Ollama (local LLM server)
```bash
ollama serve
ollama pull llama3.2
```

### Step 3 — Run the Interactive Side-by-Side Tester
```bash
python benchmark/interactive_comparison.py
```

### Step 4 — Run the Full Batch Benchmark Report
```bash
python benchmark/run_comparison.py --models llama3.2
```



---

## 🔬 How It Works

```
Your Question
    │
    ▼
[HeteroGraphEncoder (HGTConv)]  ← 9 node types, 3-layer attention, Jumping Knowledge
    │   Trained with hard-negative sampling on 6 supervision relations
    ▼
[Graph Expansion]  ← 1-hop neighbourhood of seed nodes
    │
    ▼
[Hybrid Re-Ranker]  ← GNN cosine score + lexical overlap + graph degree
    │
    ▼
[Ollama LLM (llama3.2)]  ← Grounded by retrieved API symbols
    │
    ▼
[Active Validator]  ← 6 checks (C0–C5) for code correctness
    │
    ▼
✅ Grounded, Validated PyTorch Code
```

---

## 📊 Benchmark Results (vs llama3.2)

Scoring formula: `Final = 0.4 × Retrieval + 0.6 × (0.5 × Grounding + 0.5 × Validity)`

| System | Avg Retrieval | Avg Grounding | Avg Validity | **Avg Final** |
|--------|-------------|------------|------------|------------|
| 🏆 **Structural-Coder** | 0.79 | 0.60 | 0.80 | **0.75** |
| llama3.2 (Standalone) | 0.00 | 0.03 | 0.78 | **0.24** |

See `benchmark/outputs/comparison_report.md` for full per-query details.

---

## 🧬 GNN Architecture (Technical)

The `src/graph_rag/gnn_encoder.py` module implements these components ported directly from `notebooks/gnn_encoder_improved.ipynb`:

### `HeteroGraphEncoder`
- **Input projection** per node type (`nn.Linear` → `nn.LayerNorm`)
- **3-layer HGTConv** (Heterogeneous Graph Transformer) with 4 attention heads
- **Residual pass-through** for isolated node types (prevents dead gradients)
- **Jumping Knowledge (JK)** connections: all layer outputs concatenated → final linear

### `LowRankBilinearDecoder`
- Factored bilinear link scorer: `score = (z_src @ U) · (z_dst @ V)` (rank=32)
- Much fewer parameters than full bilinear — prevents overfitting on rare relations

### Training Strategy
- **Supervision relations**: `IMPLEMENTS`, `CONTAINS`, `HAS_PARAM`, `CALLS`, `RELATED_TO`, `REPLACES`
- **Message-passing relations**: All above + `EXPLAINS`, `REFERENCES`
- **Hard Negative Sampling**: Degree-distribution weighted, true-positive filtered
- **Gradient Clipping**: `max_norm=2.0` for stable HGT training

---

## 💡 Design Principle: Decoupled GNN & Pipeline

The **GNN Encoder** and the **Integration Pipeline** are strictly decoupled. You can retrain, swap, or improve the GNN without touching a single line of the pipeline code. The pipeline simply reads the cached `artifacts/research_gnn_embeddings.json` file.

---

## 👤 Authors

- **Amitesh Sinha** — Benchmarking, Pipeline Integration, Evaluation Engine
- **Rahul Anand** — GNN Architecture (`gnn_encoder_improved.ipynb`)
- **Mohit** — Validation Pipeline, Data Engineering
