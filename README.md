# 🧠 Structural-Coder
### A Smart PyTorch Code Generator powered by a Knowledge Graph + Heterogeneous GNN

---

## 🤔 What is this project?

Imagine you want to write PyTorch code but don't know which exact function names, parameters, or class names to use. This project does that **FOR YOU — automatically**.

It uses a massive **knowledge graph** (24,485 PyTorch API concepts, 47,958 connections) and a **Heterogeneous Graph Neural Network (HGTConv)** to find and assemble the most relevant code for any question you ask.

---

## 🆚 How Are We Better Than a Regular AI Chatbot?

| Feature | Structural-Coder (Ours) | Standalone AI (e.g. llama3.1:8b) |
|---------|------------------------|-------------------------------|
| PyTorch knowledge graph (24K nodes, 47K edges) | ✅ | ❌ |
| Heterogeneous GNN (9 distinct node types) | ✅ HGTConv | ❌ |
| Grounded in real API symbols | ✅ Always | ❌ Often hallucinates |
| Fully-qualified import paths (e.g. `torch.nn.Module`) | ✅ From GNN output | ❌ Guesses |
| Code validation (C0–C5 checks) | ✅ Automatic | ❌ Never |
| **Avg Final Score (benchmark)** | **🏆 0.65+** | **0.28** |

We won **all 10 out of 10** benchmark queries against llama3.1:8b.

---

## 📁 Folder Structure

```
Structural-Coder/
│
├── README.md                      ← You are here
├── requirements.txt
│
├── data/                          ← Input Data (Knowledge Graph)
│   ├── nodes.csv                  ← 24,485 PyTorch API nodes (Id, Label, Name, URL)
│   └── edges.csv                  ← 47,958 connections (Source, Target, Type)
│
├── outputs/                       ← GNN Training Outputs (from gnn_encoder_improved.ipynb)
│   ├── gnn_embeddings.jsonl       ← 24,485 node embeddings (256-D) with display_name + node_type
│   ├── gnn_embeddings.pt          ← Same embeddings as PyTorch dict (faster to load)
│   ├── best_model.pt              ← Trained HGTConv model weights
│   ├── hetero_metadata.json       ← Node types + edge types metadata
│   ├── hetero_graph.pt            ← Full HeteroData graph object
│   ├── train_graph.pt             ← Training split of graph
│   ├── split_state.pt             ← Train/val/test split state
│   └── training_summary.json      ← Metrics, thresholds, training history
│
├── src/
│   ├── graph_rag/                 ← Heterogeneous GNN retrieval engine
│   │   ├── gnn_encoder.py         ← HGTConv model + embedding loaders
│   │   └── retriever.py           ← Topological anchor + neighborhood retrieval
│   ├── integration_pipeline/      ← Validation + graph loading
│   │   ├── graph_loader.py        ← CSV graph reader (Node, Edge, CsvGraph)
│   │   ├── validator.py           ← C0-C5 active code checks
│   │   └── pipeline.py            ← Integration orchestrator
│   └── research_pipeline/         ← Research evaluation orchestrator
│       └── pipeline.py            ← Main pipeline: loading, retrieval, LLM, scoring
│
├── benchmark/
│   ├── interactive_comparison.py  ← Live side-by-side tester
│   ├── run_comparison.py          ← Full batch benchmark
│   ├── queries/queries.json       ← 10 test questions
│   └── outputs/                   ← Live benchmark results
│
├── notebooks/
│   └── gnn_encoder_improved.ipynb ← Source of the HGTConv architecture + training
│
└── tests/
```

---

## 📊 Data Integration Map

This diagram shows exactly how each data file flows through the system:

```
data/nodes.csv ──────────────┐
  24,485 nodes               │
  Fields: Id, Label, Name,   ├──→ graph_loader.py ──→ CsvGraph (in-memory graph)
  URL                        │                            │
data/edges.csv ──────────────┘                            │
  47,958 edges                                            │
  Fields: Source, Target,                                  │
  Type                                                    │
                                                          ▼
outputs/gnn_embeddings.jsonl ──→ load_embeddings_from_jsonl()
  24,485 embeddings (256-D)       │
  Fields: element_id,             ├──→ node_ids + embeddings tensor
  node_type, display_name,        │
  embedding                       ├──→ display_names dict ──→ Enriches CsvGraph node names
                                  │    (e.g. "SymInt" → "torch.SymInt")
                                  │
                                  ▼
                            GraphRAGRetriever
                              │
                              ├── Phase 1: Lexical Anchor Discovery (text match)
                              ├── Phase 2: Topological Neighborhood Expansion (GNN cosine)
                              ├── Phase 3: Hybrid Re-Ranking (text + GNN + degree + type)
                              │
                              ▼
                         Retrieved Context (top 5 nodes)
                              │
                              ▼
                    _build_ollama_prompt() ── Splits APIs vs Concepts
                              │
                              ▼
                    Ollama LLM (llama3.1:8b) ── Advisory prompt
                              │
                              ▼
                    Active Validator (C0–C5)
                              │
                              ▼
                    ✅ Grounded, Validated PyTorch Code
```

### What each output file does

| File | Used at Runtime? | Purpose |
|------|-----------------|---------|
| `gnn_embeddings.jsonl` | ✅ **Yes** | Pre-computed 256-D vectors for all 24,485 nodes. Contains `display_name` (fully-qualified Python paths like `torch.nn.Module`) used to enrich node names for the LLM. |
| `gnn_embeddings.pt` | ✅ Optional | Same embeddings in PyTorch dict format — faster to load than JSONL. |
| `best_model.pt` | ❌ No | The trained HGTConv model weights. Used during training to *produce* the embeddings, not needed at inference. |
| `hetero_metadata.json` | ❌ No | Node/edge type schema. Reference only. |
| `hetero_graph.pt` | ❌ No | Full HeteroData graph. Used during GNN training only. |
| `train_graph.pt` | ❌ No | Training split. Used during GNN training only. |
| `split_state.pt` | ❌ No | Train/val/test split. Used during GNN training only. |
| `training_summary.json` | ❌ No | Metrics + history from GNN training (AUC, accuracy per relation). |

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start Ollama (local LLM server)
```bash
OLLAMA_MODELS="/path/to/models" ollama serve
ollama pull llama3.1:8b
```

### Step 3 — Run the Interactive Side-by-Side Tester
```bash
cd Structural-Coder
../.venv/bin/python benchmark/interactive_comparison.py --model llama3.1:8b
```

### Step 4 — Run the Full Batch Benchmark Report
```bash
../.venv/bin/python benchmark/run_comparison.py --models llama3.1:8b
```

---

## 🔬 How It Works (Architecture)

### Step 1: Lexical Anchor Discovery
The user's query text (e.g. "flash attention with fallback") is tokenized and keyword-searched against all 24,485 nodes. This produces 1–4 **Anchor Nodes** — the exact entry points into the graph.

### Step 2: Topological Neighborhood Expansion (GNN)
Each Anchor Node's pre-computed 256-D vector (from `gnn_embeddings.jsonl`) is retrieved. We compute cosine similarity against all 24,485 vectors to find the **structurally nearest neighbors** — APIs that share edges in the documentation graph, even if they share zero text keywords.

### Step 3: Hybrid Re-Ranking
All candidates (lexical matches + topological neighbors + 1-hop graph expansion) are scored with:
- **GNN cosine similarity** (structural proximity)
- **Lexical overlap** (text matching, weighted 2x)
- **Graph degree** (well-connected nodes preferred)
- **Node type bonus**: `API_Class/Function/Method` boosted +1.0, `API_Endpoint` demoted -0.5

### Step 4: LLM Prompt Construction
Top 5 nodes are split into:
- 💻 **Valid PyTorch APIs** (advisory — use only if relevant)
- 📚 **Conceptual Context** (READ-ONLY, do not import)

The prompt instructs the LLM: "If the APIs seem irrelevant, IGNORE THEM and rely on your own knowledge."

### Step 5: Active Validation (C0–C5)
Generated code is validated with 6 checks: syntax, import safety, API correctness, type checking, runtime execution, and compilation.

---

## 🧬 GNN Architecture (Technical)

The `src/graph_rag/gnn_encoder.py` module implements these components ported directly from `notebooks/gnn_encoder_improved.ipynb`:

### Node Types (9)
`API_Class`, `API_Function`, `API_Method`, `API_Parameter`, `API_Endpoint`, `CodeSnippet`, `Concept`, `DeprecatedAPI`, `PyTorchConcept`

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

### Training Metrics (from `training_summary.json`)
| Relation Type | ROC AUC | Accuracy |
|--------------|---------|----------|
| API_Endpoint→API_Function | 1.000 | 1.000 |
| API_Class→API_Method | 0.863 | 0.833 |
| API_Endpoint→API_Class | 0.811 | 0.857 |
| API_Endpoint→CodeSnippet | 0.672 | 0.650 |
| Overall | 0.668 | 0.701 |

---

## 💡 Design Principle: Decoupled GNN & Pipeline

The **GNN Encoder** and the **Integration Pipeline** are strictly decoupled. You can retrain, swap, or improve the GNN without touching a single line of the pipeline code. The pipeline simply reads the cached `outputs/gnn_embeddings.jsonl` (or `.pt`) file and enriches node names from the `display_name` field.

---

## 👤 Authors

- **Amitesh Sinha** — Benchmarking, Pipeline Integration, Evaluation Engine
- **Rahul Anand** — GNN Architecture (`gnn_encoder_improved.ipynb`)
- **Mohit** — Validation Pipeline, Data Engineering
