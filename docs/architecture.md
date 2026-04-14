# 🏗️ Architecture

**Path:** `docs/architecture.md`

---

## System Overview

Structural-Coder is a Graph-RAG code generation system. It compares two approaches:

- **System A** — Standalone LLM (no context, pure training memory)
- **System B** — GNN predicts API path → LLM generates with context

```
System A:  Query → LLM (zero context) → Code A → Score

System B:  Query → GNN Decoder (best_model.pt)
                 → Predict top-K API nodes
                 → LLM (with GNN context) → Code B → Score

Compare scores. System B should win.
```

---

## Source Code Layout (`src/`)

Every main class has its own file. All helpers are in one file.

```
src/
├── config.py            ← MODEL_NAME (change LLM in one line)
├── pipeline.py          ← 🎯 Global entry point: System A, System B, compare()
│
├── graph_loader.py      ← CsvGraph, Node, Edge (reads nodes.csv + edges.csv)
├── gnn_encoder.py       ← HeteroGraphEncoder, LowRankBilinearDecoder (model def)
├── query_encoder.py     ← GNNQueryPredictor (live GNN inference per query)
├── retriever.py         ← GraphRAGRetriever (topological anchor + cosine expansion)
├── validator.py         ← ActiveValidator (C0–C5 code checks)
│
├── helpers.py           ← Combined utilities:
│                            GraphRetriever (lexical baseline)
│                            call_ollama(), build_gnn_prompt()
│                            token_hit_score(), grounding_score()
│                            validity_score(), compute_final_score()
│
├── graph_rag/           ← Legacy copies (kept for backward compat)
├── integration_pipeline/← Legacy copies
└── research_pipeline/   ← Legacy copies
```

### Class → File Mapping

| Class | File | What It Does |
|-------|------|-------------|
| `Pipeline` | `pipeline.py` | `system_a()`, `system_b()`, `compare()` |
| `CsvGraph` | `graph_loader.py` | Reads CSV → nodes/edges + adjacency lists |
| `GNNQueryPredictor` | `query_encoder.py` | Hash query → project through trained weights → decoder score |
| `GraphRAGRetriever` | `retriever.py` | Lexical anchor → cosine expansion → hybrid rank |
| `ActiveValidator` | `validator.py` | C0–C5 code safety checks |
| `HeteroGraphEncoder` | `gnn_encoder.py` | HGTConv model definition (for training) |
| `GraphRetriever` | `helpers.py` | Lexical keyword search (ablation baseline) |

---

## How System B Works (per query)

```
1. Hash query text → 1024-D feature vector
2. Project through trained input_proj layer → 256-D query vector
3. Score query vs ALL 24,485 node embeddings via bilinear decoder:
     score = (query @ U) · (node @ V)
   U, V are trained [256, 32] matrices from best_model.pt
4. Lexical boost: nodes matching query keywords get +2.0 per overlap
5. Top 10 nodes = "GNN-predicted API path"
6. Feed predicted nodes as advisory context to Ollama LLM
7. LLM generates code using GNN context
```

---

## Data Files

```
data/
├── nodes.csv      ← 24,485 PyTorch API nodes (Id, Label, Name, URL)
└── edges.csv      ← 47,958 relationships (Source, Target, Type)

outputs/
├── gnn_embeddings.pt     ← 256-D vectors per node (used at runtime)
├── gnn_embeddings.jsonl  ← Same vectors + display_name (for enrichment)
├── best_model.pt         ← Trained model weights (used for inference!)
├── hetero_metadata.json  ← Node/edge type schema
└── training_summary.json ← Metrics from training
```

---

## Config (`src/config.py`)

```python
MODEL_NAME: str = "llama3.1:8b"  # Change this ONE line to switch the LLM
```

---

## Running

```bash
# Single query comparison
python -m src.pipeline "How to use torch.compile?"

# Interactive comparison
python benchmark/interactive_comparison.py

# Batch benchmark (all 10 queries)
python benchmark/run_comparison.py
```
