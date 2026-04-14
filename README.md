# 🧠 Structural-Coder
### PyTorch Code Generator: Standalone LLM vs GNN-Augmented LLM

---

## What This Project Does

Given a PyTorch coding question, we compare two code generation systems:

| | System A: Standalone LLM | System B: GNN + LLM |
|---|---|---|
| **Input** | Query → LLM | Query → GNN model → LLM |
| **Context** | ❌ None (pure LLM memory) | ✅ GNN predicts relevant API nodes |
| **GNN Weight** | ❌ Not used | ✅ Trained decoder from `best_model.pt` |
| **Knowledge Graph** | ❌ None | ✅ 24,485 nodes, 47,958 edges |

System B uses the **trained GNN model** to predict which PyTorch APIs are relevant to the query, then feeds them as context to the same LLM.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Start Ollama
ollama serve
ollama pull llama3.1:8b

# Run a single comparison
python -m src.pipeline "How to use torch.compile with dynamic shapes?"

# Interactive mode
python benchmark/interactive_comparison.py

# Batch benchmark (all 10 queries)
python benchmark/run_comparison.py
```

---

## Source Code (`src/`)

Each main class = one file. All helpers = one file.

```
src/
├── config.py            ← Change MODEL_NAME to switch LLM (one line)
├── pipeline.py          ← 🎯 Entry point: system_a(), system_b(), compare()
│
├── graph_loader.py      ← CsvGraph, Node, Edge
├── gnn_encoder.py       ← HeteroGraphEncoder (model definition)
├── query_encoder.py     ← GNNQueryPredictor (live GNN inference)
├── retriever.py         ← GraphRAGRetriever (topological expansion)
├── validator.py         ← ActiveValidator (C0–C5 code checks)
│
└── helpers.py           ← GraphRetriever, Ollama caller, prompts, scoring
```

---

## How System B Works

```
Query: "flash attention with fallback"
    │
    ▼
[1] Hash query → 1024-D vector (same hashing as GNN training)
    │
    ▼
[2] Project through trained input_proj → 256-D (from best_model.pt)
    │
    ▼
[3] Bilinear decoder scores: (query @ U) · (node @ V)
    Score against all 24,485 nodes using trained U, V matrices
    │
    ▼
[4] Top-10 nodes = GNN-predicted API path:
    → torch.backends.is_flash_attention_available
    → torch.nn.functional.scaled_dot_product_attention
    → torch.backends.can_use_flash_attention
    │
    ▼
[5] Feed predicted APIs to Ollama LLM as advisory context
    │
    ▼
[6] LLM generates grounded, validated code
```

---

## Scoring

```
System A: Final = 0.3 × Validity
System B: Final = 0.4 × Retrieval + 0.6 × (0.5 × Grounding + 0.5 × Validity)
```

| Metric | What It Measures |
|--------|-----------------|
| **Retrieval** | Query keywords covered by GNN-predicted nodes |
| **Grounding** | Retrieved API names that appear in generated code |
| **Validity** | C0–C5 code safety checks passed |

---

## 📚 Detailed Documentation

| Document | What It Covers |
|----------|---------------|
| [architecture.md](docs/architecture.md) | System architecture, class mapping, data flow |
| [data_integration.md](docs/data_integration.md) | Every data file explained |
| [gnn_technical.md](docs/gnn_technical.md) | HGTConv model, training, embeddings |
| [benchmark_guide.md](docs/benchmark_guide.md) | How to run and interpret benchmarks |
| [validation_checks.md](docs/validation_checks.md) | C0–C5 checks explained |
| [deprecated_code.md](docs/deprecated_code.md) | What was moved to `_deprecated/` |
