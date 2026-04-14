# Data Integration Guide — Structural-Coder

> This document explains **every single data file** in the project, how it was produced, and exactly how it feeds into the live system.

---

## Table of Contents

1. [Overview](#overview)
2. [Input Data: `data/` folder](#input-data-data-folder)
3. [GNN Outputs: `outputs/` folder](#gnn-outputs-outputs-folder)
4. [How Files Connect: The Full Pipeline](#how-files-connect-the-full-pipeline)
5. [Code-Level Integration Map](#code-level-integration-map)
6. [What Each Source File Does](#what-each-source-file-does)

---

## Overview

Structural-Coder works in two phases:

| Phase | When | What Happens |
|-------|------|-------------|
| **Offline Training** | Once (via `gnn_encoder_improved.ipynb`) | Reads `nodes.csv` + `edges.csv` → builds a PyTorch Geometric HeteroData graph → trains HGTConv → exports embeddings to `outputs/` |
| **Online Inference** | Every query | Reads `nodes.csv` + `edges.csv` + `outputs/gnn_embeddings.jsonl` → finds relevant APIs → generates code via Ollama LLM |

---

## Input Data: `data/` folder

### `data/nodes.csv` — The Knowledge Graph Nodes

**24,485 rows.** Each row is one PyTorch concept scraped from official PyTorch documentation.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Id` | int | Unique node identifier (0–24484) | `22715` |
| `Label` | string | Node type category (9 types) | `API_Class` |
| `Name` | string | Human-readable name | `SymInt` |
| `URL` | string | Link to official PyTorch docs page | `https://docs.pytorch.org/...` |

**The 9 Node Types:**

| Label | Count | What It Represents |
|-------|-------|-------------------|
| `Concept` | ~17,943 | General documentation topics, tutorials, guides |
| `API_Endpoint` | ~2,747 | Documentation page URLs (one per HTML page) |
| `CodeSnippet` | ~1,985 | Example code blocks from docs |
| `API_Parameter` | ~1,119 | Function/method parameters (e.g., `dim`, `dtype`) |
| `API_Method` | ~271 | Class methods (e.g., `Module.forward`) |
| `API_Function` | ~264 | Standalone functions (e.g., `torch.randn`) |
| `API_Class` | ~142 | Classes (e.g., `torch.nn.Module`) |
| `DeprecatedAPI` | ~7 | Deprecated APIs |
| `PyTorchConcept` | ~7 | Core architectural concepts (TorchDynamo, TorchInductor, etc.) |

**How it's loaded:** `src/integration_pipeline/graph_loader.py` → `CsvGraph.load()` reads this CSV and creates `Node` dataclass objects stored in `self.nodes: Dict[int, Node]`.

---

### `data/edges.csv` — The Knowledge Graph Edges

**47,958 rows.** Each row is a directed relationship between two nodes.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Source` | int | Source node ID | `7` |
| `Target` | int | Target node ID | `8` |
| `Type` | string | Relationship type | `EXPLAINS` |

**Edge/Relationship Types:**

| Type | What It Means |
|------|--------------|
| `EXPLAINS` | A concept page explains an API |
| `CONTAINS` | An endpoint page contains a class/function |
| `HAS_PARAM` | A function/method has a parameter |
| `IMPLEMENTS` | An endpoint implements a code snippet |
| `CALLS` | A code snippet calls an API function |
| `REFERENCES` | A page references another page |
| `RELATED_TO` | General relationship |
| `REPLACES` | A new API replaces a deprecated one |

**How it's loaded:** `CsvGraph.load()` builds two adjacency lists:
- `self.adj[source_id]` → list of outgoing edges
- `self.rev_adj[target_id]` → list of incoming edges

These adjacency lists power the 1-hop graph expansion during retrieval.

---

## GNN Outputs: `outputs/` folder

All files in `outputs/` are produced by `notebooks/gnn_encoder_improved.ipynb`. This notebook trains a Heterogeneous Graph Transformer (HGTConv) on the knowledge graph.

### Files Used at Runtime ✅

#### `outputs/gnn_embeddings.jsonl` (140 MB) — **Primary Embedding Store**

**24,485 lines.** Each line is a JSON object representing one node's trained embedding.

```json
{
  "element_id": "4:170763f8-8d9f-40c2-b8d0-f5cb2fbaa0a1:22715",
  "node_type": "API_Class",
  "display_name": "torch.SymInt",
  "embedding": [0.067, -0.018, 0.050, ...]  // 256 floats
}
```

| Field | Description |
|-------|-------------|
| `element_id` | Neo4j-style ID. We extract the node ID from the last segment after `:` (e.g., `22715`) |
| `node_type` | Same as `Label` in nodes.csv (redundant but useful for validation) |
| `display_name` | **Fully-qualified Python name** (e.g., `torch.SymInt`, `torch.nn.Module`). For `API_Endpoint` nodes, this is the URL. |
| `embedding` | 256-dimensional vector trained by HGTConv. Encodes the node's **structural position** in the graph. |

**How it's loaded:**
1. `gnn_encoder.py` → `load_embeddings_from_jsonl()` extracts `node_ids`, `embeddings` tensor, and `display_names` dict
2. `pipeline.py` → `step3_gnn_training()` patches `CsvGraph.nodes[nid].name` with the richer `display_name` (skipping URLs). This enriches bare names like `"SymInt"` → `"torch.SymInt"` so the LLM can write correct imports.
3. The 256-D vectors are L2-normalized and used for topological neighbor discovery via cosine similarity.

---

#### `outputs/gnn_embeddings.pt` (25 MB) — **Fast Embedding Store** (Optional)

Same embeddings as the JSONL but stored as a PyTorch dict:

```python
{
    "API_Class": Tensor[142, 256],
    "API_Endpoint": Tensor[2747, 256],
    "API_Function": Tensor[264, 256],
    "API_Method": Tensor[271, 256],
    "API_Parameter": Tensor[1119, 256],
    "CodeSnippet": Tensor[1985, 256],
    "Concept": Tensor[17943, 256],
    "DeprecatedAPI": Tensor[7, 256],
    "PyTorchConcept": Tensor[7, 256],
}
```

**Advantage:** ~5x faster to load than parsing 140MB JSON text.
**Disadvantage:** Does not contain `display_name`, so node name enrichment won't happen.

**How to use:** Pass `--embedding-cache outputs/gnn_embeddings.pt` to the benchmark scripts.

---

### Files NOT Used at Runtime ❌ (Training Artifacts Only)

#### `outputs/best_model.pt` (56 MB)
The trained HGTConv model weights (state dict). Used to **produce** the embeddings — not needed at inference because we use the pre-computed vectors directly.

Contains: `HeteroGraphEncoder` state dict with:
- Per-node-type input projections (`nn.Linear` + `nn.LayerNorm`)
- 3-layer HGTConv attention weights (4 heads)
- Jumping Knowledge aggregation layer
- `LowRankBilinearDecoder` weights

#### `outputs/hetero_metadata.json` (7 MB)
Node types and edge types schema used during GNN construction:
```json
{
  "node_types": ["API_Class", "API_Endpoint", ...],
  "edge_types": [["API_Class", "CONTAINS", "API_Method"], ...]
}
```

#### `outputs/hetero_graph.pt` (101 MB)
The full `HeteroData` PyTorch Geometric object containing all node features and edge indices. This is the graph that was fed into the GNN for training.

#### `outputs/train_graph.pt` (101 MB)
The training split of the hetero graph (with some edges held out for validation/test).

#### `outputs/split_state.pt` (154 KB)
Train/validation/test edge split masks.

#### `outputs/training_summary.json` (409 KB)
Full training metrics including per-epoch loss, per-relation ROC AUC, accuracy, thresholds, and MRR scores.

Key metrics:
| Relation | ROC AUC | Accuracy |
|----------|---------|----------|
| API_Endpoint → API_Function | 1.000 | 1.000 |
| API_Class → API_Method | 0.863 | 0.833 |
| API_Endpoint → API_Class | 0.811 | 0.857 |
| API_Endpoint → CodeSnippet | 0.672 | 0.650 |
| **Overall** | **0.668** | **0.701** |

---

## How Files Connect: The Full Pipeline

```
                    ┌──────────────────────────────────────┐
                    │     OFFLINE (run once)                │
                    │                                      │
  data/nodes.csv ───┤                                      │
                    ├──→ gnn_encoder_improved.ipynb         │
  data/edges.csv ───┤     │                                │
                    │     ├── Builds HeteroData graph       │
                    │     ├── Trains HGTConv (3 layers)     │
                    │     ├── Exports best_model.pt         │
                    │     └── Exports gnn_embeddings.jsonl  │
                    │                                      │
                    └──────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │     ONLINE (every query)              │
                    │                                      │
  data/nodes.csv ───┤──→ CsvGraph.load()                   │
  data/edges.csv ───┤     │  Creates in-memory graph        │
                    │     │  with adjacency lists           │
                    │     │                                 │
  outputs/          │     │                                 │
  gnn_embeddings    │     │                                 │
  .jsonl ───────────┤──→ load_embeddings_from_jsonl()       │
                    │     │  Returns:                       │
                    │     │  ├── node_ids [24485]            │
                    │     │  ├── embeddings [24485, 256]     │
                    │     │  └── display_names {id→name}     │
                    │     │                                 │
                    │     ├── Enriches graph node names      │
                    │     │   "SymInt" → "torch.SymInt"     │
                    │     │                                 │
                    │     ▼                                 │
                    │  GraphRAGRetriever                     │
                    │     │                                 │
                    │     ├── Phase 1: Lexical Anchor        │
                    │     │   Text-match query → 1-4 nodes  │
                    │     │                                 │
                    │     ├── Phase 2: Topological Expansion │
                    │     │   Anchor's 256-D vector →       │
                    │     │   cosine vs all 24,485 vectors  │
                    │     │   → top 20 structural neighbors │
                    │     │                                 │
                    │     ├── Phase 3: 1-hop Graph Walk      │
                    │     │   Follow edges from anchors     │
                    │     │                                 │
                    │     ├── Phase 4: Hybrid Re-Ranking     │
                    │     │   Score = GNN + 2×Lexical       │
                    │     │         + Degree + TypeBonus     │
                    │     │                                 │
                    │     ▼                                 │
                    │  Top 5 nodes → LLM Prompt              │
                    │     │  APIs: "Use if relevant"         │
                    │     │  Concepts: "READ-ONLY"           │
                    │     │                                 │
                    │     ▼                                 │
                    │  Ollama LLM (llama3.1:8b)              │
                    │     │                                 │
                    │     ▼                                 │
                    │  Active Validator (C0-C5 checks)       │
                    │     │                                 │
                    │     ▼                                 │
                    │  ✅ Grounded, Validated PyTorch Code   │
                    └──────────────────────────────────────┘
```

---

## Code-Level Integration Map

This table shows exactly which source file reads which data file:

| Source File | Reads | What It Extracts |
|------------|-------|-----------------|
| `src/integration_pipeline/graph_loader.py` | `data/nodes.csv` | Node objects (id, label, name, url) |
| `src/integration_pipeline/graph_loader.py` | `data/edges.csv` | Edge objects (source, target, type) + adjacency lists |
| `src/graph_rag/gnn_encoder.py` → `load_embeddings_from_jsonl()` | `outputs/gnn_embeddings.jsonl` | node_ids + 256-D embeddings + display_names |
| `src/graph_rag/gnn_encoder.py` → `load_embeddings_from_pt()` | `outputs/gnn_embeddings.pt` | node_ids + 256-D embeddings (no display_names) |
| `src/research_pipeline/pipeline.py` → `step3_gnn_training()` | Uses above loaders | Enriches CsvGraph names + creates GraphRAGRetriever |
| `src/graph_rag/retriever.py` → `retrieve()` | In-memory graph + embeddings | Lexical anchors → topological neighbors → ranked results |
| `src/research_pipeline/pipeline.py` → `_build_ollama_prompt()` | Retrieved nodes | Splits APIs vs Concepts for LLM prompt |

---

## What Each Source File Does

### `src/integration_pipeline/graph_loader.py`
- Defines `Node`, `Edge`, `CsvGraph` dataclasses
- `CsvGraph.load()` reads both CSVs into memory
- Provides `one_hop()`, `multi_hop()`, `highest_degree()` graph traversal methods

### `src/graph_rag/gnn_encoder.py`
- Defines `HeteroGraphEncoder` (HGTConv model) and `LowRankBilinearDecoder`
- `build_graph_tensor_data()` converts CSV graph to PyTorch Geometric `HeteroData`
- `train_gnn_embeddings()` trains the GNN (used if no cache exists)
- `load_embeddings_from_jsonl()` loads pre-computed embeddings + display names
- `load_embeddings_from_pt()` loads pre-computed embeddings from PT format

### `src/graph_rag/retriever.py`
- `GraphRAGRetriever` — the core retrieval engine
- Phase 1: `_global_lexical_search()` — brute-force text matching against 24K nodes
- Phase 2: Topological expansion — cosine similarity in GNN embedding space
- Phase 3: 1-hop graph walk using adjacency lists
- Phase 4: `_hybrid_rank()` — combines GNN score + lexical overlap + degree + type bonuses

### `src/research_pipeline/pipeline.py`
- `CsvFirstResearchPipeline` — orchestrates everything
- `step3_gnn_training()` — loads embeddings and enriches node names
- `_build_ollama_prompt()` — constructs the LLM prompt with API/Concept separation
- `_grounding_score()` / `_code_validity_score()` — evaluation metrics
- `_call_ollama()` — sends prompts to local Ollama server

### `benchmark/interactive_comparison.py`
- Interactive terminal: type a query, see Structural-Coder vs Standalone side-by-side

### `benchmark/run_comparison.py`
- Batch benchmark: runs all queries from `queries.json`, produces comparison report
