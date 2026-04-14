# 🧬 GNN Technical Details

**Path:** `docs/gnn_technical.md`

---

## Model: HeteroGraphEncoder

A Heterogeneous Graph Transformer (HGTConv) that produces 256-D embeddings for every node in the knowledge graph.

**Source:** `notebooks/gnn_encoder_improved.ipynb` → ported to `src/graph_rag/gnn_encoder.py`

### Architecture

```
Raw CSV Text Features (hashed → 128-dim per node)
         │
         ▼
[Input Projection]  ← Per-node-type nn.Linear + nn.LayerNorm
         │
         ▼
[HGTConv Layer 1]  ← 4-head attention across typed edges
         │
[HGTConv Layer 2]  ← Residual passthrough for isolated types
         │
[HGTConv Layer 3]
         │
         ▼
[Jumping Knowledge (JK)]  ← Concat all 3 layer outputs
         │
         ▼
[Final Linear]  → 256-dim L2-normalized embedding per node
```

### Node Types (9)

| Type | Count | What It Represents |
|------|-------|-------------------|
| `Concept` | 17,943 | General docs/tutorials |
| `API_Endpoint` | 2,747 | Documentation page URLs |
| `CodeSnippet` | 1,985 | Example code blocks |
| `API_Parameter` | 1,119 | Function parameters |
| `API_Method` | 271 | Class methods |
| `API_Function` | 264 | Standalone functions |
| `API_Class` | 142 | Classes |
| `DeprecatedAPI` | 7 | Deprecated APIs |
| `PyTorchConcept` | 7 | Core architectural concepts |

### Edge/Relation Types

| Used For | Relations |
|----------|----------|
| **Supervision (loss)** | `IMPLEMENTS`, `CONTAINS`, `HAS_PARAM`, `CALLS`, `RELATED_TO`, `REPLACES` |
| **Message-passing only** | `EXPLAINS`, `REFERENCES` |

### Training Strategy

| Technique | Purpose |
|-----------|---------|
| Hard Negative Sampling | Degree-weighted, true-positive filtered — prevents contradictory training signal |
| Gradient Clipping | `max_norm=2.0` — stabilises HGT attention |
| LowRankBilinearDecoder | Factored scorer `(z@U)·(z@V)` rank=32 — fewer parameters on rare relations |

---

## Embedding Files

### `outputs/gnn_embeddings.jsonl` (140 MB) — Primary

Each of the 24,485 lines is one JSON object:
```json
{
  "element_id": "4:...:22715",
  "node_type": "API_Class",
  "display_name": "torch.SymInt",
  "embedding": [0.067, -0.018, ...]  // 256 floats
}
```

**`display_name`** is the key enrichment field. It carries fully-qualified Python paths (`torch.SymInt`, `torch.nn.Module`). At load time, the pipeline patches bare CSV names with these qualified names so the LLM can write correct imports.

For `API_Endpoint` nodes, `display_name` is a URL — these are skipped during enrichment.

### `outputs/gnn_embeddings.pt` (25 MB) — Fast Alternative

Same vectors stored as a PyTorch dict keyed by node type:
```python
{"API_Class": Tensor[142, 256], "Concept": Tensor[17943, 256], ...}
```
~5x faster to load than JSONL. No `display_name` field — node name enrichment won't happen.

---

## Training Metrics (from `outputs/training_summary.json`)

| Relation | ROC AUC | Accuracy |
|----------|---------|----------|
| API_Endpoint → API_Function | 1.000 | 1.000 |
| API_Class → API_Method | 0.863 | 0.833 |
| API_Endpoint → API_Class | 0.811 | 0.857 |
| API_Endpoint → CodeSnippet | 0.672 | 0.650 |
| **Overall** | **0.668** | **0.701** |

---

## Re-training the GNN

If you modify `nodes.csv`/`edges.csv` and need new embeddings:

1. Open `notebooks/gnn_encoder_improved.ipynb`
2. Run all cells — it reads `data/nodes.csv` + `data/edges.csv`
3. Outputs are written to `outputs/` folder
4. The pipeline will automatically pick up the new `gnn_embeddings.jsonl`
