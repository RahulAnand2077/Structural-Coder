# Graph-RAG Retriever for PyTorch 2.x API Grounding

A Graph-RAG (Retrieval-Augmented Generation) retriever that finds relevant PyTorch API symbols from a knowledge graph using GNN-learned embeddings and hybrid ranking.

## How It Works

```
Query: "compile-safe transformer block"
                │
                ▼
┌──────────────────────────────────┐
│  1. Seed Selection               │  Find starting nodes via embedding similarity
│  2. Graph Expansion              │  Walk outward through edges to discover neighbors
│  3. Hybrid Ranking               │  Score candidates: GNN + lexical + degree
└──────────────────────────────────┘
                │
                ▼
Output: ranked list of PyTorch API symbols
        (Transformer, torch.compile, torch.nn.Linear, ...)
```

## Project Structure

```
Structural-Coder/
  run_graph_rag_pipeline.py     ← CLI entry point
  nodes.csv                     ← Knowledge graph nodes (PyTorch APIs)
  edges.csv                     ← Knowledge graph edges (relationships)
  requirements.txt              ← Python dependencies
  src/
    graph_rag/
      graph_loader.py           ← CSV → graph data structure
      gnn_encoder.py            ← GNN training + embedding cache
      retriever.py              ← Core retrieval algorithm
      pipeline.py               ← Orchestrates everything
  tests/
    test_graph_rag.py           ← 18 automated tests
```

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate    # or your venv path

# 2. Install dependencies
pip install torch>=2.2

# 3. Run with local GNN training
python3 run_graph_rag_pipeline.py \
  --allow-csv-gnn-training \
  --query "compile-safe transformer block" \
  --output-json outputs/result.json

# 4. Or run with external pre-trained embeddings
python3 run_graph_rag_pipeline.py \
  --external-embeddings path/to/gnn_embeddings.json \
  --query "compile-safe transformer block" \
  --output-json outputs/result.json
```

## Example Output

```json
{
  "query": "compile-safe transformer block",
  "seed_nodes": ["Transformer", "BlockingAsyncStager", "blocked_autorange"],
  "retrieved_symbols": ["Transformer", "torch.compile", "torch.nn.Linear", "..."],
  "retrieved_nodes": 20,
  "retrieved_edges": 74
}
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/test_graph_rag.py -v
```

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Graph Loader | `graph_loader.py` | Reads `nodes.csv`/`edges.csv` into a graph with adjacency lists |
| GNN Encoder | `gnn_encoder.py` | Trains a 2-layer GraphSAGE encoder via link prediction |
| Retriever | `retriever.py` | Seed selection → graph expansion → hybrid ranking |
| Pipeline | `pipeline.py` | Orchestrates load → embed → retrieve → output |
| CLI | `run_graph_rag_pipeline.py` | Terminal interface with all configuration flags |

## Key Design Decisions

- **Two embedding spaces**: Hash-bag vectors for seed selection (same space as queries) + GNN vectors for re-ranking (captures graph structure)
- **Cache versioning**: Embedding cache includes a SHA-256 hash of the CSV files — stale caches auto-retrain
- **Binary `.pt` format**: Embeddings saved as PyTorch binary (~57% smaller, ~10× faster than JSON)
- **Configurable ranking**: `RankingWeights` dataclass lets you tune GNN vs lexical vs degree weights

## Team

- **Amitesh** — Graph-RAG Retriever (this module)
- **Mohit** — GNN Encoder (Neo4j + torch_geometric training)
