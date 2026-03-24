# Mohit's GNN Encoder

This folder is now notebook-first to match the rest of the project.

Main file:

- `gnn_encoder.ipynb`

## What The Notebook Covers

The notebook does the full GNN part end to end:

1. pulls the knowledge graph and node embeddings from Neo4j
2. converts the graph into `torch_geometric.data.HeteroData`
3. trains a heterogeneous GNN with self-supervised link prediction
4. evaluates the trained model
5. exports graph-aware node embeddings
6. optionally writes those new embeddings back to Neo4j as `gnn_embedding`

## Important Idea

The original MiniLM embeddings are the input node features.

They do not need to be overwritten.

The GNN learns a new embedding on top of them:

```text
x = original text embedding
z = GNN(x, graph)
```

Best practice:

- keep the original Neo4j property as `embedding`
- store the trained graph-aware vectors as `gnn_embedding`

## Setup

Install the dependencies for this folder with:

`pip install -r requirements.txt`

## Output Files

After running the notebook, the main outputs are:

- `outputs/hetero_graph.pt`
- `outputs/hetero_metadata.json`
- `outputs/train_graph.pt`
- `outputs/split_state.pt`
- `outputs/best_model.pt`
- `outputs/training_summary.json`
- `outputs/gnn_embeddings.jsonl`
- `outputs/gnn_embeddings.pt`

## Short Team Explanation

The knowledge graph and initial text embeddings were already built by the previous stage. This notebook trains a heterogeneous GNN on that Neo4j graph and produces graph-aware embeddings for the later Graph-RAG retrieval stage.
