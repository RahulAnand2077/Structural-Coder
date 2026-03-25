"""
GNN Encoder — Embedding Builder
=================================

This file handles everything related to node embeddings:

    1. FEATURE HASHING:  Convert node text → fixed-size vectors (hash-bag)
    2. GNN TRAINING:     Train a GraphSAGE encoder on the graph structure
    3. CACHE MANAGEMENT: Save/load embeddings in binary .pt format

The embeddings it produces are used by retriever.py for ranking nodes.

Note: This is a standalone reimplementation of Mohit's torch_geometric
      GNN encoder, using only pure PyTorch (no external GNN library needed).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_loader import CsvGraph


# ──────────────────────────────────────────────────────────────────
# Data container for graph tensors
# ──────────────────────────────────────────────────────────────────

@dataclass
class GraphTensorData:
    """Holds all tensor data needed for GNN training."""
    node_ids: list[int]        # original node IDs from CSV
    edge_index: torch.Tensor   # shape [2, E] — all edges as (src, dst) pairs
    x: torch.Tensor            # shape [N, feature_dim] — node feature vectors

    @property
    def id_to_index(self) -> dict[int, int]:
        """Map from node ID → tensor row index."""
        return {nid: i for i, nid in enumerate(self.node_ids)}


# ──────────────────────────────────────────────────────────────────
# Feature Hashing: text → vector
# ──────────────────────────────────────────────────────────────────

def _hashed_text_features(text: str, dim: int) -> torch.Tensor:
    """Convert text to a fixed-size vector using hash-bag encoding.

    How it works:
        1. Split text into tokens
        2. Hash each token with SHA-256 (deterministic)
        3. Use hash % dim to pick a vector position
        4. Increment that position
        5. Normalize to unit length

    Example:
        "torch compile linear" → [0, 0.57, 0, ..., 0.82, 0, ...]
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    for tok in text.lower().split():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = torch.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ──────────────────────────────────────────────────────────────────
# GraphSAGE Encoder (2-layer, pure PyTorch)
# ──────────────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE encoder.

    How it works:
        Layer 1: h = ReLU(W_self · x + W_neigh · mean_aggregate(x))
        Layer 2: z = W_self · h + W_neigh · mean_aggregate(h)
        Output:  normalize(z) → unit vectors for cosine similarity
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 96) -> None:
        super().__init__()
        # Layer 1: input → hidden
        self.lin_self_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh_1 = nn.Linear(in_dim, hidden_dim)
        # Layer 2: hidden → output
        self.lin_self_2 = nn.Linear(hidden_dim, out_dim)
        self.lin_neigh_2 = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _mean_aggregate(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute mean of neighbor features for each node."""
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        deg = torch.zeros(h.size(0), 1, device=h.device)
        agg.index_add_(0, dst, h[src])
        deg.index_add_(0, dst, torch.ones((src.numel(), 1), device=h.device))
        deg = torch.clamp(deg, min=1.0)
        return agg / deg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass: input features → output embeddings."""
        # Layer 1: combine self features + neighbor features.
        n1 = self._mean_aggregate(x, edge_index)
        h = F.relu(self.lin_self_1(x) + self.lin_neigh_1(n1))
        # Layer 2: refine with another round of neighbor aggregation.
        n2 = self._mean_aggregate(h, edge_index)
        z = self.lin_self_2(h) + self.lin_neigh_2(n2)
        # Normalize to unit vectors (for cosine similarity).
        return F.normalize(z, p=2, dim=1)


class LinkPredictor(nn.Module):
    """Predict whether an edge exists between two nodes (dot product)."""
    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        src = z[edge_pairs[0]]
        dst = z[edge_pairs[1]]
        return (src * dst).sum(dim=1)


# ──────────────────────────────────────────────────────────────────
# Build tensor data from CSV graph
# ──────────────────────────────────────────────────────────────────

def build_graph_tensor_data(
    graph: CsvGraph,
    feature_dim: int = 128,
    allow_csv_training: bool = False,
) -> GraphTensorData:
    """Convert a loaded CsvGraph into tensors for GNN training.

    Args:
        graph: Already-loaded CsvGraph instance.
        feature_dim: Size of hash-bag feature vectors.
        allow_csv_training: Safety flag (must be True to proceed).
    """
    if not allow_csv_training:
        raise ValueError(
            "CSV snapshots (nodes.csv/edges.csv) are plotting/retrieval artifacts and are "
            "not allowed for GNN training unless --allow-csv-gnn-training is explicitly set."
        )

    # Map each node ID to a consecutive index (0, 1, 2, ...).
    node_ids = sorted(graph.nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build feature matrix: one hash-bag vector per node.
    x = torch.zeros((len(node_ids), feature_dim), dtype=torch.float32)
    for nid, idx in id_to_idx.items():
        node = graph.nodes[nid]
        text = f"{node.label} {node.name} {node.url}"
        x[idx] = _hashed_text_features(text, feature_dim)

    # Build edge index: bidirectional for message passing.
    edges = []
    for e in graph.edges:
        if e.source in id_to_idx and e.target in id_to_idx:
            edges.append((id_to_idx[e.source], id_to_idx[e.target]))
            edges.append((id_to_idx[e.target], id_to_idx[e.source]))  # reverse

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return GraphTensorData(node_ids=node_ids, edge_index=edge_index, x=x)


# ──────────────────────────────────────────────────────────────────
# Negative sampling (avoids real edges)
# ──────────────────────────────────────────────────────────────────

def _sample_negative_pairs(
    num_nodes: int,
    num_samples: int,
    positive_edge_index: torch.Tensor,
) -> torch.Tensor:
    """Sample random node pairs that are NOT existing edges.

    This is important for GNN training quality — if negatives
    accidentally include real edges, the model gets confused.
    """
    # Build a set of all real edges for fast lookup.
    pos_set: set[tuple[int, int]] = set()
    if positive_edge_index.numel() > 0:
        src_list = positive_edge_index[0].tolist()
        dst_list = positive_edge_index[1].tolist()
        pos_set = set(zip(src_list, dst_list))

    neg_src: list[int] = []
    neg_dst: list[int] = []
    attempts = 0
    max_attempts = num_samples * 4

    while len(neg_src) < num_samples and attempts < max_attempts:
        batch = max(256, num_samples - len(neg_src))
        s = torch.randint(0, num_nodes, (batch,)).tolist()
        d = torch.randint(0, num_nodes, (batch,)).tolist()
        for si, di in zip(s, d):
            if si == di:                   # skip self-loops
                continue
            if (si, di) in pos_set:        # skip real edges
                continue
            neg_src.append(si)
            neg_dst.append(di)
            if len(neg_src) >= num_samples:
                break
        attempts += batch

    # Fallback for extremely dense graphs.
    while len(neg_src) < num_samples:
        neg_src.append(torch.randint(0, num_nodes, (1,)).item())
        neg_dst.append(torch.randint(0, num_nodes, (1,)).item())

    return torch.stack(
        [torch.tensor(neg_src[:num_samples], dtype=torch.long),
         torch.tensor(neg_dst[:num_samples], dtype=torch.long)],
        dim=0,
    )


# ──────────────────────────────────────────────────────────────────
# GNN Training
# ──────────────────────────────────────────────────────────────────

def train_gnn_embeddings(
    data: GraphTensorData,
    hidden_dim: int = 128,
    out_dim: int = 96,
    epochs: int = 20,
    lr: float = 1e-3,
    max_pos_edges_per_epoch: int = 6000,
    seed: int = 42,
) -> tuple[GraphSAGEEncoder, torch.Tensor]:
    """Train a GraphSAGE encoder and return the learned embeddings.

    Training objective: link prediction (predict existing edges).
    The resulting embeddings capture graph structure.

    Returns:
        (trained_encoder, embeddings_tensor)
    """
    torch.manual_seed(seed)
    in_dim = data.x.size(1)
    encoder = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    predictor = LinkPredictor()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )

    pos_edges = data.edge_index
    if pos_edges.size(1) == 0:
        # No edges → just return features passed through the encoder.
        with torch.no_grad():
            z = encoder(data.x, data.edge_index)
        return encoder, z

    # Training loop.
    for epoch in range(max(1, epochs)):
        encoder.train()
        optimizer.zero_grad()

        # Forward pass: get node embeddings.
        z = encoder(data.x, data.edge_index)

        # Sample a batch of positive edges (subsample if too many).
        if pos_edges.size(1) > max_pos_edges_per_epoch:
            idx = torch.randperm(pos_edges.size(1))[:max_pos_edges_per_epoch]
            pos_batch = pos_edges[:, idx]
        else:
            pos_batch = pos_edges

        # Sample negative edges (pairs that should NOT have high similarity).
        neg_batch = _sample_negative_pairs(z.size(0), pos_batch.size(1), pos_edges)

        # Compute loss: positives should score high, negatives should score low.
        pos_logits = predictor(z, pos_batch)
        neg_logits = predictor(z, neg_batch)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

    # Final embeddings (no gradient needed).
    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)
    return encoder, z


# ──────────────────────────────────────────────────────────────────
# Cache versioning
# ──────────────────────────────────────────────────────────────────

def compute_graph_hash(nodes_csv: str | Path, edges_csv: str | Path) -> str:
    """SHA-256 hash of CSV file contents.

    Used to detect when the graph data has changed, so we know
    whether cached embeddings are still valid.
    """
    h = hashlib.sha256()
    for fpath in (nodes_csv, edges_csv):
        h.update(Path(fpath).read_bytes())
    return h.hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────
# Save / Load embeddings
# ──────────────────────────────────────────────────────────────────

def save_embeddings(
    path: str | Path,
    node_ids: list[int],
    embeddings: torch.Tensor,
    input_features: torch.Tensor,
    graph_hash: str = "",
) -> None:
    """Save embeddings to binary .pt file.

    Stores:
        - node_ids: which nodes these embeddings belong to
        - embeddings: GNN-learned vectors (for re-ranking)
        - input_features: hash-bag vectors (for seed selection)
        - graph_hash: for detecting stale caches
    """
    target = Path(path).with_suffix(".pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "node_ids": node_ids,
        "embeddings": embeddings.cpu(),
        "input_features": input_features.cpu(),
        "graph_hash": graph_hash,
    }, target)


def load_embeddings(
    path: str | Path,
) -> tuple[list[int], torch.Tensor, torch.Tensor | None, str]:
    """Load embeddings, trying .pt first, then legacy .json.

    Returns: (node_ids, embeddings, input_features_or_None, graph_hash)
    """
    pt_path = Path(path).with_suffix(".pt")
    json_path = Path(path).with_suffix(".json")

    # Try binary format first (faster, smaller).
    if pt_path.exists():
        payload = torch.load(pt_path, weights_only=False)
        return (
            payload["node_ids"],
            payload["embeddings"],
            payload.get("input_features"),
            payload.get("graph_hash", ""),
        )

    # Fall back to legacy JSON format.
    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        node_ids = [int(n) for n in raw["node_ids"]]
        embeddings = torch.tensor(raw["embeddings"], dtype=torch.float32)
        return node_ids, embeddings, None, ""

    raise FileNotFoundError(f"No embedding cache found at {pt_path} or {json_path}")
