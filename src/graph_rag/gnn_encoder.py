from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.integration_pipeline.graph_loader import CsvGraph


@dataclass
class GraphTensorData:
    node_ids: list[int]
    edge_index: torch.Tensor
    x: torch.Tensor

    @property
    def id_to_index(self) -> dict[int, int]:
        return {nid: i for i, nid in enumerate(self.node_ids)}


class GraphSAGEEncoder(nn.Module):
    """Minimal GraphSAGE-style encoder implemented in pure torch."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 96) -> None:
        super().__init__()
        self.lin_self_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_self_2 = nn.Linear(hidden_dim, out_dim)
        self.lin_neigh_2 = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _mean_aggregate(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index shape: [2, E] with directed src->dst
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        deg = torch.zeros(h.size(0), 1, device=h.device)
        agg.index_add_(0, dst, h[src])
        deg.index_add_(0, dst, torch.ones((src.numel(), 1), device=h.device))
        deg = torch.clamp(deg, min=1.0)
        return agg / deg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n1 = self._mean_aggregate(x, edge_index)
        h = F.relu(self.lin_self_1(x) + self.lin_neigh_1(n1))
        n2 = self._mean_aggregate(h, edge_index)
        z = self.lin_self_2(h) + self.lin_neigh_2(n2)
        return F.normalize(z, p=2, dim=1)


class LinkPredictor(nn.Module):
    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        src = z[edge_pairs[0]]
        dst = z[edge_pairs[1]]
        return (src * dst).sum(dim=1)


def _hashed_text_features(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = torch.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_graph_tensor_data(nodes_csv: str | Path, edges_csv: str | Path, feature_dim: int = 128) -> GraphTensorData:
    graph = CsvGraph(nodes_csv, edges_csv)
    graph.load()

    node_ids = sorted(graph.nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    x = torch.zeros((len(node_ids), feature_dim), dtype=torch.float32)
    for nid, idx in id_to_idx.items():
        node = graph.nodes[nid]
        text = f"{node.label} {node.name} {node.url}"
        x[idx] = _hashed_text_features(text, feature_dim)

    edges = []
    for e in graph.edges:
        if e.source in id_to_idx and e.target in id_to_idx:
            # bi-directional edges for message passing stability
            edges.append((id_to_idx[e.source], id_to_idx[e.target]))
            edges.append((id_to_idx[e.target], id_to_idx[e.source]))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return GraphTensorData(node_ids=node_ids, edge_index=edge_index, x=x)


def _sample_negative_pairs(num_nodes: int, num_samples: int) -> torch.Tensor:
    src = torch.randint(0, num_nodes, (num_samples,), dtype=torch.long)
    dst = torch.randint(0, num_nodes, (num_samples,), dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def train_gnn_embeddings(
    data: GraphTensorData,
    hidden_dim: int = 128,
    out_dim: int = 96,
    epochs: int = 20,
    lr: float = 1e-3,
    max_pos_edges_per_epoch: int = 6000,
    seed: int = 42,
) -> tuple[GraphSAGEEncoder, torch.Tensor]:
    torch.manual_seed(seed)
    encoder = GraphSAGEEncoder(in_dim=data.x.size(1), hidden_dim=hidden_dim, out_dim=out_dim)
    predictor = LinkPredictor()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    pos_edges = data.edge_index
    if pos_edges.size(1) == 0:
        with torch.no_grad():
            z = encoder(data.x, data.edge_index)
        return encoder, z

    for _ in range(max(1, epochs)):
        encoder.train()
        optimizer.zero_grad()

        z = encoder(data.x, data.edge_index)
        if pos_edges.size(1) > max_pos_edges_per_epoch:
            idx = torch.randperm(pos_edges.size(1))[:max_pos_edges_per_epoch]
            pos_batch = pos_edges[:, idx]
        else:
            pos_batch = pos_edges

        neg_batch = _sample_negative_pairs(z.size(0), pos_batch.size(1))
        pos_logits = predictor(z, pos_batch)
        neg_logits = predictor(z, neg_batch)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)
    return encoder, z


def save_embeddings(path: str | Path, node_ids: list[int], embeddings: torch.Tensor) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "node_ids": node_ids,
        "embeddings": embeddings.cpu().tolist(),
    }
    target.write_text(json.dumps(payload), encoding="utf-8")


def load_embeddings(path: str | Path) -> tuple[list[int], torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    node_ids = [int(n) for n in payload["node_ids"]]
    embeddings = torch.tensor(payload["embeddings"], dtype=torch.float32)
    return node_ids, embeddings
