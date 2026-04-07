from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData

from src.integration_pipeline.graph_loader import CsvGraph

# -- HETEROGENEOUS MODELS PORTED FROM NOTEBOOK --

def edge_type_to_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)

class HeteroGraphEncoder(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 96,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.node_types      = metadata[0]
        self.edge_types      = metadata[1]
        self.num_layers      = num_layers
        self.hidden_channels = hidden_channels

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, hidden_channels, bias=True)
            for nt in self.node_types
        })
        self.input_norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_channels) for nt in self.node_types
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=4,
                )
            )
            self.norms.append(
                nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in self.node_types})
            )

        jk_dim = hidden_channels * (1 + num_layers)
        self.jk_lin = nn.ModuleDict({
            nt: nn.Linear(jk_dim, out_channels) for nt in self.node_types
        })

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        projected: dict[str, torch.Tensor] = {
            nt: self.dropout(F.relu(self.input_norms[nt](self.input_proj[nt](x))))
            for nt, x in x_dict.items()
        }

        xs_all: dict[str, list[torch.Tensor]] = {nt: [h] for nt, h in projected.items()}
        hidden = projected

        for layer_idx, conv in enumerate(self.convs):
            out_dict = conv(hidden, edge_index_dict)
            new_hidden: dict[str, torch.Tensor] = {}

            for node_type in self.node_types:
                if node_type in out_dict:
                    feats   = out_dict[node_type]
                    normed  = self.norms[layer_idx][node_type](feats)
                    activated = self.dropout(F.relu(normed))
                    new_hidden[node_type] = activated
                else:
                    new_hidden[node_type] = hidden[node_type]

                xs_all[node_type].append(new_hidden[node_type])

            hidden = new_hidden

        return {
            nt: self.jk_lin[nt](torch.cat(xs, dim=-1))
            for nt, xs in xs_all.items()
        }

class LowRankBilinearDecoder(nn.Module):
    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        embedding_dim: int,
        rank: int = 32,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.U = nn.ParameterDict({
            edge_type_to_key(et): nn.Parameter(torch.empty(embedding_dim, rank))
            for et in edge_types
        })
        self.V = nn.ParameterDict({
            edge_type_to_key(et): nn.Parameter(torch.empty(embedding_dim, rank))
            for et in edge_types
        })
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for u, v in zip(self.U.values(), self.V.values()):
            nn.init.xavier_uniform_(u)
            nn.init.xavier_uniform_(v)

    def forward(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        key = edge_type_to_key(edge_type)
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]
        return ((src_z @ self.U[key]) * (dst_z @ self.V[key])).sum(dim=-1)

# -- HOMOGENOUS WRAPPERS FOR COMPATIBILITY --

@dataclass
class GraphTensorData:
    node_ids: list[int]
    hetero_data: HeteroData
    id_to_hetero: dict[int, tuple[str, int]]
    hetero_to_id: dict[str, list[int]]

def _hashed_text_features(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = torch.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def build_graph_tensor_data(
    nodes_csv: str | Path,
    edges_csv: str | Path,
    feature_dim: int = 128,
    allow_csv_training: bool = False,
) -> GraphTensorData:
    graph = CsvGraph(nodes_csv, edges_csv)
    graph.load()

    node_ids = sorted(graph.nodes.keys())
    
    # Group by label
    grouped_nodes = defaultdict(list)
    id_to_hetero = {}
    for nid in node_ids:
        node = graph.nodes[nid]
        lbl = node.label if node.label else "Unknown"
        grouped_nodes[lbl].append((nid, node))

    data = HeteroData()
    hetero_to_id = {}
    
    for lbl, nodes in grouped_nodes.items():
        x = torch.zeros((len(nodes), feature_dim), dtype=torch.float32)
        nids = []
        for i, (nid, node) in enumerate(nodes):
            text = f"{node.label} {node.name} {node.url}"
            x[i] = _hashed_text_features(text, feature_dim)
            id_to_hetero[nid] = (lbl, i)
            nids.append(nid)
        data[lbl].x = x
        hetero_to_id[lbl] = nids

    edge_buckets = defaultdict(list)
    for e in graph.edges:
        if e.source in id_to_hetero and e.target in id_to_hetero:
            src_lbl, src_idx = id_to_hetero[e.source]
            dst_lbl, dst_idx = id_to_hetero[e.target]
            rel = e.relation if e.relation else "RELATED_TO"
            edge_buckets[(src_lbl, rel, dst_lbl)].append([src_idx, dst_idx])
            # Reverse edge
            edge_buckets[(dst_lbl, f"rev_{rel}", src_lbl)].append([dst_idx, src_idx])

    for etype, pairs in edge_buckets.items():
        data[etype].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()

    return GraphTensorData(
        node_ids=node_ids, 
        hetero_data=data, 
        id_to_hetero=id_to_hetero, 
        hetero_to_id=hetero_to_id
    )

DEFAULT_SUPERVISION_RELATIONS = {
    "IMPLEMENTS", "CONTAINS", "HAS_PARAM", "CALLS", "RELATED_TO", "REPLACES",
}

def sample_hard_negative_edges(
    edge_index: torch.Tensor,
    num_dst_nodes: int,
    num_samples: int,
) -> torch.Tensor:
    if num_samples <= 0 or edge_index.size(1) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    pos_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    dst_indices = edge_index[1]
    degrees = torch.bincount(dst_indices, minlength=num_dst_nodes).float() + 1.0
    weights = degrees ** 0.75
    probs = weights / weights.sum()

    collected_src = []
    collected_dst = []
    oversample_factor = 3
    repeats = (num_samples * oversample_factor // edge_index.size(1)) + 1
    src_pool = edge_index[0].repeat(repeats)

    for _ in range(5):
        remaining = num_samples - len(collected_src)
        if remaining <= 0:
            break
        neg_dsts = torch.multinomial(probs, remaining * oversample_factor, replacement=True)
        srcs = src_pool[:len(neg_dsts)]
        for s, d in zip(srcs.tolist(), neg_dsts.tolist()):
            if (s, d) not in pos_set:
                collected_src.append(s)
                collected_dst.append(d)
            if len(collected_src) >= num_samples:
                break

    if not collected_src:
        return torch.empty((2, 0), dtype=torch.long)

    src_t = torch.tensor(collected_src[:num_samples], dtype=torch.long)
    dst_t = torch.tensor(collected_dst[:num_samples], dtype=torch.long)
    return torch.stack([src_t, dst_t], dim=0)

def train_gnn_embeddings(
    data: GraphTensorData,
    hidden_dim: int = 128,
    out_dim: int = 96,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(seed)
    hdata = data.hetero_data
    metadata = hdata.metadata()

    # Isolate supervision relations to prevent structural bleed
    supervised_edge_types = [
        et for et in metadata[1]
        if et[1] in DEFAULT_SUPERVISION_RELATIONS
    ]
    if not supervised_edge_types:
        supervised_edge_types = metadata[1]
    
    encoder = HeteroGraphEncoder(
        metadata=metadata, 
        in_channels=hdata[metadata[0][0]].x.size(1), 
        hidden_channels=hidden_dim, 
        out_channels=out_dim
    )
    predictor = LowRankBilinearDecoder(edge_types=supervised_edge_types, embedding_dim=out_dim)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    for _ in range(max(1, epochs)):
        encoder.train()
        optimizer.zero_grad()
        z_dict = encoder(hdata.x_dict, hdata.edge_index_dict)
        
        loss = 0.0
        for edge_type in supervised_edge_types:
            ei = hdata[edge_type].edge_index
            if ei.size(1) == 0:
                continue
            
            # Predict positive edges
            pos_logits = predictor(z_dict, edge_type, ei)
            
            # True-positive filtered hard negative sampling
            num_dst = z_dict[edge_type[2]].size(0)
            neg_ei = sample_hard_negative_edges(ei, num_dst, ei.size(1))
            
            if neg_ei.size(1) > 0:
                neg_ei = neg_ei.to(ei.device)
                neg_logits = predictor(z_dict, edge_type, neg_ei)
                pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
                loss += pos_loss + neg_loss
            else:
                loss += F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            
        if hasattr(loss, "backward"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=2.0)
            optimizer.step()

    encoder.eval()
    with torch.no_grad():
        z_dict = encoder(hdata.x_dict, hdata.edge_index_dict)
        
    # Re-flatten for homogeneous GraphRAGRetriever
    flat_z = torch.zeros((len(data.node_ids), out_dim), dtype=torch.float32)
    id_to_dist = {nid: i for i, nid in enumerate(data.node_ids)}
    
    for lbl, z in z_dict.items():
        nids = data.hetero_to_id[lbl]
        for i, nid in enumerate(nids):
            flat_z[id_to_dist[nid]] = z[i]
            
    # Normalize globally for absolute cosine sim
    flat_z = F.normalize(flat_z, p=2, dim=1)

    return encoder, flat_z

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

def load_embeddings_from_jsonl(path: str | Path) -> tuple[list[int], torch.Tensor]:
    """Load embeddings from neo4j-style jsonl export."""
    node_ids = []
    embeddings = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # element_id is like "4:170763f8-8d9f-40c2-b8d0-f5cb2fbaa0a1:22717"
            raw_id = data["element_id"]
            nid = int(raw_id.rsplit(":", 1)[-1])
            emb = data["embedding"]
            node_ids.append(nid)
            embeddings.append(emb)
    
    return node_ids, torch.tensor(embeddings, dtype=torch.float32)

def load_query_encoder(
    metadata_path: str | Path,
    weights_path: str | Path,
    in_channels: int = 128,
    hidden_channels: int = 256,
    out_channels: int = 256
) -> HeteroGraphEncoder | None:
    """Load the HeteroGraphEncoder for query embedding alignment."""
    try:
        mp = Path(metadata_path)
        wp = Path(weights_path)
        if not mp.exists() or not wp.exists():
            return None

        # metadata is expected to have 'node_types' and 'edge_types'
        meta_dict = json.loads(mp.read_text(encoding="utf-8"))
        node_types = meta_dict.get("node_types", ["PyTorchConcept"])
        # edge_types from json comes as lists of 3 strings; convert to tuple
        edge_types = [tuple(e) for e in meta_dict.get("edge_types", [])]
        
        model = HeteroGraphEncoder(
            metadata=(node_types, edge_types),
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        )
        
        state = torch.load(str(wp), map_location="cpu", weights_only=False)
        # Handle dict from saving logic "model_state"
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
            
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Failed to load query encoder: {e}")
        return None
