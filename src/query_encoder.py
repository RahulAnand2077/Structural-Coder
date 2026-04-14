"""
GNN Query Predictor
===================
Uses the trained GNN model (best_model.pt) to predict which API path a query needs.

Instead of just cosine similarity on pre-computed vectors, this module uses the
trained bilinear decoder weights to score how likely it is that a query "connects"
to each API node — the same scoring function the GNN was trained to optimize.

Usage:
    predictor = GNNQueryPredictor(
        model_path="outputs/best_model.pt",
        embeddings_pt="outputs/gnn_embeddings.pt",
        nodes_csv="data/nodes.csv",
    )
    path = predictor.predict_path("flash attention with fallback", top_k=10)
    # Returns: [(node_id, score), ...] — the GNN-predicted API path
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PredictedPath:
    """The GNN's predicted API path for a query."""
    query: str
    nodes: list[tuple[int, float]]  # (node_id, decoder_score)
    query_embedding: torch.Tensor   # 256-D query vector


class GNNQueryPredictor:
    """Uses the trained GNN decoder to predict which API nodes a query needs.
    
    Architecture:
        1. Hash query text → 1024-D feature vector (same hashing as training)
        2. Project through trained input_proj layer → 256-D query vector
        3. Score query against all node embeddings using trained bilinear decoder
        4. Return top-K nodes ranked by decoder score
    
    This uses the ACTUAL trained model weights (best_model.pt) for inference,
    not just pre-computed cosine similarity.
    """

    def __init__(
        self,
        model_path: str | Path,
        embeddings_pt: str | Path,
        nodes_csv: str | Path,
        query_node_type: str = "Concept",
    ) -> None:
        self.query_node_type = query_node_type
        
        # ── Load model weights ──────────────────────────────────
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        state = checkpoint["model_state"]
        config = checkpoint.get("model_config", {})
        self.thresholds = checkpoint.get("thresholds", {})
        
        self.in_channels = None  # detected from weights
        self.hidden_channels = config.get("hidden_channels", 256)
        self.out_channels = config.get("out_channels", 256)
        self.decoder_rank = config.get("decoder_rank", 32)
        self.scored_edge_types = [tuple(et) for et in config.get("scored_edge_types", [])]
        
        # ── Extract input projection for query node type ────────
        proj_key = f"encoder.input_proj.{query_node_type}.weight"
        norm_w_key = f"encoder.input_norms.{query_node_type}.weight"
        norm_b_key = f"encoder.input_norms.{query_node_type}.bias"
        
        if proj_key not in state:
            raise ValueError(f"No input_proj weights for node type '{query_node_type}' in model")
        
        self.input_proj_weight = state[proj_key]  # [hidden, in_channels]
        self.input_proj_bias = state[f"encoder.input_proj.{query_node_type}.bias"]
        self.input_norm_weight = state[norm_w_key]
        self.input_norm_bias = state[norm_b_key]
        self.in_channels = self.input_proj_weight.shape[1]  # 1024
        
        # ── Extract decoder U/V matrices ────────────────────────
        self.decoder_U: dict[str, torch.Tensor] = {}
        self.decoder_V: dict[str, torch.Tensor] = {}
        for key, val in state.items():
            if key.startswith("decoder.U."):
                edge_key = key[len("decoder.U."):]
                self.decoder_U[edge_key] = val
            elif key.startswith("decoder.V."):
                edge_key = key[len("decoder.V."):]
                self.decoder_V[edge_key] = val
        
        print(f"  ✅ Loaded model: in={self.in_channels}, hidden={self.hidden_channels}, "
              f"out={self.out_channels}, decoder_rank={self.decoder_rank}")
        print(f"  ✅ Decoder edge types: {list(self.decoder_U.keys())}")
        
        # ── Load pre-computed node embeddings ───────────────────
        self._load_embeddings(embeddings_pt, nodes_csv)
    
    def _load_embeddings(self, embeddings_pt: str | Path, nodes_csv: str | Path) -> None:
        """Load pre-computed 256-D node embeddings grouped by type."""
        import csv
        
        emb_dict = torch.load(str(embeddings_pt), map_location="cpu", weights_only=False)
        
        # Group CSV nodes by label to get ID mapping
        grouped: dict[str, list[int]] = {}
        with open(nodes_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("Label", "Unknown")
                grouped.setdefault(lbl, []).append(int(row["Id"]))
        
        # Store per-type embeddings and ID mappings
        self.type_embeddings: dict[str, torch.Tensor] = {}
        self.type_node_ids: dict[str, list[int]] = {}
        
        total = 0
        for node_type, tensor in emb_dict.items():
            csv_ids = grouped.get(node_type, [])
            if len(csv_ids) != tensor.size(0):
                print(f"  ⚠️  PT {node_type}: {tensor.size(0)} vectors vs {len(csv_ids)} CSV nodes — skipping")
                continue
            self.type_embeddings[node_type] = F.normalize(tensor, dim=-1)
            self.type_node_ids[node_type] = csv_ids
            total += len(csv_ids)
        
        print(f"  ✅ Loaded {total} node embeddings across {len(self.type_embeddings)} types")

    def _hash_text(self, text: str) -> torch.Tensor:
        """Hash text to feature vector — same method used during GNN training."""
        vec = torch.zeros(self.in_channels, dtype=torch.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.in_channels] += 1.0
        norm = torch.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _project_query(self, text: str) -> torch.Tensor:
        """Project query text through the trained input_proj → 256-D vector.
        
        This uses the SAME projection weights the GNN was trained with,
        ensuring the query lands in the same embedding space as the graph nodes.
        """
        # Hash → 1024-D
        x = self._hash_text(text).unsqueeze(0)  # [1, 1024]
        
        # Input projection (same as encoder.input_proj.Concept)
        h = F.linear(x, self.input_proj_weight, self.input_proj_bias)  # [1, 256]
        
        # LayerNorm (same as encoder.input_norms.Concept)
        h = F.layer_norm(h, [self.hidden_channels], self.input_norm_weight, self.input_norm_bias)
        
        # ReLU + L2 normalize (same as training forward pass)
        h = F.relu(h)
        h = F.normalize(h, dim=-1)
        
        return h.squeeze(0)  # [256]

    def predict_path(
        self,
        query: str,
        top_k: int = 10,
        target_types: Optional[list[str]] = None,
        graph: Optional[object] = None,
    ) -> PredictedPath:
        """Predict which API nodes are most relevant to the query.
        
        Combines:
        1. Trained bilinear decoder: score = (query @ U) · (node @ V)
        2. Lexical boost: nodes whose names match query tokens get a bonus
        
        Args:
            query: Natural language query
            top_k: Number of nodes to return
            target_types: Node types to score against. Default = all callable types
            graph: Optional CsvGraph for lexical boosting against node names
        """
        if target_types is None:
            target_types = ["API_Class", "API_Function", "API_Method", "API_Endpoint", "CodeSnippet"]
        
        # Project query → 256-D using trained weights
        query_vec = self._project_query(query)
        
        # Extract query tokens for lexical boosting
        query_tokens = set(query.lower().replace(".", " ").replace("_", " ").split())
        query_tokens -= {"a", "an", "the", "for", "with", "and", "or", "to", "in", "of", "is"}
        
        all_scored: list[tuple[int, float]] = []
        
        for target_type in target_types:
            if target_type not in self.type_embeddings:
                continue
            
            node_embeddings = self.type_embeddings[target_type]  # [N, 256]
            node_ids = self.type_node_ids[target_type]
            
            # GNN decoder scoring
            decoder_key = self._find_decoder_key(self.query_node_type, target_type)
            
            if decoder_key and decoder_key in self.decoder_U:
                U = self.decoder_U[decoder_key]  # [256, 32]
                V = self.decoder_V[decoder_key]  # [256, 32]
                query_proj = query_vec @ U  # [32]
                node_proj = node_embeddings @ V  # [N, 32]
                decoder_scores = (node_proj * query_proj.unsqueeze(0)).sum(dim=-1)  # [N]
            else:
                decoder_scores = node_embeddings @ query_vec  # [N] cosine fallback
            
            # Combine decoder score with lexical boost
            for i, nid in enumerate(node_ids):
                score = float(decoder_scores[i])
                
                # Lexical boost: if node name matches query tokens, boost heavily
                if graph and nid in graph.nodes:
                    node = graph.nodes[nid]
                    name_tokens = set(node.name.lower().replace(".", " ").replace("_", " ").split())
                    overlap = query_tokens & name_tokens
                    if overlap:
                        score += len(overlap) * 2.0  # Strong lexical boost
                    
                    # Bonus for callable API types
                    if node.label in ("API_Class", "API_Function", "API_Method"):
                        score += 0.5
                
                all_scored.append((nid, score))
        
        all_scored.sort(key=lambda x: x[1], reverse=True)
        
        return PredictedPath(
            query=query,
            nodes=all_scored[:top_k],
            query_embedding=query_vec,
        )
    
    def _find_decoder_key(self, src_type: str, dst_type: str) -> Optional[str]:
        """Find the best decoder edge type key for scoring src→dst.
        
        Strategy:
        1. Direct match: src__*__dst
        2. Any edge ending in dst_type (use the one closest to our query context)
        3. None (fallback to cosine)
        """
        # Direct match
        for key in self.decoder_U:
            parts = key.split("__")
            if len(parts) == 3 and parts[0] == src_type and parts[2] == dst_type:
                return key
        
        # Any edge targeting dst_type
        for key in self.decoder_U:
            parts = key.split("__")
            if len(parts) == 3 and parts[2] == dst_type:
                return key
        
        # Any edge from dst_type (reverse — score dst as source)
        for key in self.decoder_U:
            parts = key.split("__")
            if len(parts) == 3 and parts[0] == dst_type:
                return key
        
        return None
