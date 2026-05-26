"""Tests for the Graph-RAG retriever pipeline.

Run:  python -m pytest tests/test_graph_rag.py -v
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import torch

from src.graph_rag.gnn_encoder import (
    GraphTensorData,
    _hashed_text_features,
    _sample_negative_pairs,
    build_graph_tensor_data,
    compute_graph_hash,
    load_embeddings,
    save_embeddings,
    train_gnn_embeddings,
)
from src.graph_rag.retriever import GraphRAGRetriever, RankingWeights
from src.graph_rag.graph_loader import CsvGraph


# ---------------------------------------------------------------------------
# Fixtures: tiny CSV graph (5 nodes, 4 edges)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_csv_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal nodes.csv and edges.csv."""
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"

    with nodes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Id", "Label", "Name", "URL"])
        w.writeheader()
        w.writerow({"Id": "0", "Label": "API_Module", "Name": "torch.compile", "URL": "https://pytorch.org/compile"})
        w.writerow({"Id": "1", "Label": "API_Function", "Name": "torch.nn.Linear", "URL": "https://pytorch.org/nn"})
        w.writerow({"Id": "2", "Label": "PyTorchConcept", "Name": "Transformer", "URL": "https://pytorch.org/transformer"})
        w.writerow({"Id": "3", "Label": "Guide", "Name": "compile guide", "URL": "https://pytorch.org/guide"})
        w.writerow({"Id": "4", "Label": "API_Class", "Name": "torch.optim.Adam", "URL": "https://pytorch.org/optim"})

    with edges_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Source", "Target", "Type"])
        w.writeheader()
        w.writerow({"Source": "0", "Target": "1", "Type": "CALLS"})
        w.writerow({"Source": "1", "Target": "2", "Type": "IMPLEMENTS"})
        w.writerow({"Source": "2", "Target": "3", "Type": "EXPLAINS"})
        w.writerow({"Source": "3", "Target": "4", "Type": "REFERENCES"})

    return nodes_csv, edges_csv


@pytest.fixture()
def tiny_graph(tiny_csv_dir: tuple[Path, Path]) -> CsvGraph:
    nodes_csv, edges_csv = tiny_csv_dir
    g = CsvGraph(nodes_csv, edges_csv)
    g.load()
    return g


# ---------------------------------------------------------------------------
# Tests: feature hashing
# ---------------------------------------------------------------------------

class TestFeatureHashing:
    def test_normalized(self):
        vec = _hashed_text_features("compile safe transformer", 128)
        assert vec.shape == (128,)
        assert torch.isclose(torch.norm(vec), torch.tensor(1.0), atol=1e-5)

    def test_empty_string(self):
        vec = _hashed_text_features("", 64)
        assert torch.norm(vec).item() == 0.0

    def test_deterministic(self):
        a = _hashed_text_features("torch linear", 128)
        b = _hashed_text_features("torch linear", 128)
        assert torch.equal(a, b)

    def test_different_inputs_differ(self):
        a = _hashed_text_features("compile", 128)
        b = _hashed_text_features("optimizer", 128)
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# Tests: query embedding (should match hashed_text_features space)
# ---------------------------------------------------------------------------

class TestQueryEmbedding:
    def test_normalized(self):
        vec = GraphRAGRetriever._query_embedding("compile safe transformer block", 128)
        assert vec.shape == (128,)
        assert torch.isclose(torch.norm(vec), torch.tensor(1.0), atol=1e-5)

    def test_matches_feature_space(self):
        """Query embedding should use the same hash function as node features."""
        q = GraphRAGRetriever._query_embedding("compile", 128)
        f = _hashed_text_features("compile", 128)
        # Single-token case: they must be identical.
        assert torch.allclose(q, f, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: build_graph_tensor_data
# ---------------------------------------------------------------------------

class TestBuildGraphTensorData:
    def test_shapes(self, tiny_graph: CsvGraph):
        data = build_graph_tensor_data(tiny_graph, feature_dim=64, allow_csv_training=True)
        assert len(data.node_ids) == 5
        assert data.x.shape == (5, 64)
        assert data.edge_index.shape[0] == 2
        # 4 edges × 2 (bi-directional) = 8
        assert data.edge_index.shape[1] == 8

    def test_rejects_without_flag(self, tiny_graph: CsvGraph):
        with pytest.raises(ValueError, match="not allowed"):
            build_graph_tensor_data(tiny_graph, allow_csv_training=False)


# ---------------------------------------------------------------------------
# Tests: negative sampling
# ---------------------------------------------------------------------------

class TestNegativeSampling:
    def test_excludes_positives(self):
        pos = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        pos_set = set(zip(pos[0].tolist(), pos[1].tolist()))
        neg = _sample_negative_pairs(5, 10, pos)
        neg_pairs = list(zip(neg[0].tolist(), neg[1].tolist()))
        for pair in neg_pairs:
            assert pair not in pos_set, f"Negative sample {pair} is a positive edge"

    def test_output_shape(self):
        pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        neg = _sample_negative_pairs(10, 20, pos)
        assert neg.shape == (2, 20)


# ---------------------------------------------------------------------------
# Tests: GNN training
# ---------------------------------------------------------------------------

class TestGNNTraining:
    def test_produces_embeddings(self, tiny_graph: CsvGraph):
        data = build_graph_tensor_data(tiny_graph, feature_dim=32, allow_csv_training=True)
        encoder, z = train_gnn_embeddings(data, hidden_dim=32, out_dim=16, epochs=3)
        assert z.shape == (5, 16)
        # Embeddings should be normalized.
        norms = torch.norm(z, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_pt(self, tmp_path: Path):
        node_ids = [0, 1, 2]
        emb = torch.randn(3, 16)
        inp = torch.randn(3, 32)
        save_embeddings(tmp_path / "test_emb", node_ids, emb, inp, graph_hash="abc123")
        loaded_ids, loaded_emb, loaded_inp, loaded_hash = load_embeddings(tmp_path / "test_emb")
        assert loaded_ids == node_ids
        assert torch.allclose(loaded_emb, emb)
        assert loaded_inp is not None
        assert torch.allclose(loaded_inp, inp)
        assert loaded_hash == "abc123"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Tests: cache versioning
# ---------------------------------------------------------------------------

class TestCacheVersioning:
    def test_hash_deterministic(self, tiny_csv_dir: tuple[Path, Path]):
        nodes_csv, edges_csv = tiny_csv_dir
        h1 = compute_graph_hash(nodes_csv, edges_csv)
        h2 = compute_graph_hash(nodes_csv, edges_csv)
        assert h1 == h2

    def test_hash_changes_on_modification(self, tiny_csv_dir: tuple[Path, Path]):
        nodes_csv, edges_csv = tiny_csv_dir
        h1 = compute_graph_hash(nodes_csv, edges_csv)
        # Append a node.
        with nodes_csv.open("a", encoding="utf-8") as f:
            f.write('5,API_Extra,torch.sum,https://pytorch.org/sum\n')
        h2 = compute_graph_hash(nodes_csv, edges_csv)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Tests: retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_retrieve_returns_context(self, tiny_graph: CsvGraph):
        data = build_graph_tensor_data(tiny_graph, feature_dim=32, allow_csv_training=True)
        _, emb = train_gnn_embeddings(data, hidden_dim=32, out_dim=16, epochs=3)
        retriever = GraphRAGRetriever(
            graph=tiny_graph,
            node_ids=data.node_ids,
            embeddings=emb,
            input_features=data.x,
        )
        ctx = retriever.retrieve("compile safe transformer", top_k=3, seed_k=2, expansion_hops=1)
        assert len(ctx.nodes) > 0
        assert isinstance(ctx.seed_nodes, list)
        assert ctx.query == "compile safe transformer"

    def test_custom_weights(self, tiny_graph: CsvGraph):
        data = build_graph_tensor_data(tiny_graph, feature_dim=32, allow_csv_training=True)
        _, emb = train_gnn_embeddings(data, hidden_dim=32, out_dim=16, epochs=3)
        weights = RankingWeights(gnn=0.0, lexical=10.0)  # lexical-only ranking
        retriever = GraphRAGRetriever(
            graph=tiny_graph,
            node_ids=data.node_ids,
            embeddings=emb,
            input_features=data.x,
            ranking_weights=weights,
        )
        ctx = retriever.retrieve("compile", top_k=3, seed_k=2)
        # "compile" should match nodes with "compile" in their text.
        names = [n.name for n in ctx.nodes]
        assert any("compile" in nm.lower() for nm in names)

    def test_fallback_without_input_features(self, tiny_graph: CsvGraph):
        """When input_features is None (legacy cache), retriever should still work."""
        data = build_graph_tensor_data(tiny_graph, feature_dim=32, allow_csv_training=True)
        _, emb = train_gnn_embeddings(data, hidden_dim=32, out_dim=16, epochs=3)
        retriever = GraphRAGRetriever(
            graph=tiny_graph,
            node_ids=data.node_ids,
            embeddings=emb,
            input_features=None,  # legacy fallback
        )
        ctx = retriever.retrieve("transformer", top_k=3, seed_k=2)
        assert len(ctx.nodes) > 0
