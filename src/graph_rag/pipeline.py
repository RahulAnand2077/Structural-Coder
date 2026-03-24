from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.integration_pipeline.graph_loader import CsvGraph
from src.integration_pipeline.validator import ActiveValidator

from .generator import TransformerGeneratorBridge
from .gnn_encoder import (
    build_graph_tensor_data,
    load_embeddings,
    save_embeddings,
    train_gnn_embeddings,
)
from .retriever import GraphRAGRetriever


@dataclass
class GraphRAGRunResult:
    query: str
    seed_nodes: list[str]
    retrieved_symbols: list[str]
    retrieved_nodes: int
    retrieved_edges: int
    validation: dict | None
    generated_code: str


class StructuralCoderGraphRAG:
    def __init__(self, nodes_csv: str | Path, edges_csv: str | Path) -> None:
        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)

        self.graph = CsvGraph(self.nodes_csv, self.edges_csv)
        self.graph.load()

        self._node_ids: list[int] | None = None
        self._embeddings = None

        self.generator = TransformerGeneratorBridge()
        self.validator = ActiveValidator()

    def prepare_embeddings(
        self,
        embedding_cache: str | Path,
        retrain: bool = False,
        epochs: int = 20,
        hidden_dim: int = 128,
        out_dim: int = 96,
    ) -> None:
        cache_path = Path(embedding_cache)
        if cache_path.exists() and not retrain:
            node_ids, embeddings = load_embeddings(cache_path)
            self._node_ids = node_ids
            self._embeddings = embeddings
            return

        data = build_graph_tensor_data(self.nodes_csv, self.edges_csv, feature_dim=hidden_dim)
        _, embeddings = train_gnn_embeddings(
            data,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            epochs=epochs,
        )
        save_embeddings(cache_path, data.node_ids, embeddings)
        self._node_ids = data.node_ids
        self._embeddings = embeddings

    def run(
        self,
        query: str,
        top_k: int = 20,
        seed_k: int = 4,
        expansion_hops: int = 1,
        target_hardware: str = "H100",
        validate: bool = True,
    ) -> GraphRAGRunResult:
        if self._node_ids is None or self._embeddings is None:
            raise RuntimeError("Embeddings are not prepared. Call prepare_embeddings() first.")

        retriever = GraphRAGRetriever(
            graph=self.graph,
            node_ids=self._node_ids,
            embeddings=self._embeddings,
        )
        context = retriever.retrieve(query=query, top_k=top_k, seed_k=seed_k, expansion_hops=expansion_hops)
        generated = self.generator.generate(query=query, context=context)

        validation_payload: dict | None = None
        if validate:
            report = self.validator.validate(generated.code, target_hardware=target_hardware)
            validation_payload = {
                "passed": report.passed,
                "passed_with_skips": report.passed_with_skips,
                "passed_strict": report.passed_strict,
                "checks": [asdict(c) for c in report.checks],
            }

        return GraphRAGRunResult(
            query=query,
            seed_nodes=context.seed_nodes,
            retrieved_symbols=generated.used_symbols[: max(1, top_k)],
            retrieved_nodes=len(context.nodes),
            retrieved_edges=len(context.edges),
            validation=validation_payload,
            generated_code=generated.code,
        )

    @staticmethod
    def to_json(result: GraphRAGRunResult) -> str:
        return json.dumps(asdict(result), indent=2)
