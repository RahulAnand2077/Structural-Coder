from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.graph_rag.gnn_encoder import (
    build_graph_tensor_data,
    save_embeddings,
    train_gnn_embeddings,
)
from src.graph_rag.retriever import GraphRAGRetriever
from src.integration_pipeline.graph_loader import CsvGraph, Node
from src.integration_pipeline.retriever import GraphRetriever
from src.integration_pipeline.validator import ActiveValidator


@dataclass
class StepReport:
    name: str
    details: dict


class CsvFirstResearchPipeline:
    """Research workflow that is strictly anchored to nodes.csv and edges.csv."""

    def __init__(self, nodes_csv: str | Path, edges_csv: str | Path, embedding_cache: str | Path) -> None:
        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)
        self.embedding_cache = Path(embedding_cache)

        self.graph = CsvGraph(self.nodes_csv, self.edges_csv)
        self.graph.load()

        self._gnn_node_ids: list[int] | None = None
        self._gnn_embeddings = None
        self._gnn_retriever: GraphRAGRetriever | None = None
        self.validator = ActiveValidator()

    def step1_data_scrape(self) -> StepReport:
        details = {
            "input_nodes_csv": str(self.nodes_csv),
            "input_edges_csv": str(self.edges_csv),
            "nodes_loaded": len(self.graph.nodes),
            "edges_loaded": len(self.graph.edges),
            "note": "CSV-first mode: tutorials are consumed from existing snapshots (nodes.csv/edges.csv).",
        }
        return StepReport(name="Data Scrape", details=details)

    def step2_graph_construction(self) -> StepReport:
        relation_histogram: dict[str, int] = {}
        for edge in self.graph.edges:
            key = (edge.relation or "RELATED_TO").upper()
            relation_histogram[key] = relation_histogram.get(key, 0) + 1

        top_relations = sorted(relation_histogram.items(), key=lambda x: x[1], reverse=True)[:10]
        details = {
            "graph_source": "nodes.csv + edges.csv",
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "top_relations": [{"relation": rel, "count": count} for rel, count in top_relations],
        }
        return StepReport(name="Graph Construction", details=details)

    def step3_gnn_training(
        self,
        hidden_dim: int = 128,
        out_dim: int = 96,
        epochs: int = 20,
    ) -> StepReport:
        if self._gnn_retriever is not None:
            return StepReport(name="GNN Training", details={"status": "already_loaded"})

        if self.embedding_cache.exists():
            query_encoder = None
            if self.embedding_cache.suffix == ".jsonl":
                from src.graph_rag.gnn_encoder import load_embeddings_from_jsonl, load_query_encoder
                self._gnn_node_ids, self._gnn_embeddings = load_embeddings_from_jsonl(self.embedding_cache)
                # Auto-load the corresponding best_model.pt to properly embed query texts if available
                query_encoder = load_query_encoder(
                    metadata_path=self.embedding_cache.parent / "hetero_metadata.json",
                    weights_path=self.embedding_cache.parent / "best_model.pt",
                )
            else:
                from src.graph_rag.gnn_encoder import load_embeddings
                self._gnn_node_ids, self._gnn_embeddings = load_embeddings(self.embedding_cache)

            self._gnn_retriever = GraphRAGRetriever(
                graph=self.graph,
                node_ids=self._gnn_node_ids,
                embeddings=self._gnn_embeddings,
                query_encoder=query_encoder,
            )
            return StepReport(name="GNN Training", details={"status": "loaded_from_cache", "cache_path": str(self.embedding_cache), "query_encoder_loaded": query_encoder is not None})

        tensor_data = build_graph_tensor_data(
            self.nodes_csv,
            self.edges_csv,
            feature_dim=hidden_dim,
            allow_csv_training=True,
        )
        _, embeddings = train_gnn_embeddings(
            tensor_data,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            epochs=epochs,
        )

        save_embeddings(self.embedding_cache, tensor_data.node_ids, embeddings)

        self._gnn_node_ids = tensor_data.node_ids
        self._gnn_embeddings = embeddings
        self._gnn_retriever = GraphRAGRetriever(
            graph=self.graph,
            node_ids=self._gnn_node_ids,
            embeddings=self._gnn_embeddings,
        )

        details = {
            "nodes_for_training": len(tensor_data.node_ids),
            "embedding_dim": int(embeddings.size(1)),
            "epochs": epochs,
            "cache_path": str(self.embedding_cache),
            "status": "newly_trained",
        }
        return StepReport(name="GNN Training", details=details)

    def step4_ollama_integration(
        self,
        query: str,
        model: str = "llama3.2",
        top_k: int = 20,
        seed_k: int = 4,
        expansion_hops: int = 1,
    ) -> StepReport:
        self._ensure_gnn_retriever()
        assert self._gnn_retriever is not None

        context = self._gnn_retriever.retrieve(
            query=query,
            top_k=top_k,
            seed_k=seed_k,
            expansion_hops=expansion_hops,
        )
        prompt = self._build_ollama_prompt(query, context.nodes)
        answer, source = self._call_ollama(prompt=prompt, model=model)

        details = {
            "query": query,
            "model": model,
            "ollama_source": source,
            "seed_nodes": context.seed_nodes,
            "retrieved_nodes": len(context.nodes),
            "retrieved_edges": len(context.edges),
            "answer": answer,
        }
        return StepReport(name="Integration", details=details)

    def step5_ablation_study(
        self,
        queries: Iterable[str],
        top_k: int = 20,
        model: str = "llama3.2",
        target_hardware: str = "H100",
        use_ollama: bool = True,
        retrieval_weight: float = 0.4,
        generation_weight: float = 0.6,
    ) -> StepReport:
        self._ensure_gnn_retriever()
        assert self._gnn_retriever is not None

        lexical_retriever = GraphRetriever(self.graph)

        gnn_final_scores: list[float] = []
        lexical_final_scores: list[float] = []
        gnn_retrieval_scores: list[float] = []
        lexical_retrieval_scores: list[float] = []
        gnn_grounding_scores: list[float] = []
        lexical_grounding_scores: list[float] = []
        gnn_validity_scores: list[float] = []
        lexical_validity_scores: list[float] = []
        per_query: list[dict] = []
        total = 0

        for query in queries:
            q = query.strip()
            if not q:
                continue

            gnn_ctx = self._gnn_retriever.retrieve(query=q, top_k=top_k, seed_k=4, expansion_hops=1)
            lex_ctx = lexical_retriever.retrieve(query=q, mode="detective", top_k=top_k, ranking_profile="balanced")

            gnn_retrieval = self._token_hit_score(q, gnn_ctx.nodes)
            lexical_retrieval = self._token_hit_score(q, lex_ctx.nodes)

            gnn_answer = self._generate_answer_for_context(
                query=q,
                nodes=gnn_ctx.nodes,
                model=model,
                use_ollama=use_ollama,
            )
            lexical_answer = self._generate_answer_for_context(
                query=q,
                nodes=lex_ctx.nodes,
                model=model,
                use_ollama=use_ollama,
            )

            gnn_grounding = self._grounding_score(gnn_answer["answer"], gnn_ctx.nodes, query=q)
            lexical_grounding = self._grounding_score(lexical_answer["answer"], lex_ctx.nodes, query=q)

            gnn_validity = self._code_validity_score(gnn_answer["answer"], target_hardware=target_hardware)
            lexical_validity = self._code_validity_score(lexical_answer["answer"], target_hardware=target_hardware)

            gnn_generation = 0.5 * gnn_grounding + 0.5 * gnn_validity
            lexical_generation = 0.5 * lexical_grounding + 0.5 * lexical_validity

            gnn_final = retrieval_weight * gnn_retrieval + generation_weight * gnn_generation
            lexical_final = retrieval_weight * lexical_retrieval + generation_weight * lexical_generation

            gnn_retrieval_scores.append(gnn_retrieval)
            lexical_retrieval_scores.append(lexical_retrieval)
            gnn_grounding_scores.append(gnn_grounding)
            lexical_grounding_scores.append(lexical_grounding)
            gnn_validity_scores.append(gnn_validity)
            lexical_validity_scores.append(lexical_validity)
            gnn_final_scores.append(gnn_final)
            lexical_final_scores.append(lexical_final)

            per_query.append(
                {
                    "query": q,
                    "gnn": {
                        "source": gnn_answer["source"],
                        "retrieval": gnn_retrieval,
                        "grounding": gnn_grounding,
                        "validity": gnn_validity,
                        "final": gnn_final,
                    },
                    "lexical": {
                        "source": lexical_answer["source"],
                        "retrieval": lexical_retrieval,
                        "grounding": lexical_grounding,
                        "validity": lexical_validity,
                        "final": lexical_final,
                    },
                }
            )
            total += 1

        if total == 0:
            raise ValueError("Ablation requires at least one non-empty query.")

        gnn_mean, gnn_std = self._mean_std(gnn_final_scores)
        lexical_mean, lexical_std = self._mean_std(lexical_final_scores)
        rel_gain = self._relative_gain(gnn_mean, lexical_mean)

        details = {
            "query_count": total,
            "metric": "weighted_final_score",
            "top_k": top_k,
            "use_ollama": use_ollama,
            "weights": {
                "retrieval": retrieval_weight,
                "generation": generation_weight,
                "grounding_inside_generation": 0.5,
                "validity_inside_generation": 0.5,
            },
            "gnn_score": gnn_mean,
            "lexical_score": lexical_mean,
            "gnn_std": gnn_std,
            "lexical_std": lexical_std,
            "component_means": {
                "gnn_retrieval": self._mean(gnn_retrieval_scores),
                "lexical_retrieval": self._mean(lexical_retrieval_scores),
                "gnn_grounding": self._mean(gnn_grounding_scores),
                "lexical_grounding": self._mean(lexical_grounding_scores),
                "gnn_validity": self._mean(gnn_validity_scores),
                "lexical_validity": self._mean(lexical_validity_scores),
            },
            "relative_gain_over_lexical": rel_gain,
            "per_query": per_query,
        }
        return StepReport(name="Ablation Study", details=details)

    def run_ablation_experiment(
        self,
        queries: Iterable[str],
        top_k: int = 20,
        model: str = "llama3.2",
        target_hardware: str = "H100",
    ) -> dict:
        baseline = self.step5_ablation_study(
            queries=queries,
            top_k=top_k,
            model=model,
            target_hardware=target_hardware,
            use_ollama=False,
        ).details
        live = self.step5_ablation_study(
            queries=queries,
            top_k=top_k,
            model=model,
            target_hardware=target_hardware,
            use_ollama=True,
        ).details

        delta = {
            "gnn_score_delta": live["gnn_score"] - baseline["gnn_score"],
            "lexical_score_delta": live["lexical_score"] - baseline["lexical_score"],
            "relative_gain_delta": live["relative_gain_over_lexical"] - baseline["relative_gain_over_lexical"],
            "gnn_grounding_delta": live["component_means"]["gnn_grounding"] - baseline["component_means"]["gnn_grounding"],
            "gnn_validity_delta": live["component_means"]["gnn_validity"] - baseline["component_means"]["gnn_validity"],
            "lexical_grounding_delta": live["component_means"]["lexical_grounding"] - baseline["component_means"]["lexical_grounding"],
            "lexical_validity_delta": live["component_means"]["lexical_validity"] - baseline["component_means"]["lexical_validity"],
        }

        return {
            "Experiment": {
                "query_count": baseline["query_count"],
                "metric": baseline["metric"],
                "baseline_no_ollama": baseline,
                "live_with_ollama": live,
                "delta_live_minus_baseline": delta,
            }
        }

    def run_full(
        self,
        query: str,
        queries_for_ablation: Iterable[str],
        model: str = "llama3.2",
        hidden_dim: int = 128,
        out_dim: int = 96,
        epochs: int = 20,
        top_k: int = 20,
        target_hardware: str = "H100",
        ablation_use_ollama: bool = True,
    ) -> dict:
        reports = [
            self.step1_data_scrape(),
            self.step2_graph_construction(),
            self.step3_gnn_training(hidden_dim=hidden_dim, out_dim=out_dim, epochs=epochs),
            self.step4_ollama_integration(query=query, model=model, top_k=top_k),
            self.step5_ablation_study(
                queries=queries_for_ablation,
                top_k=top_k,
                model=model,
                target_hardware=target_hardware,
                use_ollama=ablation_use_ollama,
            ),
        ]
        return {r.name: r.details for r in reports}

    def _ensure_gnn_retriever(self) -> None:
        if self._gnn_retriever is None:
            self.step3_gnn_training()

    # Evaluator stopwords — same list as retriever to keep scoring consistent
    _EVAL_STOPWORDS = {
        "torch", "pytorch", "python", "write", "create", "implement", "using",
        "with", "from", "script", "code", "how", "the", "for", "that", "this",
        "show", "use", "model", "module", "function", "class", "method",
        "import", "setup", "define", "build", "make", "get", "set", "run",
        "example", "simple", "basic", "custom", "specific", "apply", "add",
    }

    @classmethod
    def _query_tokens(cls, query: str) -> set[str]:
        return {
            tok.lower() for tok in query.replace(".", " ").replace("_", " ").split()
            if len(tok.strip()) > 2 and tok.lower() not in cls._EVAL_STOPWORDS
        }

    def _token_hit_score(self, query: str, nodes: list[Node]) -> float:
        """Retrieval score: what fraction of query keywords appear in retrieved nodes."""
        tokens = self._query_tokens(query)
        if not tokens or not nodes:
            return 0.0

        matched_tokens: set[str] = set()
        for node in nodes:
            text = f"{node.label} {node.name} {node.url}".lower()
            for token in tokens:
                if token in text:
                    matched_tokens.add(token)
        # Fraction of query-specific keywords covered by at least one node
        return min(1.0, len(matched_tokens) / len(tokens))

    def _generate_answer_for_context(
        self,
        query: str,
        nodes: list[Node],
        model: str,
        use_ollama: bool,
    ) -> dict:
        if not use_ollama:
            return {
                "answer": self._baseline_answer_from_context(query, nodes),
                "source": "baseline-template",
            }

        prompt = self._build_ollama_prompt(query, nodes)
        answer, source = self._call_ollama(prompt=prompt, model=model)
        return {"answer": answer, "source": source}

    @staticmethod
    def _baseline_answer_from_context(query: str, nodes: list[Node]) -> str:
        symbols = [n.name.strip() for n in nodes if n.name.strip()][:5]
        symbol_line = ", ".join(symbols) if symbols else "torch.nn.Module"
        return (
            f"Query: {query}\n"
            f"Grounded symbols: {symbol_line}\n\n"
            "```python\n"
            "import torch\n\n"
            "def build_block(dim: int = 128):\n"
            "    layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8)\n"
            "    return torch.compile(layer)\n"
            "```"
        )

    def _grounding_score(self, answer: str, nodes: list[Node], query: str = "") -> float:
        """Score grounding with query-relevance filtering.
        
        Only nodes whose names have lexical overlap with the query count
        toward grounding. This prevents inflated scores when the retriever
        fetches irrelevant APIs that the LLM blindly uses.
        Also checks URL-derived names for documentation nodes.
        """
        import re as _re
        query_tokens = self._query_tokens(query) if query else set()
        symbols = []
        for n in nodes:
            name = n.name.strip().lower()
            # Also try URL-derived name for documentation nodes
            if not name and n.url:
                match = _re.search(r'/([^/]+?)\.html$', n.url)
                if match:
                    name = match.group(1).lower()
            if not name:
                continue
            # Grounding is only expected for executable API nodes
            lbl = n.label or ""
            if not lbl.startswith("API_"):
                continue
            # If we have query tokens, only count query-relevant nodes
            if query_tokens:
                name_tokens = set(name.replace('.', ' ').replace('_', ' ').split())
                if not query_tokens & name_tokens:
                    continue
            symbols.append(name)
        if not symbols:
            return 0.0
        answer_l = answer.lower()
        unique_symbols = []
        seen: set[str] = set()
        for s in symbols:
            if s in seen:
                continue
            seen.add(s)
            unique_symbols.append(s)
        probe = unique_symbols[:min(12, len(unique_symbols))]
        if not probe:
            return 0.0
        hits = 0
        for s in probe:
            base_name = s.split('.')[-1]
            if s in answer_l or base_name in answer_l:
                hits += 1
        return min(1.0, hits / len(probe))

    def _code_validity_score(self, answer: str, target_hardware: str) -> float:
        """Validity: fraction of C0-C5 checks that passed.
        
        Skipped checks (e.g., C5 on macOS) count as 0.5 instead of 1.0
        to avoid inflating scores for untested compile behaviour.
        """
        code = self._extract_code_candidate(answer)
        if not code.strip():
            return 0.0
        report = self.validator.validate(code, target_hardware=target_hardware)
        total = max(1, len(report.checks))
        score = 0.0
        for c in report.checks:
            if c.status == "pass":
                score += 1.0
            elif c.status == "skipped":
                score += 0.5  # partial credit — not fully verified
            # "fail" contributes 0.0
        return min(1.0, score / total)

    @staticmethod
    def _extract_code_candidate(answer: str) -> str:
        block = re.search(r"```(?:python)?\n(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
        if block:
            return block.group(1).strip()
        return answer.strip()

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @classmethod
    def _mean_std(cls, values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = cls._mean(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, var ** 0.5

    @staticmethod
    def _relative_gain(new_value: float, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return (new_value - baseline) / baseline

    @staticmethod
    def _build_ollama_prompt(query: str, nodes: list[Node]) -> str:
        import re as _re
        lines = [
            "You are a PyTorch 2.x coding assistant.",
            "Below is a list of PyTorch resources retrieved from official documentation.",
            "ADVISORY RULE: ONLY use the provided APIs if they fit naturally. A simple, clean, runnable solution is better than forcing obscure APIs! If the tools seem irrelevant, IGNORE THEM completely and rely on your own baseline knowledge.",
            "Keep your codebase as direct and simple as possible. Do not overcomplicate.",
            "WARNING: Items marked as concepts/tutorials are NOT callable modules. DO NOT attempt to blindly import `import <name>` unless you are certain it is a standard PyTorch module.",
            "Always write complete, runnable Python code with proper standard imports.",
            "",
            f"User query: {query}",
            ""
        ]
        
        api_lines = []
        concept_lines = []
        
        idx = 0
        # TRUNCATE: Only provide top 5 high-precision nodes to avoid noise overwhelming small LLMs
        for node in nodes[:5]:
            name = node.name.strip()
            url_hint = ""
            if not name and node.url:
                match = _re.search(r'/([^/]+?)\.html$', node.url)
                if match:
                    name = match.group(1)
            if not name:
                continue
            
            if node.url:
                url_hint = f"  (docs: {node.url})"
            label = node.label.replace('API_', '').replace('_', ' ').title() if node.label else ''
            idx += 1
            entry = f"  {idx}. {name} [{label}]{url_hint}"
            
            if node.label and node.label.startswith("API_"):
                api_lines.append(entry)
            else:
                concept_lines.append(entry)
                
        if api_lines:
            lines.append("💻 Valid PyTorch APIs (Advisory - ONLY use if relevant):")
            lines.extend(api_lines)
            lines.append("")
            
        if concept_lines:
            lines.append("📚 Conceptual Context (READ-ONLY Context. Do not import as code!):")
            lines.extend(concept_lines)
            lines.append("")
            
        lines.append("Write complete working Python code. Include all necessary imports.")
        lines.append("Use ```python code blocks for your code.")
        return "\n".join(lines)

    @staticmethod
    def _call_ollama(prompt: str, model: str) -> tuple[str, str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url="http://127.0.0.1:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            text = parsed.get("response", "").strip()
            if not text:
                text = "Ollama returned an empty response."
            return text, "ollama"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            fallback = (
                "Ollama unavailable, returning retrieval-grounded fallback summary: "
                f"{type(exc).__name__}: {exc}"
            )
            return fallback, "fallback"


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2)
