from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .graph_loader import CsvGraph
from .retriever import GraphRetriever
from .validator import ActiveValidator


@dataclass
class CombinedResult:
    mode: str
    query: str
    selected_node: str
    ranking_profile: str
    top_k: int
    retrieved_nodes: int
    retrieved_edges: int
    validation: Optional[dict]
    self_heal_applied: bool
    repair_iterations: int
    repair_history: list[str]
    repair_actions: list[list[str]]


class CombinedPipeline:
    """Non-invasive integration layer: uses his CSV outputs + our validation/retrieval."""

    def __init__(self, nodes_csv: str | Path, edges_csv: str | Path) -> None:
        self.graph = CsvGraph(nodes_csv, edges_csv)
        self.graph.load()
        self.retriever = GraphRetriever(self.graph)
        self.validator = ActiveValidator()

    def run(
        self,
        query: str,
        mode: str = "detective",
        top_k: int = 80,
        ranking_profile: str = "balanced",
        generated_code: Optional[str] = None,
        target_hardware: str = "H100",
        enable_self_heal: bool = True,
        max_repair_iterations: int = 3,
    ) -> CombinedResult:
        ctx = self.retriever.retrieve(query, mode=mode, top_k=top_k, ranking_profile=ranking_profile)

        report_dict = None
        healed = False
        repair_iterations = 0
        repair_history: list[str] = []
        repair_actions: list[list[str]] = []
        if generated_code is not None:
            current_code = generated_code
            report = self.validator.validate(current_code, target_hardware=target_hardware)
            if enable_self_heal:
                while not report.passed and repair_iterations < max(1, max_repair_iterations):
                    failing_levels = [c.level for c in report.checks if not c.passed]
                    repair_history.append(f"iteration={repair_iterations + 1}; failing={','.join(failing_levels)}")
                    healed_code, actions = self.validator.apply_targeted_fixes_with_actions(
                        current_code,
                        report,
                        target_hardware=target_hardware,
                    )
                    repair_actions.append(actions)
                    if healed_code == current_code:
                        repair_history.append("stopped=no_change")
                        break
                    current_code = healed_code
                    healed = True
                    repair_iterations += 1
                    report = self.validator.validate(current_code, target_hardware=target_hardware)

                if report.passed:
                    repair_history.append("stopped=passed")
                elif repair_iterations >= max(1, max_repair_iterations):
                    repair_history.append("stopped=max_iterations")

            report_dict = {
                "passed": report.passed,
                "passed_with_skips": report.passed_with_skips,
                "passed_strict": report.passed_strict,
                "checks": [asdict(c) for c in report.checks],
            }

        return CombinedResult(
            mode=mode,
            query=query,
            selected_node=ctx.selected_node,
            ranking_profile=ctx.ranking_profile,
            top_k=ctx.top_k,
            retrieved_nodes=len(ctx.nodes),
            retrieved_edges=len(ctx.edges),
            validation=report_dict,
            self_heal_applied=healed,
            repair_iterations=repair_iterations,
            repair_history=repair_history,
            repair_actions=repair_actions,
        )

    @staticmethod
    def to_json(result: CombinedResult) -> str:
        return json.dumps(asdict(result), indent=2)
