from __future__ import annotations

from dataclasses import dataclass

from .retriever import RetrievedContext


@dataclass
class GenerationResult:
    query: str
    code: str
    used_symbols: list[str]


class TransformerGeneratorBridge:
    """Generator integration point.

    This integrates retrieval context into generation prompts and emits
    PyTorch 2.x idiomatic scaffold code. A model-backed generator can be
    dropped in here without changing retrieval/validation wiring.
    """

    def generate(self, query: str, context: RetrievedContext) -> GenerationResult:
        symbols = self._extract_symbols(context)
        symbol_comments = "\n".join([f"# - {s}" for s in symbols[:12]]) or "# - torch.compile"

        code = (
            "import torch\n\n"
            "# Retrieved Graph-RAG evidence symbols:\n"
            f"{symbol_comments}\n\n"
            "def build_model(input_dim: int = 128, output_dim: int = 128):\n"
            "    model = torch.nn.Sequential(\n"
            "        torch.nn.Linear(input_dim, 256),\n"
            "        torch.nn.GELU(),\n"
            "        torch.nn.Linear(256, output_dim),\n"
            "    )\n"
            "    model = torch.compile(model)\n"
            "    if torch.cuda.is_available():\n"
            "        model = model.to(\"cuda\")\n"
            "    return model\n\n"
            "def run_once(batch: int = 4, dim: int = 128):\n"
            "    model = build_model(dim, dim)\n"
            "    x = torch.randn(batch, dim)\n"
            "    if torch.cuda.is_available():\n"
            "        x = x.to(\"cuda\")\n"
            "    return model(x)\n\n"
            "if __name__ == \"__main__\":\n"
            "    y = run_once()\n"
            "    _ = y.shape\n"
        )

        return GenerationResult(query=query, code=code, used_symbols=symbols)

    @staticmethod
    def _extract_symbols(context: RetrievedContext) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for n in context.nodes:
            candidate = n.name.strip()
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            symbols.append(candidate)
        return symbols
