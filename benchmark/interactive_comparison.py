"""
interactive_comparison.py
=========================
Interactive terminal: type any PyTorch query, see System A vs System B side-by-side.

Usage:
  python benchmark/interactive_comparison.py --model llama3.1:8b
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_NAME
from src.pipeline import Pipeline, _print_result


def main():
    parser = argparse.ArgumentParser(description="LLM vs GNN+LLM Interactive Comparison")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    args = parser.parse_args()

    pipe = Pipeline()

    while True:
        try:
            query = input("\n📝 Enter your PyTorch query (or 'exit'): ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            result = pipe.compare(query, model=args.model)
            _print_result(result)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
