"""
Structural-Coder Configuration
===============================
Change the LLM model in ONE place here. Every module reads from this file.
"""

# ─── LLM Model ────────────────────────────────────────────────────────
# Change this single line to switch the entire system to a different model.
# Examples: "llama3.1:8b", "codellama:7b", "mistral:7b", "deepseek-coder:6.7b"
MODEL_NAME: str = "llama3.1:8b"

# ─── Retrieval ─────────────────────────────────────────────────────────
TOP_K: int = 20          # Max nodes retrieved per query
SEED_K: int = 4          # Number of lexical anchor seeds
EXPANSION_HOPS: int = 1  # Graph expansion depth
PROMPT_NODE_LIMIT: int = 5  # Max nodes injected into LLM prompt

# ─── Paths ─────────────────────────────────────────────────────────────
NODES_CSV: str = "data/nodes.csv"
EDGES_CSV: str = "data/edges.csv"
EMBEDDING_CACHE: str = "outputs/gnn_embeddings.jsonl"

# ─── GNN Training ─────────────────────────────────────────────────────
HIDDEN_DIM: int = 128
OUT_DIM: int = 96
EPOCHS: int = 20

# ─── Evaluation ────────────────────────────────────────────────────────
TARGET_HARDWARE: str = "H100"
RETRIEVAL_WEIGHT: float = 0.4
GENERATION_WEIGHT: float = 0.6
