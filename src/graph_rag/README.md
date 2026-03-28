# 🔍 `src/graph_rag/` — The Smart Searcher (Graph-RAG Engine)

This module is the **brain** of the system. It knows how to search our PyTorch
knowledge graph like a detective — starting from the most relevant entry point,
then following connections to discover related concepts.

---

## Files in This Folder

| File | What It Does (Plain English) |
|------|------------------------------|
| `graph_loader.py` | Reads `nodes.csv` and `edges.csv` and builds the graph in memory |
| `gnn_encoder.py` | Trains a Graph Neural Network to give every node a smart "fingerprint" |
| `retriever.py` | Uses those fingerprints to find nodes closest to your question |
| `pipeline.py` | Runs all the above steps in the right order |
| `generator.py` | Converts retrieved nodes into a code answer |

---

## 🧠 How the Heterogeneous GNN Works (HGTConv)

Our system uses a **Heterogeneous Graph Transformer (HGTConv)** to learn highly sophisticated, graph-aware embeddings (fingerprints) for all 24,000 PyTorch nodes.

### 🧬 Breaking Down "Heterogeneous"
Traditional GNNs (like GraphSAGE) treat every node and edge exactly the same. Our network respects the **schema** of the PyTorch ecosystem. It maintains distinct neural weights depending on whether a node is a `Concept`, `API_Function`, or `CodeSnippet`, and uniquely routes messages across specific relationships (e.g., `IMPLEMENTS`, `CALLS`, `HAS_PARAM`).

### ⚙️ The Mathematical Training Loop
The GNN trains itself using **Self-Supervised Link Prediction**. We hide a fraction of the edges and force the network to guess if a connection exists:

1. **Structural Message Passing**: The network propagates context using dense documentation edges (`EXPLAINS`, `REFERENCES`).
2. **Strict Supervision Routing**: The network explicitly evaluates Cross-Entropy Loss **only** on Semantic Edges (`IMPLEMENTS`, `CONTAINS`, `HAS_PARAM`, `CALLS`, `RELATED_TO`, `REPLACES`). This prevents generic documentation ties from destroying the semantic accuracy of the embeddings.
3. **True-Positive Filtered Hard Negatives**: To prevent graph memorization, the network calculates degree-distribution probabilities to hunt for "hard negative" connections. We apply strict True-Positive filtering to ensure valid edges are never accidentally penalized.
4. **Split Weight Decay Regularization**: The Low-Rank Bilinear Decoder matrices (`U` and `V`) are pushed through a 10x multiplier on their Adam weight decay compared to the encoder parameters. This mathematically stabilizes the gradients on exceedingly rare, specialized relations.
5. **Residual Fallbacks**: The 3-layer depth tensor network automatically falls back to its previous hidden state if a node becomes dynamically isolated during batched training, keeping backwards gradients universally alive.

By passing messages through this mathematically hardened network for 40 epochs, the output `[24485, 256]` tensor coordinate map allows us to bypass legacy keyword searches and retrieve PyTorch APIs by pure, absolute structural meaning.

---

## 🗺️ Retrieval Strategy

1. **Seed Selection** — find the 4 best starting nodes using GNN cosine similarity
2. **Graph Expansion** — follow 1-hop edges to gather related neighbours
3. **Hybrid Re-Ranking** — combine GNN score + keyword hits + graph degree to produce final top-K list

---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
