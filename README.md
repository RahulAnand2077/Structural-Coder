# The Structural Coder
**Graph-Augmented Retrieval for Idiomatic PyTorch 2.x Code Generation**

Solves the "version-hallucination" problem in LLM code generation by using a Knowledge Graph of the PyTorch 2.x API to guide a Transformer-based generator toward modern, idiomatic syntax.

---

## What's in here

| File | What it does |
|------|-------------|
| `web_crawler.ipynb` | Crawls `docs.pytorch.org/docs/stable/` via BFS, extracts API entities, code snippets, and concept text, writes everything to Neo4j |
| `creating_embed.ipynb` | Pulls all 24,485 nodes from Neo4j, encodes each with `all-MiniLM-L6-v2` (384-dim), writes embeddings back to Neo4j |
| `nodes.csv` | Exported snapshot of all graph nodes |
| `edges.csv` | Exported snapshot of all graph relationships |
| `index.html` | PyTorch 2.10 docs index — used by the crawler to seed the BFS queue |

---

## Knowledge Graph

Built from the full PyTorch 2.10 stable documentation.

| Stat | Value |
|------|-------|
| Nodes | 24,485 |
| Relationships | 47,958 |
| Embedding dim | 384 |
| Embedding model | `all-MiniLM-L6-v2` |

**Node types:** `API_Function`, `API_Class`, `API_Method`, `API_Parameter`,
`API_Endpoint`, `CodeSnippet`, `Concept`, `DeprecatedAPI`, `PyTorchConcept`

**Edge types:** `CONTAINS`, `CALLS`, `INHERITS`, `HAS_PARAM`, `IMPLEMENTS`, `EXPLAINS`, `REPLACES`, `REFERENCES`, `RELATED_TO`

---

## Reproducing the graph

**Requirements**
```
neo4j>=5.0
requests
beautifulsoup4
sentence-transformers
```

**Steps**
1. Start Neo4j at `bolt://localhost:7687`
2. Run `web_crawler.ipynb` to populate the DB
3. Run `creating _embed.ipynb` to compute node embeddings

---

## Status

- [x] Knowledge Graph construction
- [x] Node embeddings
- [ ] GNN encoder
- [ ] Graph-RAG retriever
- [ ] Code generator integration