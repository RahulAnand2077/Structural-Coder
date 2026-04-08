# 🏆 `benchmark/` — Proof That Our System Wins

This folder contains everything needed to **compare our system against AI chatbots**
like llama3.1:8b, codellama, etc. that you can download and run locally.

---

## Files

| File/Folder | What It Does |
|-------------|-------------|
| `run_comparison.py` | Main script — runs both systems on the same questions |
| `queries/queries.json` | 10 real PyTorch coding challenge questions |
| `outputs/comparison.json` | All raw results as numbers (JSON format) |
| `outputs/comparison_report.md` | Human-readable results table |

---

## 🧮 How the Scoring Works

Each answer is graded on three things:

| Metric | Meaning | Max Score |
|--------|---------|-----------|
| **Retrieval** | Did we find relevant PyTorch APIs? | 1.0 |
| **Grounding** | Did the answer actually USE those APIs? | 1.0 |
| **Validity** | Does the code pass all 6 safety checks? | 1.0 |
| **Final** | `0.4×retrieval + 0.6×(0.5×grounding + 0.5×validity)` | 1.0 |

> **Important**: For the standalone LLM (llama3.1:8b), retrieval = 0 because it cannot
> search any graph. It only knows what was in its training data.

---

## ▶️ How to Run the Benchmark

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Test against llama3.1:8b (already downloaded)
python benchmark/run_comparison.py --models llama3.1:8b

# 3. Test multiple models (if you download more)
ollama pull codellama
python benchmark/run_comparison.py --models llama3.1:8b,codellama
```

---

## 📊 Our Results (Real Live Benchmark)

| System | Score |
|--------|-------|
| 🏆 Structural-Coder (Ours) | **0.75** |
| llama3.1:8b (Standalone) | 0.24 |

**We won all 10 / 10 queries.**


---
### 💡 Architectural Note: Decoupled GNN & Pipeline
Please note that the **GNN Coder (Retriever)** and the **Integration Pipeline** are strictly decoupled. 
Any changes, experiments, or updates made to the GNN Coder (such as changing embedding dimensions, training algorithms, or node attributes) will **not** break the core pipeline. The pipeline simply consumes the resulting `research_gnn_embeddings.json` artifact, guaranteeing the system remains perfectly intact and modular.
