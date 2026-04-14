# 🛡️ Validation Checks (C0–C5)

**Path:** `docs/validation_checks.md`  
**Source code:** `src/integration_pipeline/validator.py`

---

## The 6 Checks

Every piece of generated code passes through these checks in order:

| Check | Name | What It Tests | Fail Action |
|-------|------|--------------|-------------|
| **C0** | Import Check | Does the code import `torch`? | Auto-injects `import torch` |
| **C1** | Syntax Check | Can Python parse the code without errors? | Returns syntax error details |
| **C2** | API Existence | Do the PyTorch functions/classes actually exist in `torch`? | Reports unresolved names |
| **C3** | Compile Probe | Can `torch.compile()` accept the model? | Skipped on macOS (no inductor) |
| **C4** | Device Guard | Is CUDA used safely with `torch.cuda.is_available()` checks? | Auto-wraps in device guard |
| **C5** | Dtype Safety | Are dtypes mixed safely (no float16/bfloat16 errors)? | Skipped if no GPU |

---

## Scoring

```python
score = 0.0
for check in report.checks:
    if check.status == "pass":
        score += 1.0
    elif check.status == "skipped":
        score += 0.5   # partial credit — not fully verified
    # "fail" → 0.0
validity = score / len(report.checks)
```

Skipped checks (e.g., C3/C5 on macOS with no GPU) get **0.5** instead of 1.0 to avoid inflating scores for untested behaviour.

---

## Self-Healing Loop

When validation fails, the `ActiveValidator` can automatically fix common issues:

```
Generate Code → Validate → FAIL
                    │
                    ▼
            Apply Targeted Fixes
            (inject torch import, add device guard, etc.)
                    │
                    ▼
            Re-validate → PASS? → Done
                    │
                    ▼ (still failing)
            Retry (up to 3 iterations)
```

The self-healing is used by `src/integration_pipeline/pipeline.py` (`CombinedPipeline.run()`).  
The research pipeline does **not** use self-healing — it measures raw code quality.

---

## Where Validation Is Called

| Caller | File | Self-Heal? |
|--------|------|-----------|
| `_code_validity_score()` | `src/research_pipeline/pipeline.py` | ❌ No — measures raw quality |
| `CombinedPipeline.run()` | `src/integration_pipeline/pipeline.py` | ✅ Yes — up to 3 repair iterations |
| Benchmark scoring | `benchmark/run_comparison.py` | ❌ No — measures raw quality |
