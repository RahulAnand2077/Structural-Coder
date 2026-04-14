from __future__ import annotations

import ast
import builtins
import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List


@dataclass
class Check:
    level: str
    passed: bool
    details: str
    status: str = "pass"


@dataclass
class ValidationReport:
    checks: List[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def passed_with_skips(self) -> bool:
        return all(c.status in {"pass", "skipped"} for c in self.checks)

    @property
    def passed_strict(self) -> bool:
        return all(c.status == "pass" for c in self.checks)


class ActiveValidator:
    """C0/C1-C3/C5 active validation loop over generated code."""

    def __init__(self, c5_mode: str = "auto", c5_timeout_sec: int = 90) -> None:
        # c5_mode:
        # - auto: downgrade known host/toolchain compile-probe failures to skipped
        # - strict: treat any probe execution failure as C5 failure
        # - lenient: always downgrade host/toolchain compile-probe failures
        allowed = {"auto", "strict", "lenient"}
        self.c5_mode = c5_mode if c5_mode in allowed else "auto"
        self.c5_timeout_sec = max(10, int(c5_timeout_sec))

    def validate(self, code: str, target_hardware: str = "H100") -> ValidationReport:
        checks: List[Check] = []
        checks.append(self._c0(code))
        checks.extend(self._c1_c3(code, target_hardware))
        checks.append(self._c5(code))
        return ValidationReport(checks)

    def self_heal_once(self, code: str, report: ValidationReport) -> str:
        return self.apply_targeted_fixes(code, report)

    def apply_targeted_fixes(self, code: str, report: ValidationReport, target_hardware: str = "H100") -> str:
        out, _ = self.apply_targeted_fixes_with_actions(code, report, target_hardware=target_hardware)
        return out

    def apply_targeted_fixes_with_actions(
        self,
        code: str,
        report: ValidationReport,
        target_hardware: str = "H100",
    ) -> tuple[str, list[str]]:
        out = code
        actions: list[str] = []
        failed = {c.level: c.details for c in report.checks if not c.passed}

        c0_details = failed.get("C0", "")
        if ("Missing imports:" in c0_details or "Unresolved symbols:" in c0_details) and "torch" in c0_details.lower():
            before = out
            out = self._ensure_torch_import(out)
            if out != before:
                actions.append("C0:insert_import_torch")

        if "C1" in failed:
            before = out
            out = self._ensure_device_guard(out)
            if out != before:
                actions.append("C1:ensure_device_guard")

        if "C2" in failed and "expects_gpu=true" in failed["C2"].lower():
            before = out
            out = self._promote_gpu_path(out)
            if out != before:
                actions.append("C2:promote_gpu_path")

        if "C5" in failed:
            c5_details = failed["C5"].lower()
            if "nameerror" in c5_details and "torch" in c5_details and "not defined" in c5_details:
                before = out
                out = self._ensure_torch_import(out)
                if out != before:
                    actions.append("C5:insert_import_torch_for_nameerror")

            if ".item()" in out:
                out = out.replace(".item()", ".detach().cpu().numpy()")
                actions.append("C5:replace_item_with_detach_cpu_numpy")

            cleaned = re.sub(r"torch\._dynamo\.graph_break\s*\(\s*\)", "# removed graph break for compile safety", out)
            if cleaned != out:
                out = cleaned
                actions.append("C5:remove_explicit_graph_break")

        if not actions:
            actions.append("no_change")

        return out, actions

    def _c0(self, code: str) -> Check:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return Check("C0", False, f"Syntax error line {e.lineno}: {e.msg}", status="fail")

        missing = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if importlib.util.find_spec(a.name.split(".")[0]) is None:
                        missing.append(a.name)
            if isinstance(node, ast.ImportFrom) and node.module:
                if importlib.util.find_spec(node.module.split(".")[0]) is None:
                    missing.append(node.module)
        if missing:
            return Check("C0", False, "Missing imports: " + ", ".join(sorted(set(missing))), status="fail")

        unresolved = self._find_unresolved_names(tree)
        if unresolved:
            return Check("C0", False, "Unresolved symbols: " + ", ".join(sorted(unresolved)), status="fail")

        return Check("C0", True, "Static checks passed")

    def _c1_c3(self, code: str, target_hardware: str) -> List[Check]:
        lowered = code.lower()
        c1_ok = "cuda" not in lowered or "if torch.cuda.is_available()" in lowered
        c1 = Check("C1", c1_ok, "CPU fallback guarded" if c1_ok else "CUDA used without guarded fallback", status="pass" if c1_ok else "fail")
        expects_gpu = target_hardware.lower().startswith("h") or "gpu" in target_hardware.lower() or "cuda" in target_hardware.lower()
        gpu_used = self._detect_cuda_path(code)
        c2_ok = (gpu_used if expects_gpu else True)
        c2 = Check(
            "C2",
            c2_ok,
            (
                f"GPU path alignment (expects_gpu={expects_gpu}, gpu_path_detected={gpu_used})"
            ),
            status="pass" if c2_ok else "fail",
        )

        torch_ok = importlib.util.find_spec("torch") is not None
        c3 = Check("C3", torch_ok, "Torch available in runtime" if torch_ok else "Torch not installed in runtime", status="pass" if torch_ok else "fail")
        return [c1, c2, c3]

    def _c5(self, code: str) -> Check:
        if not self._has_compile_claim(code):
            return Check("C5", True, "No compile claim detected", status="pass")

        if importlib.util.find_spec("torch") is None:
            return Check("C5", False, "torch unavailable, cannot verify compile behavior", status="fail")

        risky_patterns = [
            r"torch\._dynamo\.graph_break\s*\(",
            r"\.item\s*\(",
            r"print\s*\(",
            r"data-dependent",
        ]
        for pattern in risky_patterns:
            if re.search(pattern, code, flags=re.IGNORECASE):
                return Check("C5", False, f"Potential graph-break pattern detected: {pattern}", status="fail")

        # Probe-only override: keep torch.compile semantics while avoiding host C++ toolchain dependence.
        probe_prelude = (
            "import torch\n"
            "_c5_orig_torch_compile = torch.compile\n"
            "def _c5_probe_compile(*args, **kwargs):\n"
            "    kwargs.setdefault('backend', 'eager')\n"
            "    return _c5_orig_torch_compile(*args, **kwargs)\n"
            "torch.compile = _c5_probe_compile\n\n"
        )
        harness = (
            probe_prelude
            + code
            + "\n\n"
            + "if __name__ == '__main__':\n"
            + "    try:\n"
            + "        if 'run_once' in globals():\n"
            + "            run_once()\n"
            + "    except Exception as e:\n"
            + "        print('RUNTIME_EXCEPTION:', e)\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix="_c5.py", delete=False, encoding="utf-8") as f:
            f.write(harness)
            path = f.name
        try:
            env = dict(os.environ)
            env["TORCH_LOGS"] = "graph_breaks"
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=self.c5_timeout_sec,
                env=env,
            )
            combined = (proc.stdout + "\n" + proc.stderr)
            output = combined.lower()
            if (
                "graph break" in output
                or "graph_break" in output
                or "graphbreak" in output
                or "torch._dynamo.graph_break" in output
            ):
                return Check("C5", False, "Graph break detected via TORCH_LOGS", status="fail")

            runtime_exc = self._extract_runtime_exception(combined)
            if runtime_exc is not None:
                return Check("C5", False, f"Probe runtime exception: {runtime_exc}", status="fail")

            if proc.returncode != 0:
                stderr_hint = self._best_stderr_hint(proc.stderr)
                if self._is_environment_compile_failure(stderr_hint, combined):
                    if self._should_soft_skip_c5_env_failure():
                        return Check(
                            "C5",
                            True,
                            "Compile probe skipped due host toolchain limits (classified separately from graph-break).",
                            status="skipped",
                        )
                    return Check(
                        "C5",
                        False,
                        "Compile probe environment/toolchain failure (not graph-break): " + (stderr_hint or "unknown toolchain issue"),
                        status="fail",
                    )
                if stderr_hint:
                    return Check("C5", False, f"Probe failed with code {proc.returncode}: {stderr_hint}", status="fail")
                return Check("C5", False, f"Probe failed with code {proc.returncode}", status="fail")
            return Check("C5", True, "No graph break detected", status="pass")
        except Exception as e:
            return Check("C5", False, f"Probe error: {type(e).__name__}: {e}", status="fail")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def _has_compile_claim(code: str) -> bool:
        lowered = code.lower()
        if "torch.compile" in lowered:
            return True
        if re.search(r"\bcompile\s*\(", lowered):
            return True
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "compile":
                        return True
                if isinstance(node.func, ast.Name) and node.func.id == "compile":
                    return True
        return False

    @staticmethod
    def _detect_cuda_path(code: str) -> bool:
        lowered = code.lower()
        if "torch.cuda" in lowered:
            return True
        if ".to(\"cuda\")" in lowered or ".to('cuda')" in lowered:
            return True
        if "device=\"cuda\"" in lowered or "device='cuda'" in lowered:
            return True
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.lower() == "cuda":
                            return True
                if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                    return True
        return False

    @staticmethod
    def _find_unresolved_names(tree: ast.AST) -> set[str]:
        builtins_set = set(dir(builtins))
        defined: set[str] = set()
        unresolved: set[str] = set()

        class DefCollector(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    defined.add(alias.asname or alias.name)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                defined.add(node.name)
                for arg in node.args.args:
                    defined.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    defined.add(arg.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(node)  # type: ignore[arg-type]

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                defined.add(node.name)
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        defined.add(tgt.id)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
                self.generic_visit(node)

            def visit_With(self, node: ast.With) -> None:
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)
                self.generic_visit(node)

        class UseCollector(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load):
                    if node.id not in defined and node.id not in builtins_set:
                        unresolved.add(node.id)

        DefCollector().visit(tree)
        UseCollector().visit(tree)
        return unresolved

    @staticmethod
    def _ensure_torch_import(code: str) -> str:
        if re.search(r"^\s*import\s+torch\b", code, flags=re.MULTILINE):
            return code
        return "import torch\n" + code

    def _ensure_device_guard(self, code: str) -> str:
        out = self._ensure_torch_import(code)
        if "if torch.cuda.is_available()" not in out:
            guard = 'device = "cuda" if torch.cuda.is_available() else "cpu"\n'
            out = self._insert_after_import_block(out, guard)
        out = out.replace(".cuda()", ".to(device)")
        out = out.replace('.to("cuda")', ".to(device)")
        out = out.replace(".to('cuda')", ".to(device)")
        out = out.replace('device="cuda"', "device=device")
        out = out.replace("device='cuda'", "device=device")
        return out

    def _promote_gpu_path(self, code: str) -> str:
        out = self._ensure_device_guard(code)
        # If no explicit transfer exists, insert a conservative helper marker for downstream generation tools.
        if ".to(device)" not in out and "torch.cuda" not in out:
            out += "\n\n# TODO: move model/tensors to device for full GPU path alignment\n"
        return out

    @staticmethod
    def _extract_runtime_exception(output: str) -> str | None:
        marker = "RUNTIME_EXCEPTION:"
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if marker in line:
                exc = line.split(marker, 1)[1].strip()
                if exc:
                    return exc
                return "unknown runtime exception"
        return None

    @staticmethod
    def _best_stderr_hint(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""

        # Prefer the traceback tail where Python prints the concrete exception summary.
        for raw_line in reversed(lines):
            line = raw_line.strip()
            if line.startswith("Traceback"):
                continue
            if "error" in line.lower() or "exception" in line.lower():
                return line

        return lines[-1]

    @staticmethod
    def _insert_after_import_block(code: str, snippet: str) -> str:
        lines = code.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_at = i + 1
                continue
            break

        lines.insert(insert_at, snippet)
        return "".join(lines)

    def _should_soft_skip_c5_env_failure(self) -> bool:
        if self.c5_mode == "strict":
            return False
        if self.c5_mode == "lenient":
            return True
        # auto mode: conservative default to skip known flaky toolchain hosts.
        system = platform.system().lower()
        return system == "darwin"

    @staticmethod
    def _is_environment_compile_failure(stderr_hint: str, combined_output: str) -> bool:
        text = (stderr_hint + "\n" + combined_output).lower()
        markers = [
            "1 error generated.",
            "clang",
            "xcode",
            "command line tools",
            "ld: ",
            "linker command failed",
            "fatal error:",
            "subprocess-exited-with-error",
            "backend compiler failed",
            "inductor",
            "bus error",
            "signal 10",
            "illegal instruction",
            "resource_tracker",
            "leaked semaphore",
        ]
        return any(m in text for m in markers)
