"""Behavioral and structural clustering of candidate outputs.

Behavioral clustering treats two candidates as equivalent iff they produce
identical outputs on a set of probe inputs. This is the equivalence relation
recommended in the BTC paper (Section 3.3).

Structural (AST) clustering is provided for ablation purposes only — the
paper shows it produces spurious abstentions on code-completion tasks.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple

# ============================================================
# Behavioral clustering (recommended)
# ============================================================

def cluster_by_behavior(
    candidates_code: Sequence[str],
    probe_runner: Callable[[str], Tuple],
) -> List[Tuple]:
    """Compute behavioral cluster keys for a population of code candidates.

    Each candidate is run through `probe_runner`, which returns a tuple of
    output strings (stringified outputs of the candidate's function on a
    fixed set of probe inputs). Candidates with identical tuples belong to
    the same behavioral cluster.

    Parameters
    ----------
    candidates_code : Sequence[str]
        Code strings (function definitions or full programs).
    probe_runner : Callable[[str], Tuple]
        Function that takes a code string and returns the tuple of outputs.
        See `make_python_probe_runner` for a default implementation.

    Returns
    -------
    list of tuple
        Cluster key for each candidate, in the same order as the input.
    """
    return [tuple(probe_runner(c)) for c in candidates_code]


def make_python_probe_runner(
    entry_point: str,
    probe_args: Sequence[str],
    timeout: float = 5.0,
    prompt_prefix: str = "",
) -> Callable[[str], Tuple]:
    """Build a probe runner for Python function candidates.

    The returned callable executes each candidate in a subprocess sandbox,
    invokes `<entry_point>(<args>)` for every entry in `probe_args`, and
    returns a tuple of stringified outputs (or `__ERR__:<ExceptionName>`
    for failed evaluations).

    Parameters
    ----------
    entry_point : str
        Name of the function to call (e.g. "is_palindrome").
    probe_args : Sequence[str]
        Each entry is a Python source-code expression for the call arguments
        (e.g. "[1, 2, 3], 0.5"). The call is `entry_point(<arg_expr>)`.
    timeout : float
        Subprocess timeout in seconds.
    prompt_prefix : str
        Optional prefix to prepend to the candidate (e.g. the original task
        prompt) when the candidate is a body without imports.
    """
    if not probe_args:
        def _empty(_: str) -> Tuple:
            return ("__NO_PROBES__",)
        return _empty

    def _run(code: str) -> Tuple:
        inputs_repr = ", ".join(repr(s) for s in probe_args)
        body = (
            "results = []\n"
            f"for args_str in [{inputs_repr}]:\n"
            "    try:\n"
            f"        out = eval(f'{entry_point}({{args_str}})')\n"
            "        results.append(repr(out))\n"
            "    except Exception as e:\n"
            "        results.append('__ERR__:' + type(e).__name__)\n"
            "print('|||'.join(results))\n"
        )
        for full_code in [code, prompt_prefix + "\n" + code]:
            runner_code = full_code + "\n\n" + body
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(runner_code)
                tmp_path = f.name
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True, text=True, timeout=timeout,
                )
                if proc.returncode == 0 and "|||" in proc.stdout:
                    return tuple(proc.stdout.strip().split("|||"))
            except subprocess.TimeoutExpired:
                return ("__TIMEOUT__",)
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return ("__UNDEFINED__",)

    return _run


CALL_PATTERN = re.compile(r"\b{name}\s*\((.*?)\)\s*==", re.DOTALL)
CANDIDATE_PATTERN = re.compile(r"\bcandidate\s*\((.*?)\)\s*==", re.DOTALL)


def extract_probe_args_from_tests(
    test_source: str, entry_point: Optional[str] = None,
    max_probes: int = 4,
) -> List[str]:
    """Extract probe argument expressions from test assertions.

    Looks for `candidate(args) == ...` and `<entry_point>(args) == ...`
    patterns. Returns the first `max_probes` non-trivial matches.
    """
    matches: List[str] = []
    for m in CANDIDATE_PATTERN.finditer(test_source):
        s = m.group(1).strip()
        if 0 < len(s) < 200:
            matches.append(s)
    if not matches and entry_point:
        pat = CALL_PATTERN.pattern.replace("{name}", re.escape(entry_point))
        for m in re.finditer(pat, test_source, re.DOTALL):
            s = m.group(1).strip()
            if 0 < len(s) < 200:
                matches.append(s)
    return matches[:max_probes]


# ============================================================
# Structural (AST) clustering — ablation only
# ============================================================

def cluster_by_ast(candidates_code: Sequence[str]) -> List[str]:
    """Compute AST-hash cluster keys.

    Provided for ablation against behavioral clustering. The BTC paper shows
    this produces spurious abstentions on code-completion tasks because
    behaviorally identical implementations have distinct ASTs.
    """
    keys: List[str] = []
    for c in candidates_code:
        cleaned = re.sub(r"```\w*\n?", "", c).replace("```", "").strip()
        try:
            tree = ast.parse(cleaned, mode="exec")
            keys.append(ast.dump(tree, annotate_fields=False))
        except SyntaxError:
            keys.append(re.sub(r"\s+", "", cleaned))
    return keys
