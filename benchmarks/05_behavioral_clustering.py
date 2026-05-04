"""
Fix architetturale: passare da AST clustering (strutturale) a OUTPUT clustering
(comportamentale). Riusa la cache di 04, ricomputa il governor.

Insight critico dai dati di 04:
  HE/08 Roman: vanilla ✗, 5/5 candidati corretti, MA AST hash li separa in 5
              cluster → σ=1 → trust crolla → ABSTAIN (false positive).

Soluzione: per ogni candidato, esegui su K test inputs sintetici, cluster per
sequenza di output. Due implementazioni AST-diverse ma behaviorally-identical
finiscono nello stesso cluster → σ=0 → trust alto.

Pipeline:
1. Leggi cache di 04 (candidates_cache.jsonl) — niente nuove chiamate Qwen
2. Per OGNI problema: estrai 3-5 test inputs canonici dai test cases esistenti
3. Per OGNI candidato: esegui su test inputs, ottieni tuple di output
4. Cluster candidati per output_tuple identica
5. Ricomputa σ_calib = entropia su cluster behaviorali
6. Threshold sweep [0.40..0.80] e bootstrap CI

Predizione: false-abstention rate cala drammaticamente. Threshold ottimale
si sposta verso 0.5-0.6 con net_prec stabile.
"""
import ast
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from typing import List, Dict, Tuple, Optional

random.seed(42)

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "candidates_cache.jsonl")


# ============================================================
# Test input extraction
# ============================================================
# Per ogni problem, estrarre gli input dai test asserts ("assert f(x,y) == z")
# usando AST parsing del test string. Più robusto di regex.

CALL_RE = re.compile(r"assert\s+(\w+)\s*\((.*?)\)\s*==", re.DOTALL)


def extract_test_inputs_from_asserts(tests: List[str]) -> Tuple[str, List[str]]:
    """Ritorna (function_name, list_of_call_arg_strings)."""
    fn_name = None
    inputs = []
    for t in tests:
        m = CALL_RE.search(t)
        if m:
            fn_name = m.group(1)
            args = m.group(2).strip()
            inputs.append(args)
    return fn_name, inputs


# ============================================================
# Esegui candidato su test inputs, ritorna tuple di output (o exception flag)
# ============================================================
def run_behavioral_probe(code: str, fn_name: str, input_strs: List[str],
                          timeout: float = 5.0) -> Tuple:
    """Esegui code, chiama fn_name su ogni input. Ritorna tuple di output
    (stringificate). Su exception/timeout, sostituisce con sentinel '__ERR__'."""
    if not fn_name or not input_strs:
        return ("__NO_INPUTS__",)
    runner = code + "\n\nimport json, sys\n"
    runner += "results = []\n"
    runner += f"for args in [{','.join(repr(s) for s in input_strs)}]:\n"
    runner += "    try:\n"
    runner += f"        out = eval(f'{fn_name}({{args}})')\n"
    runner += "        results.append(repr(out))\n"
    runner += "    except Exception as e:\n"
    runner += "        results.append('__ERR__:' + type(e).__name__)\n"
    runner += "print('|||'.join(results))\n"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                      encoding="utf-8") as f:
        f.write(runner)
        tmp = f.name
    try:
        proc = subprocess.run([sys.executable, tmp],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return tuple(proc.stdout.strip().split("|||"))
        return ("__RUNTIME_ERR__", proc.stderr[:100] if proc.stderr else "")
    except subprocess.TimeoutExpired:
        return ("__TIMEOUT__",)
    except Exception as e:
        return (f"__EXC__:{type(e).__name__}",)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ============================================================
# Behavioral governor
# ============================================================
def behavioral_governance(candidates_with_outputs: List[Dict],
                          threshold: float = 0.6) -> Dict:
    """Governor che cluster per OUTPUT TUPLE invece che per AST.

    candidates_with_outputs: list of {code, self_confidence, passed, output_tuple}
    """
    valid = [c for c in candidates_with_outputs if c.get("output_tuple")]
    if not valid:
        return {"action": "ABSTAIN", "trust": 0.0, "reason": "no valid"}

    n = len(valid)
    # Cluster by output tuple
    counts = Counter(c["output_tuple"] for c in valid)
    n_clusters = len(counts)

    # PPV: media self-confidence
    ppv = sum(c["self_confidence"] for c in valid) / n

    # σ_calib: entropia sui cluster behaviorali
    if n_clusters <= 1:
        sigma_calib = 0.0
    else:
        H = -sum((cnt/n) * math.log(cnt/n) for cnt in counts.values() if cnt > 0)
        H_max = math.log(n_clusters)
        sigma_calib = max(0.0, H / max(H_max, 1e-9))

    T_comp = 0.5 + 1.0 * (1.0 - ppv)
    trust = ppv * math.exp(-sigma_calib * T_comp)

    # Modal cluster: il più frequente. Pick il primo candidato di quel cluster.
    modal_tuple, modal_count = counts.most_common(1)[0]
    modal_cand = next(c for c in valid if c["output_tuple"] == modal_tuple)

    action = "ADMIT" if trust >= threshold else "ABSTAIN"
    return {
        "action": action, "trust": trust, "ppv": ppv,
        "sigma_calib": sigma_calib, "T_comp": T_comp,
        "modal_passed": modal_cand["passed"],
        "n_clusters": n_clusters,
        "modal_agreement": modal_count / n,
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    n = len(values)
    if n == 0:
        return (0, 0)
    means = []
    for _ in range(n_boot):
        sample = [values[random.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2))])


# ============================================================
# Main: ricarica cache, esegui behavioral probes, sweep
# ============================================================
def main():
    # Load problems from 03 + 04
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    he = import_module("03_humaneval_mini")
    full = import_module("04_full_bench")
    problems_by_id = {p["id"]: p for p in full.ALL_PROBLEMS}

    # Load cache
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: cache not found at {CACHE_PATH}")
        print("Run 04_full_bench.py first to generate the cache.")
        return
    cache = []
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                cache.append(json.loads(line))
            except Exception:
                pass
    print(f"Loaded {len(cache)} cached records.")
    print()

    # Per ogni record, esegui behavioral probe
    print("Running behavioral probes (eseguendo ogni candidato su test inputs)...")
    enriched = []
    for rec in cache:
        pid = rec["id"]
        if pid not in problems_by_id:
            continue
        p = problems_by_id[pid]
        fn_name, inputs = extract_test_inputs_from_asserts(p["tests"])
        cands = []
        for c in rec["candidates"]:
            output_tuple = run_behavioral_probe(c["code"], fn_name, inputs)
            cands.append({
                "code": c["code"],
                "self_confidence": c["self_confidence"],
                "passed": c["passed"],
                "output_tuple": output_tuple,
            })
        enriched.append({
            "id": pid,
            "vanilla": rec["vanilla"],
            "candidates_b": cands,
        })
        n_clusters = len(set(c["output_tuple"] for c in cands))
        v_ok = "✓" if rec["vanilla"]["passed"] else "✗"
        n_pass = sum(1 for c in cands if c["passed"])
        print(f"  {pid}: vanilla {v_ok}  cand_pass={n_pass}/5  "
              f"behavioral_clusters={n_clusters}")
    print()

    # Threshold sweep
    print("=" * 78)
    print("BEHAVIORAL CLUSTERING — Threshold sweep")
    print("=" * 78)
    print(f"{'Threshold':>10}  {'pass@1':>8}  {'abstain':>8}  {'halluc':>7}  {'net_prec':>9}")
    print("-" * 78)
    n = len(enriched)
    v_pass = sum(1 for r in enriched if r["vanilla"]["passed"])
    print(f"{'(vanilla)':>10}  {v_pass}/{n}  {'    -':>8}  {n-v_pass:>3}/{n}  {v_pass/n:>9.2%}")

    summary = []
    rows_csv = []
    for thr in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        passed = abstain = halluc = 0
        for r in enriched:
            d = behavioral_governance(r["candidates_b"], threshold=thr)
            if d["action"] == "ABSTAIN":
                abstain += 1
            elif d["modal_passed"]:
                passed += 1
            else:
                halluc += 1
            rows_csv.append({
                "problem": r["id"], "threshold": thr,
                "vanilla_passed": r["vanilla"]["passed"],
                "hybrid_passed": d.get("modal_passed", False),
                "abstained": d["action"] == "ABSTAIN",
                "trust": d["trust"],
                "n_clusters": d.get("n_clusters", 0),
            })
        admitted = passed + halluc
        net_prec = passed / admitted if admitted else 0.0
        print(f"{thr:>10.2f}  {passed}/{n}  {abstain:>3}/{n}  {halluc:>3}/{n}  {net_prec:>9.2%}")
        summary.append({"threshold": thr, "pass": passed, "abstain": abstain,
                        "halluc": halluc, "net_prec": net_prec, "n": n})

    # Bootstrap CI for vanilla and best threshold
    print()
    print("=" * 78)
    print("BOOTSTRAP 95% CI (n_boot=2000)")
    print("=" * 78)
    vanilla_vals = [1 if r["vanilla"]["passed"] else 0 for r in enriched]
    v_lo, v_hi = bootstrap_ci(vanilla_vals)
    print(f"VANILLA pass@1 = {v_pass/n:.2%}  CI=[{v_lo:.2%}, {v_hi:.2%}]")

    # Best threshold = quello con highest net_prec MA admitted >= 50%
    eligible = [s for s in summary if (s["pass"] + s["halluc"]) >= n * 0.5]
    if eligible:
        best = max(eligible, key=lambda s: (s["net_prec"], s["pass"]))
        bt = best["threshold"]
        admitted_vals = []
        for r in enriched:
            d = behavioral_governance(r["candidates_b"], threshold=bt)
            if d["action"] == "ADMIT":
                admitted_vals.append(1 if d["modal_passed"] else 0)
        if admitted_vals:
            b_lo, b_hi = bootstrap_ci(admitted_vals)
            print(f"HYBRID@{bt} net_prec = {best['net_prec']:.2%}  "
                  f"CI=[{b_lo:.2%}, {b_hi:.2%}]  "
                  f"(on {len(admitted_vals)}/{n} admitted, "
                  f"abstain {best['abstain']}/{n})")

    # Save CSV
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_behavioral.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
