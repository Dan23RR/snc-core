"""
06: HumanEval ufficiale (164 problemi) — paper-grade benchmark.

Pipeline:
1. Scarica HumanEval.jsonl.gz da openai/human-eval (one-time, ~80KB)
2. Per ogni problema: vanilla (T=0) + K=5 hybrid (T=0.8)
3. Sandbox = formato HumanEval ufficiale (prompt + code + test + check())
4. Behavioral probe = estrai inputs da `assert candidate(args) == expected` regex
5. Cache: candidates_cache_he.jsonl (riusabile, resumable)
6. Threshold sweep [0.40..0.80] offline + bootstrap 95% CI

Tempo:
- CPU mid-tier: ~30-60s per chiamata × 6 (1 vanilla + 5 hybrid) × 164 ≈ 8-16 ore
- GPU: ~5-10s per chiamata ≈ 1-3 ore
- Per test rapido: MAX_PROBLEMS=20

Run:
    python 06_humaneval_full.py
    # Subset:
    $env:MAX_PROBLEMS=50; python 06_humaneval_full.py
"""
import csv
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from typing import List, Dict, Tuple

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
LOCAL_HE = os.path.join(HERE, "humaneval.jsonl")
CACHE_PATH = os.path.join(HERE, "candidates_cache_he.jsonl")
RESULTS_CSV = os.path.join(HERE, "bench_humaneval.csv")

# Reuse moduli precedenti
sys.path.insert(0, HERE)
from importlib import import_module
v2 = import_module("02b_snc_qwen_v2")
he_mini = import_module("03_humaneval_mini")


# ============================================================
# Download HumanEval (one-time)
# ============================================================
def ensure_humaneval():
    if os.path.exists(LOCAL_HE):
        return
    print(f"Downloading HumanEval from {HUMANEVAL_URL}...")
    try:
        with urllib.request.urlopen(HUMANEVAL_URL, timeout=60) as r:
            gz_data = r.read()
        # The URL serves gz content (sometimes auto-decompressed by urllib).
        try:
            decompressed = gzip.decompress(gz_data)
        except (OSError, gzip.BadGzipFile):
            decompressed = gz_data  # already plaintext
        with open(LOCAL_HE, "wb") as f:
            f.write(decompressed)
        print(f"  Saved to {LOCAL_HE} ({len(decompressed)} bytes)")
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        print(f"  Manual fallback: scarica https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz")
        print(f"  decomprimi e salva come {LOCAL_HE}")
        sys.exit(1)


def load_problems() -> List[Dict]:
    problems = []
    with open(LOCAL_HE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ============================================================
# Sandbox HumanEval-style
# ============================================================
def run_humaneval_sandbox(candidate_code: str, problem: Dict,
                          timeout: float = 10.0) -> Dict:
    """Runs: prompt + candidate_code + test + check(entry_point).

    HumanEval test field defines `check(candidate)` function.
    candidate_code may include the prompt or just the body — try both.
    """
    entry = problem["entry_point"]
    test = problem["test"]
    prompt = problem["prompt"]

    # 3 try strategies (most-permissive last):
    strategies = [
        candidate_code,                                        # candidate full def
        prompt + "\n" + candidate_code,                        # prompt + body
        prompt + "\n    " + candidate_code.replace("\n", "\n    "),  # body indented
    ]

    last_err = None
    for code in strategies:
        full = code + "\n\n" + test + f"\n\ncheck({entry})\nprint('__OK__')\n"
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                          encoding="utf-8") as f:
            f.write(full)
            tmp = f.name
        try:
            proc = subprocess.run([sys.executable, tmp],
                                  capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0 and "__OK__" in proc.stdout:
                return {"passed": True, "error": None}
            last_err = (proc.stderr or proc.stdout)[:300]
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return {"passed": False, "error": last_err}


# ============================================================
# Behavioral probe per HumanEval
# ============================================================
ASSERT_CALL_RE = re.compile(
    r"(?:candidate|" + r"\w+" + r")\s*\(\s*([^=]*?)\s*\)\s*==",
    re.DOTALL
)


def extract_humaneval_inputs(problem: Dict, max_inputs: int = 4) -> List[str]:
    """Estrae stringhe-argomento dai test del problema. Cerca pattern
    `candidate(args)==` nel campo test."""
    test = problem.get("test", "")
    matches = []
    # primo pattern: candidate(...)
    for m in re.finditer(r"candidate\s*\((.*?)\)\s*==", test, re.DOTALL):
        s = m.group(1).strip()
        if 0 < len(s) < 200:  # sanity
            matches.append(s)
    # secondo: entry_point(...)
    if not matches:
        entry = problem["entry_point"]
        for m in re.finditer(rf"\b{re.escape(entry)}\s*\((.*?)\)\s*==", test, re.DOTALL):
            s = m.group(1).strip()
            if 0 < len(s) < 200:
                matches.append(s)
    return matches[:max_inputs]


def behavioral_probe_humaneval(candidate_code: str, problem: Dict,
                                inputs: List[str],
                                timeout: float = 5.0) -> Tuple:
    """Esegue candidate code, chiama entry_point su ogni input, ritorna tuple
    di output stringificati."""
    if not inputs:
        return ("__NO_INPUTS__",)
    entry = problem["entry_point"]
    prompt = problem["prompt"]

    # Costruisci runner: prova prima senza prompt prefix; se entry_point non è
    # definito, fallback con prompt prefix.
    inputs_repr = ", ".join(repr(s) for s in inputs)
    body = (
        f"results = []\n"
        f"for args_str in [{inputs_repr}]:\n"
        f"    try:\n"
        f"        out = eval(f'{entry}({{args_str}})')\n"
        f"        results.append(repr(out))\n"
        f"    except Exception as e:\n"
        f"        results.append('__ERR__:' + type(e).__name__)\n"
        f"print('|||'.join(results))\n"
    )

    for code in [candidate_code, prompt + "\n" + candidate_code]:
        runner = code + "\n\n" + body
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                          encoding="utf-8") as f:
            f.write(runner)
            tmp = f.name
        try:
            proc = subprocess.run([sys.executable, tmp],
                                  capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0 and "|||" in proc.stdout:
                return tuple(proc.stdout.strip().split("|||"))
        except subprocess.TimeoutExpired:
            return ("__TIMEOUT__",)
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return ("__UNDEFINED__",)


# ============================================================
# Cache I/O
# ============================================================
def load_cache() -> Dict[str, Dict]:
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[rec["id"]] = rec
            except Exception:
                pass
    return cache


def append_cache(rec: Dict):
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================
# Generate (vanilla + K=5 candidates) for one problem
# ============================================================
def generate_for_problem(p: Dict) -> Dict:
    pid = p["task_id"]
    prompt_for_llm = p["prompt"] + "\n# Completa la funzione sopra. Solo codice Python valido."

    # Vanilla
    t0 = time.time()
    r_v = v2.call_ollama(prompt_for_llm, system="", temperature=0.0, num_predict=400)
    v_text = r_v.get("response", "").strip()
    v_code = he_mini.extract_code(v_text)
    v_pass = run_humaneval_sandbox(v_code, p)["passed"]
    v_tokens = r_v.get("eval_count", 0)
    v_time = time.time() - t0

    # Hybrid K=5
    t0 = time.time()
    cands_raw = v2.generate_K_candidates(prompt_for_llm, K=5, temperature=0.8)
    h_time = time.time() - t0
    inputs = extract_humaneval_inputs(p)
    cands = []
    for c in cands_raw:
        code = he_mini.extract_code(c["answer"])
        sb = run_humaneval_sandbox(code, p)
        out_tuple = behavioral_probe_humaneval(code, p, inputs)
        cands.append({
            "code": code,
            "self_confidence": c["self_confidence"],
            "tokens": c["tokens"],
            "passed": sb["passed"],
            "output_tuple": list(out_tuple),  # JSON-serializable
        })
    return {
        "id": pid,
        "vanilla": {
            "code": v_code, "passed": v_pass, "tokens": v_tokens,
            "elapsed": v_time,
        },
        "candidates": cands,
        "hybrid_elapsed": h_time,
    }


# ============================================================
# Behavioral governor (ricalca 05)
# ============================================================
def behavioral_governance(candidates: List[Dict], threshold: float) -> Dict:
    valid = [c for c in candidates if c.get("output_tuple")]
    if not valid:
        return {"action": "ABSTAIN", "trust": 0.0, "modal_passed": False,
                "n_clusters": 0}
    n = len(valid)
    counts = Counter(tuple(c["output_tuple"]) for c in valid)
    n_clusters = len(counts)
    ppv = sum(c["self_confidence"] for c in valid) / n
    if n_clusters <= 1:
        sigma = 0.0
    else:
        H = -sum((cnt/n) * math.log(cnt/n) for cnt in counts.values() if cnt > 0)
        H_max = math.log(n_clusters)
        sigma = max(0.0, H / max(H_max, 1e-9))
    T_comp = 0.5 + 1.0 * (1.0 - ppv)
    trust = ppv * math.exp(-sigma * T_comp)
    modal_tuple, _ = counts.most_common(1)[0]
    modal_cand = next(c for c in valid if tuple(c["output_tuple"]) == modal_tuple)
    action = "ADMIT" if trust >= threshold else "ABSTAIN"
    return {
        "action": action, "trust": trust, "modal_passed": modal_cand["passed"],
        "n_clusters": n_clusters, "ppv": ppv, "sigma": sigma,
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values: List[int], n_boot: int = 2000, alpha: float = 0.05):
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [values[random.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2))])


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion z-test for the difference p1 - p2."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p_hat = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
    return (p1 - p2) / max(se, 1e-9)


# ============================================================
# Main
# ============================================================
def main():
    ensure_humaneval()
    problems = load_problems()
    print(f"HumanEval: {len(problems)} problems loaded.")
    MAX_PROBLEMS = int(os.environ.get("MAX_PROBLEMS", len(problems)))
    problems = problems[:MAX_PROBLEMS]
    print(f"Processing {len(problems)} problems "
          f"(MAX_PROBLEMS env to limit; default = all).")

    cache = load_cache()
    print(f"Cache: {len(cache)} records loaded from {CACHE_PATH}")

    # Generation phase
    t_start = time.time()
    n_done = 0
    n_total = len(problems)
    for i, p in enumerate(problems, 1):
        pid = p["task_id"]
        if pid in cache:
            continue
        elapsed = time.time() - t_start
        eta = (elapsed / max(n_done, 1)) * (n_total - i + 1) if n_done > 0 else 0
        print(f"[{i}/{n_total}] {pid} (done={n_done}, ETA={eta/60:.1f} min)", flush=True)
        try:
            rec = generate_for_problem(p)
            append_cache(rec)
            cache[pid] = rec
            n_done += 1
            v_ok = "✓" if rec["vanilla"]["passed"] else "✗"
            n_pass = sum(1 for c in rec["candidates"] if c["passed"])
            n_clust = len(set(tuple(c["output_tuple"]) for c in rec["candidates"]))
            print(f"   vanilla {v_ok}  cand_pass={n_pass}/5  clusters={n_clust}",
                  flush=True)
        except Exception as e:
            print(f"   ERROR: {e}", flush=True)

    # Threshold sweep
    print()
    print("=" * 78)
    print(f"THRESHOLD SWEEP — n={len(cache)}")
    print("=" * 78)
    n = len(cache)
    v_pass = sum(1 for r in cache.values() if r["vanilla"]["passed"])
    v_rate = v_pass / n
    print(f"{'Threshold':>10}  {'pass@1':>9}  {'abstain':>9}  {'halluc':>8}  {'net_prec':>9}")
    print("-" * 78)
    print(f"{'(vanilla)':>10}  {v_pass}/{n}  {'    -':>9}  {n-v_pass:>3}/{n}  {v_rate:>9.2%}")

    summary = []
    rows_csv = []
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        passed = abstain = halluc = 0
        for rec in cache.values():
            d = behavioral_governance(rec["candidates"], threshold=thr)
            if d["action"] == "ABSTAIN":
                abstain += 1
            elif d["modal_passed"]:
                passed += 1
            else:
                halluc += 1
            rows_csv.append({
                "problem": rec["id"], "threshold": thr,
                "vanilla_passed": rec["vanilla"]["passed"],
                "hybrid_passed": d["modal_passed"],
                "abstained": d["action"] == "ABSTAIN",
                "trust": d["trust"], "n_clusters": d["n_clusters"],
            })
        admitted = passed + halluc
        net_prec = passed / admitted if admitted else 0.0
        print(f"{thr:>10.2f}  {passed}/{n}  {abstain:>3}/{n}  {halluc:>3}/{n}  {net_prec:>9.2%}")
        summary.append({"threshold": thr, "pass": passed, "abstain": abstain,
                        "halluc": halluc, "net_prec": net_prec, "admitted": admitted})

    # Bootstrap CI + significance test
    print()
    print("=" * 78)
    print("BOOTSTRAP 95% CI + SIGNIFICANCE")
    print("=" * 78)
    vanilla_vec = [1 if r["vanilla"]["passed"] else 0 for r in cache.values()]
    v_lo, v_hi = bootstrap_ci(vanilla_vec)
    print(f"VANILLA pass@1 = {v_rate:.2%}  CI=[{v_lo:.2%}, {v_hi:.2%}]")

    # Halluc rate vanilla vs hybrid (per ogni threshold con coverage >=70%)
    print()
    print(f"{'Threshold':>10}  {'vanilla_halluc_rate':>20}  {'hybrid_halluc_rate':>20}  {'z_stat':>8}")
    print("-" * 78)
    halluc_v_count = n - v_pass
    halluc_v_rate = halluc_v_count / n
    for s in summary:
        if s["admitted"] >= n * 0.7:
            halluc_h_rate = s["halluc"] / s["admitted"]  # halluc rate AMONG ADMITTED
            z = two_proportion_z(halluc_v_rate, n, halluc_h_rate, s["admitted"])
            sig = "SIG" if abs(z) > 1.96 else "n.s."
            print(f"{s['threshold']:>10.2f}  "
                  f"{halluc_v_rate:>20.2%}  {halluc_h_rate:>20.2%}  "
                  f"{z:+.2f} [{sig}]")

    # Save
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nResults: {RESULTS_CSV}")
    print(f"Cache (resumable): {CACHE_PATH}")


if __name__ == "__main__":
    main()
