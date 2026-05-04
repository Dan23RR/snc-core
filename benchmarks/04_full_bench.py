"""
Full benchmark: 30 problemi HumanEval-style + threshold sweep + bootstrap CI.

Strategia:
1. Genera K=5 candidati per problema, salva in JSONL cache
2. Genera baseline vanilla (T=0) per problema
3. Esegui sandbox per OGNI candidato (non solo modal) e cache risultato
4. Applica governor OFFLINE a 5 threshold diversi: 0.40, 0.50, 0.60, 0.70, 0.80
5. Bootstrap 95% CI per pass@1 e net_precision

Output:
    candidates_cache.jsonl    — cache delle generazioni (per re-runs offline)
    bench_results_full.csv    — tutti i risultati per (problema, threshold)
    pareto_summary.txt        — Pareto curve precision vs abstention

Tempo atteso: ~40-90 min (30 problemi × 6 chiamate Qwen ≈ 180 chiamate).
Per ridurre: imposta `MAX_PROBLEMS = 15` per test rapido.
"""
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
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
v2 = import_module("02b_snc_qwen_v2")
he = import_module("03_humaneval_mini")

random.seed(42)


# ============================================================
# 20 nuovi problemi (estendono i 10 di 03 → 30 totali)
# ============================================================
NEW_PROBLEMS = [
    {
        "id": "HE/11_reverse_words",
        "prompt": '''def reverse_words(s: str) -> str:
    """Inverti l'ordine delle parole in una frase. Spazi multipli → singolo.
    Esempio: reverse_words('hello world') == 'world hello'
    """
''',
        "tests": [
            "assert reverse_words('hello world') == 'world hello'",
            "assert reverse_words('a b c') == 'c b a'",
            "assert reverse_words('') == ''",
        ],
    },
    {
        "id": "HE/12_factorial",
        "prompt": '''def factorial(n: int) -> int:
    """Fattoriale di n. factorial(0)=1, factorial(5)=120.
    """
''',
        "tests": [
            "assert factorial(0) == 1",
            "assert factorial(5) == 120",
            "assert factorial(10) == 3628800",
        ],
    },
    {
        "id": "HE/13_unique_chars",
        "prompt": '''def has_unique_chars(s: str) -> bool:
    """True se tutti i caratteri sono distinti. Case-sensitive.
    """
''',
        "tests": [
            "assert has_unique_chars('abcdef') == True",
            "assert has_unique_chars('hello') == False",
            "assert has_unique_chars('') == True",
        ],
    },
    {
        "id": "HE/14_sum_digits",
        "prompt": '''def sum_digits(n: int) -> int:
    """Somma delle cifre di un intero non negativo.
    """
''',
        "tests": [
            "assert sum_digits(0) == 0",
            "assert sum_digits(123) == 6",
            "assert sum_digits(9999) == 36",
        ],
    },
    {
        "id": "HE/15_count_words",
        "prompt": '''def count_words(s: str) -> int:
    """Numero di parole separate da whitespace.
    """
''',
        "tests": [
            "assert count_words('hello world') == 2",
            "assert count_words('') == 0",
            "assert count_words('one  two   three') == 3",
        ],
    },
    {
        "id": "HE/16_remove_vowels",
        "prompt": '''def remove_vowels(s: str) -> str:
    """Rimuove vocali a/e/i/o/u (case-insensitive).
    """
''',
        "tests": [
            "assert remove_vowels('hello') == 'hll'",
            "assert remove_vowels('AEIOU') == ''",
            "assert remove_vowels('xyz') == 'xyz'",
        ],
    },
    {
        "id": "HE/17_is_prime",
        "prompt": '''def is_prime(n: int) -> bool:
    """True se n è primo (n >= 2). False altrimenti incluso n<2.
    """
''',
        "tests": [
            "assert is_prime(2) == True",
            "assert is_prime(17) == True",
            "assert is_prime(1) == False",
            "assert is_prime(15) == False",
        ],
    },
    {
        "id": "HE/18_running_max",
        "prompt": '''def running_max(nums: list) -> list:
    """Massimo cumulativo: ogni posizione è il max degli elementi fino a lì.
    Esempio: running_max([3,1,4,1,5,9,2,6]) == [3,3,4,4,5,9,9,9]
    """
''',
        "tests": [
            "assert running_max([3,1,4,1,5,9,2,6]) == [3,3,4,4,5,9,9,9]",
            "assert running_max([]) == []",
            "assert running_max([5]) == [5]",
        ],
    },
    {
        "id": "HE/19_caesar",
        "prompt": '''def caesar_cipher(s: str, shift: int) -> str:
    """Cifra di Cesare: ogni lettera (a-z, A-Z) shifta di `shift`. Non-letter invariati.
    Esempio: caesar_cipher('abc', 1) == 'bcd'
    """
''',
        "tests": [
            "assert caesar_cipher('abc', 1) == 'bcd'",
            "assert caesar_cipher('xyz', 3) == 'abc'",
            "assert caesar_cipher('Hello, World!', 5) == 'Mjqqt, Btwqi!'",
        ],
    },
    {
        "id": "HE/20_intersection",
        "prompt": '''def intersection(a: list, b: list) -> list:
    """Elementi presenti in entrambe le liste, senza duplicati, ordinati.
    """
''',
        "tests": [
            "assert intersection([1,2,3], [2,3,4]) == [2,3]",
            "assert intersection([], [1,2]) == []",
            "assert intersection([1,1,2], [2,2,3]) == [2]",
        ],
    },
    {
        "id": "HE/21_max_consecutive_ones",
        "prompt": '''def max_consecutive_ones(nums: list) -> int:
    """Lunghezza massima di una run consecutiva di 1.
    """
''',
        "tests": [
            "assert max_consecutive_ones([1,1,0,1,1,1]) == 3",
            "assert max_consecutive_ones([0,0,0]) == 0",
            "assert max_consecutive_ones([1]) == 1",
        ],
    },
    {
        "id": "HE/22_pow_mod",
        "prompt": '''def pow_mod(base: int, exp: int, mod: int) -> int:
    """Esponenziazione modulare: (base ** exp) % mod, efficiente.
    """
''',
        "tests": [
            "assert pow_mod(2, 10, 1000) == 24",
            "assert pow_mod(3, 5, 7) == 5",
            "assert pow_mod(7, 0, 10) == 1",
        ],
    },
    {
        "id": "HE/23_zip_to_dict",
        "prompt": '''def zip_to_dict(keys: list, values: list) -> dict:
    """Crea dict da liste parallele. Se mismatch, prendi min length.
    """
''',
        "tests": [
            "assert zip_to_dict(['a','b'], [1,2]) == {'a':1, 'b':2}",
            "assert zip_to_dict([], []) == {}",
            "assert zip_to_dict(['a','b','c'], [1,2]) == {'a':1, 'b':2}",
        ],
    },
    {
        "id": "HE/24_count_substring",
        "prompt": '''def count_substring(s: str, sub: str) -> int:
    """Numero di occorrenze (anche sovrapposte) di `sub` in `s`.
    Esempio: count_substring('aaaa', 'aa') == 3
    """
''',
        "tests": [
            "assert count_substring('aaaa', 'aa') == 3",
            "assert count_substring('hello', 'l') == 2",
            "assert count_substring('abc', 'd') == 0",
        ],
    },
    {
        "id": "HE/25_chunks",
        "prompt": '''def chunks(lst: list, n: int) -> list:
    """Spezza la lista in chunk di dimensione n (l'ultimo può essere più piccolo).
    """
''',
        "tests": [
            "assert chunks([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]",
            "assert chunks([], 3) == []",
            "assert chunks([1,2,3], 5) == [[1,2,3]]",
        ],
    },
    {
        "id": "HE/26_word_freq",
        "prompt": '''def word_freq(s: str) -> dict:
    """Conta occorrenze di ogni parola (split by whitespace, case-insensitive).
    """
''',
        "tests": [
            "assert word_freq('a b a') == {'a':2, 'b':1}",
            "assert word_freq('') == {}",
            "assert word_freq('Hello hello') == {'hello':2}",
        ],
    },
    {
        "id": "HE/27_levenshtein",
        "prompt": '''def edit_distance(a: str, b: str) -> int:
    """Distanza di Levenshtein (minimo numero di insertion/deletion/substitution).
    """
''',
        "tests": [
            "assert edit_distance('kitten', 'sitting') == 3",
            "assert edit_distance('', 'abc') == 3",
            "assert edit_distance('same', 'same') == 0",
        ],
    },
    {
        "id": "HE/28_matrix_transpose",
        "prompt": '''def transpose(m: list) -> list:
    """Trasposta di matrice rettangolare.
    Esempio: transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]
    """
''',
        "tests": [
            "assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]",
            "assert transpose([[1]]) == [[1]]",
            "assert transpose([]) == []",
        ],
    },
    {
        "id": "HE/29_first_unique",
        "prompt": '''def first_unique_char(s: str) -> int:
    """Indice del primo carattere non ripetuto, -1 se nessuno.
    """
''',
        "tests": [
            "assert first_unique_char('leetcode') == 0",
            "assert first_unique_char('aabbcc') == -1",
            "assert first_unique_char('loveleetcode') == 2",
        ],
    },
    {
        "id": "HE/30_binary_search",
        "prompt": '''def binary_search(arr: list, target: int) -> int:
    """Indice del target in array ordinato, -1 se assente.
    """
''',
        "tests": [
            "assert binary_search([1,2,3,4,5], 3) == 2",
            "assert binary_search([1,2,3,4,5], 6) == -1",
            "assert binary_search([], 1) == -1",
        ],
    },
]

ALL_PROBLEMS = he.PROBLEMS + NEW_PROBLEMS  # 10 + 20 = 30


# ============================================================
# Cached generation
# ============================================================
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "candidates_cache.jsonl")


def load_cache():
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


def append_cache(rec):
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def generate_for_problem(p, cache):
    if p["id"] in cache:
        return cache[p["id"]]
    prompt = p["prompt"] + "\n# Implementa la funzione. Solo codice Python valido."
    # Vanilla
    t0 = time.time()
    r_vanilla = v2.call_ollama(prompt, system="", temperature=0.0, num_predict=400)
    vanilla_text = r_vanilla.get("response", "").strip()
    vanilla_code = he.extract_code(vanilla_text)
    vanilla_tokens = r_vanilla.get("eval_count", 0)
    vanilla_time = time.time() - t0
    # Hybrid candidates K=5
    t0 = time.time()
    cands = v2.generate_K_candidates(prompt, K=5, temperature=0.8)
    for c in cands:
        c["code"] = he.extract_code(c["answer"])
    hybrid_time = time.time() - t0
    # Sandbox: test ogni candidato + vanilla
    vanilla_pass = he.run_in_sandbox(vanilla_code, p["tests"])["passed"]
    cand_results = []
    for c in cands:
        sb = he.run_in_sandbox(c["code"], p["tests"])
        cand_results.append({
            "code": c["code"],
            "self_confidence": c["self_confidence"],
            "tokens": c["tokens"],
            "passed": sb["passed"],
        })
    rec = {
        "id": p["id"],
        "vanilla": {
            "code": vanilla_code,
            "passed": vanilla_pass,
            "tokens": vanilla_tokens,
            "elapsed": vanilla_time,
        },
        "candidates": cand_results,
        "hybrid_elapsed": hybrid_time,
    }
    append_cache(rec)
    return rec


# ============================================================
# Offline governor at variable threshold
# ============================================================
def apply_governor(rec, threshold):
    cands = rec["candidates"]
    # Trasforma in formato compatibile con v2.snc_governance_v2
    fake_candidates = [{
        "answer": c["code"],
        "self_confidence": c["self_confidence"],
        "tokens": c["tokens"],
    } for c in cands]
    decision = v2.snc_governance_v2(fake_candidates, threshold=threshold)
    if decision["action"] == "ABSTAIN":
        return {"abstained": True, "passed": False, "trust": decision["trust"]}
    # Trova quale candidato è il modal e leggi il suo passed
    modal_code = decision["modal_answer"]
    matching = [c for c in cands if c["code"] == modal_code]
    if not matching:
        return {"abstained": False, "passed": False, "trust": decision["trust"]}
    return {"abstained": False, "passed": matching[0]["passed"], "trust": decision["trust"]}


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
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


# ============================================================
# Main
# ============================================================
def main():
    MAX_PROBLEMS = int(os.environ.get("MAX_PROBLEMS", len(ALL_PROBLEMS)))
    problems = ALL_PROBLEMS[:MAX_PROBLEMS]
    print(f"Benchmark on {len(problems)} problems (set MAX_PROBLEMS env var to limit)")
    print(f"Cache: {CACHE_PATH}")
    print()

    cache = load_cache()
    print(f"Loaded {len(cache)} cached records.")

    # Generate
    for i, p in enumerate(problems, 1):
        if p["id"] in cache:
            print(f"[{i}/{len(problems)}] {p['id']}: cached.")
            continue
        print(f"[{i}/{len(problems)}] {p['id']}: generating...")
        try:
            rec = generate_for_problem(p, cache)
            cache[p["id"]] = rec
            v_ok = "✓" if rec["vanilla"]["passed"] else "✗"
            n_cand_ok = sum(1 for c in rec["candidates"] if c["passed"])
            print(f"   vanilla {v_ok}  candidates_pass={n_cand_ok}/5")
        except Exception as e:
            print(f"   ERROR: {e}")

    # Threshold sweep
    print()
    print("=" * 78)
    print("THRESHOLD SWEEP (offline)")
    print("=" * 78)
    print(f"{'Threshold':>10}  {'pass@1':>8}  {'abstain':>8}  {'halluc':>7}  {'net_prec':>9}")
    print("-" * 78)

    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]
    summary = []
    rows_csv = []

    # Vanilla baseline
    v_pass = sum(1 for r in cache.values() if r["vanilla"]["passed"])
    n = len(cache)
    print(f"{'(vanilla)':>10}  {v_pass}/{n}  {'    -':>8}  {n-v_pass:>3}/{n}  {v_pass/n:>9.2%}")

    for thr in thresholds:
        passed = abstain = halluc = 0
        for rec in cache.values():
            d = apply_governor(rec, thr)
            if d["abstained"]:
                abstain += 1
            elif d["passed"]:
                passed += 1
            else:
                halluc += 1
            rows_csv.append({
                "problem": rec["id"],
                "threshold": thr,
                "vanilla_passed": rec["vanilla"]["passed"],
                "hybrid_passed": d["passed"],
                "hybrid_abstained": d["abstained"],
                "trust": d["trust"],
            })
        admitted = passed + halluc
        net_prec = passed / admitted if admitted else 0
        print(f"{thr:>10.2f}  {passed}/{n}  {abstain:>3}/{n}  {halluc:>3}/{n}  {net_prec:>9.2%}")
        summary.append({
            "threshold": thr,
            "pass": passed, "abstain": abstain, "halluc": halluc,
            "net_prec": net_prec, "n": n,
        })

    # Bootstrap CI for vanilla and best threshold
    print()
    print("=" * 78)
    print("BOOTSTRAP 95% CI (n_boot=2000)")
    print("=" * 78)
    vanilla_vals = [1 if r["vanilla"]["passed"] else 0 for r in cache.values()]
    v_lo, v_hi = bootstrap_ci(vanilla_vals)
    print(f"VANILLA pass@1 = {v_pass/n:.2%}  CI=[{v_lo:.2%}, {v_hi:.2%}]")
    # Best threshold by net_prec
    best = max(summary, key=lambda s: s["net_prec"])
    # Vector for best threshold: 1 if (admitted AND passed), 0 if (admitted AND failed). Skip abstained.
    bt = best["threshold"]
    admitted_vals = []
    for rec in cache.values():
        d = apply_governor(rec, bt)
        if not d["abstained"]:
            admitted_vals.append(1 if d["passed"] else 0)
    if admitted_vals:
        b_lo, b_hi = bootstrap_ci(admitted_vals)
        print(f"HYBRID@{bt} net_prec = {best['net_prec']:.2%}  "
              f"CI=[{b_lo:.2%}, {b_hi:.2%}]  "
              f"(on {len(admitted_vals)}/{n} admitted)")

    # Save CSV
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_results_full.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nFull CSV written to: {out}")
    print(f"Cache (re-runs offline): {CACHE_PATH}")


if __name__ == "__main__":
    main()
