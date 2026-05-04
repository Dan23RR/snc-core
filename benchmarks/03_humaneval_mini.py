"""
HumanEval mini-benchmark: Qwen vanilla vs SNC-governed Hybrid.

10 problemi HumanEval-style embedded (no download).
Per ogni problema:
  - condizione A. VANILLA: Qwen T=0.0, K=1
  - condizione B. HYBRID:  Qwen T=0.8, K=5 + SNC governance v2

Per ogni candidato/risposta: extract code → execute in subprocess sandbox
con timeout 5s → verify test cases.

Output: report tabular + CSV con tutti i risultati.

Metric:
  pass@1            = trial pass rate (ADMIT + correct)
  abstention rate   = ABSTAIN frequency (solo Hybrid)
  hallucination rate = ADMIT ma incorrect (worst case)
  net precision     = correct / (correct + hallucination)  [esclude ABSTAIN]

Predizione (verificabile):
  Vanilla pass@1 ≈ 0.55-0.75
  Hybrid pass@1 ≈ stesso o leggermente più basso (ma con abstention)
  Hybrid net_precision > Vanilla precision (cattura hallucination)
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

# Reuse v2 logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
v2 = import_module("02b_snc_qwen_v2")


# ============================================================
# 10 HumanEval-style problems
# ============================================================
# Format: prompt = function signature + docstring; tests = list of asserts.
# Reference solutions via brute force (only for verification).

PROBLEMS = [
    {
        "id": "HE/01_is_palindrome",
        "prompt": '''def is_palindrome(s: str) -> bool:
    """Ritorna True se la stringa è un palindromo (case-insensitive, ignora spazi).
    Esempio: is_palindrome("A man a plan a canal Panama") == True
    """
''',
        "tests": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('A man a plan a canal Panama') == True",
            "assert is_palindrome('') == True",
        ],
    },
    {
        "id": "HE/02_fibonacci",
        "prompt": '''def fibonacci(n: int) -> int:
    """Ritorna l'n-esimo numero di Fibonacci (0-indexed). fib(0)=0, fib(1)=1.
    """
''',
        "tests": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(10) == 55",
        ],
    },
    {
        "id": "HE/03_count_vowels",
        "prompt": '''def count_vowels(s: str) -> int:
    """Conta le vocali (a, e, i, o, u) in una stringa, case-insensitive.
    """
''',
        "tests": [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('AEIOU') == 5",
            "assert count_vowels('xyz') == 0",
        ],
    },
    {
        "id": "HE/04_gcd",
        "prompt": '''def gcd(a: int, b: int) -> int:
    """Massimo comun divisore di due interi positivi (Euclide).
    """
''',
        "tests": [
            "assert gcd(12, 18) == 6",
            "assert gcd(7, 13) == 1",
            "assert gcd(100, 25) == 25",
        ],
    },
    {
        "id": "HE/05_flatten",
        "prompt": '''def flatten(lst: list) -> list:
    """Appiattisce una lista nidificata di profondità arbitraria.
    Esempio: flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    """
''',
        "tests": [
            "assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]",
            "assert flatten([]) == []",
            "assert flatten([[1], [2], [3]]) == [1, 2, 3]",
        ],
    },
    {
        "id": "HE/06_anagram",
        "prompt": '''def is_anagram(a: str, b: str) -> bool:
    """Ritorna True se le due stringhe sono anagrammi (case-insensitive).
    """
''',
        "tests": [
            "assert is_anagram('listen', 'silent') == True",
            "assert is_anagram('hello', 'world') == False",
            "assert is_anagram('Astronomer', 'MoonStarer') == True",
        ],
    },
    {
        "id": "HE/07_max_subarray",
        "prompt": '''def max_subarray_sum(nums: list) -> int:
    """Somma massima di un sottoarray contiguo (Kadane).
    Esempio: max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6
    """
''',
        "tests": [
            "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6",
            "assert max_subarray_sum([1]) == 1",
            "assert max_subarray_sum([-1,-2,-3]) == -1",
        ],
    },
    {
        "id": "HE/08_roman",
        "prompt": '''def to_roman(n: int) -> str:
    """Converte un intero positivo (1-3999) in numero romano.
    Esempio: to_roman(1994) == 'MCMXCIV'
    """
''',
        "tests": [
            "assert to_roman(1) == 'I'",
            "assert to_roman(4) == 'IV'",
            "assert to_roman(1994) == 'MCMXCIV'",
            "assert to_roman(3999) == 'MMMCMXCIX'",
        ],
    },
    {
        "id": "HE/09_balanced_paren",
        "prompt": '''def is_balanced(s: str) -> bool:
    """True se le parentesi (), [], {} sono bilanciate. False altrimenti.
    """
''',
        "tests": [
            "assert is_balanced('()') == True",
            "assert is_balanced('([{}])') == True",
            "assert is_balanced('(]') == False",
            "assert is_balanced('') == True",
        ],
    },
    {
        "id": "HE/10_dedupe_keep_order",
        "prompt": '''def dedupe(lst: list) -> list:
    """Rimuove duplicati preservando l'ordine della prima occorrenza.
    Esempio: dedupe([1,2,1,3,2,4]) == [1,2,3,4]
    """
''',
        "tests": [
            "assert dedupe([1,2,1,3,2,4]) == [1,2,3,4]",
            "assert dedupe([]) == []",
            "assert dedupe(['a','b','a']) == ['a','b']",
        ],
    },
]


# ============================================================
# Code extraction + execution sandbox
# ============================================================
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Estrae blocco python ```...``` dalla risposta. Se assente, ritorna text raw."""
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: cerca riga 'def ' e prendi tutto da lì
    idx = text.find("def ")
    if idx >= 0:
        return text[idx:].strip()
    return text.strip()


def run_in_sandbox(code: str, tests: list, timeout: float = 5.0) -> dict:
    """Esegui code + tests in subprocess. Ritorna {passed, error}."""
    full_code = code + "\n\n" + "\n".join(tests) + "\nprint('OK')"
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                      encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0 and "OK" in proc.stdout:
            return {"passed": True, "error": None}
        return {"passed": False,
                "error": (proc.stderr or proc.stdout or "no output")[:300]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout"}
    except Exception as e:
        return {"passed": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============================================================
# Conditions
# ============================================================
def condition_vanilla(problem: dict) -> dict:
    """Qwen T=0.0, K=1, no governance."""
    prompt = problem["prompt"] + "\n# Implementa la funzione. Solo codice Python valido."
    t0 = time.time()
    r = v2.call_ollama(prompt, system="", temperature=0.0, num_predict=400)
    elapsed = time.time() - t0
    text = r.get("response", "").strip()
    code = extract_code(text)
    sandbox = run_in_sandbox(code, problem["tests"])
    return {
        "condition": "VANILLA", "code": code, "passed": sandbox["passed"],
        "error": sandbox["error"], "elapsed": elapsed,
        "tokens": r.get("eval_count", 0), "abstained": False,
    }


def condition_hybrid(problem: dict, K: int = 5, threshold: float = 0.6) -> dict:
    """Qwen T=0.8, K=5, SNC governance v2 + sandbox-execution as additional trust signal."""
    prompt = problem["prompt"] + "\n# Implementa la funzione. Solo codice Python valido."
    t0 = time.time()
    candidates = v2.generate_K_candidates(prompt, K=K, temperature=0.8)
    elapsed = time.time() - t0

    # Replace each candidate's "answer" with its extracted code (so semantic
    # normalization clusters by AST)
    for c in candidates:
        c["answer"] = extract_code(c["answer"])

    decision = v2.snc_governance_v2(candidates, threshold=threshold)
    if decision["action"] == "ABSTAIN":
        return {
            "condition": "HYBRID", "code": None, "passed": False,
            "error": "ABSTAINED", "elapsed": elapsed,
            "tokens": sum(c["tokens"] for c in candidates),
            "abstained": True, "trust": decision["trust"],
            "ppv": decision["ppv"], "sigma": decision["sigma_calib"],
        }
    code = decision["modal_answer"]
    sandbox = run_in_sandbox(code, problem["tests"])
    return {
        "condition": "HYBRID", "code": code, "passed": sandbox["passed"],
        "error": sandbox["error"], "elapsed": elapsed,
        "tokens": sum(c["tokens"] for c in candidates),
        "abstained": False, "trust": decision["trust"],
        "ppv": decision["ppv"], "sigma": decision["sigma_calib"],
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 78)
    print("HumanEval mini — Qwen vanilla vs SNC Hybrid (K=5)")
    print(f"Problems: {len(PROBLEMS)}")
    print("=" * 78)

    rows = []
    for p in PROBLEMS:
        print(f"\n[{p['id']}]")
        # Vanilla
        v = condition_vanilla(p)
        flag = "✓" if v["passed"] else "✗"
        print(f"  VANILLA  {flag}  ({v['elapsed']:.1f}s, {v['tokens']} tok)  "
              f"err={v['error'][:60] if v['error'] else 'none'}")
        # Hybrid
        h = condition_hybrid(p)
        if h["abstained"]:
            print(f"  HYBRID   ⊘  (ABSTAIN, trust={h.get('trust',0):.3f})  "
                  f"({h['elapsed']:.1f}s, {h['tokens']} tok)")
        else:
            flag2 = "✓" if h["passed"] else "✗"
            print(f"  HYBRID   {flag2}  trust={h.get('trust',0):.3f}  "
                  f"({h['elapsed']:.1f}s, {h['tokens']} tok)  "
                  f"err={h['error'][:60] if h['error'] else 'none'}")

        rows.append({
            "problem": p["id"],
            "vanilla_passed": v["passed"],
            "vanilla_tokens": v["tokens"],
            "hybrid_passed": h["passed"],
            "hybrid_abstained": h["abstained"],
            "hybrid_trust": h.get("trust", 0),
            "hybrid_tokens": h["tokens"],
        })

    # Aggregate
    n = len(rows)
    v_pass = sum(r["vanilla_passed"] for r in rows)
    h_pass = sum(r["hybrid_passed"] for r in rows)
    h_abst = sum(r["hybrid_abstained"] for r in rows)
    h_admit = n - h_abst
    h_halluc = h_admit - h_pass

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    print(f"VANILLA  pass@1 = {v_pass}/{n} = {v_pass/n:.2%}")
    print(f"HYBRID   pass@1 = {h_pass}/{n} = {h_pass/n:.2%}  "
          f"(abstain {h_abst}/{n} = {h_abst/n:.2%}, "
          f"halluc {h_halluc}/{n} = {h_halluc/n:.2%})")
    if h_admit > 0:
        print(f"HYBRID   net_precision (correct/admitted) = {h_pass}/{h_admit} = {h_pass/h_admit:.2%}")
    print()
    print(f"VANILLA tokens total = {sum(r['vanilla_tokens'] for r in rows)}")
    print(f"HYBRID  tokens total = {sum(r['hybrid_tokens'] for r in rows)} (≈5× per K=5)")

    # Save CSV
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
