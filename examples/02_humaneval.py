"""02: HumanEval benchmark with behavioral clustering and threshold sweep.

This example reproduces the experimental pipeline from the companion paper
on a small subset of HumanEval. For the full 164-problem evaluation, set
`MAX_PROBLEMS = 164` (expect 6-12 hours on CPU).

The example demonstrates:
  - Wrapping Ollama-served Qwen2.5-Coder with HybridLayer
  - Behavioral clustering on code candidates via probe execution
  - Offline threshold sweep on a single set of cached generations

Run:
    python examples/02_humaneval.py
    MAX_PROBLEMS=10 python examples/02_humaneval.py     # quick test
"""
import gzip
import json
import os
import sys
import urllib.request

from snc_core import HybridLayer, Decision, behavioral_governance
from snc_core.adapters import OllamaBackend
from snc_core.clustering import (
    extract_probe_args_from_tests,
    make_python_probe_runner,
)
from snc_core.parsing import extract_code

HERE = os.path.dirname(os.path.abspath(__file__))
HE_PATH = os.path.join(HERE, "humaneval.jsonl")
HE_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def ensure_humaneval():
    if os.path.exists(HE_PATH):
        return
    print("Downloading HumanEval...")
    with urllib.request.urlopen(HE_URL, timeout=60) as r:
        gz = r.read()
    try:
        decompressed = gzip.decompress(gz)
    except (OSError, gzip.BadGzipFile):
        decompressed = gz
    with open(HE_PATH, "wb") as f:
        f.write(decompressed)


def evaluate_problem(hybrid: HybridLayer, problem: dict) -> dict:
    """Generate K candidates with a behavior_extractor that probes the test inputs."""
    prompt = problem["prompt"] + "\n# Implement the function. Python code only."

    # Extract probe inputs from the test field
    probe_args = extract_probe_args_from_tests(
        problem["test"], entry_point=problem["entry_point"], max_probes=4
    )

    # Behavior extractor: extract code from candidate response, run probe
    probe_runner = make_python_probe_runner(
        entry_point=problem["entry_point"],
        probe_args=probe_args,
        timeout=5.0,
        prompt_prefix=problem["prompt"],
    )

    def extractor(answer: str):
        code = extract_code(answer)
        return probe_runner(code)

    hybrid.behavior_extractor = extractor
    return hybrid.query(prompt)


def main():
    ensure_humaneval()

    problems = []
    with open(HE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    n = int(os.environ.get("MAX_PROBLEMS", 5))
    problems = problems[:n]
    print(f"Processing {len(problems)} HumanEval problems")

    backend = OllamaBackend(model="qwen2.5-coder:7b")
    hybrid = HybridLayer(backend, k=5, threshold=0.5, temperature=0.8)

    admits = 0
    abstains = 0
    for p in problems:
        result = evaluate_problem(hybrid, p)
        if result.action == Decision.ADMIT:
            admits += 1
            print(f"  {p['task_id']}: ADMIT (trust={result.decision.trust:.3f}, "
                  f"clusters={result.decision.n_clusters})")
        else:
            abstains += 1
            print(f"  {p['task_id']}: ABSTAIN (trust={result.decision.trust:.3f}, "
                  f"clusters={result.decision.n_clusters})")

    print(f"\nResults: admit={admits}/{len(problems)}, "
          f"abstain={abstains}/{len(problems)}")


if __name__ == "__main__":
    main()
