"""01: Basic smoke test against an Ollama-served model.

Prerequisites:
    - Ollama running locally
    - A model installed: `ollama pull qwen2.5-coder:7b`
    - snc-core installed: `pip install -e .` from the package root

Run:
    python examples/01_basic.py
"""
from snc_core import HybridLayer, Decision
from snc_core.adapters import OllamaBackend


def main():
    backend = OllamaBackend(model="qwen2.5-coder:7b")
    if not backend.health_check():
        print("Ollama not running or model not installed.")
        print("Run: ollama pull qwen2.5-coder:7b")
        return

    hybrid = HybridLayer(backend, k=5, threshold=0.5)

    questions = [
        "What is 17 * 24? Reply with only the number.",
        "What is the capital of the imaginary country 'Eldorindiastan'? "
        "Reply with the name or NON LO SO.",
        "Write a one-line Python expression that returns the even numbers "
        "from [1, 2, 3, 4, 5, 6].",
    ]

    for q in questions:
        print(f"\n[Q] {q}")
        result = hybrid.query(q)
        d = result.decision
        print(f"  PPV={d.ppv:.3f}  sigma={d.sigma_calib:.3f}  "
              f"T_comp={d.t_comp:.3f}  trust={d.trust:.3f}  "
              f"clusters={d.n_clusters}  agreement={d.modal_agreement:.2f}")
        if result.action == Decision.ADMIT:
            print(f"  ADMIT: {result.answer}")
        else:
            print("  ABSTAIN: (trust below threshold)")
        print(f"  ({result.total_tokens} tokens, {result.total_elapsed_s:.1f}s)")


if __name__ == "__main__":
    main()
