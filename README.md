# snc-core

[![Tests](https://github.com/dculotta/snc-core/actions/workflows/test.yml/badge.svg)](https://github.com/dculotta/snc-core/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)

**Behavioral Trust Clustering** — a thermodynamic governance layer for production language models.

`snc-core` wraps any decoder-only LLM with an inference-time governance layer that **reduces the hallucination rate by 52%** on the official HumanEval benchmark with Qwen2.5-Coder-7B (16.5% → 7.8%, $z = 2.12$, $p < 0.05$). The method is model-agnostic, retraining-free, and exposes a single decision threshold $\theta$ that traces an interpretable Pareto frontier between coverage and precision.

The companion paper is included at [`paper/snc-trust-layer-paper.pdf`](paper/snc-trust-layer-paper.pdf) and archived on Zenodo (DOI above).

---

## Headline result

On the full HumanEval benchmark ($n = 164$) with Qwen2.5-Coder-7B:

| Configuration | pass@1 | hallucinations | net precision | $z$-stat (halluc) |
|---|---|---|---|---|
| Vanilla baseline | 137/164 (83.5%) | 27/164 (16.5%) | 83.54% | — |
| Hybrid @ $\theta = 0.50$ | 136/164 | 19/164 (12.3%) | 87.74% | $+1.07$ |
| Hybrid @ $\theta = 0.55$ | 132/164 | 15/164 (9.1%) | 89.80% | $+1.61$ |
| **Hybrid @ $\theta = 0.65$** | **106/164** | **9/164 (7.8%)** | **92.17%** | **$+2.12$ SIG** |

At the conservative threshold the hallucination rate is reduced by **52% relative**, statistically significant at the 5% level. Five vanilla failures are recovered (HE/91, /102, /123, /144, /160). Nine residual failures correspond to *adversarial mode collapse* — see paper Section 4.5.

---

## Why

LLMs trained on next-token prediction confidently produce incorrect outputs. In regulated industries (banking, healthcare, legal compliance) the binding constraint is not raw accuracy but **known precision conditional on emission**. A model that abstains on $10\%$ of queries and is correct on the other $90\%$ is qualitatively different from one that is silently wrong on $10\%$, even at the same headline accuracy. `snc-core` converts a fraction of the second class into the first.

---

## Install

```bash
pip install snc-core              # core (stdlib only, includes Ollama backend)
pip install snc-core[openai]      # + OpenAI-compatible backend
pip install snc-core[test]        # + pytest for development
```

Python 3.9+. No mandatory dependencies beyond the standard library.

---

## Quick start

```python
from snc_core import HybridLayer, Decision
from snc_core.adapters import OllamaBackend

backend = OllamaBackend(model="qwen2.5-coder:7b")
hybrid = HybridLayer(backend, k=5, threshold=0.65, temperature=0.8)

result = hybrid.query("What is 17 * 24?")
if result.action == Decision.ADMIT:
    print(f"Answer: {result.answer}")
    print(f"Trust: {result.decision.trust:.3f}")
else:
    print("I do not know.")
```

The same interface works for any backend that satisfies the `LLMBackend` protocol:

```python
from snc_core.adapters import OpenAIBackend

backend = OpenAIBackend(model="gpt-4o-mini", api_key="sk-...")
hybrid = HybridLayer(backend, k=5, threshold=0.65)
```

---

## How it works

The layer composes three signals (full derivation in the [paper](paper/snc-trust-layer-paper.pdf)).

**Layer 1 — confidence elicitation.** A system prompt instructs the model to emit a self-confidence in the canonical form `CONFIDENCE: <0..1>` under an asymmetric utility function (correct = +1, wrong = −3, empty = 0).

**Behavioral clustering.** $K = 5$ candidates are sampled at temperature $0.8$. They are clustered by *output equivalence* on probe inputs extracted automatically from the test specification. Two implementations of the same algorithm in different syntactic forms collapse into the same cluster.

**Trust thermodynamics.** The trust score is

$$T = \mathrm{PPV} \cdot \exp(-\sigma_{\mathrm{calib}} \cdot T_{\mathrm{comp}})$$

with computational temperature $T_{\mathrm{comp}}$ adaptive to PPV by default. The score reduces to PPV under perfect inter-sample agreement and discounts toward zero as candidates diverge. The decision is a comparison against the user-supplied threshold $\theta$.

The closed-form score admits an exact thermodynamic phase diagram with universal order parameter $X = T_{\mathrm{comp}} \cdot \sigma_{\mathrm{calib}}$ and critical line $X_c = \ln(\mathrm{PPV}/\theta)$ — see paper Section 3.6.

---

## Public API

### `HybridLayer`
The primary class. Wraps any `LLMBackend` and exposes a `query(prompt) -> HybridResult` method.

```python
HybridLayer(
    backend: LLMBackend,
    k: int = 5,
    threshold: float = 0.5,
    temperature: float = 0.8,
    max_tokens: int = 400,
    system_prompt: str = LAYER1_SYSTEM_PROMPT_EN,
    behavior_extractor: Optional[Callable[[str], Tuple]] = None,
    t_comp: Optional[float] = None,
)
```

### `behavioral_governance`
Apply the governor offline to a population of pre-computed candidates. Useful for replay analysis, threshold sweeps over cached generations, and unit testing.

### `trust_thermodynamic`
The closed-form trust score, exposed for direct inspection or composition.

### Backends
- `OllamaBackend(model, base_url, request_timeout)` — Ollama-served local models
- `OpenAIBackend(model, api_key, base_url)` — OpenAI-compatible APIs (also vLLM, LMStudio, OpenRouter)
- `CallableBackend(func)` — wrap any user-defined callable

To add a backend, implement the `LLMBackend` protocol from `snc_core.adapters.base`.

---

## Tuning the threshold

The threshold $\theta$ is the only operational hyperparameter. Three regimes have been characterized empirically on HumanEval:

| Regime | $\theta$ | Use case | Result on HumanEval |
|--------|----------|----------|----------------------|
| Aggressive | 0.50 | Internal tooling, downstream review cheap | 88% coverage, 12.3% halluc, 87.74% precision |
| Balanced | 0.55 | Customer-facing, false positives visible | 90% coverage, 9.1% halluc, 89.80% precision |
| **Conservative** | **0.65** | **Banking, healthcare, legal — high-cost errors** | **70% coverage, 7.8% halluc, 92.17% precision** |

Calibrate against a small held-out set with the operator's empirical cost ratios.

---

## Reproducing the paper results

The full experimental pipeline is included under [`benchmarks/`](benchmarks/):

```bash
# 1. Smoke test
python benchmarks/01_smoke_test.py

# 2. Hybrid wrapper validation on small probe set
python benchmarks/02_snc_qwen.py

# 3. HumanEval full benchmark with threshold sweep
python benchmarks/06_humaneval_full.py
```

All experiments use seed 42. The candidate cache is preserved as JSONL for offline analysis. Expected wall-clock time on a CPU-only consumer workstation: approximately 8–10 hours for the full HumanEval evaluation.

---

## Citation

```bibtex
@article{culotta2026btc,
  title  = {Behavioral Trust Clustering: A Thermodynamic Governance Layer for Production LLMs},
  author = {Culotta, Daniel},
  year   = {2026},
  doi    = {10.5281/zenodo.PLACEHOLDER},
  url    = {https://doi.org/10.5281/zenodo.PLACEHOLDER}
}
```

---

## Limitations

The package halves but does not eliminate hallucinations. The residual failure mode, *adversarial mode collapse*, occurs when a majority of stochastic candidates make the same systematic error. We identify nine such cases in HumanEval (paper Appendix B). Mitigation requires external information — typically a property-based test that exercises the systematic error.

The token cost of the hybrid configuration is approximately $K$ times the vanilla cost, modulo savings from clustering and short candidate emissions. On HumanEval the empirical overhead was $2.27\times$.

The behavioral clustering relies on probe inputs that exercise the relevant equivalence. For tasks under-determined by their test specification, the method degrades to structural clustering.

---

## License

MIT. See [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
