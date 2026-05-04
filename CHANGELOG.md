# Changelog

## 0.4.0 — May 2026

The 0.4 release rebuilds the package around the **Behavioral Trust Clustering** architecture validated empirically on Qwen2.5-Coder-7B (companion paper, in preparation).

### Added
- `HybridLayer`: end-to-end Generator × Governor pipeline with K-sample stochastic decoding and behavioral clustering.
- `behavioral_governance`: closed-form trust thermodynamic governor exposed as a standalone function for offline analysis.
- `trust_thermodynamic`: the closed-form trust score, $T = \mathrm{PPV} \cdot \exp(-\sigma_{\text{calib}} \cdot T_{\text{comp}})$, with adaptive computational temperature.
- `cluster_by_behavior`: behavioral (output-equivalence) clustering for code candidates.
- `make_python_probe_runner`: factory for behavioral probe runners that execute candidates in subprocess sandboxes.
- `extract_probe_args_from_tests`: regex-based extraction of probe arguments from assertion-based test specifications.
- `cluster_by_ast`: structural clustering, retained for ablation studies.
- Backend protocol `LLMBackend` with three reference implementations: `OllamaBackend`, `OpenAIBackend`, `CallableBackend`.
- Layer 1 system prompts (`LAYER1_SYSTEM_PROMPT_EN`, `LAYER1_SYSTEM_PROMPT_IT`) with calibrated utility framing.
- Strict input validation on `HybridLayer` and `trust_thermodynamic`.
- 35-test pytest suite covering governance, parsing, clustering, and end-to-end integration.

### Changed
- The trust formula default for $T_{\text{comp}}$ is now adaptive ($0.5 + (1 - \mathrm{PPV})$) rather than fixed at $1.0$. The fixed form remains available via the `t_comp` parameter.
- Confidence parsing tolerates inline `CONFIDENCE: X` markers as well as separate-line markers (regression from 0.3 found during HumanEval validation).
- The Shannon entropy normalization for $\sigma_{\text{calib}}$ now divides by $\log(n_{\text{clusters}})$ rather than $\log(K)$, eliminating a numerical artefact at perfect agreement.

### Removed
- Legacy `Substrate` and `Cascade` classes from 0.1.x; their functionality is subsumed by `HybridLayer` and the standalone `behavioral_governance`.
- The `@verified` and `@with_abstention` decorators; replaced by direct `HybridLayer.query` invocation, which is more flexible and easier to instrument.

### Fixed
- `sigma_calib` no longer becomes negative under perfect agreement due to the $\epsilon = 10^{-9}$ inside the logarithm of the 0.3 implementation.
- The cascade did not propagate the modal answer correctly when the modal cluster was singleton; this case is now handled.

### Migration from 0.1.x
The 0.1.x API is incompatible with 0.4. The migration is straightforward:

```python
# 0.1.x
from snc_core import Cascade
cascade = Cascade(substrates=[s1, s2, s3], threshold=0.5)
result = cascade.evaluate(prompt)

# 0.4
from snc_core import HybridLayer
from snc_core.adapters import OllamaBackend
hybrid = HybridLayer(OllamaBackend(model="..."), k=5, threshold=0.5)
result = hybrid.query(prompt)
```

Concretely, the 0.4 single-backend / K-sample model replaces the 0.1.x multi-substrate / single-call model. The two are not isomorphic: 0.4 is the architecture validated in the companion paper, and we recommend it for new projects.

## 0.1.0 — March 2026
Initial public release.
