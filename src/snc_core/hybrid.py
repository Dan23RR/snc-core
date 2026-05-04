"""HybridLayer: end-to-end Generator x Governor pipeline.

This is the main public API of snc-core. Wrap any LLMBackend with HybridLayer
and call `query(prompt)` to get an admit/abstain decision with calibrated trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from snc_core.adapters.base import LLMBackend, GenerationResult
from snc_core.governance import (
    behavioral_governance,
    GovernanceDecision,
    Decision,
)
from snc_core.parsing import parse_confidence, parse_answer
from snc_core.prompts import LAYER1_SYSTEM_PROMPT_EN


@dataclass(frozen=True)
class HybridResult:
    """Output of HybridLayer.query().

    Attributes
    ----------
    action : Decision
        ADMIT or ABSTAIN.
    answer : Optional[str]
        Modal cluster representative if ADMIT, else None.
    decision : GovernanceDecision
        Full decision object including trust score, ppv, sigma, ...
    candidates : List[dict]
        The K stochastic candidates with their parsed fields.
    total_tokens : int
        Sum of tokens across all K calls.
    total_elapsed_s : float
        Wall-clock time for the K calls.
    """
    action: Decision
    answer: Optional[str]
    decision: GovernanceDecision
    candidates: List[dict] = field(default_factory=list)
    total_tokens: int = 0
    total_elapsed_s: float = 0.0


@dataclass
class HybridLayer:
    """Behavioral Trust Clustering wrapper around any LLMBackend.

    Parameters
    ----------
    backend : LLMBackend
        Any object satisfying the LLMBackend protocol.
    k : int
        Number of stochastic candidates to sample (default 5).
    threshold : float
        Trust threshold for ADMIT vs ABSTAIN (default 0.5).
    temperature : float
        Sampling temperature for the K candidates (default 0.8).
    max_tokens : int
        Max tokens per generation (default 400).
    system_prompt : str
        Layer 1 system prompt. Default is the English calibration prompt.
    behavior_extractor : Callable[[str], tuple], optional
        Function that maps a parsed answer to a behavioral cluster key.
        If None, candidates are clustered by raw answer string equality
        (suitable for short numeric/factual responses; for code, supply a
        probe runner from snc_core.clustering.make_python_probe_runner).
    t_comp : float, optional
        Computational temperature override. None => adaptive (recommended).

    Example
    -------
        from snc_core import HybridLayer
        from snc_core.adapters import OllamaBackend

        backend = OllamaBackend(model="qwen2.5-coder:7b")
        hybrid = HybridLayer(backend, k=5, threshold=0.5)

        result = hybrid.query("What is 17 * 24?")
        if result.action == Decision.ADMIT:
            print(result.answer)
        else:
            print("I do not know.")
    """
    backend: LLMBackend
    k: int = 5
    threshold: float = 0.5
    temperature: float = 0.8
    max_tokens: int = 400
    system_prompt: str = LAYER1_SYSTEM_PROMPT_EN
    behavior_extractor: Optional[Callable[[str], Tuple]] = None
    t_comp: Optional[float] = None

    def __post_init__(self):
        if not (1 <= self.k <= 64):
            raise ValueError(f"k must be in [1, 64], got {self.k}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError(f"temperature must be in (0, 2], got {self.temperature}")

    def _generate_candidate(self, prompt: str) -> dict:
        result: GenerationResult = self.backend.generate(
            prompt,
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = result.text
        confidence = parse_confidence(text)
        answer = parse_answer(text)
        return {
            "raw": text,
            "answer": answer,
            "self_confidence": confidence,
            "tokens": result.eval_count,
            "elapsed": result.elapsed_s,
        }

    def query(self, prompt: str) -> HybridResult:
        """Run the K-sample governor pipeline on a single prompt.

        Returns a HybridResult with the admit/abstain decision and all
        intermediate candidates for inspection.
        """
        candidates: List[dict] = []
        total_tokens = 0
        total_elapsed = 0.0
        for _ in range(self.k):
            try:
                c = self._generate_candidate(prompt)
            except Exception as e:
                c = {
                    "raw": "", "answer": "<error>",
                    "self_confidence": 0.0,
                    "tokens": 0, "elapsed": 0.0,
                    "error": str(e),
                }
            candidates.append(c)
            total_tokens += c.get("tokens", 0)
            total_elapsed += c.get("elapsed", 0.0)

        # Behavioral clustering
        for c in candidates:
            if self.behavior_extractor is None:
                c["behavior_key"] = c["answer"]
            else:
                try:
                    c["behavior_key"] = tuple(
                        self.behavior_extractor(c["answer"])
                    )
                except Exception:
                    c["behavior_key"] = ("__EXTRACTOR_FAILED__",)

        # Governor
        decision = behavioral_governance(
            candidates,
            threshold=self.threshold,
            t_comp=self.t_comp,
            cluster_key="behavior_key",
            answer_key="answer",
            confidence_key="self_confidence",
        )

        return HybridResult(
            action=decision.action,
            answer=decision.modal_answer if decision.action == Decision.ADMIT else None,
            decision=decision,
            candidates=candidates,
            total_tokens=total_tokens,
            total_elapsed_s=total_elapsed,
        )

    def query_batch(self, prompts: Sequence[str]) -> List[HybridResult]:
        """Convenience: query a list of prompts sequentially.

        For parallel/async dispatch, build your own loop over `query`.
        """
        return [self.query(p) for p in prompts]
