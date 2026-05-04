"""Trust thermodynamics governor.

Core formula (closed-form):
    T = PPV * exp(-sigma_calib * T_comp)

Backward-compatibility limits:
- sigma_calib = 0  (perfect inter-sample agreement)  -> T = PPV
- T_comp = 0      (no meta-uncertainty discount)     -> T = PPV
- sigma_calib = 1, T_comp large                       -> T -> 0 (max conservative)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence


class Decision(str, Enum):
    """Output of the governor."""
    ADMIT = "ADMIT"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class GovernanceDecision:
    """Result of applying the governor over a population of candidates.

    Attributes
    ----------
    action : Decision
        ADMIT if trust >= threshold, else ABSTAIN.
    trust : float
        Trust score in [0, 1] (typically) — see trust_thermodynamic().
    ppv : float
        Mean self-reported confidence over the candidate population.
    sigma_calib : float
        Normalized Shannon entropy over behavioral clusters.
    t_comp : float
        Computational temperature used for the trust score.
    n_clusters : int
        Number of distinct behavioral clusters in the population.
    modal_answer : Optional[Any]
        Answer of the modal cluster's representative; None if ABSTAIN with no fallback.
    modal_agreement : float
        Fraction of candidates in the modal cluster.
    """
    action: Decision
    trust: float
    ppv: float
    sigma_calib: float
    t_comp: float
    n_clusters: int
    modal_answer: Optional[Any]
    modal_agreement: float


def trust_thermodynamic(ppv: float, sigma_calib: float,
                        t_comp: Optional[float] = None) -> float:
    """Closed-form trust score.

    T = PPV * exp(-sigma_calib * T_comp)

    Parameters
    ----------
    ppv : float
        Mean self-reported confidence in [0, 1].
    sigma_calib : float
        Normalized entropy of cluster distribution in [0, 1].
    t_comp : float, optional
        Computational temperature. If None, defaults to the adaptive form
        T_comp = 0.5 + (1 - PPV), which amplifies the discount when self-confidence
        is low.

    Returns
    -------
    float
        Trust score in [0, 1] (under standard inputs).
    """
    if not (0.0 <= ppv <= 1.0):
        raise ValueError(f"ppv must be in [0, 1], got {ppv}")
    if not (0.0 <= sigma_calib <= 1.0 + 1e-9):
        raise ValueError(f"sigma_calib must be in [0, 1], got {sigma_calib}")
    if t_comp is None:
        t_comp = 0.5 + (1.0 - ppv)
    return ppv * math.exp(-sigma_calib * t_comp)


def _shannon_entropy_normalized(cluster_sizes: Sequence[int]) -> float:
    """Shannon entropy over a discrete distribution, normalized by log(n_clusters).

    Returns 0.0 when there is exactly one cluster.
    """
    n_clusters = len(cluster_sizes)
    if n_clusters <= 1:
        return 0.0
    total = sum(cluster_sizes)
    if total <= 0:
        return 0.0
    H = -sum((c / total) * math.log(c / total)
             for c in cluster_sizes if c > 0)
    H_max = math.log(n_clusters)
    if H_max <= 0:
        return 0.0
    return max(0.0, H / H_max)


def behavioral_governance(
    candidates: Sequence[dict],
    *,
    threshold: float = 0.5,
    t_comp: Optional[float] = None,
    cluster_key: str = "behavior_key",
    answer_key: str = "answer",
    confidence_key: str = "self_confidence",
) -> GovernanceDecision:
    """Apply the thermodynamic governor over a population of candidates.

    Each candidate is a dict with at minimum:
      - candidate[answer_key]:       the parsed answer (any hashable or non-hashable)
      - candidate[confidence_key]:   self-reported confidence in [0, 1]
      - candidate[cluster_key]:      behavioral cluster key (any hashable)

    The cluster_key is the result of behavioral clustering: candidates whose
    cluster_key compare equal are treated as semantically equivalent.

    Parameters
    ----------
    candidates : Sequence[dict]
        Population of stochastic candidates.
    threshold : float
        Trust threshold for ADMIT vs ABSTAIN. Default 0.5.
    t_comp : float, optional
        Computational temperature. None => adaptive (0.5 + (1 - PPV)).
    cluster_key, answer_key, confidence_key : str
        Dict keys for the three required fields.

    Returns
    -------
    GovernanceDecision
    """
    if not candidates:
        return GovernanceDecision(
            action=Decision.ABSTAIN, trust=0.0, ppv=0.0, sigma_calib=0.0,
            t_comp=0.0, n_clusters=0, modal_answer=None, modal_agreement=0.0,
        )

    n = len(candidates)
    confidences = [c.get(confidence_key, 0.5) for c in candidates]
    ppv = sum(confidences) / n

    cluster_keys = [c.get(cluster_key) for c in candidates]
    counts = Counter(cluster_keys)
    n_clusters = len(counts)
    sigma_calib = _shannon_entropy_normalized(list(counts.values()))

    actual_t_comp = t_comp if t_comp is not None else 0.5 + (1.0 - ppv)
    trust = ppv * math.exp(-sigma_calib * actual_t_comp)

    modal_key, modal_count = counts.most_common(1)[0]
    modal_candidate = next(
        c for c in candidates if c.get(cluster_key) == modal_key
    )
    modal_answer = modal_candidate.get(answer_key)
    modal_agreement = modal_count / n

    action = Decision.ADMIT if trust >= threshold else Decision.ABSTAIN

    return GovernanceDecision(
        action=action, trust=trust, ppv=ppv, sigma_calib=sigma_calib,
        t_comp=actual_t_comp, n_clusters=n_clusters,
        modal_answer=modal_answer if action == Decision.ADMIT else None,
        modal_agreement=modal_agreement,
    )
