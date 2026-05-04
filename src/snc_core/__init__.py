"""snc-core: Behavioral Trust Clustering for production LLMs.

Public API:

    from snc_core import HybridLayer, Decision
    from snc_core.adapters import OllamaBackend

    llm = OllamaBackend(model="qwen2.5-coder:7b")
    hybrid = HybridLayer(llm, k=5, threshold=0.5)

    result = hybrid.query("Quanto fa 17 * 24?")
    if result.action == Decision.ADMIT:
        print(result.answer)
    else:
        print("NON LO SO")

For governance over already-generated candidates (offline analysis):

    from snc_core import behavioral_governance

    decision = behavioral_governance(candidates, threshold=0.5)

For probing user-defined behavioral equivalence:

    from snc_core.clustering import cluster_by_behavior
"""
from __future__ import annotations

from snc_core.hybrid import HybridLayer, HybridResult
from snc_core.governance import (
    behavioral_governance,
    trust_thermodynamic,
    GovernanceDecision,
    Decision,
)
from snc_core.clustering import cluster_by_behavior, cluster_by_ast
from snc_core.parsing import parse_confidence, parse_answer
from snc_core.prompts import LAYER1_SYSTEM_PROMPT_EN, LAYER1_SYSTEM_PROMPT_IT

__version__ = "0.4.0"

__all__ = [
    "HybridLayer",
    "HybridResult",
    "behavioral_governance",
    "trust_thermodynamic",
    "GovernanceDecision",
    "Decision",
    "cluster_by_behavior",
    "cluster_by_ast",
    "parse_confidence",
    "parse_answer",
    "LAYER1_SYSTEM_PROMPT_EN",
    "LAYER1_SYSTEM_PROMPT_IT",
    "__version__",
]
