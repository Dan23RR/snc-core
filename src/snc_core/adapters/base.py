"""LLMBackend protocol and shared types for backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationResult:
    """Single generation from an LLM backend.

    Attributes
    ----------
    text : str
        The model's response, including any CONFIDENCE: tag.
    eval_count : int
        Number of generated tokens (best-effort estimate per backend).
    elapsed_s : float
        Wall-clock time of the generation, in seconds.
    raw : dict
        The raw response object from the backend, for debugging.
    """
    text: str
    eval_count: int = 0
    elapsed_s: float = 0.0
    raw: Optional[dict] = None


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol that every backend must satisfy."""

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> GenerationResult:
        """Single generation. Must not stream."""
        ...
