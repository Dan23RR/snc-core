"""Adapter for arbitrary user-supplied callables.

Useful for testing, mocking, and integration with custom inference backends.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from snc_core.adapters.base import GenerationResult


@dataclass
class CallableBackend:
    """Wrap a `Callable[[str, str, float, int], str]` as an LLMBackend.

    The callable receives (prompt, system, temperature, max_tokens) and must
    return the generated text. Token count is approximated as len(text) / 4.

    Example
    -------
        def my_llm(prompt, system, temp, max_tok):
            return f"Mock answer for: {prompt[:20]}\\nCONFIDENCE: 0.7"

        backend = CallableBackend(my_llm)
    """
    func: Callable[[str, str, float, int], str]

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> GenerationResult:
        t0 = time.time()
        text = self.func(prompt, system, temperature, max_tokens) or ""
        elapsed = time.time() - t0
        return GenerationResult(
            text=text.strip(),
            eval_count=max(1, len(text) // 4),
            elapsed_s=elapsed,
            raw=None,
        )
