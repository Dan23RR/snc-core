"""OpenAI-compatible backend (also works with vLLM, LMStudio, etc.)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from snc_core.adapters.base import GenerationResult


@dataclass
class OpenAIBackend:
    """Backend for OpenAI-compatible chat-completions API.

    Works with the official OpenAI API and any provider that exposes the
    `/v1/chat/completions` endpoint (vLLM, LMStudio, OpenRouter, ...).

    Parameters
    ----------
    model : str
        Model name (e.g. "gpt-4o-mini", "claude-3-5-sonnet").
    api_key : str, optional
        API key. If None, the underlying client picks it up from environment.
    base_url : str, optional
        Override the API endpoint (for vLLM, LMStudio, etc.).
    """
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires 'openai>=1.0'. "
                "Install with: pip install snc-core[openai]"
            ) from e
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        resp = self._client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        eval_count = getattr(usage, "completion_tokens", 0) if usage else 0
        return GenerationResult(
            text=text.strip(),
            eval_count=eval_count,
            elapsed_s=elapsed,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
