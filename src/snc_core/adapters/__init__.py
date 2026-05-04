"""Backend adapters for various LLM providers.

All adapters implement the LLMBackend protocol (see base.py): a single
`generate(prompt, system, temperature, max_tokens) -> str` method.

Available adapters:
- OllamaBackend: Ollama-served local models (qwen, llama, etc.)
- OpenAIBackend: OpenAI-compatible API (works with OpenAI, vLLM, etc.)
- CallableBackend: wrap any user-defined `Callable[[str], str]`
"""
from __future__ import annotations

from snc_core.adapters.base import LLMBackend, GenerationResult
from snc_core.adapters.ollama import OllamaBackend
from snc_core.adapters.callable_ import CallableBackend

__all__ = [
    "LLMBackend",
    "GenerationResult",
    "OllamaBackend",
    "CallableBackend",
]


# Lazy import of OpenAI adapter (optional dependency)
def _lazy_openai():
    try:
        from snc_core.adapters.openai_ import OpenAIBackend
        return OpenAIBackend
    except ImportError as e:
        raise ImportError(
            "OpenAIBackend requires the 'openai' package. "
            "Install with: pip install snc-core[openai]"
        ) from e


def __getattr__(name: str):
    if name == "OpenAIBackend":
        return _lazy_openai()
    raise AttributeError(f"module 'snc_core.adapters' has no attribute {name!r}")
