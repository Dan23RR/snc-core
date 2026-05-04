"""Parse confidence and answer from model output.

The Layer 1 system prompt instructs the model to emit a self-confidence in
the canonical form `CONFIDENCE: <number>`. This module extracts it robustly,
whether on a separate line or inline.
"""
from __future__ import annotations

import re

CONFIDENCE_RE = re.compile(
    r"CONFIDENCE\s*[:=]\s*(-?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_confidence(text: str, default: float = 0.5) -> float:
    """Extract confidence in [0, 1]. Returns `default` if absent or unparseable.

    The regex tolerates both separate-line and inline forms:
      "CONFIDENCE: 0.9"          -> 0.9
      "Answer.\\nCONFIDENCE: 0.7" -> 0.7
      "I don't know. CONFIDENCE: 0.0" -> 0.0

    Negative inputs are clamped to 0; values above 1 are clamped to 1.
    """
    if not text:
        return default
    m = CONFIDENCE_RE.search(text)
    if not m:
        return default
    try:
        v = float(m.group(1))
    except (ValueError, TypeError):
        return default
    return max(0.0, min(1.0, v))


def parse_answer(text: str) -> str:
    """Strip any `CONFIDENCE: X` clause from the text.

    The remaining string is the model's primary answer. Code fences are NOT
    stripped here (use `extract_code` if you want only the code block).
    """
    if not text:
        return ""
    return CONFIDENCE_RE.sub("", text).strip()


def extract_code(text: str) -> str:
    """Extract a Python code block from a model response.

    Tries, in order:
      1. ```python ... ``` fenced block
      2. ``` ... ``` unfenced block
      3. text from the first `def ` keyword onwards
      4. raw text
    """
    if not text:
        return ""
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    if idx >= 0:
        return text[idx:].strip()
    return text.strip()
