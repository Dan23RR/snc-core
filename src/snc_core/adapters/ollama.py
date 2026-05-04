"""Ollama backend (stdlib only — uses urllib.request)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from snc_core.adapters.base import GenerationResult


@dataclass
class OllamaBackend:
    """Backend for Ollama-served local models.

    Parameters
    ----------
    model : str
        Model tag as known to Ollama (e.g. "qwen2.5-coder:7b").
    base_url : str
        URL of the Ollama HTTP server. Default "http://localhost:11434".
    request_timeout : float
        Per-request timeout in seconds.
    """
    model: str
    base_url: str = "http://localhost:11434"
    request_timeout: float = 180.0

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> GenerationResult:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama at {self.base_url} not reachable: {e}"
            ) from e
        elapsed = time.time() - t0
        return GenerationResult(
            text=body.get("response", "").strip(),
            eval_count=body.get("eval_count", 0),
            elapsed_s=elapsed,
            raw=body,
        )

    def health_check(self) -> bool:
        """Verify that Ollama is up and the model is available."""
        try:
            with urllib.request.urlopen(
                f"{self.base_url.rstrip('/')}/api/tags", timeout=10
            ) as r:
                tags = json.loads(r.read().decode("utf-8"))
                models = [m["name"] for m in tags.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False
