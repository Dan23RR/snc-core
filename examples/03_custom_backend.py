"""03: Implementing a custom backend.

The LLMBackend protocol requires only one method: `generate(prompt, system,
temperature, max_tokens) -> GenerationResult`. Any object that implements
this signature can be passed to HybridLayer.

This example shows three patterns:
  A. Wrapping a synchronous HTTP client (e.g. for a private inference server)
  B. Wrapping a callable directly via CallableBackend
  C. Implementing the protocol manually as a class
"""
import time
import json
import urllib.request

from snc_core import HybridLayer, Decision
from snc_core.adapters import CallableBackend
from snc_core.adapters.base import GenerationResult


# ============================================================
# A. Custom backend class (e.g. for vLLM or LMStudio behind /v1/completions)
# ============================================================
class CustomHTTPBackend:
    """Backend for a self-hosted /v1/completions style endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def generate(self, prompt, *, system="", temperature=0.7, max_tokens=400):
        full_prompt = (system + "\n\n" + prompt) if system else prompt
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode("utf-8"))
        return GenerationResult(
            text=body["choices"][0]["text"].strip(),
            eval_count=body.get("usage", {}).get("completion_tokens", 0),
            elapsed_s=time.time() - t0,
            raw=body,
        )


# ============================================================
# B. Wrapping a callable
# ============================================================
def my_oracle(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    """A user-defined function that returns text."""
    # Replace with your inference logic
    if "17 * 24" in prompt:
        return "408\nCONFIDENCE: 1.0"
    return "I don't know\nCONFIDENCE: 0.1"


def main():
    # Pattern B: simplest path for testing or for inference logic that
    # already exists as a function.
    backend = CallableBackend(my_oracle)
    hybrid = HybridLayer(backend, k=3, threshold=0.5)
    r = hybrid.query("What is 17 * 24?")
    print(f"Action: {r.action.value}, Answer: {r.answer}")

    # Pattern A: wrapping a self-hosted server. Disabled by default.
    # backend = CustomHTTPBackend(
    #     base_url="http://localhost:8000",
    #     model="my-model",
    # )
    # hybrid = HybridLayer(backend, k=5, threshold=0.5)


if __name__ == "__main__":
    main()
