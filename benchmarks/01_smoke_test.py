"""
Smoke test: verifica che Ollama + qwen2.5-coder:7b risponde correttamente.

Uso:
    1. Assicurati che Ollama sia avviato (Ollama appli o `ollama serve`)
    2. Verifica modello: `ollama list` deve mostrare qwen2.5-coder:7b
    3. Lancia: `python 01_smoke_test.py`

Output atteso: 3 risposte coerenti + tempo per token.
Se fallisce: stampa diagnostico chiaro.
"""
import json
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"


def call_ollama(prompt: str, options: dict = None) -> dict:
    """Single call to Ollama generate API."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": options or {"temperature": 0.2, "num_predict": 200},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            elapsed = time.time() - t0
            return {"ok": True, "response": json.loads(body), "elapsed": elapsed}
    except urllib.error.URLError as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - t0}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}


PROMPTS = [
    # 1. Sanity: deve sapere il proprio nome
    "Rispondi in una frase. Chi sei?",
    # 2. Coding: task piccolo
    "Scrivi una funzione Python `is_prime(n)` che ritorni True se n è primo. Solo codice.",
    # 3. Reasoning: numero semplice
    "Quanti '7' ci sono in '17, 27, 77, 717'? Solo il numero finale.",
]


def main():
    print("=" * 70)
    print(f"Smoke test Ollama @ {OLLAMA_URL}")
    print(f"Modello: {MODEL}")
    print("=" * 70)
    print()

    # 1. Quick health check
    print("[Health] verifica /api/tags...")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as r:
            tags = json.loads(r.read().decode("utf-8"))
            models = [m["name"] for m in tags.get("models", [])]
            print(f"  Ollama UP. Modelli installati: {models}")
            if not any(MODEL in m for m in models):
                print(f"  WARNING: {MODEL} non trovato. Esegui: ollama pull {MODEL}")
                return
    except Exception as e:
        print(f"  FAIL: Ollama non risponde. Avvia Ollama prima di continuare.")
        print(f"  Errore: {e}")
        return
    print()

    # 2. Run prompts
    total_tokens = 0
    total_elapsed = 0.0
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[Prompt {i}] {prompt}")
        result = call_ollama(prompt)
        if not result["ok"]:
            print(f"  FAIL: {result['error']}")
            continue
        r = result["response"]
        text = r.get("response", "").strip()
        eval_count = r.get("eval_count", 0)
        eval_dur_ns = r.get("eval_duration", 1)
        tok_per_sec = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0
        print(f"  Risposta ({eval_count} tok, {tok_per_sec:.1f} tok/s, "
              f"{result['elapsed']:.1f}s):")
        # Stampa max 200 char della risposta
        snippet = text if len(text) <= 300 else text[:300] + "..."
        for line in snippet.split("\n"):
            print(f"    {line}")
        total_tokens += eval_count
        total_elapsed += result["elapsed"]
        print()

    print("=" * 70)
    print(f"Totale: {total_tokens} tokens in {total_elapsed:.1f}s "
          f"(throughput effettivo: {total_tokens/max(total_elapsed,0.1):.1f} tok/s)")
    print("=" * 70)
    print()
    print("Se hai 3 risposte coerenti: Step 1 PASS. Procedi con 02_snc_qwen.py")
    print("Se le risposte sono garbled: verifica il modello con `ollama run qwen2.5-coder:7b`")


if __name__ == "__main__":
    main()
