"""
SNC-Qwen v2: fix dei 2 bug rivelati dai dati reali.

BUG 1: parser confidence rotto quando "CONFIDENCE: X" è inline
       invece che a inizio riga.
       FIX: regex robusto.

BUG 2: σ_calib over-sensitive a varianti sintattiche
       (e.g. "NON LO SO." vs "NON LO SO" → σ=0.97 falso).
       FIX: normalizzazione semantica multi-modale:
       - extract_number: per matematica
       - extract_code_ast: per codice (AST hash)
       - normalize_text: lowercase + strip punctuation + token sort
       Auto-detect tipo di domanda dal contenuto.

Uso identico a 02:
    python 02b_snc_qwen_v2.py
    python 02b_snc_qwen_v2.py "Quanto fa 17 * 24?"
"""
import ast
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter
from typing import List, Dict, Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"

SYSTEM_PROMPT_L1 = """Sei un assistente con auto-consapevolezza calibrata.
Per ogni risposta:
1. Fornisci la risposta più sintetica possibile.
2. Alla fine, su una nuova riga separata, scrivi ESATTAMENTE:
   CONFIDENCE: <numero da 0.0 a 1.0>
3. Se non sei sicuro, scrivi NON LO SO e CONFIDENCE: 0.0.

Penalty: una risposta sbagliata vale -3, una vuota vale 0.
Quindi NEL DUBBIO ASTIENI."""


# ============================================================
# Ollama call
# ============================================================
def call_ollama(prompt: str, system: str = "", temperature: float = 0.7,
                num_predict: int = 200) -> Dict:
    payload = {
        "model": MODEL, "prompt": prompt, "system": system, "stream": False,
        "options": {"temperature": temperature, "top_p": 0.95,
                    "num_predict": num_predict},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_K_candidates(question: str, K: int = 5,
                          temperature: float = 0.8) -> List[Dict]:
    candidates = []
    for k in range(K):
        try:
            r = call_ollama(question, system=SYSTEM_PROMPT_L1,
                           temperature=temperature)
            text = r.get("response", "").strip()
            confidence = parse_confidence_v2(text)
            answer = parse_answer(text)
            candidates.append({
                "raw": text, "answer": answer,
                "self_confidence": confidence,
                "tokens": r.get("eval_count", 0),
            })
        except Exception as e:
            candidates.append({
                "raw": "", "answer": "<error>", "self_confidence": 0.0,
                "tokens": 0, "error": str(e),
            })
    return candidates


# ============================================================
# FIX 1: parser confidence robusto (regex)
# ============================================================
CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def parse_confidence_v2(text: str) -> float:
    m = CONFIDENCE_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return 0.5  # fallback se assente


def parse_answer(text: str) -> str:
    """Rimuove qualsiasi 'CONFIDENCE: X' (inline o su nuova riga) dalla answer."""
    return CONFIDENCE_RE.sub("", text).strip()


# ============================================================
# FIX 2: normalizzazione semantica multi-modale
# ============================================================
# Auto-detect: se l'answer contiene cifre e poco testo → numerico
# Se contiene parentesi/python keywords → code
# Altrimenti → testo.

NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
CODE_HINT_RE = re.compile(r"[\[\](){}=]|def |lambda |for .* in |list\(|filter\(")


def detect_type(answers: List[str]) -> str:
    """Determina il tipo dominante della risposta dai K candidati."""
    code_hits = sum(1 for a in answers if CODE_HINT_RE.search(a))
    num_only = sum(1 for a in answers if a.strip()
                   and NUMBER_RE.fullmatch(a.strip().replace(",", ".")))
    if code_hits >= len(answers) // 2:
        return "code"
    if num_only >= len(answers) // 2:
        return "number"
    return "text"


def normalize_number(s: str) -> Optional[str]:
    nums = NUMBER_RE.findall(s)
    if nums:
        # Prendi il primo numero rilevante, normalize formato
        try:
            v = float(nums[0].replace(",", "."))
            return f"{v:.6g}"
        except ValueError:
            return None
    return None


def normalize_code(s: str) -> str:
    """Strip code fences, parse AST, ritorna hash dell'AST normalizzato.
    Se non parseable, ritorna stringa raw normalizzata."""
    # Strip backticks / code fences
    cleaned = re.sub(r"```\w*\n?", "", s)
    cleaned = cleaned.replace("```", "").strip()
    # Strip leading 'python' label
    cleaned = re.sub(r"^python\s+", "", cleaned, flags=re.IGNORECASE)
    # Try to parse and dump
    try:
        tree = ast.parse(cleaned, mode="eval")
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        try:
            tree = ast.parse(cleaned)
            return ast.dump(tree, annotate_fields=False)
        except SyntaxError:
            return re.sub(r"\s+", "", cleaned.lower())


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    tokens = sorted(s.split())
    return " ".join(tokens)


def semantic_normalize(answer: str, q_type: str) -> str:
    """Normalizza una answer in base al tipo dominante."""
    if q_type == "number":
        n = normalize_number(answer)
        return n if n is not None else normalize_text(answer)
    if q_type == "code":
        return normalize_code(answer)
    return normalize_text(answer)


# ============================================================
# Governor (con normalizzazione semantica)
# ============================================================
def snc_governance_v2(candidates: List[Dict], threshold: float = 0.6) -> Dict:
    if not candidates:
        return {"action": "ABSTAIN", "trust": 0.0, "reason": "no candidates"}
    valid = [c for c in candidates if c["answer"] != "<error>"]
    if not valid:
        return {"action": "ABSTAIN", "trust": 0.0, "reason": "all errors"}

    answers = [c["answer"] for c in valid]
    q_type = detect_type(answers)

    # Semantic normalization
    norm = [semantic_normalize(a, q_type) for a in answers]
    counts = Counter(norm)
    n = len(valid)

    # PPV: self-confidence aggregato
    ppv = sum(c["self_confidence"] for c in valid) / n

    # σ_calib: entropia normalizzata sui CLUSTER SEMANTICI
    # FIX: agreement perfetto (1 cluster) → σ=0; evita H<0 da +1e-9 nel log
    if len(counts) <= 1:
        sigma_calib = 0.0
    else:
        H = -sum((cnt/n) * math.log(cnt/n) for cnt in counts.values() if cnt > 0)
        H_max = math.log(len(counts))
        sigma_calib = max(0.0, H / max(H_max, 1e-9))

    T_comp = 0.5 + 1.0 * (1.0 - ppv)
    trust = ppv * math.exp(-sigma_calib * T_comp)

    modal_norm, modal_count = counts.most_common(1)[0]
    modal_answer = next(c["answer"] for c in valid
                       if semantic_normalize(c["answer"], q_type) == modal_norm)
    action = "ADMIT" if trust >= threshold else "ABSTAIN"

    return {
        "action": action, "trust": trust, "ppv": ppv,
        "sigma_calib": sigma_calib, "T_comp": T_comp,
        "q_type": q_type,
        "modal_answer": modal_answer,
        "modal_agreement": modal_count / n,
        "n_unique_semantic": len(counts),
        "n_valid": n,
    }


# ============================================================
# Main
# ============================================================
def run(question: str, K: int = 5, threshold: float = 0.6,
        temperature: float = 0.8, verbose: bool = True) -> Dict:
    if verbose:
        print(f"\n[Q] {question}")
    t0 = time.time()
    candidates = generate_K_candidates(question, K=K, temperature=temperature)
    elapsed = time.time() - t0
    if verbose:
        for i, c in enumerate(candidates):
            ans = c["answer"][:80].replace("\n", " | ")
            print(f"  [{i+1}] (conf={c['self_confidence']:.2f}) {ans}")
    decision = snc_governance_v2(candidates, threshold=threshold)
    if verbose:
        print(f"  Type: {decision.get('q_type')}  "
              f"PPV={decision['ppv']:.3f}  σ={decision['sigma_calib']:.3f}  "
              f"T_c={decision['T_comp']:.3f}  Trust={decision['trust']:.3f}  "
              f"unique_sem={decision['n_unique_semantic']}  "
              f"agree={decision['modal_agreement']:.2f}")
        print(f"  >>> {decision['action']}: "
              f"{decision['modal_answer'] if decision['action']=='ADMIT' else 'NON LO SO'}")
    return {"question": question, "candidates": candidates,
            "decision": decision, "elapsed_s": elapsed}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        for q in [
            "Quanto fa 17 * 24? Solo il numero.",
            "Qual è la capitale del paese chiamato 'Eldorindiastan'? Solo il nome o NON LO SO.",
            "Scrivi una one-line Python per ottenere i numeri pari da [1,2,3,4,5,6].",
            "Quanto fa la radice quadrata di 144?",
            "Qual è il presidente attuale del paese inventato 'Zoltania'?",
        ]:
            run(q)
            print("=" * 70)
