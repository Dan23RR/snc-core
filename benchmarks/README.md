# SNC-Qwen Runtime — pipeline operativa

Setup per testare empiricamente l'architettura **Hybrid Generator × Governor** (SNC-50)
su un LLM reale (Qwen2.5-Coder 7B via Ollama).

## Prerequisiti

1. Ollama installato e in esecuzione: https://ollama.ai
2. Modello scaricato: `ollama pull qwen2.5-coder:7b` (già fatto)
3. Python 3.9+ (no dipendenze esterne — solo stdlib)

## Sequenza di esecuzione

### Step 1 — Smoke test (5 min)
```
python 01_smoke_test.py
```
Verifica che Ollama risponde + Qwen genera testo coerente. **Gate**: 3 risposte coerenti, throughput tokens/sec ragionevole (atteso 20-80 tok/s su CPU mid-tier, 100-300 su GPU).

### Step 2 — SNC governance wrapper (30 min)
```
python 02_snc_qwen.py
# oppure con domanda custom:
python 02_snc_qwen.py "Quanto fa 17 * 24?"
```
Implementa **Hybrid SNC-50**:
- **Generator**: Qwen genera K=5 candidati a T=0.8 (alta diversità)
- **Governor**: SNC misura trust via thermodynamics
  - PPV = self-confidence media tra K
  - σ_calib = entropia di disagreement tra K answers
  - T_comp = 0.5 + 1.0·(1−PPV) adattivo
  - Trust = PPV · exp(−σ·T_comp)
- **Decision**: ADMIT se Trust ≥ 0.6, altrimenti ABSTAIN

**Gate atteso**:
- Domanda nota (17·24): 5 candidati concordano → high trust → ADMIT con risposta corretta
- Domanda inventata (Eldorindiastan): candidati disagree → low trust → ABSTAIN
- Domanda media (one-line Python): trust medio, ADMIT se modal coerente

### Step 3 — HumanEval mini (1-2 ore) — DA SCRIVERE DOPO STEP 2

Pipeline: 20 problemi HumanEval, run 3 condizioni:
- A. **Qwen vanilla** (T=0.0, single sample)
- B. **Qwen + Layer 1** (system prompt confidence, T=0.0)
- C. **Qwen + Hybrid SNC** (K=5 + governor)

Metriche: pass@1, abstention rate, hallucination rate (output sintatticamente valido ma incorrect).

### Step 4 — Report numerico

Confronto tabular: Δpass@1 vs vanilla, false-positive abstention rate, costo computazionale (5× tokens vs 1×).

Predizione (verificabile):
- Vanilla baseline: pass@1 ≈ 0.55-0.70 (Qwen2.5-Coder 7B su HumanEval)
- Hybrid SNC: stesso pass@1 sui problemi ammessi + abstention 10-20% sui difficili
- **Net effect**: precision più alta a spese di recall (corretto per regulated industries)

## Struttura file

```
snc_qwen_runtime/
├── README.md           # questo file
├── 01_smoke_test.py    # health check Ollama+Qwen
├── 02_snc_qwen.py      # Hybrid wrapper (Generator+Governor)
└── 03_humaneval.py     # benchmark (da scrivere dopo Step 2)
```

## Troubleshooting

**Errore "Connection refused"**: Ollama non avviato. Apri app Ollama o `ollama serve`.

**Errore "model not found"**: `ollama pull qwen2.5-coder:7b`.

**Risposte garbled / incoerenti**: prova `ollama run qwen2.5-coder:7b "Hello"` per verificare il modello.

**Lentezza**: K=5 candidati su CPU sono lenti. Riduci a K=3 o usa GPU.

## Logica del wrapper SNC (02_snc_qwen.py)

```python
def snc_governance(candidates):
    ppv = mean([c.self_confidence for c in candidates])
    sigma = entropy(distribution_of_normalized_answers)
    T_comp = 0.5 + 1.0 * (1 - ppv)
    trust = ppv * exp(-sigma * T_comp)
    if trust >= 0.6:
        return ADMIT, modal_answer(candidates)
    else:
        return ABSTAIN, "NON LO SO"
```

Backward compatibility con SNC-CORE: questa è la formula di Trust Thermodynamics
(SNC-45) applicata al caso K=5 invece di K substrate. La novità qui è che PPV è
self-reported dal modello (Layer 1), non oracolare.
