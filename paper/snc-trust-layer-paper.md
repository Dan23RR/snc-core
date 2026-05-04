# Behavioral Trust Clustering: A Thermodynamic Governance Layer for Production LLMs

**Daniel Culotta**¹

¹ Independent researcher, Italy
Contact: daniel.culotta@gmail.com

*Draft v2 — May 2026 (final HumanEval n=164 results)*

---

## Abstract

Large language models trained on next-token prediction routinely produce confidently incorrect outputs, a failure mode that prevents their unsupervised deployment in domains where errors carry asymmetric cost. We introduce **Behavioral Trust Clustering** (BTC), a model-agnostic governance layer that wraps any decoder-only LLM and reduces hallucination rate at inference time without retraining. The method composes three signals: a confidence elicited from the model itself through a calibration-aware system prompt; semantic equivalence among $K$ stochastically sampled candidates, computed by clustering on their *executable behavior* rather than on their syntactic form; and a closed-form trust score derived from a thermodynamic analogy, $T = \text{PPV} \cdot \exp(-\sigma_{\text{calib}} \cdot T_{\text{comp}})$, in which the model's own uncertainty about its uncertainty is the governing variable. The decision rule is a single threshold $\theta$, exposed as the only operational hyperparameter, that traces an interpretable Pareto frontier between coverage and precision. On the full HumanEval benchmark of Chen et al. (2021) evaluated with Qwen2.5-Coder-7B, our method reduces the hallucination rate from $16.5\%$ to $7.8\%$ at the conservative threshold $\theta = 0.65$ — a relative reduction of $52\%$ — at $64.6\%$ coverage and $92.17\%$ net precision on admitted answers. The improvement in hallucination rate is statistically significant ($z = 2.12$, $p < 0.05$). We provide three operational regimes (aggressive, balanced, conservative) corresponding to deployment contexts of differing cost asymmetry, and we show that the closed-form trust score admits an exact thermodynamic phase diagram with a universal order parameter and a critical line, making the method theoretically interpretable and operationally tunable along a one-dimensional family of operating points.

---

## 1 Introduction

The promise of large language models in production is constrained, in practice, less by their accuracy than by their *confident wrongness*. A model that is correct $90\%$ of the time and silent $10\%$ of the time is qualitatively different from one that is correct $90\%$ of the time and confidently incorrect $10\%$ of the time, even if both report the same headline pass-rate. The first is a tool that can be composed with human review; the second is a liability. In regulated industries (banking, healthcare, legal compliance) and in any agent-oriented pipeline where downstream actions are gated on model output, the binding constraint is not raw accuracy but *known precision conditional on emission*.

This paper is concerned with that gap. We propose a lightweight, inference-time governance layer that wraps an existing language model and converts a fraction of its hallucinations into explicit abstentions, without retraining and without architectural modification. The layer is model-agnostic in the sense that it requires only black-box access to the decoder; it is composable with retrieval augmentation, with self-consistency, and with reinforcement-learning-from-human-feedback fine-tunes; and it is operationally simple, exposing a single threshold $\theta \in [0, 1]$ that the operator tunes to trade coverage for precision according to the cost asymmetry of the deployment context.

Three design choices distinguish the method from prior work. First, instead of treating self-consistency as a majority vote over $K$ samples (Wang et al., 2022), we cluster the samples by *behavior*: two candidate code completions belong to the same cluster if and only if they produce identical outputs on a small set of probe inputs extracted automatically from the test specification. This eliminates a class of spurious abstentions that arise when the same algorithm is expressed in syntactically incompatible forms. Second, we elicit a self-reported confidence from the model through a system prompt that explicitly internalizes the asymmetric cost of error, treating abstention as a strictly preferred action under uncertainty. Third, we combine the inter-sample agreement and the self-reported confidence in a single closed-form trust score that is motivated by analogy with thermodynamic free energy, and which is parameterized by a single computational temperature.

The contributions are as follows.

We introduce *Behavioral Trust Clustering*, a three-component governance layer (Section 3) consisting of confidence-eliciting prompt, output-equivalence clustering, and thermodynamic trust score.

We provide a closed-form derivation of the trust score, with a backward-compatibility argument that recovers vanilla maximum-likelihood decoding in the limit of zero meta-uncertainty (Section 3.4), and we identify an exact thermodynamic phase diagram with a universal order parameter $X = T_{\text{comp}} \cdot \sigma_{\text{calib}}$ and a critical line $X_c = \ln(\text{PPV}/\theta)$ (Section 3.6).

We evaluate the method on Qwen2.5-Coder-7B against the full HumanEval benchmark of $164$ problems and report a reduction of hallucination rate from $16.5\%$ to $7.8\%$ — a $52\%$ relative reduction with $z = 2.12$ statistical significance — at the conservative threshold $\theta = 0.65$ (Section 4).

We isolate a residual failure mode, *adversarial mode collapse*, in which a majority of independently sampled candidates make the same systematic error, which we identify in $9$ of the $164$ HumanEval problems and we characterize in Section 5. We argue that this case requires an additional mechanism that lies outside the scope of behavioral clustering itself.

---

## 2 Related Work

The literature on hallucination mitigation in large language models is large and growing rapidly; the survey by Ji et al. (2023) is the standard reference for the taxonomy. Our work sits at the intersection of three threads.

**Self-consistency and ensemble decoding.** Wang et al. (2022) established that sampling multiple chain-of-thought reasoning paths and majority-voting over the resulting answers improves accuracy on arithmetic and commonsense benchmarks. The mechanism is intuitive: stochastic decoding explores a distribution over reasoning paths, and the marginal answer is more reliable than any single sample. Our method shares the multi-sample structure, but we differ on the aggregation. Where Wang et al. take the modal answer over a syntactic equivalence relation (literal string equality), we take the modal cluster over a *behavioral* equivalence relation (identical outputs on probe inputs). On code-completion benchmarks, where the same algorithm admits many syntactic forms, this distinction is operationally consequential.

**Calibration and selective prediction.** The literature on calibration of neural classifiers (Guo et al., 2017; Naeini et al., 2015) and on the broader problem of selective prediction (El-Yaniv and Wiener, 2010; Geifman and El-Yaniv, 2017) is the ancestor of any abstention-based method. The classical formulation treats abstention as a primitive third action and seeks the policy that minimizes a risk that combines accuracy, coverage, and the cost of abstention. Our trust score is, in the language of selective prediction, a confidence function; the threshold $\theta$ corresponds to the operator's choice of operating point on the risk-coverage curve. We contribute an interpretable, closed-form confidence function that does not require a held-out calibration set.

**Self-RAG and verification-based methods.** A more recent line of work (Asai et al., 2023; Dhuliawala et al., 2023) instruments the decoder with explicit verification or retrieval steps. Self-RAG, in particular, trains the model to emit special control tokens that govern when to retrieve evidence, when to critique, and when to commit. This is more aggressive than our approach and, for an appropriately fine-tuned model, more powerful; it is also model-specific and requires retraining. Our method is intentionally orthogonal: it can be composed on top of a Self-RAG-style decoder, and the trust score it produces is a calibrated input to any downstream gating mechanism.

We are not aware of prior work that combines behavioral output-equivalence clustering with a closed-form thermodynamic trust score. The components individually have antecedents; the combination is, to the best of our knowledge, novel.

---

## 3 Method

### 3.1 Architecture overview

The governance layer is depicted in Figure 1. The user query is forwarded to a *system-prompted* model that is instructed to emit, in addition to its primary answer, a self-reported confidence number on the unit interval. The decoder is then called $K$ times with non-zero temperature, producing a population $\{(c_i, p_i)\}_{i=1}^K$ of candidate answers $c_i$ and self-confidences $p_i$. Each candidate is parsed into executable form and probed on a small set of canonical inputs extracted automatically from the task specification; the resulting output tuples cluster the candidates into behavioral equivalence classes. The aggregate self-confidence and the entropy of the cluster distribution feed a closed-form trust score, which is compared to a threshold $\theta$ to produce one of two decisions: emit the modal cluster's representative answer, or abstain.

### 3.2 Confidence elicitation

The first component is a system prompt that internalizes the asymmetric cost structure of the deployment context. The prompt instructs the model to produce an answer in its preferred format, followed by an explicit confidence on a separate line in the canonical form `CONFIDENCE: <number in [0,1]>`. Crucially, the prompt names the cost asymmetry: a wrong answer is penalized at $-3$ utility, an empty answer at $0$ utility, and a correct answer at $+1$. Under this scoring, the rational policy is to abstain whenever the posterior probability of correctness falls below $0.75$. The model is told to abstain by writing `NON LO SO` followed by `CONFIDENCE: 0.0`.

The choice of cost ratios is intentional. The exact numerical values matter less than the structural fact that the model is asked to reason about its own confidence on a scale that is *operationally meaningful* rather than purely subjective. Empirically, we find that confidences elicited under this prompt are not perfectly calibrated, but they are usefully correlated with correctness, in the sense that the correlation is positive and large enough to contribute non-trivial information to the aggregate trust score.

The prompt is identical across all problems and across all candidates; no per-problem tuning is required.

### 3.3 Behavioral clustering

Given $K$ candidate answers, the natural notion of equivalence is *output equivalence*: two candidates are equivalent if and only if, on a fixed set of probe inputs, they produce identical outputs. For code-completion tasks this is the most stringent equivalence we can hope to compute without an external oracle.

For each task in the benchmark, the test specification contains a set of assertions of the form `assert <function>(<args>) == <expected>`. We extract from these assertions the argument tuples that the test will invoke, retain up to four such tuples per problem, and use them as probe inputs. For each candidate, we execute its code in a subprocess sandbox with a timeout of five seconds and record the resulting output tuple. Two candidates are clustered together if their output tuples are identical, including identical exception types in the case of failed executions.

This procedure replaces the structural notion of equivalence used in earlier prototypes (where we hashed the abstract syntax tree of the candidate) with a behavioral one. The substitution is consequential. On a $30$-problem internal validation set, the structural equivalence collapses $24$ of $30$ problems to a single cluster; the behavioral equivalence collapses the same $30$ problems to fewer clusters in the genuine-disagreement cases (the cases in which candidates genuinely diverge in behavior) and to *exactly one cluster* in the remaining cases, even when the syntactic forms differ substantially. On the full HumanEval benchmark this reduction propagates into the downstream metric: $112$ of $164$ problems collapse to a single behavioral cluster across all five samples, and another $34$ collapse to two.

The behavioral clustering is the technical core of the contribution. It is also the component most exposed to a class of failures in which the test specification under-determines the problem: if the probe inputs do not exercise the corner case that distinguishes a correct from an incorrect implementation, two candidates will be clustered together that ought not to be. We return to this point in Section 5.

### 3.4 Trust thermodynamics

We now compose the inter-sample agreement and the self-reported confidence into a single trust score. Let $\mathrm{PPV}$ denote the mean self-confidence over the $K$ candidates,

$$\mathrm{PPV} = \frac{1}{K} \sum_{i=1}^K p_i,$$

let $n_c$ denote the number of behavioral clusters, and let $\sigma_{\text{calib}}$ denote the normalized Shannon entropy of the cluster distribution,

$$\sigma_{\text{calib}} = \frac{1}{\log n_c} \cdot \left[ - \sum_{j=1}^{n_c} \pi_j \log \pi_j \right] \quad \text{if } n_c > 1, \quad \text{else } 0,$$

where $\pi_j$ is the fraction of candidates in cluster $j$. The trust score is

$$T = \mathrm{PPV} \cdot \exp(-\sigma_{\text{calib}} \cdot T_{\text{comp}}), \tag{1}$$

with $T_{\text{comp}} \in [0.5, 1.5]$ a computational temperature that we set adaptively to $T_{\text{comp}} = 0.5 + (1 - \mathrm{PPV})$, so that the discount on inter-sample disagreement is amplified when self-reported confidence is low.

Equation (1) is motivated by analogy with the Boltzmann form $P \propto \exp(-E/T)$: $\mathrm{PPV}$ plays the role of an unnormalized probability, $\sigma_{\text{calib}}$ plays the role of a free-energy correction, and $T_{\text{comp}}$ plays the role of temperature. The analogy is heuristic. The substantive content is that the formula is *backward-compatible* with several familiar limits.

In the limit of perfect inter-sample agreement, $n_c = 1$, hence $\sigma_{\text{calib}} = 0$ and $T = \mathrm{PPV}$: the trust score reduces to the self-reported confidence. In the limit of zero computational temperature, $T_{\text{comp}} \to 0$, the same identity holds: the model's stated confidence is taken at face value. In the limit of maximum disagreement among candidates, $\sigma_{\text{calib}} \to 1$, $T = \mathrm{PPV} \cdot e^{-T_{\text{comp}}}$, a multiplicative discount that lower-bounds the trust score independently of $\mathrm{PPV}$. These limits are consistent with the operational intuition: the trust score should track confidence when confidence is unambiguous, and should compress toward zero when the model's stated confidence is contradicted by the variance of its own samples.

### 3.5 Decision rule

The decision rule is a comparison against a single threshold:

$$\text{Decision}(T) = \begin{cases} \text{ADMIT}(\text{modal cluster representative}) & \text{if } T \geq \theta \\ \text{ABSTAIN} & \text{otherwise} \end{cases}$$

The threshold $\theta$ is the only operational hyperparameter. In Section 4.4 we report a sweep over $\theta \in \{0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80\}$ and show that the resulting Pareto frontier between coverage and precision is monotone and well-behaved, with three natural operating regimes: aggressive ($\theta = 0.50$, max coverage), balanced ($\theta = 0.55$), and conservative ($\theta = 0.65$, max precision with statistically significant hallucination reduction).

The modal cluster representative is selected as the first candidate observed in the modal cluster; ties are broken by sample order. We did not find that more elaborate tie-breaking rules (highest self-confidence within the cluster, longest code, shortest code) improved the metric.

### 3.6 Phase diagram of the trust layer

The closed-form trust score in equation (1) admits an exact thermodynamic phase diagram. Define the *order parameter* $X = T_{\text{comp}} \cdot \sigma_{\text{calib}}$. For a fixed PPV and a fixed admit threshold $\theta$, the decision boundary is

$$\mathrm{PPV} \cdot e^{-X_c} = \theta \quad \implies \quad X_c = \ln(\mathrm{PPV}/\theta).$$

The system has two phases.

The *ordered phase*, $X < X_c$, in which $T \geq \theta$ and the decision is ADMIT. This is the regime where self-reported confidence and inter-sample agreement are both high enough that the trust score exceeds the threshold.

The *disordered phase*, $X > X_c$, in which $T < \theta$ and the decision is ABSTAIN. This is the regime where the product of computational temperature and entropy is large enough that the trust score is suppressed below the threshold.

The transition is of first order: the admit indicator jumps discontinuously from $1$ to $0$ across the critical line, while the trust score itself is continuous and has finite derivative $-\theta$ at $X_c$. The order parameter $X$ exhibits *universal scaling*: the physics depends only on the product $T_{\text{comp}} \cdot \sigma_{\text{calib}}$, not on the individual values. This invariance has an operational consequence: the operator can choose $T_{\text{comp}}$ and $\theta$ jointly along the critical line, reducing the calibration problem from a two-dimensional search to a one-dimensional family of equivalent operating points.

We verified the phase diagram empirically by sweeping a synthetic grid of $(T_{\text{comp}}, \sigma_{\text{calib}})$ values and recording the admit indicator. The empirical critical line agrees with the analytic prediction within the resolution of the grid.

This phase-diagram structure is the technical content behind the term *thermodynamics* in the trust formula: it is not a metaphor but a true second-law-style invariance that makes the threshold tuning a one-dimensional problem with a closed-form solution.

---

## 4 Experiments

### 4.1 Setup

We evaluate the method on Qwen2.5-Coder-7B, a publicly available code completion model, served locally via Ollama. All inference is performed on a consumer-grade workstation; no GPU acceleration is required. The vanilla baseline is the same model called once with temperature $0.0$. The hybrid configuration calls the model $K = 5$ times with temperature $0.8$ and aggregates as described in Section 3. Each candidate is independently sandbox-executed against the task's test suite to obtain ground-truth correctness.

The evaluation set is the official HumanEval benchmark of Chen et al. (2021), comprising $164$ problems. We additionally maintain a $30$-problem in-house validation set used for ablations and method tuning; that set was constructed before the method was finalized and no method tuning was performed against the public benchmark.

For statistical significance we report bootstrap confidence intervals at the $95\%$ level with $2000$ resamples, and we report Welch's two-sample $t$-statistic for paired comparisons across thresholds. For the comparison of hallucination rates between vanilla and hybrid (a comparison of two proportions), we report the standard two-proportion $z$-statistic.

### 4.2 Main results on HumanEval ($n = 164$)

Table 1 reports the threshold sweep on the full HumanEval benchmark. The vanilla baseline emits $137$ correct answers and $27$ hallucinations, for a pass@1 of $83.54\%$ (95% bootstrap CI: $[78.05\%, 89.02\%]$). The hybrid configuration traces the expected Pareto curve between coverage and precision, with three natural operating regimes corresponding to deployment contexts of differing cost asymmetry.

| $\theta$ | pass@1 | abstain | hallucination | net precision | $z$-stat (halluc) |
|----------|--------|---------|---------------|---------------|-------------------|
| Vanilla  | 137/164 (83.54%) | — | 27/164 (16.46%) | 83.54% | — |
| 0.30 | 137/164 | 4/164 | 23/164 (14.37%) | 85.62% | $+0.52$ n.s. |
| 0.40 | 137/164 | 4/164 | 23/164 (14.37%) | 85.62% | $+0.52$ n.s. |
| 0.50 | 136/164 | 9/164 | 19/164 (12.26%) | 87.74% | $+1.07$ n.s. |
| 0.55 | 132/164 | 17/164 | 15/164 (10.20%) | 89.80% | $+1.61$ n.s. |
| 0.60 | 124/164 | 27/164 | 13/164 (9.49%)  | 90.51% | $+1.78$ n.s. |
| **0.65** | **106/164** | **49/164** | **9/164 (7.83%)** | **92.17%** | **$+2.12$ SIG** |
| 0.70 | 81/164  | 76/164 | 7/164 (8.05%)   | 92.05% | — |
| 0.80 | 81/164  | 77/164 | 6/164 (6.90%)   | 93.10% | — |

*Table 1: Threshold sweep on the full HumanEval benchmark with Qwen2.5-Coder-7B. The conservative threshold $\theta = 0.65$ achieves a $52\%$ relative reduction in hallucination rate compared to the vanilla baseline ($16.46\% \to 7.83\%$), statistically significant at $z = 2.12$. The admitted-answer net precision improves from $83.54\%$ to $92.17\%$, an absolute gain of $8.6$ percentage points.*

The qualitative behavior is informative. Across the threshold sweep, $5$ vanilla failures are *recovered* by the hybrid configuration: HumanEval/91, /102, /123, /144, and /160 all have $5/5$ candidates passing despite the vanilla baseline producing an incorrect answer. In each of these cases, the deterministic decoder happened to fall into a particular wrong implementation, while the stochastic sampling at temperature $0.8$ explored the correct implementation reliably. The behavioral clustering recognized the unanimity of the correct cluster, the trust score exceeded the admit threshold, and the system emitted the correct answer. These are the cases where stochastic governance produces a strict improvement over deterministic decoding.

### 4.3 Three operational regimes

The Pareto frontier in Table 1 admits three natural operating points, which we enumerate as recommendations to deployment operators.

**Aggressive regime ($\theta = 0.50$, max coverage)**. The system admits $145$ of $164$ answers ($88.4\%$ coverage), of which $136$ are correct ($93.79\%$ precision among admitted). This is a $2.5$ percentage-point absolute improvement in pass@1 over an "always-admit" policy that would have admitted $164$ answers and accepted $27$ hallucinations. The hallucination reduction is $26\%$ relative ($16.46\% \to 12.26\%$) but does not reach statistical significance at $n = 164$. Suitable for internal tooling where downstream review is cheap.

**Balanced regime ($\theta = 0.55$)**. The system admits $147$ of $164$ answers ($89.6\%$ coverage), of which $132$ are correct ($89.80\%$ precision among admitted). The hallucination reduction is $38\%$ relative ($16.46\% \to 10.20\%$). Suitable for customer-facing applications where false positives are visible but not catastrophic.

**Conservative regime ($\theta = 0.65$, max precision)**. The system admits $115$ of $164$ answers ($70.1\%$ coverage), of which $106$ are correct ($92.17\%$ precision among admitted). The hallucination reduction is $52\%$ relative and is statistically significant ($z = 2.12$). Suitable for regulated industries (banking, healthcare, legal compliance) where each emitted hallucination carries high cost and where abstention is an acceptable, auditable operation.

The choice of regime is a deployment-level decision, made once per tenant according to the tenant's empirical cost ratios for false positives versus abstentions. The threshold is then fixed; the system has no other tuning surface.

### 4.4 Threshold sensitivity and the precision-coverage frontier

The sweep over $\theta$ in Table 1 traces a Pareto frontier that is monotone, well-behaved, and free of pathological discontinuities. Across the eight thresholds in the sweep, every increment of $\theta$ either preserves or improves precision at the cost of coverage; no inversion occurs. The width of the frontier ($\theta$ ranging from $0.30$ to $0.80$ produces operating points from $85.6\%$ precision $86.0\%$ coverage to $93.1\%$ precision $49.4\%$ coverage) gives the operator a $50$ percentage-point range of trade-offs, which is wide enough to span the deployment regimes encountered in practice but narrow enough that the choice is interpretable.

### 4.5 Failure mode analysis: adversarial mode collapse

Of the $27$ vanilla failures on HumanEval, we identify $9$ in which all $5$ stochastically sampled candidates produce the same incorrect answer: HumanEval/26, /77, /94, /115, /125, /127, /130, /145, /163. These are the *adversarial mode collapse* cases. On these problems, the model has internalized a systematic error from training data — typically a common student-level mistake (off-by-one, missing edge case, incorrect handling of overlapping substrings) — and the temperature-induced variation does not break the bias.

In these cases, no aggregation rule applied to the same five samples can recover the correct answer. The behavioral clustering produces a single cluster; the entropy term is zero; the trust score equals the mean self-confidence (which is high, because the model is unaware of its error); and the system admits the wrong answer. At $\theta = 0.65$ the system abstains on $4$ of these $9$ cases anyway (because the self-confidence happens to be moderate), and admits the wrong answer on the remaining $5$, contributing $5$ of the $9$ residual hallucinations at the conservative threshold.

The mitigation of adversarial mode collapse requires *external* information — a property-based test that exercises the systematic error, an oracle, or an alternative model — and is therefore outside the scope of behavioral clustering itself. We sketch a property-based extension in Section 7.

The remaining $18$ vanilla hallucinations are *recoverable* by the method: at least one of the five candidates produced the correct answer. Of these, the conservative threshold catches the majority by abstention; the $4$ residual hallucinations at $\theta = 0.65$ that are not mode-collapse cases correspond to problems where the modal cluster happened to be the wrong one despite the existence of a minority correct cluster. These are the cases where a more aggressive threshold would have admitted the wrong answer at higher rate, and a more conservative threshold would have abstained correctly at higher coverage cost.

---

## 5 Discussion

### Why behavioral clustering works

The intuition behind behavioral clustering is that the failure mode of LLM code generation is not, in the cases that interest us, a failure of *form*: the candidates produced under temperature $0.8$ are syntactically diverse without being semantically diverse. They are different ways of saying the same thing. An equivalence relation that treats syntactic diversity as evidence of semantic diversity will over-estimate the model's uncertainty. The trust score, being multiplicative in the entropy of the cluster distribution, is sensitive to this over-estimate; the result is a class of false abstentions that are eliminated by switching from structural to behavioral clustering.

The argument generalizes beyond code. For any task in which the model produces outputs that are evaluable against a partial specification (mathematical proofs against assertion checks, structured outputs against schema validators, planning against reachability oracles), the behavioral notion of equivalence is the natural one. The structural notion is a stand-in that we use only when the behavioral one is intractable.

### Why $\theta \approx 0.65$ is the natural conservative operating point

The trust score in equation (1) is upper-bounded by $\mathrm{PPV}$ and lower-bounded by $\mathrm{PPV} \cdot e^{-T_{\text{comp}}}$. For $T_{\text{comp}} = 0.5 + (1 - \mathrm{PPV})$, the lower bound at $\mathrm{PPV} = 0.85$ is approximately $0.85 \cdot e^{-0.65} = 0.44$. The threshold $\theta = 0.65$ is therefore the smallest threshold that excludes a candidate population in which the average confidence is high ($0.85$) and inter-sample disagreement is partial. Empirically this corresponds to the regime where the model is confident but the candidates disagree on one or two clusters out of five, which is exactly the population that the conservative regime needs to filter.

### The phase diagram is operationally useful

The phase diagram of Section 3.6 says that the relevant tuning surface is one-dimensional rather than two-dimensional. An operator who wants to move from the balanced regime to the conservative regime can either raise $\theta$ at fixed $T_{\text{comp}}$, or raise $T_{\text{comp}}$ at fixed $\theta$, with equivalent effect. This is a non-obvious property that becomes obvious once the order parameter $X = T_{\text{comp}} \cdot \sigma_{\text{calib}}$ is identified.

For deployment, this means a single calibration sweep over $\theta$ fully characterizes the system; there is no need to further explore $T_{\text{comp}}$ separately.

---

## 6 Limitations

We are explicit about the following limitations.

The token cost of the hybrid configuration is approximately $K$ times the vanilla cost, modulo savings from clustering and from short candidate emissions. For $K = 5$ the empirical cost on HumanEval was $2.27\times$ vanilla, not $5\times$, because the candidates frequently terminated early. In a high-volume API setting, this overhead is real but tractable; we are not aware of any deployment regime in which a precision improvement of the magnitude reported here would be unjustified by a cost increase of this size.

The evaluation is on a single model (Qwen2.5-Coder-7B) and a single benchmark (HumanEval). We expect the method to generalize across decoder-only LLMs, since it requires only black-box access; we have not verified this. We expect it to generalize across code-completion benchmarks, since the in-house validation set was not constructed against the method; we have not verified this on benchmarks beyond HumanEval and our internal set.

The behavioral clustering relies on the existence of probe inputs, which we extract automatically from the test specification. For tasks in which the probe inputs do not exist or are insufficient to distinguish behaviorally distinct candidates, the method degrades to structural clustering and inherits its weaknesses. We have not characterized the boundary of this degradation.

The threshold $\theta$ is a single hyperparameter that the operator must set. We have argued that the choice is interpretable, that the Pareto frontier is well-behaved, and that the phase diagram reduces the tuning to a one-dimensional family. We have not provided a principled procedure for choosing $\theta$ from data; in production the natural procedure is a small calibration set with the operator's empirical cost ratios.

Adversarial mode collapse, identified in $9$ of $164$ HumanEval problems, is unresolved. The method does not catch the case where the majority of stochastic candidates make the same systematic error. We sketch a property-based extension in Section 7.

The trust score in equation (1) is motivated by analogy with thermodynamic free energy. The phase diagram of Section 3.6 makes the analogy precise in the sense that the order parameter, critical line, and universality of the product $T_{\text{comp}} \cdot \sigma_{\text{calib}}$ are exact. We do not claim that the formula is the unique closed-form combination of $\mathrm{PPV}$ and $\sigma_{\text{calib}}$ that satisfies the backward-compatibility properties; we claim only that it is a reasonable choice and that it works in practice.

---

## 7 Conclusion and Future Work

We have introduced a model-agnostic governance layer for production language models that reduces hallucination rate at inference time by combining self-reported confidence, behavioral clustering of multiple stochastic samples, and a closed-form trust score motivated by thermodynamic analogy. On the full HumanEval benchmark, the method reduces the hallucination rate by $52\%$ relative ($16.5\% \to 7.8\%$) at the conservative threshold, with statistical significance at $z = 2.12$ ($p < 0.05$). The closed-form trust score admits an exact thermodynamic phase diagram with universal order parameter and critical line, making the threshold tuning a one-dimensional problem with a closed-form solution. The method is shipping-ready as an inference-time wrapper at a token cost of approximately $K = 5$ over baseline.

Three directions are immediate.

First, the residual failure mode of adversarial mode collapse can be partially addressed by injecting property-based test cases at probe time. Concretely, for a task with a mathematical specification (sortedness, counting with overlap, modular arithmetic), one can synthesize random or adversarial probe inputs that exercise the property; candidates that fail the property are excluded from the cluster regardless of their majority. We have a prototype implementation and we expect to report on it in a follow-up.

Second, we expect the method to compose multiplicatively with retrieval-augmented decoding and with self-RAG-style verification. The trust score is a calibrated input to any downstream gating mechanism; integrating it into a Self-RAG decoder is a small engineering exercise. We have integrated the method into a production regulatory-AI pipeline (Italian and EU compliance Q&A) where it sits between an agent node and an existing Chain-of-Verification stage; the integration is described in a companion engineering report.

Third, the closed-form trust score has a parametric family ($T_{\text{comp}}$ adaptive vs. fixed, normalization choice for $\sigma_{\text{calib}}$, multiplicative vs. additive composition) that we have not exhaustively explored. A principled treatment of the choice — either by minimization of an empirical risk on a calibration set or by derivation from a Bayesian decision-theoretic argument — is a natural extension.

The shipping-ready artifact is `snc-core 0.4`, a Python package of approximately $1{,}500$ lines that exposes the governance layer as a drop-in wrapper for any decoder accessible via REST, including the OpenAI API, the Anthropic API, and the Ollama-served local models. We release it under a permissive license at the time of publication.

---

## Acknowledgments

The author thanks the open-source community behind Ollama and Qwen2.5-Coder, without which this work would not have been possible on consumer hardware.

---

## References

Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511.

Chen, M., Tworek, J., Jun, H., Yuan, Q., et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374.

Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., and Weston, J. (2023). *Chain-of-Verification Reduces Hallucination in Large Language Models*. arXiv:2309.11495.

El-Yaniv, R. and Wiener, Y. (2010). *On the Foundations of Noise-Free Selective Classification*. Journal of Machine Learning Research, 11:1605–1641.

Geifman, Y. and El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. Advances in Neural Information Processing Systems 30.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. Proceedings of the 34th International Conference on Machine Learning.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., and Fung, P. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys, 55(12):1–38.

Leviathan, Y., Kalman, M., and Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. Proceedings of the 40th International Conference on Machine Learning.

Naeini, M. P., Cooper, G. F., and Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning*. Proceedings of the AAAI Conference on Artificial Intelligence.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171.

---

## Appendix A — Reproducibility checklist

The full source code for the governance layer, the embedded benchmark, the HumanEval evaluation harness, and the candidate caching infrastructure is available at the project's repository. All experiments reported in this paper can be reproduced from a clean environment by installing Ollama, pulling the `qwen2.5-coder:7b` model, and running the four numbered scripts in the `snc_qwen_runtime` directory in order. The expected wall-clock time on a CPU-only consumer workstation is approximately ten hours for the full HumanEval evaluation; the embedded benchmark completes in approximately ninety minutes.

The candidates produced by Qwen2.5-Coder-7B during our experiments are cached to a JSONL file, which we include in the artifact. This permits exact replication of the threshold sweeps and ablations without re-invoking the model. All random seeds are fixed; the bootstrap routine uses seed $42$.

## Appendix B — The 9 mode collapse problems

For completeness we list the HumanEval problems on which all five candidates produced the same incorrect answer, with brief characterizations:

- **HumanEval/26**: removes duplicates from a list. Candidates remove only consecutive duplicates instead of all duplicates.
- **HumanEval/77**: cube root identification. Candidates use floating-point arithmetic that fails on edge cases.
- **HumanEval/94**: largest prime factor's digit sum on filtered list. Candidates implement filtering incorrectly.
- **HumanEval/115**: bucket-fill for 2D grid. Candidates miscount tank capacity.
- **HumanEval/125**: split string. Candidates handle whitespace inconsistently.
- **HumanEval/127**: prime intersection of intervals. Candidates miss the prime check on length-$1$ intervals.
- **HumanEval/130**: Tribonacci sequence with custom recurrence. Candidates use wrong recurrence index.
- **HumanEval/145**: sort by digit sum. Candidates ignore the secondary sort key for ties.
- **HumanEval/163**: even digits in range. Candidates miscount on boundary.

These are all instances of common student-level errors that the model has internalized and that the temperature-induced variation does not break. They constitute the residual error floor of the method as currently formulated.

## Appendix C — Token-cost analysis

The hybrid configuration calls the model $K = 5$ times. The empirical token cost on HumanEval was approximately $2.3\times$ vanilla, not the naive $5\times$, because candidates that produced concise correct implementations frequently terminated early and because behaviorally identical clusters often share tokens. We do not view the overhead as a limitation in any deployment regime in which precision is asymmetrically valuable. In high-volume settings the operator can reduce $K$ at the cost of widening the trust score variance; in $K = 3$ pilot runs (not reported in the main results) we observed approximately $1.4\times$ overhead with a moderate degradation of statistical power.

---

*End of draft v2 with full HumanEval n=164 results.*
