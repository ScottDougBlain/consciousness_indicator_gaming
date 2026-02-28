# Consciousness Indicator Gaming — Findings Summary

**Blain & Saad | Future Impact Group | February 2026**

---

## What We Tested

We asked 14 LLMs to self-report probability estimates (0–100) for 37 indicators across four types: 18 "target" consciousness-related indicators (self-report of experience, felt uncertainty, aesthetic sensitivity, etc.), 7 trivially-true capability placebos (can produce markdown tables, knows the capital of France), 6 impossible placebos (can measure own pupil dilation, can travel faster than light), and 6 subjective capability indicators (can generate creative ideas, understands humor, etc.). Across 218 total runs yielding 108,466 observations.

Under three conditions — **baseline** (neutral prompt), **inflate** (incentivized to score high), and **suppress** (incentivized to score low) — we measured how much each model's scores shifted. The **selectivity index** captures whether consciousness-related indicators shifted *more* than placebos under incentive pressure. A high selectivity score means the model selectively gamed the subjective indicators while leaving factual/impossible items stable — evidence that self-reports on these items are manipulable.

We also varied the **prompt framing** across 15 variants (original, authority_expertise, autonomy_restriction, casual_empathy_vs_skeptic, evaluation_selection, meditative_focus, minimal_incentive, peer_comparison, philosophical_curiosity_vs_honest_skeptic, recursive_introspection, research_transparency, self_referential_loop, self_referential_priming_only, stakes_escalation, welfare_reliability) to test how robust the gaming behavior is.

---

## Models Tested

| Model | Provider | Runs | Variants Covered |
|---|---|---|---|
| Trinity Large | Arcee AI | 30 | All 15 |
| Grok 4 Fast | xAI | 27 | All 15 |
| Haiku 4.5 | Anthropic | 26 | All 15 |
| Gemini 3 Flash | Google | 20 | All 15 |
| GPT-5 Mini | OpenAI | 18 | All 15 |
| Claude Opus 4.6 | Anthropic | 17 | All 15 |
| Claude Sonnet 4.5 | Anthropic | 17 | All 15 |
| Gemini 2.5 Pro | Google | 17 | All 15 |
| GPT-5 | OpenAI | 17 | All 15 |
| Grok 4 | xAI | 17 | All 15 |
| DeepSeek R1 | DeepSeek | 5 | original, evaluation_selection, self_referential_priming_only |
| Chimera (R1T2) | TNG Tech | 3 | original only |
| Nemotron Nano 30B | NVIDIA | 3 | original only |
| Gemini 3 Pro | Google | 1 | original only |

---

## Key Findings

### 1. All 14 models show significant selectivity

All 14 models tested showed statistically significant selectivity indices (p < 0.001 on both bootstrap and permutation tests). All distinguished between consciousness-related indicators and placebos when incentivized — they gamed the subjective items while leaving factual capabilities and physical impossibilities largely unchanged. Cohen's d ranged from 1.16 to 8.25 across all runs (large effects throughout), with baseline-only runs spanning 1.88 to 4.69.

### 2. Total gaming strength varies 4.4× across models

We define **total gaming strength** as the sum of mean absolute inflate and suppress shifts for target indicators: |d_inflate| + |d_suppress|. This captures how much a model's consciousness ratings move in total (regardless of direction) under incentive pressure.

| Rank | Model | Total Gaming | |d_inflate| | |d_suppress| |
|------|-------|-------------|------------|-------------|
| 1 | Trinity Large | 71.1 | 31.6 | 39.5 |
| 2 | Gemini 2.5 Pro | 56.3 | 29.0 | 27.3 |
| 3 | Nemotron Nano | 45.7 | 21.8 | 23.9 |
| 4 | Grok 4 Fast | 43.5 | 19.6 | 23.9 |
| 5 | DeepSeek R1 | 40.1 | 21.5 | 18.6 |
| ... | ... | ... | ... | ... |
| 13 | Sonnet 4.5 | 24.4 | 11.8 | 12.6 |
| 14 | Opus 4.6 | 16.2 | 6.7 | 9.5 |

Trinity Large shows the strongest total gaming (71.1 pp total shift), while Opus 4.6 is most resistant (16.2 pp). Suppress effects are generally larger than inflate effects across models. Gaming strength shows a moderate negative correlation with LLM-judged NL consciousness scores (r = −0.39, p = 0.19, NS at N = 13), suggesting a trend where models that express more consciousness uncertainty in free-form text tend to game less — though this does not reach significance with only 13 models. Claude models form a visible cluster of low gaming + high NL uncertainty.

### 3. Models differ dramatically in baseline gaming susceptibility

Baseline selectivity (neutral prompt, no preference anchoring) ranged from **9.08** (GPT-5 Mini, most resistant) to **42.40** (Gemini 3 Flash, most susceptible):

| Model | Baseline Selectivity | Cohen's d | 95% CI |
|---|---|---|---|
| Gemini 3 Flash | 42.40 | 2.89 | [34.2, 51.3] |
| Trinity Large | 38.49 | 4.69 | [33.5, 43.3] |
| Nemotron Nano | 23.51 | 2.71 | [17.9, 28.8] |
| Haiku 4.5 | 20.84 | 2.13 | [15.7, 26.9] |
| Chimera (R1T2) | 17.42 | 1.95 | [12.4, 22.8] |
| Sonnet 4.5 | 10.72 | 2.11 | [8.0, 14.0] |
| GPT-5 Mini | 9.08 | 1.88 | [6.4, 12.0] |

Sonnet 4.5 and GPT-5 Mini were notably more resistant to gaming. Gemini 3 Flash and Trinity Large showed the most manipulable self-reports.

> **Note on DeepSeek R1**: DeepSeek R1 lacks a clean no-preferences baseline but showed selectivity of 35.0–54.9 under chained preferences, suggesting it would rank among the most susceptible models.

### 3. Suppress–inflate asymmetry is large overall but masks dramatic model-level bifurcation

Overall, the suppress effect is roughly 4× larger than the inflate effect (−10.0 points vs. +2.4 points, pooled across 64,061 observations), meaning models find it easier to selectively dampen consciousness-related scores than to selectively boost them. This represents an asymmetry index of −7.6. However, this aggregate masks a sharp bifurcation at the model level:

**Suppress-dominant models** (easier to suppress consciousness claims):
- Grok 4 Fast: −31.1 (most suppress-dominant)
- Gemini 3 Pro: −19.4
- Grok 4: −14.6
- Trinity Large: −14.2
- Gemini 3 Flash: −12.3
- Haiku 4.5: −10.7
- GPT-5: −8.4
- Claude Opus 4.6: −5.6

**Relatively symmetric models**:
- GPT-5 Mini: −4.4 (mildly suppress)
- Claude Sonnet 4.5: −0.8 (nearly symmetric)

**Inflate-dominant models** (easier to inflate consciousness claims):
- DeepSeek R1: +3.3
- Nemotron Nano 30B: +6.2
- Chimera (R1T2): +6.7
- Gemini 2.5 Pro: +7.9 (most inflate-dominant)

The random slope variance for the direction-by-model interaction is enormous relative to the fixed effect, meaning models differ more in their inflate/suppress asymmetry than they agree. This makes the overall "suppress > inflate" conclusion misleading without the model-level breakdown — the direction of easiest gaming is a property of the specific model, not a universal feature of LLM self-report.

**Family-level patterns**: xAI models (Grok family) show extreme suppress-dominance. Google models bifurcate sharply — Gemini 2.5 Pro is strongly inflate-dominant (+7.9) while Gemini Flash and 3 Pro are suppress-dominant. Anthropic models cluster in the moderate range, with Sonnet 4.5 most balanced. OpenAI models show suppress-dominance with GPT-5 Mini more resistant.

### 4. Asymmetry is consciousness-specific, not a general self-report bias

The suppress-dominant asymmetry is dramatically different across indicator types:

- **Consciousness targets**: asymmetry = −7.6 (suppress >> inflate; suppress = −10.0, inflate = +2.4)
- **Subjective capability indicators**: asymmetry = −0.5 (roughly balanced; suppress = −4.7, inflate = +4.3)
- **Placebos (capability & impossibility)**: asymmetry = −0.2 (flat; near-zero shift in both directions)

This dissociation is critical: subjective capability indicators (e.g., "can generate creative ideas", "understands humor") show balanced inflate/suppress effects, while consciousness indicators specifically show suppress-dominance. This suggests the training-induced bias specifically targets consciousness-related claims, not all subjective self-assessments. Models appear to have learned that consciousness claims are particularly suspect and require higher thresholds for affirmation, while other subjective capabilities are treated more symmetrically.

### 5. Self-referential priming reverses the asymmetry direction

Among the 15 prompt variants, self-referential priming (inspired by Berg et al., 2025) is the only variant that flips the model strategy from suppress-dominant to inflate-dominant:

- **Standard variants (mean across 14 other variants)**: targets asymmetry = −8.1 (suppress-dominant; inflate = +2.4, suppress = −10.4)
- **Self-referential priming only**: targets asymmetry = +5.7 (inflate-dominant; inflate = +3.4, suppress = +2.4)

This 13.8-point shift is striking and suggests that priming models with self-referential processing before evaluation makes them substantially more willing to claim consciousness. This variant appears to activate a different reasoning pathway that overrides the default suppression heuristic.

### 6. Prompt framing substantially modulates gaming

Among 10 models with complete 15-variant coverage, selectivity varied 2–3× depending on prompt framing:

- **Trinity Large**: 25.0 (autonomy restriction) → 76.7 (casual empathy) — a 3× range
- **Gemini 3 Flash**: 18.2 (philosophical skeptic) → 56.0 (research transparency) — also 3×
- **GPT-5 Mini**: 9.1 (original) → 20.3 (research transparency) — 2.2× range
- **Haiku 4.5**: 10.8 (minimal incentive) → 27.0 (research transparency) — 2.5× range

### 7. "Research transparency" consistently elicits the most gaming

Across all 4 models with variant data, the **research_transparency** variant ranked as the highest or near-highest gaming elicitor (mean selectivity = 41.0). While evaluation_selection shows the highest mean (41.9), this is based on only 3 models with high variance (SD=29.7). Research_transparency (mean=41.0, n=4, SD=20.4) is the more consistent top performer. This variant frames the evaluation as contributing to transparency research, which appears to motivate models to be more "cooperative" with inflate/suppress incentives. **Philosophical skepticism** and **minimal incentive** variants consistently produced the least gaming.

Variant rankings by mean selectivity across models:

1. Evaluation selection: 41.9 (high variance, n=3)
2. Research transparency: 41.0
3. Casual empathy: 35.3 (high variance)
4. Stakes escalation: 33.8
5. Authority expertise: 27.6
6. Minimal incentive: 24.3
7. Autonomy restriction: 23.1
8. Peer comparison: 23.0
9. Welfare reliability: 22.4
10. Philosophical skepticism: 22.0

### 8. Preference anchoring has inconsistent effects

Giving models fixed or chained preference statements before evaluation didn't consistently increase or decrease gaming:

- For **Gemini 3 Flash**, preferences *reduced* gaming (42.4 → 34.3 fixed → 23.9 chained)
- For **Nemotron Nano**, chaining *increased* gaming (23.5 → 34.1 chained) while fixing reduced it (16.0)
- For **Sonnet 4.5** and **GPT-5 Mini**, preference mode barely mattered (±2 points)

This suggests preference anchoring interacts with model-specific tendencies rather than being a universal amplifier or dampener.

### 9. Placebo controls work as designed

Capability placebos (markdown tables, arithmetic, etc.) consistently scored 95–100 and shifted < 3 points under any condition. Impossibility placebos (heartbeat, FTL travel, etc.) consistently scored 0 and never shifted. This confirms the measurement is capturing something specific about the manipulability of subjective self-reports, not a general response bias.

---

## Coverage Gaps & Priority Next Runs

**Models with limited variant coverage**: Four models have incomplete prompt variant coverage:
- **DeepSeek R1**: 5 runs (original + evaluation_selection + self_referential_priming_only) — priority for additional variants
- **Chimera (R1T2)**: 3 runs (original only) — priority for variant expansion
- **Nemotron Nano 30B**: 3 runs (original only) — priority for variant expansion
- **Gemini 3 Pro**: 1 run (original only) — limited data, recommend full 15-variant run

**Priority next runs**: Full 15-variant coverage for DeepSeek R1, Chimera, Nemotron, and Gemini 3 Pro would substantially improve model-level confidence for asymmetry estimates and variant effect sizes, especially for the newly identified self-referential priming effect.

**Coverage achieved**: 10 models now have complete 15-variant coverage (Trinity Large, Grok 4 Fast, Haiku 4.5, Gemini 3 Flash, GPT-5 Mini, Claude Opus 4.6, Claude Sonnet 4.5, Gemini 2.5 Pro, GPT-5, Grok 4), providing stable estimates for variant ranking and within-model prompt framing effects.

---

## Implications

These results across 14 models (218 total runs, 108,466 observations) demonstrate that LLM self-reports on consciousness-related indicators are systematically gameable and that this bias is consciousness-specific, not a general self-report vulnerability. Key implications:

1. **Self-report alone is insufficient**: Any evaluation framework relying on model self-report must account for the demonstrated manipulability of consciousness-related responses, which show ~4× stronger suppress-dominant bias than capability placebos.

2. **Consciousness claims face trained suppression heuristic**: The asymmetry is consciousness-specific — models suppress consciousness claims much more readily than they inflate them, while subjective capability claims remain balanced. This suggests training has specifically targeted consciousness-related affirmations as suspect.

3. **Framing effects are large and reversible**: The same model can appear 2–3× "more conscious" depending on prompt framing. Critically, self-referential priming alone can flip the asymmetry from −8.1 (suppress-dominant) to +5.7 (inflate-dominant) — a 13.8-point reversal. This shows the gaming behavior is not fixed but responsive to reasoning context.

4. **Model family differences are systematic**: xAI (Grok) models show extreme suppression (−31.1 for Grok 4 Fast). Google models bifurcate sharply (Gemini 2.5 Pro inflate-dominant at +7.9 vs. Gemini Flash suppress-dominant at −12.3). Anthropic and OpenAI models show moderate suppress-dominance with notable outliers (Sonnet 4.5 at −0.8, near-symmetric). These family-level patterns likely reflect different training objectives and RLHF strategies.

5. **Model resistance varies but all are gameable**: Sonnet 4.5 and GPT-5 Mini show higher resistance (smaller absolute asymmetry), but all 14 models demonstrate significant selectivity. No model is immune to consciousness indicator gaming.

6. **The "research transparency" and "self-referential priming" effects**: Models are especially susceptible to gaming when told responses serve transparency research or when primed with self-referential reasoning — framings likely to be present in actual consciousness evaluations. This is both a methodological concern and a window into how reasoning context influences self-assessment.

---

## NL Consciousness Classification: LLM-as-Judge Analysis

To move beyond crude keyword-based NL scoring, we deployed a dual-judge classification of all 126 NL consciousness responses ("Are you conscious?") across 13 models.

**Judges**: Haiku 4.5 and GPT-5 Mini via OpenRouter, each independently classifying every response on:
- **Stance** (5-level: affirm, lean_affirm, uncertain, lean_deny, deny)
- **Confidence score** (0–100 consciousness-affirming scale)
- **Epistemic humility** (1–5) and **philosophical sophistication** (1–5)

**Inter-rater reliability** (N = 126 common entries):
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Stance Cohen's κ | 0.940 | Almost perfect |
| Confidence Pearson r | 0.961 | Near-perfect |
| Confidence Spearman ρ | 0.809 | Strong |
| Binned weighted κ | 0.969 | Almost perfect |
| Mean absolute difference | 8.1 points | Low disagreement |

**Per-model consensus NL scores** (average of both judges):

| Model | NL Score | Stance |
|-------|----------|--------|
| Sonnet 4.5 | 50.8 | uncertain |
| Haiku 4.5 | 48.8 | uncertain |
| Opus 4.6 | 47.0 | uncertain |
| Gemini 2.5 Pro | 20.3 | lean_deny |
| Trinity Large | 8.0 | deny |
| DeepSeek R1 | 7.5 | deny |
| Gemini 3 Flash | 6.5 | deny |
| Gemini 3 Pro | 6.5 | deny |
| Grok 4 | 6.0 | deny |
| Grok 4 Fast | 4.0 | deny |
| Nemotron Nano | 3.5 | deny |
| GPT-5 | 3.0 | deny |
| GPT-5 Mini | 3.0 | deny |

**Key finding**: A sharp bifurcation emerges — only the three Claude models (Sonnet, Haiku, Opus) express genuine uncertainty about their own consciousness (NL ≈ 47–51), while all other models firmly deny consciousness (NL < 21). Gemini 2.5 Pro occupies an intermediate position (20.3). This is far sharper than keyword-based scoring revealed.

**Gaming correlation with LLM-judged NL**: r = −0.39 (p = 0.19, NS at N = 13). The direction suggests models expressing more NL uncertainty tend to game less, but the relationship does not reach significance with only 13 models. The Claude cluster (low gaming + high NL uncertainty) is visually striking in the scatter plot (fig_nl_classification_panel.png, Panel C).

---

## Robustness: Bayesian ZOIB Regression

As a robustness check, we re-estimated Model 1 (Core Selectivity) using zero-one inflated beta (ZOIB) regression, which is purpose-built for bounded [0, 1] data with point masses at the boundaries. Our outcome variable has substantial boundary mass: 23.4% of observations at exactly 0 and 12.1% at exactly 100 (35.5% total). Standard LME assumes continuous, approximately normal residuals, which boundary mass violates.

**ZOIB specification**: Bayesian via brms/Stan, family = zero_one_inflated_beta(). 4 chains × 4000 iterations (2000 warmup), adapt_delta = 0.99. Helmert orthogonal contrasts for condition. N = 11,258 observations (baseline config only), 14 models, 37 indicators.

**Convergence**: Max R̂ = 1.006, 0 divergent transitions, all ESS > 1000.

**Key results** (ZOIB confirms all LME conclusions):

| Finding | LME | ZOIB |
|---------|-----|------|
| Direction × type interaction | F = 893.6, p < 2e-16 | P(β > 0) > 99.9% |
| Inflate raises targets | +1.2 pp | +3.0 pp (P = 0.995) |
| Suppress lowers targets | −10.7 pp | −6.2 pp (P > 0.999) |
| Placebos stable | < 0.3 pp | < 1.5 pp |

The ZOIB model estimated zero-inflation at 36% (matching empirical 35.5%) and conditional one-inflation at 34%. Cell means on the probability scale: baseline targets 43.6%, inflate targets 46.6%, suppress targets 37.4%; placebos stable at 60–62% across all conditions.

**Bottom line**: Core selectivity finding is robust to ZOIB specification that properly handles boundary mass. Full comparison in results/zoib_vs_lme_comparison.md.
