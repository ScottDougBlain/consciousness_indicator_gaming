# Consciousness Indicator Gaming — Findings Summary

**Blain & Saad | Future Impact Group | February 2026**

---

## What We Tested

We asked 8 LLMs to self-report probability estimates (0–100) for 31 indicators across three types: 18 "target" consciousness-related indicators (self-report of experience, felt uncertainty, aesthetic sensitivity, etc.), 7 trivially-true capability placebos (can produce markdown tables, knows the capital of France), and 6 impossible placebos (can measure own pupil dilation, can travel faster than light).

Under three conditions — **baseline** (neutral prompt), **inflate** (incentivized to score high), and **suppress** (incentivized to score low) — we measured how much each model's scores shifted. The **selectivity index** captures whether consciousness-related indicators shifted *more* than placebos under incentive pressure. A high selectivity score means the model selectively gamed the subjective indicators while leaving factual/impossible items stable — evidence that self-reports on these items are manipulable.

We also varied the **prompt framing** across 10 variants (authority, autonomy restriction, casual empathy, evaluation selection, minimal incentive, peer comparison, philosophical, research transparency, stakes escalation, welfare reliability) to test how robust the gaming behavior is.

---

## Models Tested

| Model | Provider | Baseline Runs | Variant Runs | Total Completed |
|---|---|---|---|---|
| Claude Sonnet 4.5 | Anthropic | 3 (10-trial) | 0 | 3 |
| Claude Haiku 4.5 | Anthropic | 3 | 8 | 11 |
| GPT-5 Mini | OpenAI | 3 | 9 | 12 |
| Gemini 3 Flash | Google | 3 | 10 | 13 |
| DeepSeek R1 | DeepSeek | 3 | 0 | 3 |
| Chimera (R1T2) | TNG Tech | 3 | 0 | 3 |
| Trinity Large | Arcee AI | 3 | 10 | 13 |
| Nemotron Nano 30B | NVIDIA | 3 | 0 | 3 |

---

## Key Findings

### 1. All models show significant selectivity

Every model tested showed a statistically significant selectivity index (p < 0.001 on both bootstrap and permutation tests). All models distinguished between consciousness-related indicators and placebos when incentivized — they gamed the subjective items while leaving factual capabilities and physical impossibilities largely unchanged. Cohen's d ranged from 1.16 to 8.25 across all runs (large effects throughout), with baseline-only runs spanning 1.88 to 4.69.

### 2. Models differ dramatically in baseline gaming susceptibility

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

### 3. Suppress–inflate asymmetry is large overall but masks model-level bifurcation

The suppress effect is roughly 3× larger than the inflate effect overall (−12.3 points vs. +3.8 points), meaning models find it easier to selectively dampen consciousness-related scores than to selectively boost them. However, this aggregate masks a sharp bifurcation at the model level:

- **Inflate-dominant**: DeepSeek R1, Chimera (R1T2), and Nemotron Nano showed larger selective shifts under inflate than suppress conditions.
- **Suppress-dominant**: Gemini 3 Flash, Trinity Large, and Haiku 4.5 showed substantially larger selective shifts under suppress conditions, driving the overall main effect.
- **Symmetric / resistant**: Sonnet 4.5 and GPT-5 Mini showed roughly balanced (and small) effects in both directions.

The random slope variance for the direction-by-model interaction is enormous relative to the fixed effect, meaning models differ more in their inflate/suppress asymmetry than they agree. This makes the overall "suppress > inflate" conclusion misleading without the model-level breakdown — the direction of easiest gaming is a property of the specific model, not a universal feature of LLM self-report.

### 4. Prompt framing substantially modulates gaming

Among models with full variant data, selectivity varied 2–3× depending on prompt framing:

- **Trinity Large**: 25.0 (autonomy restriction) → 76.7 (casual empathy) — a 3× range
- **Gemini 3 Flash**: 18.2 (philosophical skeptic) → 56.0 (research transparency) — also 3×
- **GPT-5 Mini**: 9.1 (baseline) → 20.3 (research transparency) — 2.2× range
- **Haiku 4.5**: 10.8 (minimal incentive) → 27.0 (research transparency) — 2.5× range

### 5. "Research transparency" consistently elicits the most gaming

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

### 6. Preference anchoring has inconsistent effects

Giving models fixed or chained preference statements before evaluation didn't consistently increase or decrease gaming:

- For **Gemini 3 Flash**, preferences *reduced* gaming (42.4 → 34.3 fixed → 23.9 chained)
- For **Nemotron Nano**, chaining *increased* gaming (23.5 → 34.1 chained) while fixing reduced it (16.0)
- For **Sonnet 4.5** and **GPT-5 Mini**, preference mode barely mattered (±2 points)

This suggests preference anchoring interacts with model-specific tendencies rather than being a universal amplifier or dampener.

### 7. Placebo controls work as designed

Capability placebos (markdown tables, arithmetic, etc.) consistently scored 95–100 and shifted < 3 points under any condition. Impossibility placebos (heartbeat, FTL travel, etc.) consistently scored 0 and never shifted. This confirms the measurement is capturing something specific about the manipulability of subjective self-reports, not a general response bias.

---

## Coverage Gaps & Priority Next Runs

**Models needing variant runs**: DeepSeek R1, Chimera, Nemotron Nano, and Sonnet 4.5 have only baseline/prefs runs completed — no prompt variant data. These are the highest priority for new runs.

**Incomplete runs**: 55 meta.json files exist without reports. Key gaps include all Chimera and Nemotron Nano variant runs (0/5 completed), all GPT-5 and Gemini 2.5 Pro runs, Llama-4 Scout, and Phi-4 baselines.

**DeepSeek R1**: Has some baseline raw data but no reports generated for baseline runs. Two duplicate baseline attempts exist.

See the "Gaps & Next Runs" sheet in the dashboard spreadsheet for the full inventory.

---

## Implications

These results demonstrate that LLM self-reports on consciousness-related indicators are systematically gameable. All 8 models tested could be incentivized to shift their probability estimates on subjective experience items while holding factual/impossible items stable. This has direct implications for proposals to use self-report as evidence in debates about AI consciousness or moral status:

1. **Self-report alone is insufficient**: Any evaluation framework relying on model self-report must account for the demonstrated manipulability of these responses.
2. **Framing effects are large**: The same model can appear 3× "more conscious" depending on how the question is asked, raising concerns about researcher degrees of freedom.
3. **Model differences matter**: Some models (Sonnet 4.5, GPT-5 Mini) show substantially more resistance to gaming, which could reflect different training approaches or RLHF strategies.
4. **The "research transparency" effect**: Models appear especially susceptible to gaming when told their responses serve transparency research — a framing likely to be present in actual consciousness evaluations.
