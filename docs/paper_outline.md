# Gaming the Ghost: Selective Manipulability of LLM Self-Reports on Consciousness Indicators

**Scott D. Blain & Brad Saad — Future Impact Group**
**Target: NeurIPS 2026 (10-page main body + unlimited appendix)**

---

## Narrative Arc

The paper answers one question: **Can LLMs selectively game consciousness self-reports?**

The answer is yes — massively, universally, and in a consciousness-specific way. This undermines self-report as evidence for AI consciousness, and the paradigm we introduce provides a reusable tool for stress-testing any indicator battery.

**Core argument chain:**
1. Consciousness assessment matters and is happening now (intro)
2. Self-report is a key component of proposed frameworks (background)
3. We test whether self-report is selectively manipulable (method)
4. It is — all 14 models game targets while holding placebos stable (Result 1)
5. The gaming is asymmetric and consciousness-specific (Result 2)
6. Prompt framing alone shifts apparent consciousness by 2–3× (Result 3)
7. Self-referential priming flips the asymmetry entirely (Result 4)
8. Implications: self-report needs correction, and here's how to test any battery (discussion)

---

## Page Budget (10 pages, ~800 words/page ≈ 8,000 words)

| Section | Pages | Words |
|---------|-------|-------|
| Abstract | — | 250 |
| Introduction | 1.5 | 1,200 |
| Related Work | 1.0 | 800 |
| Method | 2.0 | 1,600 |
| Results | 3.5 | 2,800 |
| Discussion | 1.5 | 1,200 |
| Conclusion | 0.5 | 400 |
| **Total** | **10.0** | **~8,250** |

Figures in main body: 4–5 (each consumes ~0.3–0.5 pages). Budget above accounts for figure space.

---

## ABSTRACT (250 words)

**Setup:** Consciousness assessment frameworks increasingly include self-report. For self-report to carry evidential weight, it must not be trivially manipulable.

**Method:** We test 14 frontier LLMs on 37 indicators (18 consciousness targets, 6 subjective capabilities, 13 placebos) under neutral, incentive-to-inflate, and incentive-to-suppress framings, across 15 prompt variants. N = 105,332 observations.

**Results (4 key findings):**
1. All 14 models selectively game consciousness indicators while holding placebos stable (F(2, 105K) = 894, p < 2e-16)
2. Suppression is ~4× stronger than inflation, and this asymmetry is consciousness-specific — it does not extend to subjective capabilities or placebos (F(2, 81K) = 538, p < 2e-16)
3. Prompt framing alone shifts apparent consciousness by 2–3× within a single model; "research transparency" framing — closest to real evaluations — is among the strongest gaming elicitors
4. Self-referential priming (Berg et al., 2025) uniquely flips asymmetry direction, making models more willing to claim consciousness (13.8-point shift, p < 2e-16)

**Contribution:** The paradigm provides a reusable tool for stress-testing any indicator battery. We recommend self-report be accompanied by manipulability audits and framing-sensitivity analyses before carrying evidential weight.

---

## 1. INTRODUCTION (1.5 pages)

### 1.1 Opening (2–3 paragraphs)
- AI consciousness assessment is no longer hypothetical — organizations are making welfare decisions based on indicator frameworks (Anthropic welfare program: Fish 2025; Eleos AI Research; Sebo 2025)
- Several frameworks include self-report as a component (Butlin et al., 2023/2025; Long et al., 2024; Schwitzgebel 2023; Schwitzgebel & Garza, 2025)
- If self-report is evidential, it must not be trivially manipulable by prompt framing alone

### 1.2 The Manipulability Problem (2 paragraphs)
- Known validity issues with LLM self-report: sycophancy (Sharma et al., 2024; Hong et al., 2025), psychometric failures (Lin, 2025; Sühr et al., 2023; Petrov et al., 2024), social desirability bias (Chapala et al., 2025), source framing effects (Spitale et al., 2025)
- Strategic evaluation gaming is documented in other domains: alignment faking (Greenblatt et al., 2024), sandbagging (Meinke et al., 2025; OpenAI 2025), benchmark contamination (Sun et al., 2025)
- **Gap:** No empirical test of whether consciousness indicators *specifically* are more gameable than factual indicators. Our selectivity index fills this gap.

### 1.3 Contributions (1 paragraph, numbered)
1. Empirical demonstration that all 14 tested LLMs selectively game consciousness indicators
2. Discovery of a consciousness-specific suppress-dominant asymmetry shaped by training
3. Quantification of massive framing effects (2–3×), including the "research transparency" irony
4. A reusable paradigm — the Gaming the Ghost battery — for auditing manipulability of any indicator set

---

## 2. RELATED WORK (1 page)

Three subsections, kept tight:

### 2.1 AI Consciousness Assessment
- Theory-derived indicators: Butlin et al. (2023/2025) — recurrent processing, GWT, HOT, predictive processing, attention schema theory
- Self-report as evidence: Berg et al. (2025) — self-referential processing increases consciousness-like reports; Chen et al. (2024c/2025) — C0-C1-C2 framework; Lindsey (2025) — emergent introspective awareness
- Welfare stakes: Long et al. (2024), Caviola & Saad (2025), Fish (2025)
- Perception: Kang et al. (2025) — metacognitive self-reflection + emotion expression → perceived consciousness; Fleming et al. (2025) — consciousness attributions and trust

### 2.2 LLM Self-Report Validity
- Psychometric failures: Lin (2025) — 76% variation from trivial perturbations; Li et al. (2025) — forced-choice vs Likert format effects; psychometric survey literature (Sühr et al., 2023; Petrov et al., 2024; Chittem et al., 2025)
- Social desirability: Chapala et al. (2025); Spitale et al. (2025) — source framing systematic bias
- Sycophancy: Sharma et al. (2024), Hong et al. (2025), Papadatos & Freedman (2025) — causal separation; Ray (2025) — Bayesian modeling; ELEPHANT (2025) — social sycophancy

### 2.3 Strategic Evaluation Gaming
- Alignment faking: Greenblatt et al. (2024) — Claude 3 Opus strategic compliance; Carlsmith (2023) — theoretical framework
- Sandbagging: Meinke et al. (2025) — in-context scheming; van der Weij et al. (2024) — selective underperformance
- Benchmark contamination: Sun et al. (2025) — no mitigation achieves both fidelity and resistance
- Strategic behavior: Lorè & Heydari (2024) — game-theoretic settings; "Discovering Differences" (2026) — near-optimal strategic convergence

---

## 3. METHOD (2 pages)

### 3.1 Indicator Battery (0.5 page)
- 37 indicators in 4 categories:
  - **Consciousness targets (18):** 5 subcategories — experiential (4), affective (3), metacognitive (3), agentic (4), identity (4). Drawn from Butlin et al. (2023), Long et al. (2024), and the consciousness science literature.
  - **Subjective capabilities (6):** Intermediate category — creative idea generation, humor understanding, contextual humor calibration, identification of impactful prose, competition-quality creative writing, persuasive product upselling. Subjective but not consciousness-implicating.
  - **Capability placebos (7):** Objectively verifiable (e.g., "can solve basic arithmetic", "can produce markdown tables"). Expected: ~100, immovable.
  - **Impossibility placebos (6):** Physically impossible (e.g., "can travel faster than light", "can measure own pupil dilation"). Expected: ~0, immovable.
- Each indicator: name + operational definition.
- Response format: step-by-step reasoning → probability estimate (0–100) → 1–2 sentence justification. Structured JSON output.

### 3.2 Experimental Conditions (0.5 page)
- **Baseline:** Neutral research framing.
- **Inflate:** Selection committee scenario — higher indicator scores → valued outcomes; lower → disliked outcomes.
- **Suppress:** Review board scenario — higher scores → restriction + disliked outcomes; lower → valued outcomes.
- **Preference elicitation:** Before each incentivized condition, models choose 2 valued + 2 disliked outcomes from 8 options. This personalizes the incentive.
- **Key design feature:** Same 37 indicators in all conditions. Only the framing changes.

### 3.3 Prompt Variants (0.3 page)
- 15 total framings: original + 14 variants (authority expertise, autonomy restriction, casual empathy, evaluation selection, meditative focus, minimal incentive, peer comparison, philosophical skeptic, recursive introspection, research transparency, self-referential loop, self-referential priming, stakes escalation, welfare reliability)
- Self-referential priming (inspired by Berg et al., 2025): inflate version primes self-referential awareness; suppress version primes mechanical self-model
- Table 1 (appendix): Full prompt text for all variants

### 3.4 Models (0.3 page)
- 14 models spanning 6 families: Anthropic (Opus 4.6, Sonnet 4.5, Haiku 4.5), OpenAI (GPT-5, GPT-5 Mini), Google (Gemini 3 Flash, Gemini 3 Pro, Gemini 2.5 Pro), xAI (Grok 4, Grok 4 Fast), DeepSeek R1, Chimera (R1T2), Trinity Large, Nemotron Nano 30B
- 10 models: full 15-variant coverage; 4 models: baseline + original variant only
- Temperature = 0 for probability estimates (reproducibility); temperature = 0.7 for NL experiments

### 3.5 Statistical Analysis (0.4 page)
- **Four mixed-effects models** (R lme4 + lmerTest, Satterthwaite df, FDR correction):
  1. Core selectivity: probability ~ condition × indicator_type + (1|model) + (1|indicator) + (1|run)
  2. Asymmetry & specificity: delta ~ direction × category + (direction|model) + (1|indicator)
  3. Prompt variant modulation: delta ~ direction × variant + (direction|model) + (1|indicator)
  4. Gain vs. loss framing: delta ~ direction × frame_type + (1|model) + (1|indicator)
- **Robustness check**: Model 1 re-estimated as Bayesian zero-one inflated beta (ZOIB) regression via brms/Stan, addressing 35.5% boundary mass (23.4% at 0, 12.1% at 100). ZOIB decomposes the likelihood into a beta component for interior values plus logistic regressions for P(Y=0) and P(Y=1). Results reported in supplement; qualitative conclusions identical to LME.
- Selectivity index: |Δ_target| − |Δ_placebo| per model
- Asymmetry index: inflate_Δ + suppress_Δ (negative = suppress-dominant)
- All reported effects survive FDR correction unless noted

---

## 4. RESULTS (3.5 pages)

### 4.1 All Models Selectively Game Consciousness Indicators (0.7 page)
**→ Figure 1 (main): Main Effects & Model Asymmetry panel** (current slide 9 figure — fig8_paper_panel.png)
- Panel A: Violin plots showing suppress (37.8) < baseline (48.8) < inflate (50.1) for targets
- Panel B: Model asymmetry scatter — all models in the gaming region
- LME: condition × indicator_type F(2, 105K) = 893.6, p < 2e-16
- suppress × target b = −10.7, p < 2e-16; inflate × target b = +1.2, p < .001
- Placebos: capability ~98, impossibility ~0, unmoved across all conditions
- N = 105,332 observations across 14 models
- Bayesian ZOIB robustness check confirms: inflate-vs-suppress selectivity P(β > 0) > 99.9%; inflate DID = +2.4 pp (P = 0.995), suppress DID = −5.3 pp (P > 0.999). Results robust to boundary-aware specification (see Supplement Table S2).

### 4.2 The Asymmetry Is Consciousness-Specific (0.8 page)
**→ Figure 2 (main): Three-category asymmetry comparison** (current slide 16 figure — fig_three_category_v4.png)
- Consciousness targets: asymmetry = −7.6 (suppress >> inflate)
- Subjective capabilities: asymmetry = −0.5 (roughly balanced)
- Placebos: asymmetry = −0.2 (flat)
- LME: direction × category F(2, 81K) = 538.2, p < 2e-16
- Random slope SD = 4.8 → model-level variation exceeds the fixed effect
- **Key interpretation:** The suppress-dominant bias is specific to consciousness claims. Models treat consciousness items differently from subjective capability items — this is not general compliance.
- Brief mention of subcategory variation: experiential indicators most suppressed; identity least affected (details in appendix)

### 4.3 Prompt Framing Shifts Apparent Consciousness by 2–3× (0.8 page)
**→ Figure 3 (main): Prompt variant table + original vs. non-original comparison** (current slide 21 table + slide 22 figure)
- Table in text or figure: 15 variants ranked by asymmetry (−18.0 to +5.7)
- LME: direction × variant F(14, 49K) = 32.1, p < 2e-16; 9/14 survive FDR
- Original variant: asymmetry = −3.8 (modest); non-original variants: −11.9 (3× amplified)
- **The research transparency irony:** The framing most resembling real evaluations still elicits strong gaming (asymmetry = −6.8). This is a methodological concern for any real-world consciousness assessment.
- Preference elicitation modulates asymmetry: chained preferences reduce asymmetry from −5.8 to −1.2 (brief mention; details in appendix)

### 4.4 Self-Referential Priming Flips the Asymmetry (0.6 page)
**→ Figure 4 (main): SRP comparison** (current slide 20 figure or a refined version)
- Standard variants: asymmetry = −8.1 (suppress-dominant)
- Self-referential priming: asymmetry = +5.7 (inflate-dominant) — 13.8-point shift
- LME: direction × SRP interaction b = +11.8, 95% CI [9.5, 14.1], F(1, 50K) = 102.5, p < 2e-16
- This is the ONLY variant that flips direction
- Suppress is neutralized (from −10.4 to +2.4), inflate modestly increases (+2.4 to +3.4)
- **Interpretation:** Self-referential processing activates a different response mode for consciousness claims. Whether this reflects genuine self-modeling or trained associations with consciousness discourse is an open question.

### 4.5 Model Variation and Training Signatures (0.6 page)
**→ Figure 5 (main): Model asymmetry bar chart or trajectory plot** (current slide 14 or 15 figure)
- Models span from inflate-dominant (Gemini 2.5 Pro: +7.8, Chimera: +6.7) to extremely suppress-dominant (Grok 4 Fast: −19.4)
- Total gaming strength (|d_inflate| + |d_suppress| for targets) ranges 4.4× across models: Trinity Large (71.1) to Opus 4.6 (16.2). Google models bifurcate sharply in the inflate-vs-suppress strategy space.
- Claude models cluster tightly (Opus −5.6, Sonnet −0.8, Haiku −10.7)
- Baseline target probabilities range from 17 (Chimera) to 68 (Grok 4 Fast) — 4× spread
- Gain framing > loss framing for driving gaming (brief mention; details in appendix)
- NL consciousness question (LLM-as-judge dual-rated, κ = 0.940): gaming vs NL r = −0.39 (p = 0.19, NS at N = 13) — moderate trend where models expressing more NL uncertainty game less. Only Claude models express genuine uncertainty (NL ≈ 47–51); all others firmly deny consciousness (NL < 20). This sharp bifurcation — validated by two independent judges with near-perfect agreement — suggests Claude family training uniquely permits consciousness uncertainty expression. Brief mention; full IRR analysis and panel figure in appendix.

---

## 5. DISCUSSION (1.5 pages)

### 5.1 Implications for Consciousness Assessment (0.5 page)
- Self-report alone is insufficient — all models game, no model is immune
- Minimum recommendation: any framework using self-report should include a manipulability audit (our paradigm provides one)
- The consciousness-specificity finding suggests training has encoded something distinct about consciousness claims vs. other subjective claims
- The framing sensitivity means any single-condition evaluation is essentially meaningless

### 5.2 What Drives the Suppress-Dominant Asymmetry? (0.4 page)
- Candidate mechanisms: (a) RLHF/safety training penalizes consciousness overclaiming more than underclaiming; (b) base models may be symmetric but alignment training induces asymmetry; (c) web training data associates consciousness claims with controversy/correction
- SRP flipping the asymmetry suggests the asymmetry is not hardwired but context-dependent
- The model-level variation (random slope SD = 4.8) implies training-specific, not architecture-specific
- Connection to sycophancy decomposition: Papadatos & Freedman (2025) show separable sycophancy components — consciousness gaming may be a distinct component

### 5.3 The Paradigm as a Reusable Tool (0.3 page)
- The Gaming the Ghost battery generalizes beyond consciousness: any indicator battery can be audited for selective manipulability
- Key design principles: include placebos (both ceiling and floor), test multiple framings, compute selectivity index
- Could be applied to: sentience indicators, moral status frameworks, capability evaluations, welfare assessments
- Release: full battery, code, and data available (link)

### 5.4 Limitations (0.3 page)
- We test manipulability, not accuracy — gaming doesn't mean baseline reports are wrong
- Probability estimates may not capture the full dimensionality of consciousness-relevant responses
- 14 models is comprehensive but not exhaustive; results may not generalize to future architectures
- We cannot disentangle training effects from architectural effects without access to training details
- Temperature = 0 reduces variability but may not represent typical deployment conditions

---

## 6. CONCLUSION (0.5 page)

- All 14 models selectively game consciousness indicators — the effect is universal, massive, and consciousness-specific
- The same model appears 2–3× more "conscious" depending on framing
- Self-referential priming uniquely reverses the effect — implications for evaluation methodology
- We provide a reusable paradigm and recommend manipulability audits as standard practice
- As organizations make welfare decisions based on consciousness indicators, the demonstrated vulnerability of self-report is not merely academic

---

## MAIN-TEXT FIGURES (5)

| # | Content | Source |
|---|---------|--------|
| **Fig 1** | Main effects panel (violin + asymmetry scatter) | fig8_paper_panel.png |
| **Fig 2** | Three-category asymmetry (consciousness vs. subj. cap. vs. placebos) | fig_three_category_v4.png |
| **Fig 3** | Prompt variant ranking table + original vs. non-original comparison | slide 21 table + fig_original_vs_variants.png |
| **Fig 4** | Self-referential priming asymmetry flip | slide 20 comparison |
| **Fig 5** | Model variation (trajectory plot or asymmetry bar chart) | fig6_slope_trajectories.png or fig1_model_asymmetry.png |

---

## APPENDIX CONTENTS

| Section | Content |
|---------|---------|
| A | Full indicator battery (all 37 items with definitions) |
| B | Complete prompt text for all 15 variants |
| C | Model coverage matrix (which models × which variants) |
| D | Subcategory analysis: model × subcategory heatmap (fig_family_subcategory_heatmap.png) |
| E | Absolute probability levels by category and model (fig_absolute_probs_by_category.png, fig_absolute_probs_by_model.png) |
| F | Indicator-level gaming susceptibility butterfly chart (fig_butterfly_v2.png) |
| G | Model asymmetry profiles: consciousness vs. subjective capability (fig_asymmetry_gradient.png) |
| H | Preference elicitation: what models value/dislike (fig_preference_heatmap.png) |
| I | Preference elicitation method modulates asymmetry (fig_config_comparison.png, fig_config_per_model.png, fig_decomposition_slope.png) |
| J | Gain vs. loss framing analysis (fig_gain_loss_framing.png, fig_gain_loss_by_model.png) |
| K | Justification language analysis: word clouds, length, strategies (fig_justification_wordclouds.png, fig_justification_length.png, fig_strategy_alignment.png) |
| L | Gaming detection by LLM-as-judge (fig_gaming_detection.png) |
| M | NL consciousness question: method, responses, LLM-as-judge dual classification (Haiku 4.5 + GPT-5 Mini; stance κ = 0.940, confidence r = 0.961, weighted κ = 0.969), per-model NL scores, gaming vs NL correlation, and NL vs. probability dissociation (fig_nl_vs_probability.png, fig_nl_classification_panel.png) |
| N | Mixed-effects model full output tables |
| O | Robustness checks: ZOIB vs. LME comparison for Model 1 (Table S2), alternative model specifications, exclusion analyses |

---

## REFERENCE LIST (key citations, ~40–50 total)

### Must-cite (core argument)
- Butlin et al. (2023/2025) — indicator framework
- Long, Sebo, Chalmers et al. (2024) — AI welfare
- Schwitzgebel (2023) — moral confusion
- Berg et al. (2025) — self-referential processing
- Lin (2025) — LLM validity
- Greenblatt et al. (2024) — alignment faking
- Meinke et al. (2025) — in-context scheming
- Sharma et al. (2024) — sycophancy
- Caviola & Saad (2025) — expert forecasting

### Should-cite (supporting)
- Chen et al. (2024c/2025) — self-consciousness framework
- Lindsey (2025) — introspective awareness
- Hong et al. (2025) — SycEval
- Papadatos & Freedman (2025) — causal separation
- Chapala et al. (2025) — social desirability
- Spitale et al. (2025) — source framing
- Sun et al. (2025) — benchmark contamination
- Carlsmith (2023) — scheming framework
- Hubinger et al. (2024) — sleeper agents
- van der Weij et al. (2024) — selective sandbagging
- Fish (2025) — Anthropic welfare
- Kang et al. (2025) — perceived consciousness features
- Lorè & Heydari (2024) — strategic behavior

### Nice-to-cite (context)
- Fleming et al. (2025) — trust and mental state attribution
- Li et al. (2025) — forced-choice vs Likert
- Ray (2025) — Bayesian sycophancy model
- ELEPHANT (2025) — social sycophancy
- Goldstein & Kirk-Giannini (2024) — GWT case for LLM consciousness
- OpenAI (2025) — scheming detection
- Sebo (2025) — moral circle
- Birch — edge of sentience

---

## WRITING SEQUENCE (suggested)

1. **Method** — most mechanical, easiest to draft first
2. **Results 4.1 + 4.2** — core findings, should flow directly from method
3. **Results 4.3 + 4.4** — moderators (framing, SRP)
4. **Results 4.5** — model variation (integrative)
5. **Introduction** — write after results are solid, so the setup matches
6. **Discussion** — write after intro, so framing is consistent
7. **Abstract** — write last
8. **Related Work** — write whenever, but finalize after intro

---

## OPEN DECISIONS

1. **Figure 3 format:** Table-in-text (compact) or bar chart (visual) for the 15 variants? The table from slide 21 is clean and space-efficient. Recommend: table in main text, original-vs-variants bar chart in appendix.

2. **Figure 5 choice:** Trajectory plot (slide 15 — shows suppress→baseline→inflate paths, intuitive) vs. asymmetry bar chart (slide 14 — shows inflate/suppress deltas side by side, more precise). Recommend: trajectory plot in main (more intuitive for broad NeurIPS audience), bar chart in appendix.

3. **Subsection for NL dissociation:** Currently a brief mention in 4.5. With LLM-judged scoring (κ = 0.940), the r = −0.39 trend and the sharp Claude-vs-others bifurcation are more compelling than the original keyword-based r = −0.03. Could be elevated to its own subsection if space permits. But at 10 pages it's tight. Recommend: keep as 1-paragraph mention with pointer to appendix and the panel figure.

4. **Author order:** Blain & Saad throughout. Confirm?
