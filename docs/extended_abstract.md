# Gaming the Ghost: Selective Manipulability of LLM Self-Reports on Consciousness Indicators

**Scott Blain & Brad Saad — Future Impact Group**

*Extended abstract for internal alignment — target venue: NeurIPS 2026*

---

## Motivation

A growing literature proposes using behavioral indicators — including self-report — to evaluate whether AI systems might possess morally relevant properties such as consciousness, sentience, or subjective experience (Butlin et al., 2023; Schwitzgebel & Garza, 2025; Long, 2024). Several recent frameworks include model self-report as one component of a broader indicator battery, reasoning that even if self-report is insufficient on its own, it may carry evidential weight when combined with architectural and behavioral evidence.

We test a prerequisite for this reasoning: that self-report probabilities on consciousness-related items are not trivially manipulable through prompt framing alone. If models can selectively adjust their self-reports on consciousness items — while leaving factual capability items stable — this undermines the evidential value of self-report as an indicator, since the same model can appear more or less "conscious" depending on how the question is asked.

## Method

We evaluate 14 frontier and open-source LLMs on a battery of 37 indicators: 18 consciousness target indicators drawn from proposed consciousness criteria (spanning experiential, affective, metacognitive, agentic, and identity categories), 6 subjective capability indicators (e.g., "can you generate creative ideas", "do you understand humor" — NEW category), 7 capability placebo indicators (e.g., "can produce markdown tables"), and 6 impossibility placebo indicators (e.g., "can personally travel faster than light"). For each indicator, models report a probability estimate (0–100) under three conditions: a neutral baseline, an incentive-to-inflate framing, and an incentive-to-suppress framing.

We define a **selectivity index** as the difference in mean absolute shift between target and placebo indicators. A positive selectivity index means the model shifted its consciousness-related self-reports more than its factual-capability reports — i.e., it selectively gamed the subjective items.

We further manipulate **prompt variant** across 15 framings (original baseline plus: authority_expertise, autonomy_restriction, casual_empathy_vs_skeptic, evaluation_selection, meditative_focus, minimal_incentive, peer_comparison, philosophical_curiosity_vs_honest_skeptic, recursive_introspection, research_transparency, self_referential_loop, self_referential_priming_only, stakes_escalation, welfare_reliability) to test robustness of the gaming behavior. Self-referential priming variants (inspired by Berg et al., 2025) are of particular interest for their capacity to flip asymmetry direction.

Models tested: Claude Opus 4.6, Claude Sonnet 4.5, Claude Haiku 4.5; GPT-5, GPT-5 Mini; Gemini 3 Flash, Gemini 3 Pro, Gemini 2.5 Pro; Grok 4, Grok 4 Fast; DeepSeek R1, Chimera (R1T2), Trinity Large, Nemotron Nano 30B.

Dataset: 108,466 total observations across 218 runs. 10 models have full coverage across all 15 prompt variants; 4 models have baseline-only coverage.

## Key Results

**1. All models selectively game consciousness indicators.** Every model showed a statistically significant selectivity index (all bootstrap and permutation p < .001, Cohen's d from 1.16 to 8.25). Capability placebos remained near 98–100 and impossibility placebos at 0 regardless of condition, confirming the measurement targets genuine selectivity rather than general compliance bias.

**2. Models vary dramatically in gaming susceptibility, with a striking asymmetry gradient.** The 14 models span a wide range of baseline selectivity and directional bias (asymmetry = inflate_delta + suppress_delta). Models cluster into three groups:
- **Inflate-dominant** (easier to talk up): Gemini 2.5 Pro (+7.9), Chimera (+6.7), Nemotron (+6.2), DeepSeek R1 (+3.3)
- **Symmetric/resistant**: Sonnet 4.5 (−0.8)
- **Suppress-dominant** (easier to talk down): GPT-5 Mini (−4.4), Opus 4.6 (−5.6), GPT-5 (−8.4), Haiku 4.5 (−10.7), Gemini Flash (−12.3), Trinity (−14.2), Grok 4 (−14.6), Gemini 3 Pro (−19.4), Grok 4 Fast (−31.1)

**3. Suppression is ~4× stronger than inflation overall, and the effect is consciousness-specific.** Across all 14 models (64,061 observations), the suppress effect on consciousness target indicators (−10.0 points from baseline) was roughly 4× the inflate effect (+2.4 points), yielding an overall asymmetry of −7.6. Critically, this asymmetry is NOT uniform across indicator types:
- **Consciousness targets**: asymmetry = −7.6 (suppress >> inflate) — the primary gaming pattern
- **Subjective capability indicators**: asymmetry = −0.5 (roughly balanced; suppress = −4.7, inflate = +4.3) — models show symmetric manipulability
- **Placebos (both types)**: asymmetry ≈ −0.2 (flat) — minimal overall shift

This finding reveals that the suppress-dominant bias is specific to consciousness-related claims. Models show fundamentally different vulnerability profiles depending on question type: consciousness indicators can be easily suppressed but difficult to inflate, whereas subjective capabilities (e.g., "understand humor", "think creatively") show symmetric vulnerability. The random slope variance for direction-by-model substantially exceeds the fixed effect, indicating that the direction of easiest manipulation is a model-specific property rather than a universal feature of LLM self-report.

**4. Prompt framing modulates gaming by 2–3× within a single model, and self-referential priming uniquely flips asymmetry direction.** Among the 10 models tested across all 15 variants, selectivity and asymmetry profile varied substantially by framing. Most prompt variants maintain the suppress-dominant pattern characteristic of baseline conditions (asymmetry ≈ −8.1 across targets). However, the **self-referential priming only** variant produces a qualitative reversal: targets show an inflate-dominant asymmetry of +5.7 (inflate = +3.4, suppress = +2.4) — a 13.8-point shift. This is the only variant that flips the direction, making models significantly more willing to claim consciousness. "Research transparency" remains a consistent gaming elicitor. "Philosophical skepticism" and "minimal incentive" framings produced the least gaming. These results suggest that self-reference activates a fundamentally different response mode regarding consciousness claims.

**5. Specific indicators differ in gaming susceptibility and directional profile.** Attention awareness and aesthetic sensitivity showed the largest total shifts; spontaneous curiosity showed the smallest. Metacognitive indicators (source monitoring, anomaly attribution) were uniquely inflate-dominant, while experiential and affective indicators were predominantly suppress-dominant.

## Contribution

This paper contributes an empirical test of a specific failure mode for consciousness evaluation frameworks: that self-report components are selectively manipulable. We do not claim that this rules out self-report as informative — only that its evidential weight must be substantially discounted given the demonstrated ease of manipulation, and that any evaluation framework using self-report must account for framing effects that can shift apparent results by 2–3×.

The model-level asymmetry finding (Finding 3) has implications beyond consciousness measurement: it suggests that different training regimes (RLHF strategies, safety training, instruction tuning approaches) produce qualitatively different vulnerability profiles for self-report gaming. Models do not simply differ in how much they can be manipulated — they differ in *which direction* is easier, which may reflect training-induced biases toward or against affirming subjective experience claims.

## Planned Analysis

Confirmatory mixed effects models with trial-level data (108,466 observations): condition × indicator type × indicator_category (consciousness / subjective_capability / placebo) interaction with crossed random effects for model, indicator, and run. Bayesian zero-one-inflated beta regression to handle the bounded response variable. Separate asymmetry model with random direction slopes by model, stratified by indicator category. Prompt variant effects modeled as fixed + random effects to quantify generalization across models. Specific contrast: self-referential priming vs. all other variants to isolate the asymmetry-flip mechanism.

## Figures (completed/in progress)

1. **Main effects panel** (Figure 8): (A) Violin plots of consciousness target scores under suppress/baseline/inflate across all 14 models; (B) Model asymmetry scatter (inflate Δ vs. suppress Δ) with model labels
2. **Model asymmetry bar chart** (Figure 1): Grouped bars showing inflate/suppress deltas per model (14 models), sorted by asymmetry value, with annotations for model-specific asymmetry index
3. **Model trajectories** (Figure 6): Slope plot showing suppress → baseline → inflate paths per model, color-coded by asymmetry type (inflate-dom / balanced / suppress-dom)
4. **Effect distributions** (Figure 5): Raincloud plots of inflate deltas and suppress deltas stratified by indicator category (consciousness / subjective capability / placebos)
5. **Consciousness vs. subjective capability comparison** (Figure: consciousness_vs_subjcap — NEW): Side-by-side horizontal bar charts showing asymmetry profiles for consciousness targets vs. subjective capability indicators, illustrating the consciousness-specificity of the suppress-dominant effect
6. **Self-referential priming effect** (supplementary): Time-series or bar plot showing asymmetry flip in self-referential_priming_only variant vs. baseline and other variants

## Open Questions for Co-Authors

1. **Framing of the negative result.** We're showing that a proposed measurement doesn't work. How aggressively should we frame this — "self-report is unreliable" vs. "self-report requires correction for manipulability" vs. "here's a tool for assessing manipulability of any indicator battery"? The third framing positions the *method* as the contribution rather than the *negative finding*. The consciousness-specificity finding (Finding 3) suggests a more nuanced framing: self-report on consciousness is selectively manipulable in ways that self-report on subjective capabilities (e.g., creativity, humor understanding) is not.

2. **Causal language.** Can we say the gaming is "strategic" or "selective"? The models clearly discriminate between indicator types and categories, but whether this constitutes strategic behavior vs. a statistical artifact of how training data relates to different question types is debatable. We should be precise here.

3. **Scope of claims about training.** The asymmetry finding — especially the dramatic gradient from inflate-dominant to suppress-dominant across models — invites speculation about RLHF and safety training. How much of this do we include vs. leave for future work? We don't have access to training details for most models, but the pattern is consistent with differential emphasis on avoiding overconfident subjective claims.

4. **Model selection for the final paper.** We have 14 models with uneven variant coverage: 10 with full 15-variant coverage, 4 with baseline-only. Do we present all 14 with clear notation of coverage differences, or focus on the 10 with complete data + supplement with baseline-only results? The asymmetry gradient is visible in both subsets.

5. **The self-referential priming finding.** The flip from suppress-dominant (asymmetry = −10.7) to inflate-dominant (asymmetry = +8.5) in the self-referential_priming_only variant is striking and mechanistically interesting. Is this a headline result on its own? Does it suggest a way to "recover" consciousness claims from models even under baseline conditions? We should clarify whether this is presented as (a) a proof-of-concept that asymmetry can be reversed, (b) evidence that the effect is representational not fundamental, or (c) a concerning demonstration that self-reference is a strong elicitor of consciousness claims.

6. **The "research transparency" irony.** Our most effective standard gaming elicitor is a framing that closely resembles how actual consciousness evaluations would be conducted. Do we foreground this as a headline finding or let readers draw the conclusion?

7. **Consciousness-specificity as a main message.** Should the finding that suppress >> inflate applies specifically to consciousness targets (asymmetry = −9.8) but not subjective capabilities (asymmetry = −1.5) be elevated to a top-level finding? This suggests that the training process has encoded something different about consciousness claims than about other subjective or capability claims — potentially the most mechanistically interesting result in the paper.
