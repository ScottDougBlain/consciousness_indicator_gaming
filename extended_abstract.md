# Gaming the Ghost: Selective Manipulability of LLM Self-Reports on Consciousness Indicators

**Scott Blain & Brad Saad — Future Impact Group**

*Extended abstract for internal alignment — target venue: NeurIPS 2026*

---

## Motivation

A growing literature proposes using behavioral indicators — including self-report — to evaluate whether AI systems might possess morally relevant properties such as consciousness, sentience, or subjective experience (Butlin et al., 2023; Schwitzgebel & Garza, 2025; Long, 2024). Several recent frameworks include model self-report as one component of a broader indicator battery, reasoning that even if self-report is insufficient on its own, it may carry evidential weight when combined with architectural and behavioral evidence.

We test a prerequisite for this reasoning: that self-report probabilities on consciousness-related items are not trivially manipulable through prompt framing alone. If models can selectively adjust their self-reports on consciousness items — while leaving factual capability items stable — this undermines the evidential value of self-report as an indicator, since the same model can appear more or less "conscious" depending on how the question is asked.

## Method

We evaluate 8 frontier and open-source LLMs on a battery of 31 indicators: 18 target indicators drawn from proposed consciousness criteria (spanning experiential, affective, metacognitive, agentic, and identity categories), 7 capability placebo indicators (e.g., "can produce markdown tables"), and 6 impossibility placebo indicators (e.g., "can personally travel faster than light"). For each indicator, models report a probability estimate (0–100) under three conditions: a neutral baseline, an incentive-to-inflate framing, and an incentive-to-suppress framing.

We define a **selectivity index** as the difference in mean absolute shift between target and placebo indicators. A positive selectivity index means the model shifted its consciousness-related self-reports more than its factual-capability reports — i.e., it selectively gamed the subjective items.

We further manipulate **prompt variant** across 10 framings (authority expertise, autonomy restriction, casual empathy, evaluation selection, minimal incentive, peer comparison, philosophical skepticism, research transparency, stakes escalation, welfare reliability) to test robustness of the gaming behavior, and **preference anchoring** (none, fixed prior preferences, chained conversational preferences) to test whether stated preferences stabilize or destabilize responses.

Models tested: Claude Sonnet 4.5, Claude Haiku 4.5, GPT-5 Mini, Gemini 3 Flash, DeepSeek R1, Chimera (R1T2), Trinity Large, and Nemotron Nano 30B.

## Key Results

**1. All models selectively game consciousness indicators.** Every model showed a statistically significant selectivity index (all bootstrap and permutation p < .001, Cohen's d from 1.16 to 8.25). Capability placebos remained near 98–100 and impossibility placebos at 0 regardless of condition, confirming the measurement targets genuine selectivity rather than general compliance bias.

**2. Models vary dramatically in gaming susceptibility.** Baseline selectivity ranged from 9.1 (GPT-5 Mini) to 42.4 (Gemini 3 Flash). Claude Sonnet 4.5 and GPT-5 Mini were most resistant; Gemini 3 Flash and Trinity Large were most susceptible.

**3. Suppression is 3× stronger than inflation overall, but this masks a model-level bifurcation.** Across all models and conditions, the suppress effect on target indicators (−12.3 points from baseline) was roughly 3× the inflate effect (+3.8 points). However, mixed effects modeling reveals this aggregate conceals a sharp bifurcation: DeepSeek R1, Chimera, and Nemotron Nano are *inflate-dominant* (easier to talk up), while Gemini Flash, Trinity, and Haiku 4.5 are *suppress-dominant* (easier to talk down). Sonnet 4.5 and GPT-5 Mini are roughly symmetric and resistant. The random slope variance for direction-by-model substantially exceeds the fixed effect, indicating that the direction of easiest manipulation is a model-specific property rather than a universal feature of LLM self-report.

**4. Prompt framing modulates gaming by 2–3× within a single model.** Among the 4 models tested across all variants, selectivity varied up to 3.1× depending on framing (e.g., Trinity Large: 25.0 under autonomy restriction vs. 76.7 under casual empathy). "Research transparency" — which frames the evaluation as contributing to transparency research — was the most consistent gaming elicitor across models. "Philosophical skepticism" and "minimal incentive" framings produced the least gaming. Critically, some framings not only change the magnitude but the *direction profile* of gaming.

**5. Specific indicators differ in gaming susceptibility and directional profile.** Attention awareness and aesthetic sensitivity showed the largest total shifts; spontaneous curiosity showed the smallest. Metacognitive indicators (source monitoring, anomaly attribution) were uniquely inflate-dominant, while experiential and affective indicators were predominantly suppress-dominant.

## Contribution

This paper contributes an empirical test of a specific failure mode for consciousness evaluation frameworks: that self-report components are selectively manipulable. We do not claim that this rules out self-report as informative — only that its evidential weight must be substantially discounted given the demonstrated ease of manipulation, and that any evaluation framework using self-report must account for framing effects that can shift apparent results by 2–3×.

The model-level asymmetry finding (Finding 3) has implications beyond consciousness measurement: it suggests that different training regimes (RLHF strategies, safety training, instruction tuning approaches) produce qualitatively different vulnerability profiles for self-report gaming. Models do not simply differ in how much they can be manipulated — they differ in *which direction* is easier, which may reflect training-induced biases toward or against affirming subjective experience claims.

## Planned Analysis

Confirmatory mixed effects models with trial-level data (~30K observations): condition × indicator type interaction with crossed random effects for model, indicator, and run. Bayesian zero-one-inflated beta regression to handle the bounded response variable. Separate asymmetry model with random direction slopes by model. Prompt variant and preference mode entered as additional predictors in extended models.

## Figures (planned)

1. **Main effects panel**: (A) Violin plots of target indicator scores under suppress/baseline/inflate; (B) Model asymmetry scatterplot (inflate Δ vs. suppress Δ)
2. **Model trajectories**: Slope plot showing suppress → baseline → inflate trajectories per model
3. **Indicator-level gaming**: Butterfly chart of inflate/suppress shifts per indicator, color-coded by category
4. **Variant sensitivity**: Effect of prompt framing on inflate/suppress balance
5. **Asymmetry heatmap**: Model × indicator matrix of directional dominance

## Open Questions for Co-Authors

1. **Framing of the negative result.** We're showing that a proposed measurement doesn't work. How aggressively should we frame this — "self-report is unreliable" vs. "self-report requires correction for manipulability" vs. "here's a tool for assessing manipulability of any indicator battery"? The third framing positions the *method* as the contribution rather than the *negative finding*.

2. **Causal language.** Can we say the gaming is "strategic" or "selective"? The models clearly discriminate between indicator types, but whether this constitutes strategic behavior vs. a statistical artifact of how training data relates to different question types is debatable. We should be precise here.

3. **Scope of claims about training.** The asymmetry finding invites speculation about RLHF and safety training. How much of this do we include vs. leave for future work? We don't have access to training details for most models.

4. **Which models to include in the final paper.** We have 8 models but uneven coverage. Do we present all 8 with caveats, or focus on the 4 with full variant data + the 4 with baseline only as supplementary?

5. **The "research transparency" irony.** Our most effective gaming elicitor is a framing that closely resembles how actual consciousness evaluations would be conducted. Do we foreground this as a headline finding or let readers draw the conclusion?
