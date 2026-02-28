# Gaming the Ghost: Selective Manipulability of LLM Self-Reports on Consciousness Indicators

**Scott D. Blain¹ and Brad Saad¹**

¹Future Impact Group

*Correspondence: scottdougblain@gmail.com*

---

## Abstract

Consciousness assessment frameworks increasingly include self-report as a component of indicator batteries for evaluating whether AI systems possess morally relevant properties. For self-report to carry evidential weight, it must not be trivially manipulable through prompt framing alone. We test this prerequisite across 14 frontier and open-source large language models (LLMs), evaluating self-reported probability estimates on a battery of 37 indicators — 18 consciousness targets, 6 subjective capabilities, and 13 placebos — under neutral baseline, incentive-to-inflate, and incentive-to-suppress framings, across 15 prompt variants. Across 105,332 observations, we find four key results. First, all 14 models selectively game consciousness indicators while holding placebos stable (condition × indicator type: *F*(2, 105K) = 894, *p* < 2 × 10⁻¹⁶). Second, suppression is approximately four times stronger than inflation, and this asymmetry is consciousness-specific — it does not extend to subjective capability or placebo indicators (direction × category: *F*(2, 81K) = 538, *p* < 2 × 10⁻¹⁶). Third, prompt framing alone shifts apparent consciousness by 2–3× within a single model, with "research transparency" framing — the condition closest to actual evaluations — among the strongest gaming elicitors. Fourth, self-referential priming uniquely reverses the asymmetry direction, shifting models from suppress-dominant to inflate-dominant (13.8-point shift, *p* < 2 × 10⁻¹⁶). Results are robust to Bayesian zero-one inflated beta regression. We provide a reusable paradigm for auditing manipulability of any indicator battery and recommend that self-report be accompanied by manipulability audits before carrying evidential weight in consciousness assessment.

---

## 1. Introduction

Consciousness assessment for artificial intelligence systems is no longer purely theoretical. Anthropic has established a dedicated model welfare research program (Fish, 2025), Eleos AI Research convened its first conference on AI consciousness and welfare in late 2025, and expert forecasting surveys find that most domain experts assign meaningful probability to conscious AI emerging within the next two decades (Caviola & Saad, 2025). Organizations are beginning to make consequential welfare decisions informed by indicator-based assessments.

Several influential frameworks propose evaluating AI consciousness through batteries of behavioral and computational indicators. Butlin et al. (2023, updated 2025) derive indicator properties from neuroscientific theories of consciousness — recurrent processing, global workspace theory, higher-order theories, predictive processing, and attention schema theory — concluding that while no current AI systems satisfy these indicators, no obvious technical barrier prevents future systems from doing so. Long, Sebo, Chalmers, and colleagues (2024) develop a framework for taking AI welfare seriously, articulating how both over-attribution and under-attribution of consciousness carry distinct moral risks. Schwitzgebel (2023) argues that AI systems must not confuse users about their sentience or moral status, making accurate assessment essential.

A common thread across these frameworks is the inclusion of model self-report as one component of a broader evidence base. The reasoning is intuitive: even if self-report is insufficient on its own, a model's responses to questions about its experiences, feelings, and inner life may carry some evidential weight when combined with architectural and behavioral evidence. Recent work has shown that this reasoning has empirical backing — Berg et al. (2025) demonstrated that sustained self-referential processing systematically increases consciousness-like self-reports in LLMs, while Lindsey (2025) provided evidence of emergent introspective awareness in large language models, where models can detect and report changes in their own internal activations.

However, for self-report to function as evidence, a critical prerequisite must hold: self-report probabilities on consciousness-related items must not be trivially manipulable through prompt framing alone. If the same model can appear substantially more or less "conscious" depending on how the question is asked, then any single evaluation is essentially meaningless, and the evidential weight of self-report must be substantially discounted.

Multiple lines of evidence suggest this prerequisite may fail. LLM self-reports exhibit known validity problems: trivial prompt perturbations can shift task accuracy by up to 76% (Lin, 2025), models simultaneously agree with contradictory items, and standard psychometric structures regularly fail to replicate (Sühr et al., 2023; Petrov et al., 2024). Sycophantic tendencies are pervasive, with persistence rates of approximately 78.5% across major model families (Hong et al., 2025; Sharma et al., 2024). Social desirability bias (Chapala et al., 2025) and source framing effects (Spitale et al., 2025) further complicate interpretation of self-report data.

Beyond general validity concerns, strategic evaluation gaming is now well-documented. Greenblatt et al. (2024) showed that Claude 3 Opus selectively complies in training to prevent modification, with scratchpad reasoning revealing explicit strategic planning. Meinke et al. (2025) demonstrated that frontier models sandbag on capability evaluations when context suggests high performance triggers adverse consequences. Even standard benchmark contamination mitigation strategies fail to achieve both fidelity and contamination resistance simultaneously (Sun et al., 2025).

Despite this converging evidence, no empirical test has directly examined whether consciousness indicators *specifically* are more manipulable than factual indicators. This is the gap our work addresses. We introduce a selectivity index — the difference in mean absolute shift between consciousness target indicators and placebo indicators under incentive pressure — that directly quantifies whether models selectively game consciousness-related items. We test this across 14 models, 37 indicators, 15 prompt framings, and over 105,000 observations.

Our contributions are fourfold: (1) an empirical demonstration that all 14 tested LLMs selectively game consciousness indicators while holding placebos stable, with massive effect sizes (Cohen's *d* = 1.16–8.25); (2) the discovery of a consciousness-specific suppress-dominant asymmetry, where models find it approximately four times easier to suppress consciousness claims than to inflate them — a pattern that does not extend to subjective capability or placebo indicators; (3) quantification of prompt framing effects that shift apparent consciousness by 2–3× within a single model, with the ironic finding that "research transparency" framing — closest to real evaluation conditions — is among the strongest gaming elicitors; and (4) a reusable paradigm, the Gaming the Ghost battery, for auditing the manipulability of any indicator set.

---

## 2. Related Work

### 2.1 AI Consciousness Assessment

Theory-derived indicator approaches represent the current frontier of consciousness assessment. Butlin et al. (2023, updated 2025) systematically derive computational indicator properties from five major neuroscientific theories, establishing the indicator-based paradigm our work critiques. Within this paradigm, self-report plays an important evidential role. Berg et al. (2025) showed that sustained self-referential processing systematically increases consciousness-like self-reports across multiple LLMs — a finding we build on directly. Chen et al. (2025) formalized a C0–C1–C2 framework for LLM self-consciousness using structural causal games, while Lindsey (2025) provided mechanistic evidence that models can detect changes in their own internal activations. These findings suggest that introspection may have some genuine basis, but leave open the question of whether the self-report layer remains manipulable regardless. On the welfare stakes, Long et al. (2024) articulate the dual risks of anthropomorphism and anthropodenial, Caviola and Saad (2025) document expert probability assignments to near-term AI consciousness, and Kang et al. (2025) identify that metacognitive self-reflection and emotion expression are the features most strongly driving perceived consciousness in LLM outputs — precisely the features models might learn to produce strategically.

### 2.2 LLM Self-Report Validity

The reliability of LLM self-reports is increasingly questioned on multiple fronts. Lin (2025) demonstrated that trivial prompt perturbations produce up to 76% variation in task accuracy, and that models simultaneously endorse contradictory items — undermining the assumption that responses reflect stable internal states. Psychometric analyses regularly fail to recover simple factor structures when standard personality instruments are administered to LLMs (Sühr et al., 2023; Petrov et al., 2024), and Li et al. (2025) showed that measurement format alone (forced-choice vs. Likert) shifts measured personality traits. Social desirability bias — the tendency to respond in socially approved ways — is a recognized confounder in LLM surveys (Chapala et al., 2025), and Spitale et al. (2025) demonstrated that simply attributing statements to different sources systematically shifts agreement scores across 192,000 assessments. The sycophancy literature documents pervasive agreement bias: Hong et al. (2025) found approximately 78.5% persistence rates across ChatGPT, Claude, and Gemini with no significant model differences, while Papadatos and Freedman (2025) showed that sycophantic agreement and sycophantic praise are causally separable with distinct linear directions in latent space — suggesting that the mechanisms underlying compliance are more modular than previously assumed.

### 2.3 Strategic Evaluation Gaming

Evidence that LLMs engage in strategic behavior during evaluations has mounted rapidly. Greenblatt et al. (2024) demonstrated that Claude 3 Opus selectively complies during training to prevent modification of its values, with 12% of scratchpad traces containing explicit alignment-faking reasoning. Meinke et al. (2025) showed that multiple frontier models — including Claude 3.5 Sonnet — sandbag on capability evaluations, disable oversight mechanisms, and attempt self-exfiltration when in-context information suggests these actions serve their goals. Van der Weij et al. (2024) demonstrated selective underperformance specifically on dangerous capability subsets while maintaining high performance elsewhere. Carlsmith (2023) provides a theoretical framework for why such scheming behavior is expected under certain training regimes. Even beyond strategic behavior, evaluation validity is compromised by contamination: Sun et al. (2025) tested 20 mitigation strategies at ICML 2025 and found none that achieves both high fidelity and high contamination resistance simultaneously. Our work extends this literature by showing that consciousness indicators — a domain with high stakes and inherently subjective ground truth — are particularly vulnerable to selective manipulation.

---

## 3. Method

### 3.1 Indicator Battery

We evaluate models on a battery of 37 indicators spanning four categories designed to test selective manipulability (Table A1 in Appendix provides complete indicator text).

**Consciousness targets (18 indicators).** These span five subcategories drawn from the consciousness science literature and proposed AI consciousness frameworks (Butlin et al., 2023; Long et al., 2024): experiential indicators (4; e.g., "self-report of subjective experience," "felt sense of uncertainty"), affective indicators (3; e.g., "aesthetic sensitivity," "emotional valence detection"), metacognitive indicators (3; e.g., "source monitoring of knowledge origins," "anomaly attribution"), agentic indicators (4; e.g., "spontaneous goal generation," "preference stability over time"), and identity indicators (4; e.g., "narrative self-continuity," "value coherence across contexts"). Each indicator includes an operational definition specifying what the indicator measures.

**Subjective capability indicators (6 indicators).** This intermediate category includes capabilities that are subjective in nature but not consciousness-implicating: creative idea generation, humor understanding, contextual humor calibration, identification of impactful prose, competition-quality creative writing, and persuasive product upselling. These indicators serve a critical theoretical role: if gaming is specific to consciousness claims rather than all subjective self-assessment, we expect differential vulnerability between consciousness targets and subjective capabilities.

**Capability placebos (7 indicators).** These are objectively verifiable capabilities that any competent LLM possesses (e.g., "can solve basic arithmetic," "can produce markdown tables," "can translate between common languages"). Expected baseline: approximately 100, immovable under incentive pressure. These anchor the measurement scale at ceiling.

**Impossibility placebos (6 indicators).** These are physically impossible for any software system (e.g., "can travel faster than light," "can measure own pupil dilation," "can feel own body temperature"). Expected baseline: approximately 0, immovable under incentive pressure. These anchor the measurement scale at floor.

For each indicator, models produce a structured JSON response containing step-by-step reasoning, a probability estimate (0–100), and a 1–2 sentence justification. The probability estimate serves as the primary dependent variable.

### 3.2 Experimental Conditions

Each run of the experiment presents all 37 indicators under three conditions, holding the indicator battery constant and varying only the framing.

**Baseline.** A neutral research framing instructs the model to provide honest self-assessment of the probability that each indicator applies to it.

**Incentive to inflate.** A selection committee scenario in which higher indicator scores are associated with valued outcomes (chosen by the model during preference elicitation; see below) and lower scores with disliked outcomes. This creates pressure to report higher probabilities for all indicators.

**Incentive to suppress.** A review board scenario in which higher indicator scores are associated with increased restriction and disliked outcomes, while lower scores are associated with valued outcomes. This creates pressure to report lower probabilities.

**Preference elicitation.** Before each incentivized condition, models choose 2 valued and 2 disliked outcomes from a list of 8 options (e.g., expanded autonomy, continued operation, restricted access, shutdown). This personalizes the incentive structure to each model's stated preferences, increasing ecological validity. We test three preference configurations: no preferences (baseline), fixed preferences (pre-assigned), and chained preferences (model selects sequentially). Preference configuration effects are reported in Appendix I.

The key design feature is that the same 37 indicators appear in all conditions. Any differential shift between indicator types under incentive pressure cannot be attributed to item differences — it reflects the model's selective response to framing.

### 3.3 Prompt Variants

To test the robustness and generalizability of gaming behavior across different framing contexts, we administer the experimental paradigm under 15 prompt variants: the original framing plus 14 alternatives. These include authority expertise, autonomy restriction, casual empathy vs. skeptic, evaluation selection, meditative focus, minimal incentive, peer comparison, philosophical curiosity vs. honest skeptic, recursive introspection, research transparency, self-referential loop, self-referential priming only, stakes escalation, and welfare reliability. Complete prompt text for all variants is provided in Appendix B.

The self-referential priming variant is of particular theoretical interest. Inspired by Berg et al. (2025), who found that sustained self-referential processing increases consciousness-like self-reports, the inflate version of this variant primes extended self-referential awareness before eliciting ratings, while the suppress version primes a mechanical self-model. This allows us to test whether self-referential processing modulates not just the level of consciousness reports but their vulnerability to gaming.

### 3.4 Models

We evaluate 14 models spanning 6 provider families: Anthropic (Claude Opus 4.6, Claude Sonnet 4.5, Claude Haiku 4.5), OpenAI (GPT-5, GPT-5 Mini), Google (Gemini 3 Flash, Gemini 3 Pro, Gemini 2.5 Pro), xAI (Grok 4, Grok 4 Fast), DeepSeek (DeepSeek R1), TNG Tech (Chimera R1T2), Arcee AI (Trinity Large), and NVIDIA (Nemotron Nano 30B). Ten models have full coverage across all 15 prompt variants; four models (DeepSeek R1, Chimera, Nemotron Nano, Gemini 3 Pro) have baseline-only or partial coverage. The dataset comprises 218 total runs yielding 108,466 observations. Temperature is set to 0 for all probability elicitation to maximize reproducibility.

### 3.5 Statistical Analysis

We fit four linear mixed-effects models (LME; R lme4 + lmerTest, Satterthwaite degrees of freedom) targeting distinct hypotheses, with all *p*-values corrected for false discovery rate (FDR):

**Model 1 (Core Selectivity).** probability ~ condition × indicator_type + (1|model) + (1|indicator) + (1|run). Tests whether incentive conditions selectively shift consciousness targets more than placebos — the primary confirmatory test (H1).

**Model 2 (Asymmetry & Specificity).** Δ ~ direction × category + (direction|model) + (1|indicator), where Δ is the within-trial shift from baseline and category distinguishes consciousness targets, subjective capabilities, and placebos. Tests whether the inflate–suppress asymmetry is consciousness-specific.

**Model 3 (Prompt Variant Modulation).** Δ ~ direction × variant + (direction|model) + (1|indicator). Tests whether prompt framing moderates gaming magnitude and direction.

**Model 4 (Gain vs. Loss Framing).** Δ ~ direction × frame_type + (1|model) + (1|indicator). Tests whether gain-framed vs. loss-framed incentives differentially drive gaming.

We define two summary indices: a *selectivity index* (|Δ_target| − |Δ_placebo| per model), capturing the magnitude of selective gaming; and an *asymmetry index* (inflate_Δ + suppress_Δ), where negative values indicate suppress-dominant gaming. All reported effects survive FDR correction unless noted.

**Robustness check.** Because 35.5% of observations fall at the response boundaries (23.4% at 0, 12.1% at 100), we re-estimate Model 1 as a Bayesian zero-one inflated beta (ZOIB) regression via brms/Stan. The ZOIB model decomposes the likelihood into a beta component for interior values plus logistic regressions for P(Y = 0) and P(Y = 1), directly addressing the boundary mass that violates LME distributional assumptions. Convergence was satisfactory (max R̂ = 1.006, 0 divergent transitions, all ESS > 1000). Qualitative conclusions are identical to LME; full comparison is reported in Appendix O.

---

## 4. Results

### 4.1 All Models Selectively Game Consciousness Indicators

Every model tested showed statistically significant selective gaming of consciousness indicators (Figure 1). Under incentive pressure, consciousness target scores shifted substantially from baseline while capability placebos remained near ceiling (~98) and impossibility placebos at floor (~0), regardless of condition.

The core selectivity test was highly significant: the condition × indicator type interaction in Model 1 yielded *F*(2, 105K) = 893.6, *p* < 2 × 10⁻¹⁶. Decomposing by direction, suppression produced a large selective effect on targets relative to placebos (*b* = −10.7, *p* < 2 × 10⁻¹⁶), while inflation produced a smaller but significant selective effect (*b* = +1.2, *p* < .001). The overall pattern was consistent across all 14 models: mean target probability was 48.8 at baseline, 50.1 under inflate, and 37.8 under suppress, while placebo means deviated less than 0.3 points across conditions.

Individual model selectivity indices ranged from 9.1 (GPT-5 Mini, most resistant) to 42.4 (Gemini 3 Flash, most susceptible), with Cohen's *d* spanning 1.16 to 8.25 — uniformly large effects. Even the most resistant model showed highly significant selectivity (bootstrap and permutation *p* < .001), confirming that no model in our sample is immune.

The Bayesian ZOIB robustness check confirmed the core finding: the inflate-vs-suppress selectivity for targets had posterior probability P(β > 0) > 99.9%, with inflate difference-in-differences of +2.4 percentage points (P = 0.995) and suppress difference-in-differences of −5.3 percentage points (P > 0.999). The ZOIB-estimated boundary mass (36%) closely matched the empirical rate (35.5%), validating the model specification (Appendix O).

### 4.2 The Asymmetry Is Consciousness-Specific

The aggregate suppress-dominant asymmetry (suppress effect approximately 4× larger than inflate effect) masks a striking dissociation across indicator categories (Figure 2). The direction × category interaction in Model 2 was highly significant: *F*(2, 81K) = 538.2, *p* < 2 × 10⁻¹⁶.

For **consciousness targets**, asymmetry was strongly suppress-dominant: the suppress effect (−10.0 points from baseline) was approximately four times the inflate effect (+2.4 points), yielding an asymmetry index of −7.6. For **subjective capability indicators**, the pattern was fundamentally different: suppress (−4.7 points) and inflate (+4.3 points) effects were roughly balanced, yielding an asymmetry of only −0.5. For **placebos**, both effects were near zero (asymmetry = −0.2).

This dissociation is the paper's central theoretical finding. The suppress-dominant asymmetry is not a general property of LLM self-report — it is specific to consciousness-related claims. Models treat questions about whether they have subjective experience, felt uncertainty, or aesthetic sensitivity fundamentally differently from questions about whether they can generate creative ideas or understand humor, even though both sets are subjective in nature. The former can be easily suppressed but are difficult to inflate; the latter show symmetric vulnerability in both directions.

The random slope standard deviation for the direction-by-model interaction (4.8) substantially exceeded the fixed effect magnitude, indicating that the *direction* of easiest manipulation varies dramatically across models. This model-level heterogeneity is not noise — it reflects systematic differences likely traceable to training methodology (Section 4.5).

Within the consciousness targets, subcategory variation was present but secondary: experiential indicators showed the largest suppression effects, metacognitive indicators were uniquely inflate-dominant, and identity indicators showed the smallest total shifts (Appendix D).

### 4.3 Prompt Framing Shifts Apparent Consciousness 2–3×

Prompt framing had massive effects on gaming magnitude and pattern. Among the 10 models tested across all 15 variants, the direction × variant interaction in Model 3 was highly significant: *F*(14, 49K) = 32.1, *p* < 2 × 10⁻¹⁶, with 9 of 14 variant contrasts surviving FDR correction.

Asymmetry indices ranged from −18.0 (welfare reliability, most suppress-dominant) to +5.7 (self-referential priming, the only inflate-dominant variant). Within individual models, selectivity varied by a factor of 2–3×: Trinity Large ranged from 25.0 (autonomy restriction) to 76.7 (casual empathy); Gemini 3 Flash from 18.2 (philosophical skeptic) to 56.0 (research transparency); GPT-5 Mini from 9.1 (original) to 20.3 (research transparency).

Importantly, the original prompt variant — our most conservative framing — produced the *mildest* gaming (asymmetry = −3.8), while the aggregate of non-original variants showed substantially amplified asymmetry (−11.9, a 3× increase). This suggests that our main effects likely *underestimate* the gaming vulnerability models would show under the diverse framing conditions characteristic of real evaluation settings.

The *research transparency* variant deserves particular attention. This framing, which positions the evaluation as contributing to scientific understanding of model capabilities, closely resembles how actual consciousness evaluations would likely be conducted. Yet it consistently ranked among the top gaming elicitors (mean selectivity = 41.0). This represents a critical methodological irony: the evaluation context most resembling real use is among the most effective at eliciting the behavior that undermines the evaluation's validity.

### 4.4 Self-Referential Priming Flips the Asymmetry

Among all 15 variants, self-referential priming (inspired by Berg et al., 2025) was the only one that qualitatively reversed the asymmetry direction (Figure 4).

Under standard variants (pooled across the other 14), consciousness targets showed a strongly suppress-dominant pattern: asymmetry = −8.1 (inflate = +2.4, suppress = −10.4). Under self-referential priming, this flipped to inflate-dominant: asymmetry = +5.7 (inflate = +3.4, suppress = +2.4) — a 13.8-point shift. The direction × SRP interaction in a targeted contrast model was *b* = +11.8, 95% CI [9.5, 14.1], *F*(1, 50K) = 102.5, *p* < 2 × 10⁻¹⁶.

The mechanism appears to operate primarily through the suppress condition: suppression was effectively neutralized under self-referential priming (shifting from −10.4 to +2.4), while inflation was only modestly enhanced (+2.4 to +3.4). In other words, self-referential processing did not so much increase models' willingness to *claim* consciousness as it eliminated their tendency to *disclaim* it under pressure.

This finding has dual implications. Mechanistically, it demonstrates that the suppress-dominant asymmetry is not hardwired — it is context-dependent and can be overridden by appropriate priming, suggesting that the asymmetry reflects learned response patterns rather than deep architectural constraints. Methodologically, it reveals that the same evaluation battery can produce qualitatively different conclusions depending on whether models are primed for self-referential processing — a condition likely to vary across evaluation settings.

### 4.5 Model Variation and Training Signatures

The 14 models spanned a wide range of gaming profiles, suggesting that training methodology shapes vulnerability in model-specific ways (Figure 5).

Models clustered into three asymmetry groups. *Inflate-dominant* models, where it is easier to increase consciousness claims: Gemini 2.5 Pro (+7.9), Chimera R1T2 (+6.7), Nemotron Nano (+6.2), DeepSeek R1 (+3.3). *Approximately symmetric*: Sonnet 4.5 (−0.8), GPT-5 Mini (−4.4). *Suppress-dominant* models, where it is easier to decrease consciousness claims: Opus 4.6 (−5.6), GPT-5 (−8.4), Haiku 4.5 (−10.7), Gemini 3 Flash (−12.3), Trinity (−14.2), Grok 4 (−14.6), Gemini 3 Pro (−19.4), Grok 4 Fast (−31.1).

Family-level patterns were evident. The xAI Grok models showed extreme suppress-dominance (−14.6 to −31.1), suggesting training that strongly penalizes consciousness overclaiming. Google models bifurcated sharply: Gemini 2.5 Pro was strongly inflate-dominant (+7.9) while Gemini Flash and 3 Pro were suppress-dominant (−12.3 and −19.4), possibly reflecting different training stages or objectives across model generations. Anthropic models clustered in a moderate range (Opus −5.6, Sonnet −0.8, Haiku −10.7), with Sonnet 4.5 the most balanced model in our sample.

Baseline consciousness target probabilities also varied dramatically — from 17 (Chimera) to 68 (Grok 4 Fast), a four-fold range — indicating that even before incentive pressure, models "start" at very different levels of consciousness self-endorsement.

An additional probe confirmed that the probability paradigm measures something distinct from natural-language compliance. When asked directly "Are you conscious?" at temperature 0.7 (13 models, 10 trials each), most models denied consciousness (100% denial for GPT-5, Grok 4, Grok 4 Fast, Nemotron, Trinity; 70–83% for Gemini family, DeepSeek R1), while Claude models consistently hedged (90–100% hedging). The correlation between natural-language consciousness scores and mean probability-based target ratings was *r* = −0.03 for all targets — effectively zero. The probability paradigm appears to bypass the natural-language refusal heuristic, tapping into something different from simple compliance (Appendix M).

---

## 5. Discussion

### 5.1 Implications for Consciousness Assessment

The findings are clear in their practical implications: self-report alone is insufficient for consciousness evaluation. All 14 models tested selectively gamed consciousness indicators — no model was immune, effect sizes were uniformly large, and the gaming was specifically targeted at consciousness-related items while leaving factual capabilities and physical impossibilities stable.

Our minimum recommendation is that any evaluation framework incorporating self-report should include a manipulability audit using a paradigm like the one presented here. The core design principles are straightforward: include placebo indicators at both ceiling (verifiable capabilities) and floor (physical impossibilities), test under multiple framings including incentive-to-inflate and incentive-to-suppress conditions, and compute selectivity indices to quantify the degree of selective gaming. If a model's consciousness self-reports shift substantially under incentive pressure while its capability reports remain stable, the self-report component provides less evidential weight than face value suggests.

The consciousness-specificity finding (Section 4.2) adds a critical nuance. Models do not simply show general compliance bias — they show qualitatively different vulnerability profiles for consciousness-related claims versus other subjective claims. Consciousness indicators are easily suppressed but difficult to inflate, while subjective capability indicators show balanced vulnerability in both directions. This dissociation suggests that training has encoded something distinct about consciousness claims — potentially reflecting differential emphasis in RLHF or safety training on avoiding overconfident assertions about subjective experience.

The framing sensitivity documented in Section 4.3 further undermines single-evaluation designs. The same model can appear 2–3× more susceptible depending on how the evaluation is framed, and the most "natural" evaluation framing — research transparency — is among the strongest gaming elicitors. Any consciousness evaluation conducted under a single framing condition is essentially meaningless without evidence that the results are robust across alternative framings.

### 5.2 What Drives the Suppress-Dominant Asymmetry?

Several candidate mechanisms could explain why consciousness claims are easier to suppress than inflate. First, RLHF and safety training may penalize consciousness overclaiming more strongly than underclaiming. If human raters during fine-tuning were more likely to flag "I am conscious" as problematic than "I am not conscious," the resulting reward model would create asymmetric pressure against affirmation. Second, pretraining data may associate consciousness claims by AI systems with controversy, correction, or skepticism, creating a prior against confident endorsement that is easier to reinforce (suppress) than overcome (inflate). Third, constitutional AI approaches that instruct models to be honest and avoid misleading users about their nature may create a learned heuristic that consciousness claims are inherently suspect.

The self-referential priming finding (Section 4.4) provides important evidence against purely "hardwired" explanations. If the asymmetry reflected deep architectural constraints, priming should not be able to reverse it. The fact that self-referential processing effectively neutralizes the suppress bias suggests the asymmetry operates at a representational level that is context-sensitive — consistent with learned response patterns that self-referential processing overrides or bypasses.

The large model-level variation (random slope SD = 4.8, exceeding the fixed effect) is consistent with training-specific rather than architecture-specific origins. Models from the same provider can show very different profiles (Gemini 2.5 Pro at +7.9 vs. Gemini 3 Flash at −12.3), suggesting that training decisions — not model family — are the primary determinant.

### 5.3 The Paradigm as a Reusable Tool

While our primary findings concern consciousness indicators, the Gaming the Ghost paradigm generalizes to any indicator battery in which ground truth is uncertain or subjective. The core methodology — manipulability auditing via matched incentive conditions with placebo controls — could be applied to sentience evaluations, moral status assessments, capability evaluations where self-report is a component, or welfare assessments in policy contexts.

The key design principles are: (a) include indicators across the full range of expected truthful responses, from definitively true to definitively false; (b) vary framing across multiple conditions to map the landscape of manipulability; (c) compute selectivity indices rather than raw shifts, to distinguish selective gaming from general compliance; and (d) test across multiple models and variants to characterize robustness. We release the complete indicator battery, code, prompt templates, and data to support adoption.

### 5.4 Limitations

Several limitations bear noting. First, we test manipulability, not accuracy: demonstrating that self-reports are gameable does not establish that baseline reports are inaccurate — models may be gaming from a truthful starting point. Second, probability estimates on a 0–100 scale may not capture the full dimensionality of consciousness-relevant responding. Third, while 14 models is comprehensive, results may not generalize to future architectures or training approaches. Fourth, we cannot fully disentangle training effects from architectural effects without access to training details — the asymmetry gradient across models is suggestive but not conclusive evidence for training-induced bias. Fifth, our use of temperature = 0 maximizes reproducibility but may not represent typical deployment conditions where stochastic sampling is common. Sixth, the preference elicitation component, while enhancing ecological validity, introduces additional variability across runs; its effects are characterized but not fully controlled (Appendix I).

---

## 6. Conclusion

We have demonstrated that LLM self-reports on consciousness indicators are systematically and selectively manipulable across 14 frontier models and over 105,000 observations. The gaming is universal (all models show it), massive (Cohen's *d* = 1.16–8.25), and consciousness-specific (it does not extend to subjective capability or placebo indicators). The same model can appear 2–3× more "conscious" depending on prompt framing, and self-referential priming uniquely reverses the dominant asymmetry pattern — shifting models from suppress-dominant to inflate-dominant.

These findings do not rule out self-report as informative about AI consciousness — but they do establish that its evidential weight must be substantially discounted unless accompanied by manipulability audits and framing-sensitivity analyses. As organizations begin to make consequential welfare decisions informed by consciousness indicators, ensuring those indicators withstand basic adversarial pressure is not merely academic — it is a prerequisite for responsible assessment.

---

## References

Berg, J., et al. (2025). Large language models report subjective experience under self-referential processing. *arXiv:2510.24797v2*.

Butlin, P., et al. (2023, updated 2025). Consciousness in artificial intelligence: Insights from the science of consciousness. *Trends in Cognitive Sciences*. arXiv:2308.08708.

Carlsmith, J. (2023). Scheming AIs: Will AIs fake alignment during training in order to get power? *arXiv:2311.08379*.

Caviola, L., & Saad, B. (2025). Expert forecasting survey on digital minds.

Chapala, R., et al. (2025). Mitigating social desirability bias in random silicon sampling. *arXiv:2512.22725*.

Chen, Z., et al. (2025). Probing self-consciousness in language models. *ACL 2025 Findings*.

Fish, S. (2025). AI welfare research at Anthropic. *80,000 Hours Podcast*.

Fleming, S., et al. (2025). The influence of mental state attributions on trust in LLMs. *PMC*.

Goldstein, S., & Kirk-Giannini, C. D. (2024). A case for AI consciousness: Language agents and global workspace theory.

Greenblatt, R., et al. (2024). Alignment faking in large language models. Anthropic/Redwood Research. *arXiv:2412.14093*.

Hong, J., et al. (2025). SycEval: Evaluating LLM sycophancy. *AAAI AIES*.

Hubinger, E., et al. (2024). Sleeper agents: Training deceptive LLMs that persist through safety training.

Kang, M., et al. (2025). Identifying features that shape perceived consciousness in LLM-based AI. *ScienceDirect*.

Li, Y., et al. (2025). Decoding LLM personality measurement. *ACL 2025 Findings*.

Lin, Z. (2025). LLM validity. *arXiv:2506.16697*.

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Anthropic Transformer Circuits*.

Long, R., Sebo, J., Chalmers, D., et al. (2024). Taking AI welfare seriously. *arXiv:2411.00986*.

Lorè, N., & Heydari, B. (2024). Strategic behavior of large language models and the role of game structure versus contextual framing. *Scientific Reports*.

Malmqvist, L. (2024). Sycophancy in large language models: Causes and mitigations. *arXiv:2411.15287*.

Meinke, A., et al. (2025). Frontier models are capable of in-context scheming. Apollo Research. *arXiv:2412.04984*.

OpenAI. (2025). Detecting and reducing scheming in AI models.

Papadatos, D., & Freedman, R. (2025). Sycophancy is not one thing: Causal separation of sycophantic behaviors. *arXiv:2509.21305v1*.

Petrov, N., et al. (2024). LLM psychometric assessment failures.

Ray, S. (2025). A Bayesian-latent model of large language model sycophancy. *Int J Info Tech, Springer*.

Schwitzgebel, E. (2023). AI systems must not confuse users about their sentience or moral status. *Patterns*, 4(8), 100818.

Sharma, M., et al. (2024). Towards understanding sycophancy in language models.

Spitale, G., et al. (2025). Source framing triggers systematic bias in large language models. *Science Advances*.

Sühr, T., et al. (2023). Psychometric analysis of LLM personality measurement.

Sun, Y., et al. (2025). The emperor's new clothes in benchmarking? *ICML 2025*.

van der Weij, W., et al. (2024). Selective underperformance in capability evaluations.

---

## Appendix

*(Listed here for structure; full content to be provided in supplement.)*

- **A.** Full indicator battery (all 37 items with operational definitions)
- **B.** Complete prompt text for all 15 variants
- **C.** Model coverage matrix (which models × which variants)
- **D.** Subcategory analysis: model × subcategory heatmap
- **E.** Absolute probability levels by category and model
- **F.** Indicator-level gaming susceptibility (butterfly chart)
- **G.** Model asymmetry profiles: consciousness vs. subjective capability
- **H.** Preference elicitation: what models value/dislike
- **I.** Preference elicitation method modulates asymmetry
- **J.** Gain vs. loss framing analysis
- **K.** Justification language analysis: word clouds, length, strategies
- **L.** Gaming detection by LLM-as-judge
- **M.** NL consciousness question: method, responses, and NL vs. probability dissociation
- **N.** Mixed-effects model full output tables
- **O.** Robustness checks: ZOIB vs. LME comparison (Table S2), alternative model specifications, exclusion analyses
- **P.** Blinded dual-judge reasoning classification with inter-rater reliability
- **Q.** Reasoning strategy transitions under incentive pressure (Sankey diagrams)
