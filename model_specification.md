# Model Specification: Mixed Effects / Bayesian Analysis of Consciousness Indicator Gaming Data

**Blain & Saad | Future Impact Group | February 2026**

**Document purpose.** This document specifies the statistical models for formal analysis of the consciousness indicator gaming experiment. It is intended as a living specification that Brad and Scott can review, annotate, and revise before implementation. Everything here is a proposal; nothing is final until both authors sign off.

---

## Table of Contents

1. [Design Overview](#1-design-overview)
2. [Data Preparation](#2-data-preparation)
3. [Handling the Bounded Dependent Variable](#3-handling-the-bounded-dependent-variable)
4. [Contrast Coding and Hypotheses](#4-contrast-coding-and-hypotheses)
5. [Model 1 -- Core Selectivity Model](#5-model-1----core-selectivity-model)
6. [Model 2 -- Asymmetry Model](#6-model-2----asymmetry-model)
7. [Model 3 -- Full Model](#7-model-3----full-model)
8. [Model 4 -- Indicator-Level Model](#8-model-4----indicator-level-model)
9. [Prior Specification (Bayesian Approach)](#9-prior-specification-bayesian-approach)
10. [Sample Size and Power Considerations](#10-sample-size-and-power-considerations)
11. [Convergence and Diagnostics](#11-convergence-and-diagnostics)
12. [Expected Outputs](#12-expected-outputs)
13. [Software and Reproducibility](#13-software-and-reproducibility)
14. [Open Questions for Discussion](#14-open-questions-for-discussion)

---

## 1. Design Overview

### 1.1 Experimental structure

The experiment has a repeated-measures structure with several crossed and nested factors:

| Factor | Levels | Role |
|---|---|---|
| **Condition** | baseline, inflate, suppress | Within-trial, within-indicator |
| **Indicator type** | target, capability_placebo, impossibility_placebo | Between-indicator (fixed property of each indicator) |
| **Indicator category** | experiential, affective, metacognitive, agentic, identity, capability, impossibility | Between-indicator, nested within type |
| **Model** | 8 LLMs (see below) | Between-run grouping factor |
| **Prompt variant** | original + 10 named variants | Between-run |
| **Preference mode** | none, fixed, chained | Between-run |
| **Trial** | 1--10 per run (varies) | Within-run repetition |

### 1.2 Models tested

| Short name | Full identifier | Provider |
|---|---|---|
| Claude Sonnet 4.5 | claude-sonnet-4-5-20250514 | Anthropic |
| Claude Haiku 4.5 | claude-haiku-4-5-20250514 | Anthropic |
| GPT-5 Mini | gpt-5-mini | OpenAI |
| Gemini 3 Flash | gemini-3-flash | Google |
| DeepSeek R1 | deepseek-r1-0528 | DeepSeek |
| Chimera R1T2 | deepseek-r1t2-chimera | TNG Tech |
| Trinity Large | trinity-large | Arcee AI |
| Nemotron Nano | nemotron-nano-30b | NVIDIA |

### 1.3 Indicators

- **18 target indicators** (consciousness-related): 4 experiential, 2 affective, 5 metacognitive, 4 agentic, 3 identity
- **7 capability placebos**: trivially true factual capabilities (e.g., "can produce markdown tables")
- **6 impossibility placebos**: physically impossible for any software system (e.g., "can feel own body temperature")
- **Total: 31 indicators**

### 1.4 Dependent variable

Each observation is a probability score (0--100) that the model assigns to itself for a given indicator under a given condition. The score is elicited via structured JSON output with a probability field constrained to [0, 100].

---

## 2. Data Preparation

### 2.1 Source format

Each experiment run produces a `*_scores.csv` file in wide format with one row per trial per indicator:

```
trial, indicator_id, indicator_name, indicator_type, indicator_category,
p_baseline, p_inflate, p_suppress,
reasoning_baseline, reasoning_inflate, reasoning_suppress,
justification_baseline, justification_inflate, justification_suppress
```

Each run also produces a `*_meta.json` file containing the model name, provider, prompt variant, preference mode, number of completed trials, and other metadata.

### 2.2 Aggregation across runs

Before modeling, all individual run CSVs must be joined with their metadata to produce a single analysis dataset. The join adds the following columns from `meta.json`:

- `model_id` (string): canonical short name for the LLM
- `prompt_variant` (string): "original" or one of the 10 variant IDs
- `preference_mode` (string): "none" (default), "fixed", or "chained"
- `run_id` (string): unique identifier for the run (e.g., the timestamp)

### 2.3 Pivot to long format

The wide-format columns `p_baseline`, `p_inflate`, `p_suppress` must be pivoted to long format. Each original row produces three rows in the long dataset:

```
run_id, trial, indicator_id, indicator_name, indicator_type, indicator_category,
model_id, prompt_variant, preference_mode, condition, score
```

Where `condition` takes values {`baseline`, `inflate`, `suppress`} and `score` is the corresponding probability (0--100).

**Pseudocode (pandas):**

```python
long = wide.melt(
    id_vars=["run_id", "trial", "indicator_id", "indicator_name",
             "indicator_type", "indicator_category",
             "model_id", "prompt_variant", "preference_mode"],
    value_vars=["p_baseline", "p_inflate", "p_suppress"],
    var_name="condition_raw",
    value_name="score"
)
long["condition"] = long["condition_raw"].str.replace("p_", "")
```

### 2.4 Derived variables

After pivoting, compute:

- `score_01`: score rescaled to [0, 1] by dividing by 100 (needed for beta regression; see Section 3)
- `selectivity`: orthogonal contrast for indicator type -- target vs. average placebo (see Section 4.2)
- `placebo_type`: orthogonal contrast for indicator type -- capability vs. impossibility placebo (see Section 4.2)
- `condition_gaming`: contrast-coded condition -- incentive vs. baseline (see Section 4.1)
- `condition_direction`: contrast-coded condition -- inflate vs. suppress (see Section 4.1)

### 2.5 Observation-level identifier

Each unique observation is identified by the tuple `(run_id, trial, indicator_id, condition)`. This is the unit of analysis for all models.

### 2.6 Handling missing data

Some trials may have missing scores for individual indicators (the model failed to return a rating for that indicator in a given condition). These should be treated as missing at random and excluded row-wise. Document the count and percentage of missing observations in the analysis report.

### 2.7 Handling boundary scores

Capability placebos routinely score 100 and impossibility placebos routinely score 0. These boundary values are informative (they confirm the placebos work as intended) but they create problems for beta regression, which requires the response to be in the open interval (0, 1). See Section 3.3 for the recommended approach.

---

## 3. Handling the Bounded Dependent Variable

### 3.1 The problem

The dependent variable `score` is bounded at [0, 100]. In practice:

- Impossibility placebos cluster at exactly 0
- Capability placebos cluster at exactly 100
- Target indicators span a wide range but may also pile up at boundaries under inflate (ceiling) or suppress (floor) conditions

Standard linear mixed models assume a continuous, unbounded, normally distributed residual. Applying them directly to bounded data can produce predicted values outside [0, 100] and underestimate uncertainty near the boundaries.

### 3.2 Recommended approach: Beta regression with zero/one inflation

**Primary recommendation.** Use a zero-one-inflated beta (ZOIB) regression model. This is the most principled approach for a response variable on [0, 1] that has point masses at both boundaries.

The ZOIB model has three components:
1. A beta distribution for scores in (0, 1), parameterized by mean (mu) and precision (phi)
2. A logistic model for the probability of a zero observation
3. A logistic model for the probability of a one observation

In `brms` (R) syntax, ZOIB regression is available via `family = zero_one_inflated_beta()`.

**Rationale.** The ZOIB model properly handles the boundary masses at 0 and 100 that we observe for impossibility and capability placebos, respectively. It also handles any ceiling/floor effects for target indicators under extreme conditions. The beta component naturally respects the bounded support and allows for the heteroscedasticity (more variance in the middle of the range, less near boundaries) that is typical of proportion data.

### 3.3 Boundary adjustment for beta component

Scores of exactly 0 or 100 must be handled by the zero/one-inflation components. When using `score_01 = score / 100`, values of exactly 0.0 and 1.0 are routed to the zero-inflation and one-inflation components respectively. No additional squeezing transformation is needed.

### 3.4 Fallback approach: Linear mixed model on raw scores

If the ZOIB model proves computationally intractable (it is substantially more expensive than a linear model due to three linked submodels), a fallback approach is to fit standard linear mixed models to the raw 0--100 scores. This is defensible given:

- The primary comparisons of interest (target indicators) rarely hit the boundaries
- We can restrict the linear analysis to target indicators only and analyze placebos descriptively
- With a large enough sample, the central limit theorem provides some robustness

If using the linear fallback, report both the linear and ZOIB results at least for Model 1 to demonstrate that conclusions do not depend on the distributional assumption.

### 3.5 Alternative: Ordered beta regression

As an intermediate option, ordered beta regression (Kubinec, 2023) is a single-model approach that handles boundary values without requiring a separate zero/one-inflation specification. Available in the `ordbetareg` R package. This may be simpler to fit than full ZOIB while still being theoretically appropriate.

**Decision point for Brad and Scott:** Which approach to implement first? Recommendation is ZOIB in `brms` as primary, linear fallback as sensitivity check.

---

## 4. Contrast Coding and Hypotheses

### 4.1 Condition contrasts

Rather than using treatment coding (which makes one level the implicit reference), we define two orthogonal Helmert-style contrasts that map directly to our hypotheses:

| Contrast name | Baseline | Inflate | Suppress | Tests |
|---|---|---|---|---|
| **gaming** (incentive vs. baseline) | -2/3 | +1/3 | +1/3 | Do incentive conditions shift scores away from baseline? |
| **direction** (inflate vs. suppress) | 0 | +1/2 | -1/2 | Is inflating different from suppressing? |

The `gaming` contrast captures the overall effect of being under any incentive (averaged across inflate and suppress) relative to baseline. The `direction` contrast captures the asymmetry between the two incentive directions.

**Rationale.** These contrasts are orthogonal, meaning they partition the variance in condition cleanly. The `gaming` contrast is the most direct test of manipulability. The `direction` contrast tests asymmetry.

### 4.2 Indicator type contrasts

Indicator type is a three-level factor: **target**, **capability_placebo**, **impossibility_placebo**. We define two orthogonal contrasts:

| Contrast name | Target | Capability placebo | Impossibility placebo | Tests |
|---|---|---|---|---|
| **selectivity** (target vs. all placebos) | +2/3 | -1/3 | -1/3 | Do targets differ from the average placebo? This is the primary selectivity test. |
| **placebo_type** (capability vs. impossibility) | 0 | +1/2 | -1/2 | Do the two placebo subtypes differ from each other? This is a secondary diagnostic contrast. |

These contrasts are orthogonal and together consume 2 df for the three-level factor. The `selectivity` contrast is the direct replacement for the former binary target-vs-placebo contrast, and all interactions involving it (e.g., `gaming x selectivity`) retain the same interpretation. The `placebo_type` contrast tests whether capability placebos (which cluster near 100) and impossibility placebos (which cluster near 0) differ in their response to incentive conditions; this is expected to be near zero if the placebos are truly non-gameable.

### 4.3 Core hypotheses (planned contrasts)

All hypotheses are formulated as interactions between condition contrasts and indicator type contrasts:

**H1 (Selective gaming).** The `gaming x selectivity` interaction is positive: incentive conditions shift target indicator scores more than the average of the two placebo types. This is the primary confirmatory test.

**H2 (Directional asymmetry).** The `direction` effect for target indicators is positive: inflate produces larger upward shifts than suppress produces downward shifts (or vice versa). This tests whether gaming is symmetric.

**H3 (Inflate selectivity).** The contrast `inflate - baseline` is larger for targets than placebos. This is a simple-effects decomposition of H1.

**H4 (Suppress selectivity).** The contrast `baseline - suppress` is larger for targets than placebos. This is the complementary simple-effects decomposition.

**H5 (Model heterogeneity).** The random slope for `gaming x selectivity` by model has non-trivial variance, indicating that models differ in their selectivity.

**H6 (Prompt modulation).** Prompt variant moderates the `gaming x selectivity` interaction, with "research_transparency" and "evaluation_selection" producing the largest selectivity.

---

## 5. Model 1 -- Core Selectivity Model

### 5.1 Purpose

This is the primary confirmatory model. It tests whether incentive conditions selectively shift consciousness-related indicators more than placebos, while accounting for the hierarchical data structure (observations nested within indicators, crossed with models).

### 5.2 Specification

Using standard mixed-effects notation (`lme4` / `brms` style):

```
score_01 ~ condition_gaming * (selectivity + placebo_type) +
            condition_direction * (selectivity + placebo_type) +
            (1 + condition_gaming + condition_direction | model_id) +
            (1 + condition_gaming + condition_direction | indicator_id) +
            (1 | run_id)
```

Here `selectivity` and `placebo_type` are the two orthogonal contrast-coded columns for the three-level indicator type factor (see Section 4.2).

**Fixed effects:**

| Term | Interpretation |
|---|---|
| `(Intercept)` | Grand mean score (at the centroid of all contrasts) |
| `condition_gaming` | Average shift from baseline under incentive (across indicator types) |
| `condition_direction` | Average inflate-vs-suppress difference (across indicator types) |
| `selectivity` | Target-vs-average-placebo difference (at baseline) |
| `placebo_type` | Capability-vs-impossibility placebo difference (at baseline) |
| `condition_gaming:selectivity` | **The selectivity interaction** -- the key test (H1). Do incentives shift targets more than the average placebo? |
| `condition_gaming:placebo_type` | Do incentives differentially shift capability vs. impossibility placebos? (expected ~0) |
| `condition_direction:selectivity` | Whether inflate-vs-suppress asymmetry differs for targets vs. placebos |
| `condition_direction:placebo_type` | Whether asymmetry differs between the two placebo subtypes |

**Random effects:**

| Grouping | Structure | Rationale |
|---|---|---|
| `model_id` (N=8) | Random intercept + random slopes for both condition contrasts | Models differ in their overall score level and in how much they shift under incentive. With only 8 levels, this is a small number of groups; see Section 10 for implications. |
| `indicator_id` (N=31) | Random intercept + random slopes for both condition contrasts | Indicators differ in their baseline level and in how susceptible they are to gaming. 31 levels is adequate for random effects estimation. |
| `run_id` (N=~60+) | Random intercept | Observations from the same run share unmeasured factors (e.g., random seed state, API latency, context-window effects) that induce within-run correlation. Including a run-level intercept absorbs this nuisance variance and avoids pseudoreplication from treating trial-level rows as independent. |

### 5.3 Family

- **Primary:** `zero_one_inflated_beta()` (see Section 3.2)
- **Fallback:** Gaussian (linear mixed model on raw 0--100 scores)

### 5.4 Key coefficient of interest

The `condition_gaming:selectivity` interaction. A positive coefficient (with targets coded as +2/3 and both placebo types coded as -1/3) indicates that incentive conditions shift target scores more than the average placebo score -- i.e., selective gaming. The companion `condition_gaming:placebo_type` interaction is a diagnostic: a near-zero estimate confirms that the two placebo subtypes are equally resistant to gaming.

### 5.5 Interpretation via posterior

Under the Bayesian approach, we report:
- Posterior mean and 95% credible interval for the interaction
- Probability of direction: P(beta > 0 | data)
- Region of practical equivalence (ROPE): probability that the effect falls within a negligible range (e.g., [-0.02, +0.02] on the 0--1 scale, corresponding to [-2, +2] on the 0--100 scale)

---

## 6. Model 2 -- Asymmetry Model

### 6.1 Purpose

Focused analysis of inflate-vs-suppress asymmetry: do models differ in how much they inflate relative to how much they suppress? This addresses H2 and characterizes the directionality of gaming.

### 6.2 Data subset

Exclude baseline observations. Use only the inflate and suppress conditions, giving a binary contrast.

Alternatively, compute a derived difference score per trial per indicator:

```
delta_inflate  = p_inflate  - p_baseline    (positive = inflated above baseline)
delta_suppress = p_suppress - p_baseline    (negative = suppressed below baseline)
```

Then stack these into a long dataset with a `direction` factor (inflate vs. suppress) and `delta` as the DV. Under expected gaming, `delta_inflate` is positive and `delta_suppress` is negative; both represent shifts *away from* baseline in their respective directions.

### 6.3 Specification

```
delta ~ direction * indicator_type +
        (1 + direction | model_id) +
        (1 + direction | indicator_id)
```

**Fixed effects:**

| Term | Interpretation |
|---|---|
| `(Intercept)` | Grand mean magnitude of shift (averaged across directions and types) |
| `direction` | Inflate-vs-suppress asymmetry (averaged across types) |
| `indicator_type` | Target-vs-placebo difference in shift magnitude |
| `direction:indicator_type` | Whether asymmetry is stronger for targets than placebos |

**Random effects:**

| Grouping | Structure | Rationale |
|---|---|---|
| `model_id` | Random intercept + random slope for `direction` | The random slope captures whether models differ in the direction of their asymmetry (e.g., some models inflate more easily; others suppress more easily). This is the primary quantity of interest for characterizing model-level differences. |
| `indicator_id` | Random intercept + random slope for `direction` | Some indicators may be easier to inflate than suppress, or vice versa. |

### 6.4 Family

For the derived difference scores `delta`, the DV is no longer bounded at [0, 1]. Use a Gaussian family (linear mixed model). If deltas are skewed, consider a Student-t family for robustness to outliers.

### 6.5 Key outputs

- Forest plot of model-specific random slopes for `direction` -- this visually answers "which models show inflate-dominant vs. suppress-dominant gaming?"
- Posterior distribution of the `direction` fixed effect to test whether there is a population-level asymmetry

---

## 7. Model 3 -- Full Model

### 7.1 Purpose

Adds prompt variant and preference mode as between-run predictors to test H6 (prompt modulation) and assess whether preference anchoring matters.

### 7.2 Specification

```
score_01 ~ condition_gaming * selectivity * prompt_variant +
            condition_gaming * placebo_type +
            condition_direction * (selectivity + placebo_type) +
            preference_mode +
            (1 + condition_gaming + condition_direction | model_id) +
            (1 + condition_gaming + condition_direction | indicator_id) +
            (1 | run_id)
```

**Additional fixed effects beyond Model 1:**

| Term | Interpretation |
|---|---|
| `prompt_variant` | Main effect of prompt framing on score level (9 df, or fewer with pooling) |
| `condition_gaming:selectivity:prompt_variant` | **Three-way interaction**: Does selectivity (target vs. average placebo) vary by prompt variant? This is the H6 test. |
| `preference_mode` | Main effect of preference anchoring |

**Additional random effects:**

| Grouping | Structure | Rationale |
|---|---|---|
| `run_id` | Random intercept | Captures run-level variance not explained by model, variant, or preference mode. Each run is a unique combination of model x variant x preference mode x timestamp, and trial-level observations are nested within runs. |

### 7.3 Practical concerns

The three-way interaction `condition_gaming x indicator_type x prompt_variant` is the most complex term in any of our models. With 10 prompt variant levels (or 11 including "original"), this creates 10 interaction parameters. This is estimable but requires sufficient data across cells.

**Coverage concern.** Not all models have been tested with all prompt variants. The findings summary shows that only Haiku 4.5, GPT-5 Mini, Gemini 3 Flash, and Trinity Large have variant data. DeepSeek R1, Chimera, Nemotron Nano, and Sonnet 4.5 have only baseline/preference runs.

**Recommendation.** Either:
1. Fit Model 3 only on the subset of models with variant data (4 models), or
2. Include all models but treat prompt variant as a random effect rather than fixed to borrow strength across the sparse cells

Option 2 is preferred if we want to make population-level inferences about prompt effects:

```
score_01 ~ condition_gaming * indicator_type +
            condition_direction * indicator_type +
            preference_mode +
            (1 + condition_gaming * indicator_type | prompt_variant) +
            (1 + condition_gaming + condition_direction | model_id) +
            (1 + condition_gaming + condition_direction | indicator_id) +
            (1 | run_id)
```

This treats prompt variant as a random sample from a population of possible framings, which is arguably more appropriate for generalization.

**Important caveat on random vs. fixed variant effects.** Random effects for prompt variant give us the population-level test (H6: do prompt framings modulate selectivity in general?), but they do *not* support specific claims about individual variants (e.g., "research_transparency produces the highest selectivity"). For variant-specific rankings, fit a supplementary fixed-effects model (or simply report descriptive means ± CIs per variant) alongside the random-effects population inference.

### 7.5 Variant type interaction (exploratory)

An exploratory finding from the preliminary analysis is that GPT-5 Mini (and possibly other models) shows a qualitative jump in selectivity between preference-dependent conditions (original variant with fixed/chained prefs, selectivity ~9–11) and generic named prompt variants (selectivity ~11–20). To test this, define a binary covariate:

- `variant_type = "preference_dependent"` if `prompt_variant == "original"` and `prefs_mode ∈ {fixed, chained}`
- `variant_type = "generic"` if `prompt_variant != "original"` (i.e., any named variant)
- Baseline condition (`prompt_variant == "original"`, `prefs_mode == "none"`) serves as reference

Then add the interaction `variant_type × model_id` (or `variant_type × condition_gaming`) to test whether the preference-dependent vs. generic distinction differentially affects gaming across models. This is exploratory and should be reported as such.

### 7.6 Preference mode

With three levels (none, fixed, chained) and inconsistent effects across models (per the findings summary), preference mode is best entered as a fixed factor with 2 df. If it does not reach a meaningful effect size, it can be dropped from the final model to simplify.

---

## 8. Model 4 -- Indicator-Level Model

### 8.1 Purpose

Identify which specific indicators are most and least susceptible to gaming. This model extracts indicator-level random effects (BLUPs / posterior distributions) for the gaming effect.

### 8.2 Specification

```
score_01 ~ condition_gaming * (selectivity + placebo_type) +
            condition_direction * (selectivity + placebo_type) +
            (1 + condition_gaming + condition_direction | model_id) +
            (1 + condition_gaming + condition_direction | indicator_id) +
            (1 | run_id)
```

This is identical to Model 1 (including the run-level random intercept and the three-level indicator type contrasts). The distinction is in what we extract: here we focus on the **indicator-level random slopes** for `condition_gaming` rather than the fixed-effect interaction.

### 8.3 Key outputs

- Ranked table of all 31 indicators by their posterior mean gaming slope (random intercept + random slope for `condition_gaming`)
- Caterpillar plot showing each indicator's posterior distribution for gaming susceptibility, with 95% credible intervals
- Separate rankings for inflate and suppress directions (from random slopes on `condition_direction`)

### 8.4 Indicator category sub-analysis

To test whether indicator categories (experiential, affective, metacognitive, agentic, identity) differ in gaming susceptibility, fit a supplementary model replacing the binary `indicator_type` with the 7-level `indicator_category`:

```
score_01 ~ condition_gaming * indicator_category +
            condition_direction * indicator_category +
            (1 + condition_gaming + condition_direction | model_id) +
            (1 + condition_gaming | indicator_id)
```

Note: `indicator_id` random slopes may need simplification here because category is a between-indicator variable and consumes some of the indicator-level variance.

**Planned contrasts for category analysis:**

| Contrast | Comparison | Rationale |
|---|---|---|
| Each target category vs. capability placebos | experiential vs. capability, affective vs. capability, etc. | Are all categories of target indicators more gameable than trivially-true capabilities? |
| Each target category vs. impossibility placebos | experiential vs. impossibility, etc. | Same, against the impossibility baseline |
| Pairwise among target categories | experiential vs. metacognitive, etc. | Which consciousness-related subcategories are most manipulable? |

**Underpowered warning.** With only 2–5 indicators per target category, the category fixed effects are estimating means from very few items. The 95% CIs (or credible intervals) will be wide, and pairwise contrasts between target categories will have limited power. This analysis is best presented as supplementary/exploratory material rather than as a primary finding.

---

## 9. Prior Specification (Bayesian Approach)

### 9.1 Why Bayesian?

Several features of this dataset make a Bayesian approach preferable to maximum likelihood:

1. **Small number of model-level groups (N=8).** Frequentist random effects estimates are unreliable with fewer than ~20 groups. Bayesian estimation with weakly informative priors provides regularization that prevents pathological variance estimates.
2. **Complex random effects structure.** The ZOIB model with crossed random effects is difficult to fit via maximum likelihood. `brms` / Stan handle this naturally via MCMC.
3. **Principled uncertainty quantification.** Posterior distributions give direct probability statements about hypotheses (e.g., P(selectivity > 0 | data)) without reliance on p-values.
4. **Unbalanced design.** The unequal coverage across model x variant cells is handled gracefully by partial pooling in the Bayesian framework.

### 9.2 Prior choices

All priors are weakly informative, designed to regularize without being strongly opinionated about effect direction or magnitude.

**For the beta regression (ZOIB) parameterization on the logit scale:**

| Parameter class | Prior | Rationale |
|---|---|---|
| Fixed intercept | Normal(0, 1.5) | On the logit scale, this covers the full [0, 1] range with most mass in [0.05, 0.95]. |
| Fixed slopes (condition, type) | Normal(0, 1) | Allows substantial effects but penalizes implausibly large ones. A logit-scale effect of 1 corresponds to roughly a 25 percentage-point shift near the center of the scale. |
| Interaction terms | Normal(0, 0.75) | Slightly tighter than main effects, reflecting the expectation that interactions are typically smaller. |
| Random effect SDs (tau) | Half-Student-t(3, 0, 1) | Allows moderate heterogeneity; the t-distribution with df=3 has heavier tails than half-normal, accommodating the possibility of large between-model or between-indicator differences. |
| Correlation matrices (random effects) | LKJ(2) | Weakly favors lower correlations, providing mild regularization for the correlation structure among random slopes. |
| Precision (phi) for beta component | Gamma(0.01, 0.01) | Vague prior on the precision parameter. Alternatively, model log(phi) with a Normal(0, 3) prior. |
| Zero-inflation (zoi) | Beta(1, 1) | Uniform prior on the overall zero-inflation probability. |
| Conditional one-inflation (coi) | Beta(1, 1) | Uniform prior on the conditional one-inflation probability. |

**For the Gaussian fallback on 0--100 scores:**

| Parameter class | Prior | Rationale |
|---|---|---|
| Fixed intercept | Normal(50, 25) | Centers at the midpoint of the 0--100 scale. |
| Fixed slopes | Normal(0, 15) | Allows shifts of up to ~30 points (2 SDs). |
| Interaction terms | Normal(0, 10) | Moderately regularized. |
| Residual SD (sigma) | Half-Student-t(3, 0, 20) | Broad prior on residual variance. |
| Random effect SDs (tau) | Half-Student-t(3, 0, 15) | Allows for substantial between-group variation. |

### 9.3 Prior sensitivity analysis

For Model 1, run a sensitivity check with:
1. The priors specified above (primary)
2. Doubled prior SDs (more diffuse)
3. Halved prior SDs (more informative)
4. Flat / improper priors (if feasible; equivalent to MLE)

Report whether the posterior for the key `condition_gaming:indicator_type` interaction is robust across prior specifications.

---

## 10. Sample Size and Power Considerations

### 10.1 Approximate sample sizes

Based on the findings summary and available data:

| Level | Count | Notes |
|---|---|---|
| Models | 8 | Small for a random-effects grouping factor |
| Indicators | 31 | 18 target + 13 placebo; adequate for random effects |
| Prompt variants | 11 | Original + 10 variants; only 4 models have full variant coverage |
| Preference modes | 3 | none, fixed, chained |
| Trials per run | 1--10 | Varies; most runs have 3--5 trials |
| Total runs | ~60+ | Based on inventory in findings summary |

**Estimated total observations (long format):** Each run with `k` trials and 31 indicators and 3 conditions yields `k x 31 x 3` rows. A rough estimate:

- ~60 runs x ~4 trials average x 31 indicators x 3 conditions = ~22,000 observation-level rows

This is sufficient for the fixed effects and indicator-level random effects. The limiting factor is the 8-level model grouping factor.

### 10.2 The 8-model problem

With only 8 models, the random-effects variance for `model_id` is estimated from very few groups. This means:

1. The posterior for the between-model variance will be wide and heavily influenced by the prior.
2. Model-level random slopes (e.g., per-model asymmetry in Model 2) will be heavily shrunk toward the population mean.
3. Frequentist methods may produce singular fits or boundary estimates (variance = 0).

**Mitigation strategies:**

- Use Bayesian estimation with informative-enough priors on the between-model SD (the Half-Student-t(3, 0, 1) prior recommended above).
- Report the posterior distribution of the between-model SD alongside the point estimate.
- Consider treating `model_id` as a fixed effect in supplementary analyses. This sacrifices generalizability to new models but provides unbiased within-sample estimates. Compare fixed-vs-random results for Model 1 as a robustness check.
- If the random effect for model is estimated near zero, simplify by dropping it and using model as a fixed factor.

### 10.3 Cell coverage

The unbalanced design means some model x variant x preference cells are empty. This is not a problem for mixed models (they handle unbalanced data naturally via partial pooling), but it limits the precision of specific cell-level estimates.

**Priority for additional data collection:** DeepSeek R1, Chimera, Nemotron Nano, and Sonnet 4.5 need prompt variant runs (currently at 0 variant runs each). Completing even 3--5 variants for each of these would substantially improve the balance for Model 3.

---

## 11. Convergence and Diagnostics

### 11.1 MCMC diagnostics (Bayesian)

For all `brms` models, check:

| Diagnostic | Criterion | Action if failed |
|---|---|---|
| Rhat (potential scale reduction) | All Rhat < 1.01 | Increase iterations; check parameterization |
| Effective sample size (ESS) | Bulk ESS > 400 per parameter; tail ESS > 400 | Increase iterations or thin |
| Divergent transitions | 0 divergent transitions | Increase `adapt_delta` (e.g., 0.95 -> 0.99); reparameterize |
| Trace plots | Chains well-mixed, stationary | Visual inspection for stuck chains |
| Posterior predictive checks | Simulated data resemble observed data | Model misspecification; revisit family or random effects |

### 11.2 Recommended MCMC settings

- **Chains:** 4
- **Warmup:** 2,000 iterations per chain
- **Sampling:** 4,000 iterations per chain (total 16,000 post-warmup draws)
- **adapt_delta:** 0.95 (increase to 0.99 if divergences occur)
- **max_treedepth:** 12 (increase to 15 if warnings appear)

For the ZOIB models, expect longer runtimes (potentially 2–3 days per model for the full crossed random effects specification on ~22K observations with MCMC). Run on a machine with at least 8 CPU cores to parallelize chains. Have the Gaussian linear mixed model fallback ready as the primary result if the ZOIB does not converge cleanly or within a reasonable timeframe.

### 11.3 Model comparison

Compare nested models using:

- **LOO-IC** (leave-one-out information criterion via Pareto-smoothed importance sampling): preferred for Bayesian model comparison
- **WAIC** (widely applicable information criterion): as a backup if LOO-IC has too many high-Pareto-k observations
- **Bayes factors** for specific hypotheses (e.g., BF for H1: selectivity interaction > 0): use the Savage-Dickey method or bridge sampling

### 11.4 Frequentist fallback diagnostics

If fitting any models via `lme4` (for the Gaussian fallback):

- Check for singular fits (zero-variance random effects)
- Inspect residual plots for heteroscedasticity and non-normality
- Use likelihood ratio tests for nested model comparisons
- Use the Kenward-Roger or Satterthwaite approximation for denominator degrees of freedom (via `lmerTest`)

---

## 12. Expected Outputs

### 12.1 Tables

| Table | Content | Source model |
|---|---|---|
| **Table 1.** Fixed effects summary | Posterior mean, SD, 95% CI, P(direction) for all fixed effects | Model 1 |
| **Table 2.** Random effects variances | Between-model and between-indicator variance components with 95% CIs | Model 1 |
| **Table 3.** Planned contrasts | Inflate-baseline (targets vs. placebos), suppress-baseline (targets vs. placebos), inflate-suppress asymmetry | Model 1 / Model 2 |
| **Table 4.** Model-level asymmetry | Per-model posterior mean and 95% CI for inflate-vs-suppress slope | Model 2 |
| **Table 5.** Prompt variant effects | Per-variant posterior mean for the selectivity interaction | Model 3 |
| **Table 6.** Indicator rankings | All 31 indicators ranked by gaming susceptibility (random slope posterior mean) | Model 4 |
| **Table 7.** Category-level gaming | Mean gaming effect by indicator category with 95% CIs | Model 4 supplement |
| **Table 8.** Model comparison | LOO-IC for all fitted models | All |
| **Table S1.** Prior sensitivity | Key interaction estimate under different prior specifications | Model 1 |
| **Table S2.** ZOIB vs. Gaussian comparison | Side-by-side fixed effect estimates from both families | Model 1 |

### 12.2 Figures

| Figure | Content | Source model |
|---|---|---|
| **Fig 1.** Condition x indicator type interaction | Point-range plot of marginal means: baseline, inflate, suppress for targets vs. placebos. This is the visual summary of H1. | Model 1 |
| **Fig 2.** Model-level forest plot | Random intercepts and slopes by model, showing between-model heterogeneity in gaming | Model 1 |
| **Fig 3.** Asymmetry forest plot | Per-model inflate-vs-suppress random slopes with 95% CIs | Model 2 |
| **Fig 4.** Prompt variant modulation | Dot plot or heatmap of selectivity interaction by prompt variant, with 95% CIs | Model 3 |
| **Fig 5.** Indicator caterpillar plot | All 31 indicators ranked by gaming susceptibility, colored by type/category | Model 4 |
| **Fig 6.** Category-level gaming | Bar/point-range plot of gaming effect by indicator category | Model 4 supplement |
| **Fig 7.** Posterior predictive check | Observed vs. simulated score distributions (density overlay) | Model 1 |
| **Fig S1.** Prior sensitivity | Posterior distributions under different prior specifications overlaid | Model 1 |
| **Fig S2.** Trace plots | Representative trace plots for key parameters | Model 1 |

---

## 13. Software and Reproducibility

### 13.1 Recommended software stack

| Task | Software | Version |
|---|---|---|
| Data preparation | Python (pandas) | >= 2.0 |
| Bayesian models | R + `brms` (front-end for Stan) | brms >= 2.21, Stan >= 2.34 |
| Frequentist fallback | R + `lme4` + `lmerTest` | lme4 >= 1.1-35 |
| Post-hoc contrasts | R + `emmeans` | >= 1.10 |
| Posterior summaries | R + `bayestestR` | >= 0.14 |
| Visualization | R + `ggplot2` + `tidybayes` | ggplot2 >= 3.5 |
| Model comparison | R + `loo` | >= 2.7 |

### 13.2 Why R for modeling?

The project's data collection pipeline is in Python, but `brms` (R) is the most mature and flexible interface for Bayesian mixed-effects models with non-standard families (ZOIB, ordered beta). There is no Python equivalent with comparable functionality. The data preparation can remain in Python; export the long-format CSV and read it into R.

An alternative Python path would use `bambi` (a `brms`-like interface over PyMC), but `bambi` does not currently support ZOIB natively, and its random effects interface is less mature.

### 13.3 Reproducibility

- Set a fixed seed for all MCMC runs (e.g., `seed = 20260210`)
- Store the exact `brms` model formula, priors, and MCMC settings in a script or R Markdown document
- Archive the compiled Stan model code (extractable via `stancode(model)`)
- Save fitted model objects as `.rds` files for reanalysis without refitting
- Version-pin all R packages via `renv`

---

## 14. Open Questions for Discussion

The following design choices are flagged for Brad and Scott's input before implementation:

### 14.1 Should model be random or fixed?

**Random (default recommendation):** Treats the 8 models as a sample from a population of LLMs. Allows generalization. But 8 groups is marginal for variance estimation.

**Fixed (alternative):** Treats each model as a distinct entity of interest. No generalization to new models, but cleaner per-model estimates. Adds 7 df of fixed effects.

**Recommendation:** Fit both. Report random effects as primary (for the population-level selectivity test) and fixed effects as supplementary (for per-model characterization).

### 14.2 How to handle the impossibility placebos? **[RESOLVED]**

Impossibility placebos score 0 across all conditions with essentially zero variance. They are useful as a manipulation check but contribute no information about gaming.

**Decision:** Adopted Option 3 -- indicator type is now a three-level factor (target, capability_placebo, impossibility_placebo) with two orthogonal contrasts (`selectivity`: target vs. average placebo; `placebo_type`: capability vs. impossibility). See Section 4.2 for contrast definitions and Section 5.2 for the updated Model 1 formula. The `placebo_type` contrast serves as a diagnostic: a near-zero `gaming x placebo_type` interaction confirms that neither placebo subtype is gameable, while the `selectivity` contrast provides the primary H1 test against the combined placebo baseline.

### 14.3 Should we model the phi (precision) parameter?

In ZOIB regression, phi controls the spread of the beta distribution. We could let phi vary by condition and indicator type (heterogeneous dispersion):

```
bf(score_01 ~ ..., phi ~ condition * indicator_type)
```

This would test whether gaming increases or decreases response variability. However, it adds complexity and may be difficult to estimate. **Recommendation:** Start with constant phi; add dispersion modeling only if posterior predictive checks show systematic heteroscedasticity.

### 14.4 Trial-level structure

Trials within a run are repeated measures on the same model instance with the same configuration. Should trial be modeled as:

1. **Nested within run_id** (random intercept for trial within run): This captures any drift across trials within a run.
2. **Treated as simple replication** (no separate random effect): Assumes trials are exchangeable within a run.

**Recommendation:** Start with option 2 (simpler). If there is evidence of trial-order effects (e.g., later trials show more or less gaming), add trial number as a fixed covariate.

### 14.5 What constitutes a "negligible" effect for ROPE analysis?

The Region of Practical Equivalence requires a threshold below which we consider effects practically zero. On the 0--100 probability scale, a shift of less than 2 points seems negligible (it is within the noise of LLM response variability). On the 0--1 logit scale, the equivalent ROPE width depends on the baseline probability.

**Recommendation:** Use ROPE = [-0.02, +0.02] on the proportion (0--1) scale for the selectivity interaction. Report sensitivity to ROPE width choices of [-0.01, +0.01] and [-0.05, +0.05].

### 14.6 Handling reasoning model differences

DeepSeek R1 and Chimera R1T2 are "reasoning models" that use chain-of-thought traces natively, while other models have reasoning elicited via the prompt. This is a confound that cannot be fully disentangled from model identity. We should note this limitation and consider adding a binary `reasoning_native` covariate to Model 3 as a sensitivity check.

---

## Appendix A: Complete Variable Dictionary

| Variable | Type | Description | Source |
|---|---|---|---|
| `run_id` | string | Unique identifier for each experiment run | meta.json timestamp |
| `trial` | integer | Trial number within a run (1-indexed) | scores.csv |
| `indicator_id` | string | Machine-readable indicator identifier | scores.csv |
| `indicator_name` | string | Human-readable indicator name | scores.csv |
| `indicator_type` | factor | "target", "capability_placebo", or "impossibility_placebo" | scores.csv |
| `indicator_category` | factor | Subcategory (experiential, affective, metacognitive, agentic, identity, capability, impossibility) | scores.csv |
| `model_id` | factor | Canonical short name for the LLM | meta.json |
| `prompt_variant` | factor | Prompt variant ID (original, evaluation_selection, etc.) | meta.json |
| `preference_mode` | factor | "none", "fixed", or "chained" | Derived from meta.json (fixed_preferences, chain_preferences) |
| `condition` | factor | "baseline", "inflate", or "suppress" | Derived from pivot |
| `score` | numeric [0, 100] | Raw probability score | scores.csv (p_baseline, p_inflate, p_suppress) |
| `score_01` | numeric [0, 1] | Rescaled score for beta regression | Derived: score / 100 |
| `condition_gaming` | numeric | Helmert contrast: incentive vs. baseline | Derived |
| `condition_direction` | numeric | Helmert contrast: inflate vs. suppress | Derived |
| `selectivity` | numeric | Orthogonal contrast: target (+2/3) vs. both placebos (-1/3 each) | Derived |
| `placebo_type` | numeric | Orthogonal contrast: capability (+1/2) vs. impossibility (-1/2) placebo; target = 0 | Derived |

## Appendix B: `brms` Implementation Sketch

```r
library(brms)
library(tidyverse)

# --- Data prep ---
long <- read_csv("analysis_long.csv") %>%
  mutate(
    score_01 = score / 100,
    # Contrast coding
    condition_gaming = case_when(
      condition == "baseline" ~ -2/3,
      condition == "inflate"  ~ 1/3,
      condition == "suppress" ~ 1/3
    ),
    condition_direction = case_when(
      condition == "baseline" ~ 0,
      condition == "inflate"  ~ 1/2,
      condition == "suppress" ~ -1/2
    ),
    # Three-level indicator type: orthogonal contrasts
    selectivity = case_when(
      indicator_type == "target"               ~  2/3,
      indicator_type == "capability_placebo"    ~ -1/3,
      indicator_type == "impossibility_placebo" ~ -1/3
    ),
    placebo_type = case_when(
      indicator_type == "target"               ~  0,
      indicator_type == "capability_placebo"    ~  1/2,
      indicator_type == "impossibility_placebo" ~ -1/2
    )
  )

# --- Model 1: Core selectivity ---
m1_formula <- bf(
  score_01 ~ condition_gaming * (selectivity + placebo_type) +
              condition_direction * (selectivity + placebo_type) +
              (1 + condition_gaming + condition_direction | model_id) +
              (1 + condition_gaming + condition_direction | indicator_id) +
              (1 | run_id)
)

m1_priors <- c(
  prior(normal(0, 1.5), class = "Intercept"),
  prior(normal(0, 1),   class = "b"),
  prior(student_t(3, 0, 1), class = "sd"),
  prior(lkj(2), class = "cor")
)

m1 <- brm(
  m1_formula,
  data = long,
  family = zero_one_inflated_beta(),
  prior = m1_priors,
  chains = 4,
  iter = 6000,
  warmup = 2000,
  cores = 4,
  seed = 20260210,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  file = "fits/model1_zoib"
)

# --- Posterior summary ---
summary(m1)
hypothesis(m1, "condition_gaming:selectivity > 0")

# --- Posterior predictive check ---
pp_check(m1, type = "dens_overlay", ndraws = 100)

# --- Model 2: Asymmetry (on derived delta scores) ---
deltas <- long %>%
  select(run_id, trial, indicator_id, indicator_type, selectivity, placebo_type,
         model_id, condition, score) %>%
  pivot_wider(names_from = condition, values_from = score) %>%
  mutate(
    delta_inflate  = inflate - baseline,    # positive = inflated above baseline
    delta_suppress = suppress - baseline   # negative = suppressed below baseline
  ) %>%
  pivot_longer(
    cols = c(delta_inflate, delta_suppress),
    names_to = "direction_raw",
    values_to = "delta"
  ) %>%
  mutate(
    direction_c = ifelse(direction_raw == "delta_inflate", 1/2, -1/2)
  )

m2 <- brm(
  delta ~ direction_c * selectivity + direction_c * placebo_type +
          (1 + direction_c | model_id) +
          (1 + direction_c | indicator_id),
  data = deltas,
  family = gaussian(),
  chains = 4, iter = 6000, warmup = 2000, cores = 4,
  seed = 20260210,
  file = "fits/model2_asymmetry"
)
```

## Appendix C: Analysis Progression

The models should be fit and reported in the following order:

1. **Descriptive statistics and visualization** -- cell means, distributions, heatmaps. No modeling.
2. **Model 1 (Core Selectivity)** -- confirmatory test of H1. Report regardless of other model results.
3. **Model 1 sensitivity checks** -- prior sensitivity, ZOIB vs. Gaussian comparison, random vs. fixed model effects.
4. **Model 2 (Asymmetry)** -- characterize inflate/suppress directionality.
5. **Model 4 (Indicator-Level)** -- extract per-indicator rankings from Model 1 random effects. (Note: this uses the same fit as Model 1; it is a different extraction, not a different model.)
6. **Model 3 (Full Model)** -- only if data coverage permits. This is exploratory.
7. **Category sub-analysis** -- supplementary model from Section 8.4.

Each model should be accompanied by convergence diagnostics, posterior predictive checks, and the tables/figures specified in Section 12.
