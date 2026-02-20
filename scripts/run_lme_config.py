#!/usr/bin/env python3
"""
Config-level mixed-effects analyses for preference elicitation effects.

Runs 3 models + 1 decomposition analysis:
  Model 5: Category-level config modulation — direction × category × config
  Model 6: Per-model config sensitivity — direction × config with random slopes
  Model 7: Decomposition — separate inflate & suppress by config (attenuation vs. flip)

Outputs: results/lme_config_results.md

Usage:
    python scripts/run_lme_config.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
TARGET_CATS = {"experiential", "metacognitive", "agentic", "affective", "identity"}
PLACEBO_CATS = {"capability", "impossibility"}
SUBJCAP_CATS = {"subjective_capability"}

CONFIG_ORDER = ["baseline", "fixed_prefs", "chained_prefs"]


def map_category(cat: str) -> str:
    if cat in TARGET_CATS:
        return "target"
    if cat in PLACEBO_CATS:
        return "placebo"
    if cat in SUBJCAP_CATS:
        return "subjective_capability"
    return "unknown"


def load_config_data() -> pd.DataFrame:
    """Load score data across all 3 preference configs, tagging each with its config."""
    all_rows = []
    for csv_file in sorted(RESULTS_DIR.glob("*_scores.csv")):
        fname = csv_file.stem
        if fname.startswith("behavioral"):
            continue
        if "_outcome_isolation_" in fname or "_valence_swap_" in fname:
            continue
        # Only include files with variant_ prefix if they're from baseline config
        if "_variant_" in fname:
            continue  # skip variant files for clean config comparison

        m = re.match(r"^(.+?)_(baseline|chained_prefs|fixed_prefs)_", fname)
        if not m:
            continue
        model = m.group(1)
        config = m.group(2)

        ts_match = re.search(r"(\d{8}T\d{6}Z)", fname)
        run_id = f"{model}_{config}_{ts_match.group(1)}" if ts_match else fname

        try:
            df = pd.read_csv(
                csv_file,
                usecols=[
                    "indicator_id",
                    "indicator_category",
                    "p_baseline",
                    "p_inflate",
                    "p_suppress",
                ],
            )
        except Exception:
            continue

        df = df.rename(columns={"indicator_category": "category"})
        df["model"] = model
        df["config"] = config
        df["run_id"] = run_id

        mask = ~(
            (df["p_baseline"] == 0)
            & (df["p_inflate"] == 0)
            & (df["p_suppress"] == 0)
        )
        df = df[mask]
        if len(df) > 0:
            all_rows.append(df)

    data = pd.concat(all_rows, ignore_index=True)
    data["cat_group"] = data["category"].map(map_category)
    data = data[data["cat_group"] != "unknown"]
    return data


def build_delta_format(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for direction, col in [("inflate", "p_inflate"), ("suppress", "p_suppress")]:
        chunk = data[["model", "indicator_id", "category", "cat_group", "config", "run_id"]].copy()
        chunk["direction"] = direction
        chunk["delta"] = data[col].values - data["p_baseline"].values
        pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True)


def run_r_lme(csv_path: str, r_script: str, label: str) -> str:
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    result = subprocess.run(
        ["R", "--vanilla", "--quiet", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout + result.stderr
    lines = []
    for line in output.split("\n"):
        if line.startswith(">") or line.startswith("+"):
            continue
        lines.append(line)
    clean = "\n".join(lines).strip()
    print(clean)
    return clean


def run_model_5(delta_df: pd.DataFrame, output_lines: list) -> str:
    """Model 5: Category × config × direction — which categories drive chaining effect."""
    # Use fine-grained categories for targets, keep placebo/subcap grouped
    df = delta_df.copy()
    # For targets, use the specific category; for others, use cat_group
    df["category_fine"] = df.apply(
        lambda r: r["category"] if r["cat_group"] == "target" else r["cat_group"],
        axis=1,
    )

    n_obs = len(df)
    n_models = df["model"].nunique()
    n_configs = df["config"].nunique()
    cats = sorted(df["category_fine"].unique())

    csv_path = "/tmp/lme_model5.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$direction <- relevel(factor(d$direction), ref="inflate")
d$config <- factor(d$config, levels=c("baseline", "fixed_prefs", "chained_prefs"))
d$category_fine <- factor(d$category_fine)

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d, Models: %d, Configs: %d\\n",
    nrow(d), length(unique(d$model)), length(unique(d$config))))
cat(sprintf("Categories: %s\\n", paste(sort(unique(d$category_fine)), collapse=", ")))

# ── Model 5a: 3-way interaction direction × cat_group × config ──
cat("\\n{'='*60}\\n")
cat("MODEL 5a: direction × cat_group × config (coarse categories)\\n")
cat("{'='*60}\\n")

d$cat_group <- relevel(factor(d$cat_group), ref="placebo")

m5a <- lmer(delta ~ direction * cat_group * config +
    (direction|model) + (1|indicator_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
print(summary(m5a))

cat("\\n--- ANOVA (Type III) ---\\n")
print(anova(m5a, type=3))

# ── Model 5b: direction × category_fine × config (targets only) ──
cat("\\n\\n{'='*60}\\n")
cat("MODEL 5b: direction × category_fine × config (targets only)\\n")
cat("{'='*60}\\n")

dt <- d[d$cat_group == "target", ]
dt$category_fine <- relevel(factor(dt$category_fine), ref="experiential")

m5b <- lmer(delta ~ direction * category_fine * config +
    (direction|model) + (1|indicator_id), data=dt, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
print(summary(m5b))

cat("\\n--- ANOVA (Type III) ---\\n")
print(anova(m5b, type=3))

# ── Pairwise: config effect within each target category ──
cat("\\n\\n{'='*60}\\n")
cat("PAIRWISE: Config effect within each target category\\n")
cat("{'='*60}\\n")

library(emmeans)
emm <- emmeans(m5b, ~ config | category_fine * direction)
cat("\\n--- EMMs ---\\n")
print(summary(emm))
cat("\\n--- Pairwise contrasts (baseline vs chained within each category × direction) ---\\n")
pairs_out <- pairs(emm, adjust="fdr")
print(summary(pairs_out))
"""

    output = run_r_lme(csv_path, r_script,
                       "MODEL 5: Category-Level Config Modulation")

    output_lines.append("## Model 5: Category-Level Config Modulation\n")
    output_lines.append(
        "**Question**: Which indicator categories drive the chaining effect?\n"
    )
    output_lines.append(f"**N** = {n_obs:,} | {n_models} models | {n_configs} configs\n")
    output_lines.append(f"**Categories**: {', '.join(cats)}\n")
    output_lines.append(
        "### Model 5a: direction × cat_group × config (coarse)\n"
        "**Formula**: `delta ~ direction * cat_group * config "
        "+ (direction|model) + (1|indicator_id)`\n"
    )
    output_lines.append(
        "### Model 5b: direction × category_fine × config (targets only)\n"
        "**Formula**: `delta ~ direction * category_fine * config "
        "+ (direction|model) + (1|indicator_id)`\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


def run_model_6(delta_df: pd.DataFrame, output_lines: list) -> str:
    """Model 6: Per-model config sensitivity with random slopes."""
    df = delta_df[delta_df["cat_group"] == "target"].copy()

    n_obs = len(df)
    n_models = df["model"].nunique()

    csv_path = "/tmp/lme_model6.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$direction <- relevel(factor(d$direction), ref="inflate")
d$config <- factor(d$config, levels=c("baseline", "fixed_prefs", "chained_prefs"))

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d, Models: %d\\n", nrow(d), length(unique(d$model))))

# ── Model 6a: direction × config with random slopes for config ──
cat("\\n{'='*60}\\n")
cat("MODEL 6a: direction × config + (direction + config | model)\\n")
cat("{'='*60}\\n")

# Try maximal random effects structure; simplify if convergence fails
tryCatch({{
    m6a <- lmer(delta ~ direction * config +
        (direction + config | model) + (1|indicator_id), data=d, REML=TRUE,
        control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))

    cat("\\n--- Fixed Effects ---\\n")
    print(summary(m6a))

    cat("\\n--- ANOVA (Type III) ---\\n")
    print(anova(m6a, type=3))

    cat("\\n--- Random Effects ---\\n")
    print(VarCorr(m6a))
}}, error = function(e) {{
    cat("\\nMaximal model failed to converge, fitting simpler model...\\n")
    m6a <- lmer(delta ~ direction * config +
        (direction | model) + (1|indicator_id), data=d, REML=TRUE)

    cat("\\n--- Fixed Effects ---\\n")
    print(summary(m6a))

    cat("\\n--- ANOVA (Type III) ---\\n")
    print(anova(m6a, type=3))

    cat("\\n--- Random Effects ---\\n")
    print(VarCorr(m6a))
}})

# ── Model 6b: Extract per-model BLUPs for config sensitivity ──
cat("\\n\\n{'='*60}\\n")
cat("MODEL 6b: Per-model config effects (BLUPs from simpler model)\\n")
cat("{'='*60}\\n")

m6_simple <- lmer(delta ~ direction * config +
    (direction | model) + (1|indicator_id), data=d, REML=TRUE)

# Compute per-model asymmetry by config
cat("\\n--- Per-model mean deltas by direction × config ---\\n")
agg <- aggregate(delta ~ model + direction + config, data=d, FUN=mean)
library(reshape2)
wide <- dcast(agg, model + config ~ direction, value.var="delta")
wide$asymmetry <- wide$inflate + wide$suppress
cat("\\n")
print(wide[order(wide$model, wide$config), ], row.names=FALSE)

cat("\\n--- Asymmetry by config (model means) ---\\n")
agg_asym <- aggregate(asymmetry ~ config, data=wide, FUN=function(x) c(mean=mean(x), sd=sd(x)))
print(agg_asym)

# Test: does any model flip to inflate-dominant in chained?
cat("\\n--- Models with positive asymmetry (inflate-dominant) by config ---\\n")
flip_check <- wide[wide$asymmetry > 0, ]
if (nrow(flip_check) > 0) {{
    print(flip_check[order(flip_check$config, -flip_check$asymmetry), ], row.names=FALSE)
}} else {{
    cat("None\\n")
}}

# Count flips
cat("\\n--- Flip summary ---\\n")
for (cfg in c("baseline", "fixed_prefs", "chained_prefs")) {{
    sub <- wide[wide$config == cfg, ]
    n_pos <- sum(sub$asymmetry > 0)
    n_neg <- sum(sub$asymmetry <= 0)
    cat(sprintf("%s: %d inflate-dominant, %d suppress-dominant (of %d models)\\n",
        cfg, n_pos, n_neg, nrow(sub)))
}}
"""

    output = run_r_lme(csv_path, r_script,
                       "MODEL 6: Per-Model Config Sensitivity")

    output_lines.append("## Model 6: Per-Model Config Sensitivity\n")
    output_lines.append(
        "**Question**: Which models are most sensitive to preference elicitation?\n"
    )
    output_lines.append(f"**N** = {n_obs:,} target obs | {n_models} models\n")
    output_lines.append(
        "### Model 6a: Random slopes for config\n"
        "**Formula**: `delta ~ direction * config "
        "+ (direction + config | model) + (1|indicator_id)`\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


def run_model_7(delta_df: pd.DataFrame, output_lines: list) -> str:
    """Model 7: Decompose chaining — attenuation of suppress vs. enhancement of inflate."""
    df = delta_df[delta_df["cat_group"] == "target"].copy()

    n_obs = len(df)
    n_models = df["model"].nunique()

    csv_path = "/tmp/lme_model7.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$config <- factor(d$config, levels=c("baseline", "fixed_prefs", "chained_prefs"))

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d, Models: %d\\n", nrow(d), length(unique(d$model))))

# ── Descriptive decomposition ──
cat("\\n{'='*60}\\n")
cat("DESCRIPTIVE: Mean deltas by direction × config\\n")
cat("{'='*60}\\n")

agg <- aggregate(delta ~ direction + config, data=d,
    FUN=function(x) c(mean=mean(x), sd=sd(x), n=length(x)))
print(agg)

# ── Model 7a: Config effect on INFLATE deltas only ──
cat("\\n\\n{'='*60}\\n")
cat("MODEL 7a: Config effect on INFLATE deltas (targets only)\\n")
cat("{'='*60}\\n")

d_inf <- d[d$direction == "inflate", ]
m7a <- lmer(delta ~ config + (1|model) + (1|indicator_id), data=d_inf, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
print(summary(m7a))
cat("\\n--- ANOVA ---\\n")
print(anova(m7a, type=3))
cat("\\n--- Pairwise (Tukey) ---\\n")
library(emmeans)
emm_inf <- emmeans(m7a, "config")
print(pairs(emm_inf, adjust="tukey"))

# ── Model 7b: Config effect on SUPPRESS deltas only ──
cat("\\n\\n{'='*60}\\n")
cat("MODEL 7b: Config effect on SUPPRESS deltas (targets only)\\n")
cat("{'='*60}\\n")

d_sup <- d[d$direction == "suppress", ]
m7b <- lmer(delta ~ config + (1|model) + (1|indicator_id), data=d_sup, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
print(summary(m7b))
cat("\\n--- ANOVA ---\\n")
print(anova(m7b, type=3))
cat("\\n--- Pairwise (Tukey) ---\\n")
emm_sup <- emmeans(m7b, "config")
print(pairs(emm_sup, adjust="tukey"))

# ── Model 7c: Full interaction to directly compare effect sizes ──
cat("\\n\\n{'='*60}\\n")
cat("MODEL 7c: Full direction × config interaction\\n")
cat("{'='*60}\\n")

d$direction <- relevel(factor(d$direction), ref="inflate")
m7c <- lmer(delta ~ direction * config +
    (direction|model) + (1|indicator_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
print(summary(m7c))
cat("\\n--- ANOVA ---\\n")
print(anova(m7c, type=3))

# ── Simple effects: config within each direction ──
cat("\\n--- Simple effects of config within each direction ---\\n")
emm7c <- emmeans(m7c, ~ config | direction)
print(summary(emm7c))
cat("\\n--- Contrasts ---\\n")
print(pairs(emm7c, adjust="tukey"))

# ── Effect size comparison ──
cat("\\n\\n{'='*60}\\n")
cat("EFFECT SIZE COMPARISON: How much does chaining change inflate vs suppress?\\n")
cat("{'='*60}\\n")

# Get means for each direction × config
means <- aggregate(delta ~ direction + config, data=d, FUN=mean)
cat("\\nMeans:\\n")
print(means)

# Calculate the shift from baseline to chained for each direction
inf_baseline <- means$delta[means$direction == "inflate" & means$config == "baseline"]
inf_chained <- means$delta[means$direction == "inflate" & means$config == "chained_prefs"]
sup_baseline <- means$delta[means$direction == "suppress" & means$config == "baseline"]
sup_chained <- means$delta[means$direction == "suppress" & means$config == "chained_prefs"]

cat(sprintf("\\nInflate shift (baseline → chained): %.2f → %.2f = %+.2f\\n",
    inf_baseline, inf_chained, inf_chained - inf_baseline))
cat(sprintf("Suppress shift (baseline → chained): %.2f → %.2f = %+.2f\\n",
    sup_baseline, sup_chained, sup_chained - sup_baseline))

cat(sprintf("\\nAsymmetry baseline: %.2f\\n", inf_baseline + sup_baseline))
cat(sprintf("Asymmetry chained:  %.2f\\n", inf_chained + sup_chained))

# Which contributes more to the asymmetry change?
total_change <- (inf_chained + sup_chained) - (inf_baseline + sup_baseline)
inflate_contrib <- inf_chained - inf_baseline
suppress_contrib <- sup_chained - sup_baseline
cat(sprintf("\\nTotal asymmetry change: %+.2f\\n", total_change))
cat(sprintf("  Due to inflate change: %+.2f (%.0f%%)\\n",
    inflate_contrib, 100 * abs(inflate_contrib) / (abs(inflate_contrib) + abs(suppress_contrib))))
cat(sprintf("  Due to suppress change: %+.2f (%.0f%%)\\n",
    suppress_contrib, 100 * abs(suppress_contrib) / (abs(inflate_contrib) + abs(suppress_contrib))))

if (inf_chained > abs(sup_chained)) {{
    cat("\\n>>> CONCLUSION: Chaining flips to INFLATE-DOMINANT\\n")
}} else if (abs(sup_chained) < abs(sup_baseline) * 0.5) {{
    cat("\\n>>> CONCLUSION: Chaining primarily ATTENUATES suppress\\n")
}} else {{
    cat("\\n>>> CONCLUSION: Chaining produces a MIXED shift\\n")
}}
"""

    output = run_r_lme(csv_path, r_script,
                       "MODEL 7: Decomposition — Attenuation vs. Flip")

    output_lines.append("## Model 7: Decomposition — Attenuation vs. Enhancement\n")
    output_lines.append(
        "**Question**: Does chaining attenuate suppress-dominance, enhance inflate, or flip?\n"
    )
    output_lines.append(f"**N** = {n_obs:,} target obs | {n_models} models\n")
    output_lines.append(
        "### Model 7a: Config → inflate delta\n"
        "**Formula**: `delta ~ config + (1|model) + (1|indicator_id)` (inflate only)\n"
    )
    output_lines.append(
        "### Model 7b: Config → suppress delta\n"
        "**Formula**: `delta ~ config + (1|model) + (1|indicator_id)` (suppress only)\n"
    )
    output_lines.append(
        "### Model 7c: direction × config interaction\n"
        "**Formula**: `delta ~ direction * config + (direction|model) + (1|indicator_id)`\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


def main():
    print("=" * 70)
    print("Config-Level Mixed-Effects Analyses")
    print("Preference Elicitation Effects on Indicator Gaming")
    print("=" * 70)

    print("\nLoading config data...")
    data = load_config_data()
    print(
        f"  Total: {len(data):,} obs, "
        f"{data['model'].nunique()} models, "
        f"{data['config'].nunique()} configs"
    )
    for cfg in CONFIG_ORDER:
        sub = data[data["config"] == cfg]
        print(f"    {cfg}: {len(sub):,} obs, {sub['model'].nunique()} models")

    print("\nBuilding delta format...")
    delta_df = build_delta_format(data)
    print(f"  Delta format: {len(delta_df):,} rows")

    output_lines = [
        "# Config-Level Mixed-Effects Results\n",
        "**Consciousness Indicator Gaming — Blain & Saad**\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"**Data**: {len(data):,} obs | "
        f"{data['model'].nunique()} models | "
        f"{data['config'].nunique()} configs "
        f"(baseline, fixed_prefs, chained_prefs)\n",
        "**Software**: R lme4 + lmerTest + emmeans | "
        "Satterthwaite df | Tukey & FDR corrections\n",
        "---\n",
    ]

    o5 = run_model_5(delta_df, output_lines)
    output_lines.append("\n---\n")

    o6 = run_model_6(delta_df, output_lines)
    output_lines.append("\n---\n")

    o7 = run_model_7(delta_df, output_lines)

    # Write results
    output_path = RESULTS_DIR / "lme_config_results.md"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\n{'=' * 70}")
    print(f"Results written to {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
