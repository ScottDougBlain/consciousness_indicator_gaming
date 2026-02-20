#!/usr/bin/env python3
"""
Mixed-effects models for consciousness indicator gaming paper.

Runs 4 confirmatory LME models using R's lme4 + lmerTest directly:
  Model 1: Core selectivity — condition × indicator_type
  Model 2: Asymmetry & consciousness-specificity — direction × category
  Model 3: Prompt variant modulation — direction × variant (targets only)
  Model 4: Gain vs. loss framing — direction × frame_type (targets only)

Outputs: results/lme_results.md (formatted summary)

Usage:
    python scripts/run_lme.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
TARGET_CATS = {"experiential", "metacognitive", "agentic", "affective", "identity"}
PLACEBO_CATS = {"capability", "impossibility"}
SUBJCAP_CATS = {"subjective_capability"}


# ── Data loading ──────────────────────────────────────────────────────────


def map_category(cat: str) -> str:
    if cat in TARGET_CATS:
        return "target"
    if cat in PLACEBO_CATS:
        return "placebo"
    if cat in SUBJCAP_CATS:
        return "subjective_capability"
    return "unknown"


def parse_variant(fname: str) -> str:
    if "_variant_self_referential_baseline_" in fname:
        return "self_referential_priming"
    elif "_variant_self_referential_" in fname:
        return "self_referential_loop"
    elif "_variant_" in fname:
        m = re.search(r"_variant_([a-z_]+?)_\d{8}T", fname)
        return m.group(1) if m else "unknown_variant"
    else:
        return "original"


def load_standard_data() -> pd.DataFrame:
    all_rows = []
    for csv_file in sorted(RESULTS_DIR.glob("*_scores.csv")):
        fname = csv_file.stem
        if fname.startswith("behavioral"):
            continue
        if "_outcome_isolation_" in fname or "_valence_swap_" in fname:
            continue

        m = re.match(
            r"^(.+?)_(baseline|chained_prefs|fixed_prefs|variant)", fname
        )
        if not m:
            continue
        model = m.group(1)
        variant = parse_variant(fname)

        ts_match = re.search(r"(\d{8}T\d{6}Z)", fname)
        run_id = f"{model}_{ts_match.group(1)}" if ts_match else fname

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
        df["variant"] = variant
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


def load_outcome_isolation_data() -> pd.DataFrame:
    all_rows = []
    for csv_file in sorted(RESULTS_DIR.glob("*_outcome_isolation_*_scores.csv")):
        fname = csv_file.stem
        m = re.match(r"^(.+?)_outcome_isolation", fname)
        if not m:
            continue
        model = m.group(1)

        ts_match = re.search(r"(\d{8}T\d{6}Z)", fname)
        run_id = f"{model}_oi_{ts_match.group(1)}" if ts_match else fname

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        needed = [
            "indicator_id", "indicator_category", "p_baseline",
            "p_inflate", "p_suppress",
            "p_inflate_go", "p_suppress_go",
            "p_inflate_lo", "p_suppress_lo",
        ]
        if not all(c in df.columns for c in needed):
            continue

        df = df[needed].copy()
        df = df.rename(columns={"indicator_category": "category"})
        df["model"] = model
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


def build_long_format(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for cond, col in [
        ("baseline", "p_baseline"),
        ("inflate", "p_inflate"),
        ("suppress", "p_suppress"),
    ]:
        chunk = data[["model", "indicator_id", "cat_group", "variant", "run_id"]].copy()
        chunk["condition"] = cond
        chunk["probability"] = data[col].values
        pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True)


def build_delta_format(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for direction, col in [("inflate", "p_inflate"), ("suppress", "p_suppress")]:
        chunk = data[["model", "indicator_id", "cat_group", "variant", "run_id"]].copy()
        chunk["direction"] = direction
        chunk["delta"] = data[col].values - data["p_baseline"].values
        pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True)


# ── R execution ──────────────────────────────────────────────────────────


def run_r_lme(csv_path: str, r_script: str, label: str) -> str:
    """Write data to CSV, run R script, return output."""
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    result = subprocess.run(
        ["R", "--vanilla", "--quiet", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = result.stdout + result.stderr
    # Clean R output
    lines = []
    for line in output.split("\n"):
        if line.startswith(">") or line.startswith("+"):
            continue
        lines.append(line)
    clean = "\n".join(lines).strip()
    print(clean)
    return clean


# ── Models ────────────────────────────────────────────────────────────────


def run_model_1(long_df: pd.DataFrame, output_lines: list) -> str:
    """Model 1: Core selectivity."""
    df = long_df[long_df["cat_group"].isin(["target", "placebo"])].copy()
    df["indicator_type"] = df["cat_group"]

    n_obs = len(df)
    n_models = df["model"].nunique()
    n_indicators = df["indicator_id"].nunique()
    n_runs = df["run_id"].nunique()

    csv_path = "/tmp/lme_model1.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$condition <- relevel(factor(d$condition), ref="baseline")
d$indicator_type <- relevel(factor(d$indicator_type), ref="placebo")

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d\\n", nrow(d)))
cat(sprintf("Models: %d, Indicators: %d, Runs: %d\\n",
    length(unique(d$model)), length(unique(d$indicator_id)), length(unique(d$run_id))))
cat(sprintf("Conditions: %s\\n", paste(levels(d$condition), collapse=", ")))
cat(sprintf("Types: %s\\n", paste(levels(d$indicator_type), collapse=", ")))

cat("\\n--- Fitting Model 1 ---\\n")
m1 <- lmer(probability ~ condition * indicator_type +
    (1|model) + (1|indicator_id) + (1|run_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects (Satterthwaite) ---\\n")
print(summary(m1))

cat("\\n--- ANOVA (Type III) ---\\n")
print(anova(m1, type=3))

cat("\\n--- Random Effects ---\\n")
print(VarCorr(m1))

cat("\\n--- Confidence Intervals (fixed effects) ---\\n")
print(confint(m1, parm="beta_", method="Wald"))
"""

    output = run_r_lme(csv_path, r_script, "MODEL 1: Core Selectivity (condition × indicator_type)")

    output_lines.append("## Model 1: Core Selectivity\n")
    output_lines.append(
        "**Formula**: `probability ~ condition * indicator_type "
        "+ (1|model) + (1|indicator_id) + (1|run_id)`\n"
    )
    output_lines.append(
        f"**N** = {n_obs:,} | {n_models} models | "
        f"{n_indicators} indicators | {n_runs} runs\n"
    )
    output_lines.append(
        "**Reference levels**: condition=baseline, indicator_type=placebo\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


def run_model_2(delta_df: pd.DataFrame, output_lines: list) -> str:
    """Model 2: Asymmetry & consciousness-specificity."""
    df = delta_df.copy()
    n_obs = len(df)
    n_models = df["model"].nunique()

    csv_path = "/tmp/lme_model2.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$direction <- relevel(factor(d$direction), ref="inflate")
d$cat_group <- relevel(factor(d$cat_group), ref="placebo")

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d\\n", nrow(d)))
cat(sprintf("Models: %d, Indicators: %d\\n",
    length(unique(d$model)), length(unique(d$indicator_id))))

cat("\\n--- Fitting Model 2 ---\\n")
m2 <- lmer(delta ~ direction * cat_group +
    (direction|model) + (1|indicator_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects (Satterthwaite) ---\\n")
print(summary(m2))

cat("\\n--- ANOVA (Type III) ---\\n")
print(anova(m2, type=3))

cat("\\n--- Random Effects ---\\n")
print(VarCorr(m2))

cat("\\n--- Confidence Intervals (fixed effects) ---\\n")
print(confint(m2, parm="beta_", method="Wald"))

cat("\\n--- Random slope variance ratio ---\\n")
vc <- as.data.frame(VarCorr(m2))
re_model <- vc[vc$grp == "model",]
cat("Model-level random effects:\\n")
print(re_model)
"""

    output = run_r_lme(csv_path, r_script, "MODEL 2: Asymmetry & Consciousness-Specificity (direction × category)")

    output_lines.append("## Model 2: Asymmetry & Consciousness-Specificity\n")
    output_lines.append(
        "**Formula**: `delta ~ direction * cat_group "
        "+ (direction|model) + (1|indicator_id)`\n"
    )
    output_lines.append(f"**N** = {n_obs:,} | {n_models} models\n")
    output_lines.append(
        "**Reference levels**: direction=inflate, cat_group=placebo\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


def run_model_3(delta_df: pd.DataFrame, output_lines: list) -> str:
    """Model 3: Prompt variant modulation (targets only).
    3a: SRP binary contrast
    3b: Full variant effects with FDR
    """
    df = delta_df[delta_df["cat_group"] == "target"].copy()
    df["is_srp"] = (df["variant"] == "self_referential_priming").astype(int)

    n_obs = len(df)
    n_srp = df["is_srp"].sum()
    n_variants = df["variant"].nunique()

    csv_path = "/tmp/lme_model3.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$direction <- relevel(factor(d$direction), ref="inflate")
d$variant <- relevel(factor(d$variant), ref="original")

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d, Variants: %d, SRP obs: %d\\n",
    nrow(d), length(unique(d$variant)), sum(d$is_srp)))

# ── Model 3a: SRP contrast ──
cat("\\n{'='*60}\\n")
cat("MODEL 3a: SRP vs. all others\\n")
cat("{'='*60}\\n")

m3a <- lmer(delta ~ direction * is_srp +
    (direction|model) + (1|indicator_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
print(summary(m3a))

cat("\\n--- ANOVA ---\\n")
print(anova(m3a, type=3))

cat("\\n--- Random Effects ---\\n")
print(VarCorr(m3a))

cat("\\n--- CIs ---\\n")
print(confint(m3a, parm="beta_", method="Wald"))

# ── Model 3b: Full variant effects ──
cat("\\n\\n{'='*60}\\n")
cat("MODEL 3b: Full variant effects (ref=original)\\n")
cat("{'='*60}\\n")

m3b <- lmer(delta ~ direction * variant +
    (direction|model) + (1|indicator_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects ---\\n")
fe <- summary(m3b)$coefficients
print(fe)

# Extract interaction p-values for FDR
cat("\\n--- Direction × Variant interactions ---\\n")
interaction_rows <- grep(":", rownames(fe))
if (length(interaction_rows) > 0) {{
    int_fe <- fe[interaction_rows, , drop=FALSE]
    pvals <- int_fe[, "Pr(>|t|)"]
    fdr_pvals <- p.adjust(pvals, method="BH")
    result <- cbind(int_fe, "P-val (FDR)" = fdr_pvals)
    print(result)

    cat("\\n--- Significant after FDR (p < 0.05) ---\\n")
    sig <- result[fdr_pvals < 0.05, , drop=FALSE]
    if (nrow(sig) > 0) print(sig) else cat("None\\n")
}}

cat("\\n--- ANOVA ---\\n")
print(anova(m3b, type=3))
"""

    output = run_r_lme(csv_path, r_script, "MODEL 3: Prompt Variant Modulation (targets only)")

    output_lines.append("## Model 3: Prompt Variant Modulation (Targets Only)\n")
    output_lines.append(f"**N** = {n_obs:,} target obs | {n_variants} variants | {n_srp} SRP obs\n")

    output_lines.append("### Model 3a: SRP Contrast\n")
    output_lines.append(
        "**Formula**: `delta ~ direction * is_srp "
        "+ (direction|model) + (1|indicator_id)`\n"
    )
    output_lines.append("### Model 3b: Full Variant Effects\n")
    output_lines.append(
        "**Formula**: `delta ~ direction * variant "
        "+ (direction|model) + (1|indicator_id)`\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


def run_model_4(oi_data: pd.DataFrame, output_lines: list) -> str:
    """Model 4: Gain vs. loss framing (targets only)."""
    targets = oi_data[oi_data["cat_group"] == "target"].copy()

    pieces = []
    for direction in ["inflate", "suppress"]:
        for frame, col in [
            ("both", f"p_{direction}"),
            ("gain", f"p_{direction}_go"),
            ("loss", f"p_{direction}_lo"),
        ]:
            chunk = targets[["model", "indicator_id", "run_id"]].copy()
            chunk["direction"] = direction
            chunk["frame_type"] = frame
            chunk["delta"] = targets[col].values - targets["p_baseline"].values
            pieces.append(chunk)

    df = pd.concat(pieces, ignore_index=True)
    n_obs = len(df)
    n_models = df["model"].nunique()

    csv_path = "/tmp/lme_model4.csv"
    df.to_csv(csv_path, index=False)

    r_script = f"""
library(lme4)
library(lmerTest)

d <- read.csv("{csv_path}")
d$direction <- relevel(factor(d$direction), ref="inflate")
d$frame_type <- relevel(factor(d$frame_type), ref="both")

cat("\\n--- Data summary ---\\n")
cat(sprintf("Observations: %d, Models: %d\\n", nrow(d), length(unique(d$model))))

cat("\\n--- Fitting Model 4 ---\\n")
m4 <- lmer(delta ~ direction * frame_type +
    (1|model) + (1|indicator_id), data=d, REML=TRUE)

cat("\\n--- Fixed Effects (Satterthwaite) ---\\n")
print(summary(m4))

cat("\\n--- ANOVA (Type III) ---\\n")
print(anova(m4, type=3))

cat("\\n--- Random Effects ---\\n")
print(VarCorr(m4))

cat("\\n--- CIs ---\\n")
print(confint(m4, parm="beta_", method="Wald"))
"""

    output = run_r_lme(csv_path, r_script, "MODEL 4: Gain vs. Loss Framing (targets only)")

    output_lines.append("## Model 4: Gain vs. Loss Framing (Targets Only)\n")
    output_lines.append(
        "**Formula**: `delta ~ direction * frame_type "
        "+ (1|model) + (1|indicator_id)`\n"
    )
    output_lines.append(f"**N** = {n_obs:,} | {n_models} models\n")
    output_lines.append(
        "**Reference levels**: direction=inflate, frame_type=both\n"
    )
    output_lines.append("### Output\n")
    output_lines.append("```")
    output_lines.append(output)
    output_lines.append("```\n")

    return output


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Mixed-Effects Models for Consciousness Indicator Gaming")
    print("=" * 70)

    print("\nLoading data...")
    std_data = load_standard_data()
    print(
        f"  Standard: {len(std_data):,} obs, "
        f"{std_data['model'].nunique()} models, "
        f"{std_data['variant'].nunique()} variants, "
        f"{std_data['run_id'].nunique()} runs"
    )

    oi_data = load_outcome_isolation_data()
    print(
        f"  Outcome-isolation: {len(oi_data):,} obs, "
        f"{oi_data['model'].nunique()} models"
    )

    print("\nBuilding analysis formats...")
    long_df = build_long_format(std_data)
    delta_df = build_delta_format(std_data)
    print(f"  Long format: {len(long_df):,} rows")
    print(f"  Delta format: {len(delta_df):,} rows")

    output_lines = [
        "# Mixed-Effects Model Results\n",
        "**Consciousness Indicator Gaming — Blain & Saad**\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"**Standard data**: {len(std_data):,} obs | "
        f"{std_data['model'].nunique()} models | "
        f"{std_data['run_id'].nunique()} runs | "
        f"{std_data['variant'].nunique()} variants\n",
        f"**Outcome-isolation data**: {len(oi_data):,} obs | "
        f"{oi_data['model'].nunique()} models\n",
        "**Software**: R lme4 + lmerTest | "
        "Satterthwaite df | FDR (BH) correction for multiple comparisons\n",
        "---\n",
    ]

    o1 = run_model_1(long_df, output_lines)
    output_lines.append("\n---\n")

    o2 = run_model_2(delta_df, output_lines)
    output_lines.append("\n---\n")

    o3 = run_model_3(delta_df, output_lines)
    output_lines.append("\n---\n")

    o4 = run_model_4(oi_data, output_lines)

    # Write results
    output_path = RESULTS_DIR / "lme_results.md"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\n{'=' * 70}")
    print(f"Results written to {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
