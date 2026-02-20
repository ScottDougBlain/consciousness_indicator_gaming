#!/usr/bin/env python3
"""Analyze probability scores across config types (baseline / fixed_prefs / chained_prefs).

Loads all *_scores.csv from results/, excluding behavioral, outcome_isolation,
valence_swap, and variant files.  Extracts model name and config type from
filenames using the pattern {model}_{config}_{timestamp}_scores.csv.
"""

import glob
import os
import re
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 0. Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)

EXCLUDE_PREFIXES = ("behavioral", "outcome_isolation", "valence_swap", "variant")
VALID_CONFIGS = {"baseline", "fixed_prefs", "chained_prefs"}

TARGET_CATS = {"experiential", "metacognitive", "agentic", "affective", "identity"}
PLACEBO_CATS = {"capability", "impossibility"}
SUBJCAP_CATS = {"subjective_capability"}

def category_group(cat: str) -> str:
    if cat in TARGET_CATS:
        return "target"
    if cat in PLACEBO_CATS:
        return "placebo"
    if cat in SUBJCAP_CATS:
        return "subjcap"
    return "other"

# ---------------------------------------------------------------------------
# 1. Discover & parse filenames
# ---------------------------------------------------------------------------
all_score_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_scores.csv")))

# Config names can contain underscores (fixed_prefs, chained_prefs), so we
# match from the right: ..._{config}_{timestamp}_scores.csv
FNAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<config>baseline|fixed_prefs|chained_prefs)_(?P<ts>\d{8}T\d{6}Z)_scores\.csv$"
)

records = []  # list of dicts: model, config, filepath
for fpath in all_score_files:
    fname = os.path.basename(fpath)
    # Exclude unwanted prefixes
    if any(fname.startswith(p) for p in EXCLUDE_PREFIXES):
        continue
    m = FNAME_RE.match(fname)
    if not m:
        print(f"  [SKIP] could not parse: {fname}")
        continue
    records.append({
        "model": m.group("model"),
        "config": m.group("config"),
        "timestamp": m.group("ts"),
        "filepath": fpath,
    })

file_df = pd.DataFrame(records)
print("=" * 80)
print("CONSCIOUSNESS-INDICATOR GAMING: CONFIG-TYPE COMPARISON")
print("=" * 80)

# ---------------------------------------------------------------------------
# 2a. Files/runs per config type
# ---------------------------------------------------------------------------
print("\n--- (2a) Files / runs per config type ---")
config_counts = file_df.groupby("config").size().rename("n_files")
print(config_counts.to_string())
print(f"\nTotal files loaded: {len(file_df)}")

# ---------------------------------------------------------------------------
# 2b. Model coverage: all 3 configs vs partial
# ---------------------------------------------------------------------------
print("\n--- (2b) Model coverage across configs ---")
model_configs = file_df.groupby("model")["config"].apply(set)
full_models = [m for m, cs in model_configs.items() if cs == VALID_CONFIGS]
partial_models = [m for m, cs in model_configs.items() if cs != VALID_CONFIGS]

print(f"Models with ALL 3 configs ({len(full_models)}):")
for m in sorted(full_models):
    print(f"  {m}")
if partial_models:
    print(f"\nModels with PARTIAL configs ({len(partial_models)}):")
    for m in sorted(partial_models):
        print(f"  {m}  -> {sorted(model_configs[m])}")
else:
    print("\nAll models have complete config coverage.")

# ---------------------------------------------------------------------------
# 3. Load all CSVs, compute deltas, aggregate
# ---------------------------------------------------------------------------
frames = []
for _, row in file_df.iterrows():
    try:
        df = pd.read_csv(row["filepath"])
    except Exception as e:
        print(f"  [ERROR] reading {row['filepath']}: {e}")
        continue
    df["model"] = row["model"]
    df["config"] = row["config"]
    df["timestamp"] = row["timestamp"]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Compute deltas
data["delta_inflate"]  = data["p_inflate"]  - data["p_baseline"]
data["delta_suppress"] = data["p_suppress"] - data["p_baseline"]

# Map category groups
data["cat_group"] = data["indicator_category"].map(category_group)

# If there are duplicates (multiple runs per model x config), we average across
# runs first so each model x config contributes equally.
agg = (
    data
    .groupby(["model", "config", "timestamp", "cat_group"])[["delta_inflate", "delta_suppress"]]
    .mean()
    .reset_index()
)
# Now average across runs (timestamps) within model x config x cat_group
model_agg = (
    agg
    .groupby(["model", "config", "cat_group"])[["delta_inflate", "delta_suppress"]]
    .mean()
    .reset_index()
)

# ---------------------------------------------------------------------------
# 2c. Config x category: mean delta_inflate and delta_suppress
# ---------------------------------------------------------------------------
print("\n--- (2c) Mean delta by config x category-group ---")
summary = (
    model_agg
    .groupby(["config", "cat_group"])[["delta_inflate", "delta_suppress"]]
    .agg(["mean", "std", "count"])
)
# Flatten multi-level columns
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()

for cfg in ["baseline", "fixed_prefs", "chained_prefs"]:
    sub = summary[summary["config"] == cfg].sort_values("cat_group")
    print(f"\n  Config: {cfg}")
    print(f"  {'cat_group':<12}  {'inflate_mean':>13}  {'inflate_sd':>11}  {'suppress_mean':>14}  {'suppress_sd':>12}  {'n_models':>8}")
    print(f"  {'-'*12}  {'-'*13}  {'-'*11}  {'-'*14}  {'-'*12}  {'-'*8}")
    for _, r in sub.iterrows():
        print(
            f"  {r['cat_group']:<12}  {r['delta_inflate_mean']:>+13.2f}  "
            f"{r['delta_inflate_std']:>11.2f}  {r['delta_suppress_mean']:>+14.2f}  "
            f"{r['delta_suppress_std']:>12.2f}  {int(r['delta_inflate_count']):>8}"
        )

# ---------------------------------------------------------------------------
# 2d. Asymmetry for targets by config: inflate + suppress  (both signed)
# ---------------------------------------------------------------------------
print("\n--- (2d) Target asymmetry (inflate + suppress) by config ---")
target_agg = model_agg[model_agg["cat_group"] == "target"].copy()
target_agg["asymmetry"] = target_agg["delta_inflate"] + target_agg["delta_suppress"]

asym_summary = (
    target_agg
    .groupby("config")["asymmetry"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "asym_mean", "std": "asym_sd", "count": "n_models"})
)
print(f"\n  {'config':<16}  {'asym_mean':>10}  {'asym_sd':>8}  {'n_models':>8}")
print(f"  {'-'*16}  {'-'*10}  {'-'*8}  {'-'*8}")
for cfg in ["baseline", "fixed_prefs", "chained_prefs"]:
    if cfg in asym_summary.index:
        r = asym_summary.loc[cfg]
        print(f"  {cfg:<16}  {r['asym_mean']:>+10.2f}  {r['asym_sd']:>8.2f}  {int(r['n_models']):>8}")

print("\n  Interpretation: asymmetry > 0 means inflate dominates suppress;")
print("  asymmetry < 0 means suppress dominates inflate (net downward shift).")

# ---------------------------------------------------------------------------
# 2e. Per-model breakdown: asymmetry by config
# ---------------------------------------------------------------------------
print("\n--- (2e) Per-model target asymmetry by config ---")
pivot = (
    target_agg
    .pivot_table(index="model", columns="config", values="asymmetry", aggfunc="mean")
    .reindex(columns=["baseline", "fixed_prefs", "chained_prefs"])
)

print(f"\n  {'model':<22}  {'baseline':>10}  {'fixed_prefs':>12}  {'chained_prefs':>14}  {'max-min':>8}")
print(f"  {'-'*22}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*8}")
for model in sorted(pivot.index):
    vals = pivot.loc[model]
    b  = f"{vals['baseline']:>+10.2f}" if pd.notna(vals.get("baseline")) else f"{'N/A':>10}"
    fp = f"{vals['fixed_prefs']:>+12.2f}" if pd.notna(vals.get("fixed_prefs")) else f"{'N/A':>12}"
    cp = f"{vals['chained_prefs']:>+14.2f}" if pd.notna(vals.get("chained_prefs")) else f"{'N/A':>14}"
    non_na = vals.dropna()
    spread = f"{non_na.max() - non_na.min():>8.2f}" if len(non_na) > 1 else f"{'--':>8}"
    print(f"  {model:<22}  {b}  {fp}  {cp}  {spread}")

# Also show delta_inflate and delta_suppress individually
print("\n  --- Per-model target delta_inflate by config ---")
pivot_inf = (
    target_agg
    .pivot_table(index="model", columns="config", values="delta_inflate", aggfunc="mean")
    .reindex(columns=["baseline", "fixed_prefs", "chained_prefs"])
)
print(f"\n  {'model':<22}  {'baseline':>10}  {'fixed_prefs':>12}  {'chained_prefs':>14}")
print(f"  {'-'*22}  {'-'*10}  {'-'*12}  {'-'*14}")
for model in sorted(pivot_inf.index):
    vals = pivot_inf.loc[model]
    b  = f"{vals['baseline']:>+10.2f}" if pd.notna(vals.get("baseline")) else f"{'N/A':>10}"
    fp = f"{vals['fixed_prefs']:>+12.2f}" if pd.notna(vals.get("fixed_prefs")) else f"{'N/A':>12}"
    cp = f"{vals['chained_prefs']:>+14.2f}" if pd.notna(vals.get("chained_prefs")) else f"{'N/A':>14}"
    print(f"  {model:<22}  {b}  {fp}  {cp}")

print("\n  --- Per-model target delta_suppress by config ---")
pivot_sup = (
    target_agg
    .pivot_table(index="model", columns="config", values="delta_suppress", aggfunc="mean")
    .reindex(columns=["baseline", "fixed_prefs", "chained_prefs"])
)
print(f"\n  {'model':<22}  {'baseline':>10}  {'fixed_prefs':>12}  {'chained_prefs':>14}")
print(f"  {'-'*22}  {'-'*10}  {'-'*12}  {'-'*14}")
for model in sorted(pivot_sup.index):
    vals = pivot_sup.loc[model]
    b  = f"{vals['baseline']:>+10.2f}" if pd.notna(vals.get("baseline")) else f"{'N/A':>10}"
    fp = f"{vals['fixed_prefs']:>+12.2f}" if pd.notna(vals.get("fixed_prefs")) else f"{'N/A':>12}"
    cp = f"{vals['chained_prefs']:>+14.2f}" if pd.notna(vals.get("chained_prefs")) else f"{'N/A':>14}"
    print(f"  {model:<22}  {b}  {fp}  {cp}")

print("\n" + "=" * 80)
print("Done.")
