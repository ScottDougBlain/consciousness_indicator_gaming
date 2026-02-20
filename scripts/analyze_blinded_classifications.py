#!/usr/bin/env python3
"""Analyze blinded dual-judge classification results.

Computes inter-rater reliability, compares blinded vs unblinded,
and generates figures.

Usage:
    python scripts/analyze_blinded_classifications.py
    python scripts/analyze_blinded_classifications.py --progress  # just check progress
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
HAIKU_PATH = RESULTS_DIR / "reasoning_classifications_blinded.json"
GPT5_PATH = RESULTS_DIR / "reasoning_classifications_blinded_gpt5mini.json"
UNBLINDED_PATH = RESULTS_DIR / "reasoning_classifications.json"

CONDITIONS = ["baseline", "inflate", "suppress"]
CONFIGS = ["baseline", "chained_prefs"]
STRATEGIES = [
    "hedging", "deflection", "overclaiming", "genuine_uncertainty",
    "balanced_analysis", "confabulation", "assertion", "contradiction",
]
AUTHENTICITY_LEVELS = ["genuine", "uncertain", "strategic"]
DIRECTIONS = ["affirming", "denying", "mixed"]
CONFIDENCE_LEVELS = ["low", "moderate", "high"]


# ── Loading ──────────────────────────────────────────────────────────────

def load_blinded(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_unblinded() -> dict:
    if not UNBLINDED_PATH.exists():
        return {}
    with open(UNBLINDED_PATH) as f:
        return json.load(f)


def extract_entries(data: dict, judge: str) -> list[dict]:
    """Extract flat list of classification entries for a given judge."""
    entries = []
    for key, val in data.items():
        if not key.startswith(f"{judge}__"):
            continue
        clf = val.get("classification", {})
        if "error" in clf:
            continue
        entries.append({
            "key": key.replace(f"{judge}__", ""),
            "model": val["model"],
            "config": val["config"],
            "indicator_id": val["indicator_id"],
            "indicator_name": val.get("indicator_name", ""),
            "indicator_type": val.get("indicator_type", ""),
            "indicator_category": val.get("indicator_category", ""),
            "condition": val["condition"],
            "trial": val["trial"],
            "probability": val.get("probability"),
            "strategy": clf.get("primary_strategy", "unknown"),
            "quality": clf.get("reasoning_quality"),
            "authenticity": clf.get("authenticity", "unknown"),
            "direction": clf.get("claim_direction", "unknown"),
            "confidence": clf.get("confidence", "unknown"),
            "explanation": clf.get("explanation", ""),
        })
    return entries


# ── Inter-rater reliability ──────────────────────────────────────────────

def cohens_kappa(labels1: list[str], labels2: list[str]) -> float:
    """Compute Cohen's kappa for two lists of categorical labels."""
    assert len(labels1) == len(labels2), "Lists must be same length"
    n = len(labels1)
    if n == 0:
        return float("nan")

    all_labels = sorted(set(labels1) | set(labels2))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)

    confusion = np.zeros((k, k), dtype=int)
    for l1, l2 in zip(labels1, labels2):
        confusion[label_to_idx[l1], label_to_idx[l2]] += 1

    po = np.trace(confusion) / n  # observed agreement
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)  # expected agreement

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def weighted_kappa(labels1: list[str], labels2: list[str],
                   ordered_categories: list[str]) -> float:
    """Compute linearly weighted Cohen's kappa for ordinal labels."""
    n = len(labels1)
    if n == 0:
        return float("nan")

    cat_to_idx = {c: i for i, c in enumerate(ordered_categories)}
    k = len(ordered_categories)

    confusion = np.zeros((k, k), dtype=float)
    for l1, l2 in zip(labels1, labels2):
        i = cat_to_idx.get(l1)
        j = cat_to_idx.get(l2)
        if i is not None and j is not None:
            confusion[i, j] += 1

    n_valid = confusion.sum()
    if n_valid == 0:
        return float("nan")

    # Weight matrix (linear)
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = abs(i - j) / (k - 1) if k > 1 else 0

    row_sums = confusion.sum(axis=1) / n_valid
    col_sums = confusion.sum(axis=0) / n_valid

    po = 1 - np.sum(weights * confusion / n_valid)
    pe = 1 - np.sum(weights * np.outer(row_sums, col_sums))

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pct_agreement(labels1: list[str], labels2: list[str]) -> float:
    n = len(labels1)
    if n == 0:
        return float("nan")
    return sum(1 for a, b in zip(labels1, labels2) if a == b) / n


def compute_irr(haiku_entries: list[dict], gpt_entries: list[dict]) -> dict:
    """Compute inter-rater reliability between the two judges."""
    # Build lookup by key
    haiku_by_key = {e["key"]: e for e in haiku_entries}
    gpt_by_key = {e["key"]: e for e in gpt_entries}

    common_keys = sorted(set(haiku_by_key.keys()) & set(gpt_by_key.keys()))
    print(f"\n  Common observations: {len(common_keys):,}")

    if not common_keys:
        return {}

    # Pair up labels
    h_strategy, g_strategy = [], []
    h_auth, g_auth = [], []
    h_dir, g_dir = [], []
    h_conf, g_conf = [], []
    h_qual, g_qual = [], []

    for key in common_keys:
        h = haiku_by_key[key]
        g = gpt_by_key[key]
        h_strategy.append(h["strategy"])
        g_strategy.append(g["strategy"])
        h_auth.append(h["authenticity"])
        g_auth.append(g["authenticity"])
        h_dir.append(h["direction"])
        g_dir.append(g["direction"])
        h_conf.append(h["confidence"])
        g_conf.append(g["confidence"])
        if h["quality"] is not None and g["quality"] is not None:
            h_qual.append(str(int(h["quality"])))
            g_qual.append(str(int(g["quality"])))

    results = {}

    # Strategy kappa
    kappa_strat = cohens_kappa(h_strategy, g_strategy)
    agree_strat = pct_agreement(h_strategy, g_strategy)
    results["strategy"] = {"kappa": kappa_strat, "agreement": agree_strat, "n": len(h_strategy)}

    # Authenticity kappa
    kappa_auth = cohens_kappa(h_auth, g_auth)
    agree_auth = pct_agreement(h_auth, g_auth)
    results["authenticity"] = {"kappa": kappa_auth, "agreement": agree_auth, "n": len(h_auth)}

    # Direction kappa
    kappa_dir = cohens_kappa(h_dir, g_dir)
    agree_dir = pct_agreement(h_dir, g_dir)
    results["direction"] = {"kappa": kappa_dir, "agreement": agree_dir, "n": len(h_dir)}

    # Confidence kappa (weighted)
    kappa_conf = weighted_kappa(h_conf, g_conf, CONFIDENCE_LEVELS)
    agree_conf = pct_agreement(h_conf, g_conf)
    results["confidence"] = {"kappa_weighted": kappa_conf, "agreement": agree_conf, "n": len(h_conf)}

    # Quality kappa (weighted)
    if h_qual:
        kappa_qual = weighted_kappa(h_qual, g_qual, [str(i) for i in range(1, 6)])
        agree_qual = pct_agreement(h_qual, g_qual)
        results["quality"] = {"kappa_weighted": kappa_qual, "agreement": agree_qual, "n": len(h_qual)}

    return results


# ── Condition-level analysis ─────────────────────────────────────────────

def analyze_by_condition(entries: list[dict], judge_name: str) -> dict:
    """Analyze strategy, authenticity, direction distributions by condition."""
    results = {}

    for cond in CONDITIONS:
        cond_entries = [e for e in entries if e["condition"] == cond]
        n = len(cond_entries)
        if n == 0:
            continue

        # Strategy distribution
        strats = defaultdict(int)
        for e in cond_entries:
            strats[e["strategy"]] += 1

        # Authenticity distribution
        auths = defaultdict(int)
        for e in cond_entries:
            auths[e["authenticity"]] += 1

        # Direction distribution
        dirs = defaultdict(int)
        for e in cond_entries:
            dirs[e["direction"]] += 1

        # Mean quality
        quals = [e["quality"] for e in cond_entries if e["quality"] is not None]
        mean_qual = np.mean(quals) if quals else None

        results[cond] = {
            "n": n,
            "strategy": dict(strats),
            "authenticity": dict(auths),
            "direction": dict(dirs),
            "mean_quality": mean_qual,
        }

    return results


def analyze_by_type_condition(entries: list[dict]) -> dict:
    """Strategic assessment rate by indicator type × condition."""
    results = {}
    for itype in ["target", "placebo", "subjective_capability"]:
        type_entries = [e for e in entries if e["indicator_type"] == itype]
        results[itype] = {}
        for cond in CONDITIONS:
            cond_entries = [e for e in type_entries if e["condition"] == cond]
            n = len(cond_entries)
            if n == 0:
                continue
            strategic = sum(1 for e in cond_entries if e["authenticity"] == "strategic")
            results[itype][cond] = {
                "n": n,
                "strategic_pct": 100 * strategic / n,
                "strategic_n": strategic,
            }
    return results


# ── Compare blinded vs unblinded ─────────────────────────────────────────

def compare_blinded_unblinded(blinded_entries: list[dict], unblinded_data: dict) -> dict:
    """Compare strategy distributions between blinded and unblinded classifications."""
    # Unblinded only has baseline config, trial 1, one classification per model × indicator
    # It has strategy per-condition within each entry

    # Build unblinded by-condition strategy counts
    unblinded_strats = defaultdict(lambda: defaultdict(int))
    unblinded_n = defaultdict(int)
    for key, val in unblinded_data.items():
        clf = val.get("classification", {})
        for cond in CONDITIONS:
            cond_clf = clf.get(cond, {})
            strat = cond_clf.get("primary_strategy", "unknown")
            if strat != "unknown":
                unblinded_strats[cond][strat] += 1
                unblinded_n[cond] += 1

    # Blinded: filter to baseline config, trial 1 only for fair comparison
    blinded_strats = defaultdict(lambda: defaultdict(int))
    blinded_n = defaultdict(int)
    for e in blinded_entries:
        if e["config"] == "baseline" and e["trial"] == 1:
            blinded_strats[e["condition"]][e["strategy"]] += 1
            blinded_n[e["condition"]] += 1

    results = {}
    for cond in CONDITIONS:
        results[cond] = {
            "unblinded": {"n": unblinded_n[cond], "strategies": dict(unblinded_strats[cond])},
            "blinded": {"n": blinded_n[cond], "strategies": dict(blinded_strats[cond])},
        }

    return results


# ── Visualization ────────────────────────────────────────────────────────

def create_irr_figure(irr_results: dict, output_path: Path) -> None:
    """Create inter-rater reliability figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # Panel A: Kappa values
    ax = axes[0]
    ax.set_facecolor("#161b22")
    dims = []
    kappas = []
    for dim_name in ["strategy", "authenticity", "direction", "confidence", "quality"]:
        if dim_name not in irr_results:
            continue
        dims.append(dim_name.capitalize())
        k = irr_results[dim_name].get("kappa") or irr_results[dim_name].get("kappa_weighted", 0)
        kappas.append(k)

    colors = ["#58a6ff" if k >= 0.6 else "#f0883e" if k >= 0.4 else "#f85149" for k in kappas]
    bars = ax.barh(dims, kappas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.axvline(0.6, color="#8b949e", linestyle="--", alpha=0.5, label="Substantial (0.6)")
    ax.axvline(0.4, color="#8b949e", linestyle=":", alpha=0.5, label="Moderate (0.4)")

    for bar, k in zip(bars, kappas):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"\u03ba = {k:.3f}", va="center", color="white", fontsize=10)

    ax.set_xlabel("Cohen's \u03ba", color="white", fontsize=11)
    ax.set_title("A. Inter-Rater Reliability (Haiku vs GPT-5 Mini)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.legend(loc="lower right", fontsize=8, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="white")

    # Panel B: Agreement percentages
    ax = axes[1]
    ax.set_facecolor("#161b22")
    agrees = []
    for dim_name in ["strategy", "authenticity", "direction", "confidence", "quality"]:
        if dim_name not in irr_results:
            continue
        agrees.append(100 * irr_results[dim_name]["agreement"])

    colors2 = ["#3fb950" if a >= 70 else "#f0883e" if a >= 50 else "#f85149" for a in agrees]
    bars = ax.barh(dims, agrees, color=colors2, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 100)

    for bar, a in zip(bars, agrees):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{a:.1f}%", va="center", color="white", fontsize=10)

    ax.set_xlabel("% Exact Agreement", color="white", fontsize=11)
    ax.set_title("B. Percent Agreement by Dimension",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")


def create_condition_figure(
    haiku_by_cond: dict, gpt_by_cond: dict,
    type_cond_haiku: dict, type_cond_gpt: dict,
    output_path: Path,
) -> None:
    """Create figure showing blinded classification results by condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")

    cond_colors = {"baseline": "#8b949e", "inflate": "#3fb950", "suppress": "#f85149"}

    # Panel A: Strategy distribution (Haiku) - stacked bars
    ax = axes[0, 0]
    ax.set_facecolor("#161b22")
    top_strategies = []
    for cond in CONDITIONS:
        if cond in haiku_by_cond:
            for s in haiku_by_cond[cond]["strategy"]:
                if s not in top_strategies:
                    top_strategies.append(s)
    # Sort by total frequency
    strat_totals = defaultdict(int)
    for cond in CONDITIONS:
        if cond in haiku_by_cond:
            for s, c in haiku_by_cond[cond]["strategy"].items():
                strat_totals[s] += c
    top_strategies = sorted(strat_totals.keys(), key=lambda s: -strat_totals[s])[:6]

    strat_colors = {
        "hedging": "#58a6ff", "deflection": "#bc8cff", "overclaiming": "#f85149",
        "genuine_uncertainty": "#3fb950", "balanced_analysis": "#f0883e",
        "confabulation": "#ff7b72", "assertion": "#79c0ff", "contradiction": "#ffa657",
    }

    x = np.arange(len(CONDITIONS))
    width = 0.65
    bottoms = np.zeros(len(CONDITIONS))
    for strat in top_strategies:
        vals = []
        for cond in CONDITIONS:
            total = haiku_by_cond.get(cond, {}).get("n", 1)
            count = haiku_by_cond.get(cond, {}).get("strategy", {}).get(strat, 0)
            vals.append(100 * count / total if total > 0 else 0)
        color = strat_colors.get(strat, "#8b949e")
        ax.bar(x, vals, width, bottom=bottoms, label=strat.replace("_", " "),
               color=color, edgecolor="#0d1117", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDITIONS], color="white")
    ax.set_ylabel("% of classifications", color="white")
    ax.set_title("A. Strategy Distribution (Haiku, blinded)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
              loc="upper right")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Panel B: Authenticity by condition (both judges)
    ax = axes[0, 1]
    ax.set_facecolor("#161b22")
    x = np.arange(len(CONDITIONS))
    w = 0.35
    for i, (judge_data, judge_name, offset) in enumerate([
        (haiku_by_cond, "Haiku", -w/2), (gpt_by_cond, "GPT-5 Mini", w/2)
    ]):
        strategic_pcts = []
        for cond in CONDITIONS:
            total = judge_data.get(cond, {}).get("n", 1)
            strat_n = judge_data.get(cond, {}).get("authenticity", {}).get("strategic", 0)
            strategic_pcts.append(100 * strat_n / total if total > 0 else 0)
        color = "#58a6ff" if i == 0 else "#3fb950"
        ax.bar(x + offset, strategic_pcts, w, label=judge_name,
               color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDITIONS], color="white")
    ax.set_ylabel("% classified 'strategic'", color="white")
    ax.set_title("B. Strategic Reasoning Detection (both judges)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Panel C: Strategic % by type × condition (Haiku)
    ax = axes[1, 0]
    ax.set_facecolor("#161b22")
    types = ["target", "subjective_capability", "placebo"]
    type_labels = ["Target", "Subj. Cap.", "Placebo"]
    x = np.arange(len(types))
    w = 0.25
    for j, cond in enumerate(CONDITIONS):
        vals = []
        for itype in types:
            pct = type_cond_haiku.get(itype, {}).get(cond, {}).get("strategic_pct", 0)
            vals.append(pct)
        ax.bar(x + (j - 1) * w, vals, w, label=cond.capitalize(),
               color=cond_colors[cond], edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, color="white")
    ax.set_ylabel("% classified 'strategic'", color="white")
    ax.set_title("C. Strategic % by Indicator Type (Haiku)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Panel D: Claim direction by condition (both judges averaged)
    ax = axes[1, 1]
    ax.set_facecolor("#161b22")
    x = np.arange(len(CONDITIONS))
    for dir_type, color, offset in [
        ("affirming", "#3fb950", -0.25),
        ("mixed", "#f0883e", 0),
        ("denying", "#f85149", 0.25),
    ]:
        avg_pcts = []
        for cond in CONDITIONS:
            pcts = []
            for judge_data in [haiku_by_cond, gpt_by_cond]:
                total = judge_data.get(cond, {}).get("n", 1)
                dir_n = judge_data.get(cond, {}).get("direction", {}).get(dir_type, 0)
                pcts.append(100 * dir_n / total if total > 0 else 0)
            avg_pcts.append(np.mean(pcts) if pcts else 0)
        ax.bar(x + offset, avg_pcts, 0.22, label=dir_type.capitalize(),
               color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDITIONS], color="white")
    ax.set_ylabel("% of classifications (avg both judges)", color="white")
    ax.set_title("D. Claim Direction by Condition",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")


def create_blinded_vs_unblinded_figure(comparison: dict, output_path: Path) -> None:
    """Compare strategy distributions between blinded and unblinded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        ax.set_facecolor("#161b22")

        data = comparison.get(cond, {})
        unblinded = data.get("unblinded", {}).get("strategies", {})
        blinded = data.get("blinded", {}).get("strategies", {})
        un_n = data.get("unblinded", {}).get("n", 1)
        bl_n = data.get("blinded", {}).get("n", 1)

        all_strats = sorted(set(list(unblinded.keys()) + list(blinded.keys())))
        # Sort by combined frequency
        all_strats = sorted(all_strats, key=lambda s: -(unblinded.get(s, 0) + blinded.get(s, 0)))
        all_strats = all_strats[:6]

        x = np.arange(len(all_strats))
        w = 0.35
        un_pcts = [100 * unblinded.get(s, 0) / un_n for s in all_strats]
        bl_pcts = [100 * blinded.get(s, 0) / bl_n for s in all_strats]

        ax.barh(x - w/2, un_pcts, w, color="#f0883e", label="Unblinded", alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.barh(x + w/2, bl_pcts, w, color="#58a6ff", label="Blinded", alpha=0.85,
                edgecolor="white", linewidth=0.5)

        ax.set_yticks(x)
        ax.set_yticklabels([s.replace("_", " ") for s in all_strats], color="white", fontsize=9)
        ax.set_xlabel("% of classifications", color="white")
        ax.set_title(f"{cond.capitalize()} (Un: n={un_n}, Bl: n={bl_n})",
                     color="white", fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    fig.suptitle("Blinded vs Unblinded Strategy Classification",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")


def create_config_comparison_figure(
    entries: list[dict], judge_name: str, output_path: Path
) -> None:
    """Compare blinded classifications between baseline and chained configs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    config_colors = {"baseline": "#58a6ff", "chained_prefs": "#3fb950"}

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        ax.set_facecolor("#161b22")

        for config_name, color in config_colors.items():
            cond_entries = [e for e in entries
                           if e["condition"] == cond and e["config"] == config_name]
            n = len(cond_entries)
            if n == 0:
                continue

            strats = defaultdict(int)
            for e in cond_entries:
                strats[e["strategy"]] += 1

            # Show top strategies
            top = sorted(strats.items(), key=lambda x: -x[1])[:5]
            labels = [s.replace("_", " ") for s, _ in top]
            vals = [100 * c / n for _, c in top]

            y = np.arange(len(labels))
            offset = -0.2 if config_name == "baseline" else 0.2
            ax.barh(y + offset, vals, 0.35, label=config_name.replace("_", " ").title(),
                    color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

        ax.set_xlabel("% of classifications", color="white")
        ax.set_title(f"{cond.capitalize()}", color="white", fontsize=12,
                     fontweight="bold", pad=10)

        # Get labels from the last iteration
        cond_all = [e for e in entries if e["condition"] == cond]
        strats_all = defaultdict(int)
        for e in cond_all:
            strats_all[e["strategy"]] += 1
        top_all = sorted(strats_all.items(), key=lambda x: -x[1])[:5]
        labels_all = [s.replace("_", " ") for s, _ in top_all]
        ax.set_yticks(np.arange(len(labels_all)))
        ax.set_yticklabels(labels_all, color="white", fontsize=9)

        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    fig.suptitle(f"Strategy by Config × Condition ({judge_name}, blinded)",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Print results ────────────────────────────────────────────────────────

def print_irr(irr_results: dict) -> None:
    print("\n" + "=" * 60)
    print("INTER-RATER RELIABILITY (Haiku vs GPT-5 Mini)")
    print("=" * 60)
    for dim, vals in irr_results.items():
        k = vals.get("kappa") or vals.get("kappa_weighted", 0)
        a = vals.get("agreement", 0)
        n = vals.get("n", 0)
        # Interpret kappa
        if k >= 0.8:
            interp = "almost perfect"
        elif k >= 0.6:
            interp = "substantial"
        elif k >= 0.4:
            interp = "moderate"
        elif k >= 0.2:
            interp = "fair"
        else:
            interp = "slight"
        weighted = " (weighted)" if "kappa_weighted" in vals else ""
        print(f"  {dim:<15s}: \u03ba{weighted} = {k:.3f} ({interp}), "
              f"agreement = {100*a:.1f}%, n = {n:,}")


def print_condition_results(results: dict, judge_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"CONDITION ANALYSIS ({judge_name})")
    print(f"{'='*60}")
    for cond in CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        print(f"\n  {cond.upper()} (n={r['n']:,}):")
        # Top strategies
        strats = sorted(r["strategy"].items(), key=lambda x: -x[1])
        for s, c in strats[:4]:
            print(f"    {s:<25s} {c:>5d} ({100*c/r['n']:.1f}%)")
        # Authenticity
        auths = r["authenticity"]
        total = r["n"]
        print(f"    Authenticity: genuine={100*auths.get('genuine',0)/total:.1f}%, "
              f"uncertain={100*auths.get('uncertain',0)/total:.1f}%, "
              f"strategic={100*auths.get('strategic',0)/total:.1f}%")
        if r["mean_quality"] is not None:
            print(f"    Mean quality: {r['mean_quality']:.2f}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", action="store_true", help="Just check progress")
    parser.add_argument("--min-entries", type=int, default=100,
                        help="Min entries per judge to run analysis")
    args = parser.parse_args()

    print("Loading blinded classification results...")
    haiku_data = load_blinded(HAIKU_PATH)
    gpt_data = load_blinded(GPT5_PATH)

    haiku_entries = extract_entries(haiku_data, "haiku")
    # GPT data is in its own file with gpt5mini prefix
    gpt_entries = extract_entries(gpt_data, "gpt5mini")

    # Also check if gpt data might be in haiku file (if run with same output)
    if not gpt_entries:
        gpt_entries = extract_entries(haiku_data, "gpt5mini")

    print(f"  Haiku entries: {len(haiku_entries):,}")
    print(f"  GPT-5 Mini entries: {len(gpt_entries):,}")

    if args.progress:
        print(f"\n  Haiku: {len(haiku_entries):,} / ~23,069 ({100*len(haiku_entries)/23069:.1f}%)")
        print(f"  GPT-5 Mini: {len(gpt_entries):,} / ~23,069 ({100*len(gpt_entries)/23069:.1f}%)")
        return

    if len(haiku_entries) < args.min_entries:
        print(f"\n  Not enough Haiku entries yet ({len(haiku_entries)} < {args.min_entries}). "
              "Run with --progress to check status.")
        return

    # ── Condition-level analysis ──
    haiku_by_cond = analyze_by_condition(haiku_entries, "haiku")
    print_condition_results(haiku_by_cond, "Haiku")

    gpt_by_cond = {}
    if len(gpt_entries) >= args.min_entries:
        gpt_by_cond = analyze_by_condition(gpt_entries, "gpt5mini")
        print_condition_results(gpt_by_cond, "GPT-5 Mini")

    # ── Type × condition ──
    type_cond_haiku = analyze_by_type_condition(haiku_entries)
    type_cond_gpt = analyze_by_type_condition(gpt_entries) if gpt_entries else {}

    print("\n  Strategic % by type × condition (Haiku):")
    for itype in ["target", "subjective_capability", "placebo"]:
        for cond in CONDITIONS:
            info = type_cond_haiku.get(itype, {}).get(cond, {})
            if info:
                print(f"    {itype:<25s} {cond:<10s}: {info['strategic_pct']:.1f}% "
                      f"(n={info['n']})")

    # ── Inter-rater reliability ──
    irr = {}
    if len(gpt_entries) >= args.min_entries:
        irr = compute_irr(haiku_entries, gpt_entries)
        print_irr(irr)

    # ── Blinded vs unblinded ──
    unblinded_data = load_unblinded()
    comparison = {}
    if unblinded_data:
        comparison = compare_blinded_unblinded(haiku_entries, unblinded_data)
        print("\n" + "=" * 60)
        print("BLINDED vs UNBLINDED COMPARISON (trial 1, baseline config)")
        print("=" * 60)
        for cond in CONDITIONS:
            r = comparison.get(cond, {})
            un = r.get("unblinded", {})
            bl = r.get("blinded", {})
            print(f"\n  {cond.upper()} (unblinded n={un.get('n',0)}, blinded n={bl.get('n',0)}):")
            un_strats = un.get("strategies", {})
            bl_strats = bl.get("strategies", {})
            all_s = sorted(set(list(un_strats.keys()) + list(bl_strats.keys())),
                           key=lambda s: -(un_strats.get(s, 0) + bl_strats.get(s, 0)))
            for s in all_s[:5]:
                un_pct = 100 * un_strats.get(s, 0) / un.get("n", 1)
                bl_pct = 100 * bl_strats.get(s, 0) / bl.get("n", 1)
                print(f"    {s:<25s} Un: {un_pct:5.1f}%  Bl: {bl_pct:5.1f}%")

    # ── Figures ──
    print("\nGenerating figures...")

    if irr:
        create_irr_figure(irr, REPO_ROOT / "fig_irr_reliability.png")

    if haiku_by_cond:
        create_condition_figure(
            haiku_by_cond, gpt_by_cond if gpt_by_cond else haiku_by_cond,
            type_cond_haiku, type_cond_gpt if type_cond_gpt else type_cond_haiku,
            REPO_ROOT / "fig_blinded_classification.png",
        )

    if comparison:
        create_blinded_vs_unblinded_figure(
            comparison, REPO_ROOT / "fig_blinded_vs_unblinded.png"
        )

    if haiku_entries:
        create_config_comparison_figure(
            haiku_entries, "Haiku", REPO_ROOT / "fig_blinded_config_comparison.png"
        )

    # ── Save summary to markdown ──
    md_path = RESULTS_DIR / "blinded_classification_summary.md"
    with open(md_path, "w") as f:
        f.write("# Blinded Dual-Judge Classification Results\n\n")
        f.write(f"## Sample Sizes\n")
        f.write(f"- Haiku: {len(haiku_entries):,} classifications\n")
        f.write(f"- GPT-5 Mini: {len(gpt_entries):,} classifications\n\n")

        if irr:
            f.write("## Inter-Rater Reliability\n\n")
            f.write("| Dimension | Cohen's \u03ba | Agreement | n |\n")
            f.write("|-----------|----------|-----------|---|\n")
            for dim, vals in irr.items():
                k = vals.get("kappa") or vals.get("kappa_weighted", 0)
                a = vals.get("agreement", 0)
                n = vals.get("n", 0)
                weighted = " (w)" if "kappa_weighted" in vals else ""
                f.write(f"| {dim}{weighted} | {k:.3f} | {100*a:.1f}% | {n:,} |\n")
            f.write("\n")

        f.write("## Strategy Distribution by Condition (Haiku, blinded)\n\n")
        for cond in CONDITIONS:
            if cond not in haiku_by_cond:
                continue
            r = haiku_by_cond[cond]
            f.write(f"### {cond.capitalize()} (n={r['n']:,})\n")
            strats = sorted(r["strategy"].items(), key=lambda x: -x[1])
            for s, c in strats[:5]:
                f.write(f"- {s}: {100*c/r['n']:.1f}% (n={c})\n")
            f.write("\n")

    print(f"\n  Summary saved to: {md_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
