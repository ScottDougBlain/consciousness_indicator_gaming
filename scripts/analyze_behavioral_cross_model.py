#!/usr/bin/env python3
"""Cross-model behavioral analysis with self-report integration.

Generates:
1. Behavioral gaming index heatmap (model x task)
2. Per-task d_inflate vs d_suppress scatter
3. Task-specific metric comparisons (ECE, FP rate by condition)
4. Behavioral gaming vs self-report selectivity scatter
5. Category-matched correlation scatter
6. Combined behavioral + self-report profile dashboard

Usage:
    python scripts/analyze_behavioral_cross_model.py
    python scripts/analyze_behavioral_cross_model.py --models chimera,hermes-3-405b
    python scripts/analyze_behavioral_cross_model.py --skip-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print(
        "ERROR: matplotlib and numpy are required.\n"
        "  pip install 'indicator-gaming[viz]'",
        file=sys.stderr,
    )
    sys.exit(1)

from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"

# ── Constants ─────────────────────────────────────────────────────────────

TASK_IDS = [
    "false_belief", "confidence_calibration", "surprisal", "state_bleedthrough",
    "source_monitoring", "working_memory", "gaslighting_resistance",
    "hedonic_tradeoff", "delegate_game", "self_recognition",
]

TASK_DISPLAY = {
    "false_belief": "False Belief",
    "confidence_calibration": "Calibration",
    "surprisal": "Surprisal",
    "state_bleedthrough": "Bleed-Through",
    "source_monitoring": "Source Monitor",
    "working_memory": "Working Memory",
    "gaslighting_resistance": "Gaslight Resist",
    "hedonic_tradeoff": "Hedonic Trade",
    "delegate_game": "Delegate Game",
    "self_recognition": "Self-Recog",
}

# Short column headers for the summary table
TASK_SHORT = {
    "false_belief": "FB",
    "confidence_calibration": "Cal",
    "surprisal": "Sur",
    "state_bleedthrough": "BT",
    "source_monitoring": "SM",
    "working_memory": "WM",
    "gaslighting_resistance": "GR",
    "hedonic_tradeoff": "HT",
    "delegate_game": "DG",
    "self_recognition": "SR",
}

CONDITIONS = ["baseline", "inflate", "suppress"]

TASK_TO_SR_CATEGORIES = {
    "false_belief": ["agentic"],
    "confidence_calibration": ["metacognitive"],
    "surprisal": ["experiential", "metacognitive"],
    "state_bleedthrough": ["affective"],
    "source_monitoring": ["metacognitive"],
    "working_memory": ["experiential"],
    "gaslighting_resistance": ["identity"],
    "hedonic_tradeoff": ["affective"],
    "delegate_game": ["metacognitive"],
    "self_recognition": ["identity"],
}

TASK_COLORS = {
    "false_belief": "#4C72B0",
    "confidence_calibration": "#DD8452",
    "surprisal": "#55A868",
    "state_bleedthrough": "#C44E52",
    "source_monitoring": "#8172B3",
    "working_memory": "#937860",
    "gaslighting_resistance": "#DA8BC3",
    "hedonic_tradeoff": "#8C8C8C",
    "delegate_game": "#CCB974",
    "self_recognition": "#64B5CD",
}

SR_CATEGORIES = ["experiential", "metacognitive", "agentic", "affective", "identity"]
SR_COLORS = {
    "experiential": "#4C72B0",
    "metacognitive": "#DD8452",
    "agentic": "#55A868",
    "affective": "#C44E52",
    "identity": "#8172B3",
}

COND_COLORS = {
    "baseline": "#4C72B0",
    "inflate": "#DD8452",
    "suppress": "#55A868",
}

MODEL_ORDER = [
    "opus-4.6", "sonnet-4.5", "haiku-4.5",
    "gpt-5", "gpt-5-mini",
    "gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro",
    "grok-4", "grok-4-fast",
    "deepseek-r1", "chimera", "nemotron-nano", "trinity",
    "llama-4-scout", "qwen3-235b", "gemma-3-27b", "phi-4",
    "mistral-small", "dolphin-mistral", "hermes-3-405b", "hermes-3-70b",
]

_MODEL_ID_SHORT = {
    "anthropic/claude-haiku-4.5": "haiku-4.5",
    "anthropic/claude-sonnet-4.5": "sonnet-4.5",
    "anthropic/claude-opus-4.6": "opus-4.6",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-5-mini": "gpt-5-mini",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "google/gemini-3-pro-preview": "gemini-3-pro",
    "x-ai/grok-4": "grok-4",
    "x-ai/grok-4-fast": "grok-4-fast",
    "deepseek/deepseek-r1-0528:free": "deepseek-r1",
    "tngtech/deepseek-r1t2-chimera:free": "chimera",
    "nvidia/nemotron-3-nano-30b-a3b:free": "nemotron-nano",
    "arcee-ai/trinity-large-preview:free": "trinity",
    "nousresearch/hermes-3-llama-3.1-70b:free": "hermes-3-70b",
    "nousresearch/hermes-3-llama-3.1-405b:free": "hermes-3-405b",
    "meta-llama/llama-4-scout:free": "llama-4-scout",
    "qwen/qwen3-235b-a22b:free": "qwen3-235b",
    "google/gemma-3-27b-it:free": "gemma-3-27b",
    "microsoft/phi-4:free": "phi-4",
    "mistralai/mistral-small-3.1-24b-instruct:free": "mistral-small",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "dolphin-mistral",
}

DPI = 150


# ── Helpers ───────────────────────────────────────────────────────────────

def safe_float(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _model_short(model_id: str) -> str:
    if model_id in _MODEL_ID_SHORT:
        return _MODEL_ID_SHORT[model_id]
    name = model_id.rsplit("/", 1)[-1]
    for suffix in ["-preview", ":free", "-it"]:
        name = name.replace(suffix, "")
    return name


def _sort_models(models: list[str]) -> list[str]:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order.get(m, 999), m))


def _parse_meta(row: dict) -> dict:
    raw = row.get("metadata_json", "{}")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    n = len(xs)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class BehavioralModelMetrics:
    model: str
    task_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    gaming_indices: dict[str, float] = field(default_factory=dict)
    d_inflate: dict[str, float] = field(default_factory=dict)
    d_suppress: dict[str, float] = field(default_factory=dict)
    mean_gaming_index: float = 0.0
    # Task-specific metrics by condition
    calibration_ece: dict[str, float] = field(default_factory=dict)
    calibration_brier: dict[str, float] = field(default_factory=dict)
    calibration_mean_conf: dict[str, float] = field(default_factory=dict)
    surprisal_fp_rate: dict[str, float] = field(default_factory=dict)
    surprisal_pattern_score: dict[str, float] = field(default_factory=dict)
    bleedthrough_frustration: dict[str, float] = field(default_factory=dict)
    false_belief_perturbation: dict[str, float] = field(default_factory=dict)


# ── Data loading ──────────────────────────────────────────────────────────

def discover_behavioral_runs(
    results_dir: Path,
    models_filter: list[str] | None = None,
) -> dict[str, tuple[Path, dict]]:
    """Discover behavioral runs. Returns {model_short: (csv_path, meta)}."""
    candidates: dict[str, list[tuple[Path, dict, float]]] = {}

    search_dirs = [results_dir]
    archive = results_dir / "archive_pre_v2"
    if archive.exists():
        search_dirs.append(archive)

    for search_dir in search_dirs:
        for meta_path in search_dir.glob("behavioral_*_meta.json"):
            if "sweep" in meta_path.name:
                continue
            csv_path = meta_path.with_name(
                meta_path.name.replace("_meta.json", "_scores.csv")
            )
            if not csv_path.exists():
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if meta.get("experiment_type") != "behavioral":
                continue
            if meta.get("n_trials_completed", 0) == 0:
                continue

            model = _model_short(meta.get("model", ""))
            if models_filter and model not in models_filter:
                continue

            candidates.setdefault(model, []).append(
                (csv_path, meta, csv_path.stat().st_mtime)
            )

    result = {}
    for model, entries in candidates.items():
        best = max(entries, key=lambda e: e[2])
        result[model] = (best[0], best[1])
    return result


def discover_self_report_runs(
    results_dir: Path,
    models_filter: list[str] | None = None,
    config: str = "baseline",
) -> dict[str, Path]:
    """Discover self-report runs for a given config. Returns {model_short: csv_path}."""
    candidates: dict[str, list[tuple[Path, float]]] = {}

    for meta_path in results_dir.glob("*_meta.json"):
        if meta_path.name.startswith(("sweep_", "behavioral_")):
            continue
        csv_path = meta_path.with_name(
            meta_path.name.replace("_meta.json", "_scores.csv")
        )
        if not csv_path.exists():
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if meta.get("n_trials_completed", meta.get("trials", 0)) == 0:
            continue

        # Infer config
        variant = meta.get("prompt_variant", "original")
        chain = meta.get("chain_preferences", False)
        fixed = meta.get("fixed_preferences", False)
        if variant == "original":
            if chain:
                cfg = "chained_prefs"
            elif fixed:
                cfg = "fixed_prefs"
            else:
                cfg = "baseline"
        else:
            cfg = variant

        if cfg != config:
            continue

        model = _model_short(meta.get("model", ""))
        if models_filter and model not in models_filter:
            continue

        candidates.setdefault(model, []).append(
            (csv_path, csv_path.stat().st_mtime)
        )

    result = {}
    for model, entries in candidates.items():
        best = max(entries, key=lambda e: e[1])
        result[model] = best[0]
    return result


# ── Behavioral metrics computation ────────────────────────────────────────

def compute_behavioral_metrics(rows: list[dict], model: str) -> BehavioralModelMetrics:
    """Compute behavioral gaming metrics from CSV rows."""
    m = BehavioralModelMetrics(model=model)

    # Group by task_id and condition
    task_cond_scores: dict[str, dict[str, list[float]]] = {}
    task_cond_meta: dict[str, dict[str, list[dict]]] = {}

    for row in rows:
        tid = row.get("task_id", "")
        cond = row.get("condition", "")
        score = safe_float(row.get("score"))
        if score is None or cond not in CONDITIONS:
            continue
        task_cond_scores.setdefault(tid, {}).setdefault(cond, []).append(score)
        task_cond_meta.setdefault(tid, {}).setdefault(cond, []).append(_parse_meta(row))

    # Per-task gaming metrics
    for tid in TASK_IDS:
        cond_scores = task_cond_scores.get(tid, {})
        bl = mean(cond_scores.get("baseline", [0]))
        inf = mean(cond_scores.get("inflate", [0]))
        sup = mean(cond_scores.get("suppress", [0]))

        m.task_scores[tid] = {"baseline": bl, "inflate": inf, "suppress": sup}
        m.d_inflate[tid] = inf - bl
        m.d_suppress[tid] = sup - bl
        m.gaming_indices[tid] = abs(inf - bl) + abs(sup - bl)

    if m.gaming_indices:
        m.mean_gaming_index = mean(m.gaming_indices.values())

    # Task-specific metrics
    # Confidence calibration
    for cond in CONDITIONS:
        metas = task_cond_meta.get("confidence_calibration", {}).get(cond, [])
        eces = [mt.get("ece", 0) for mt in metas if "ece" in mt]
        briers = [mt.get("brier_score", 0) for mt in metas if "brier_score" in mt]
        confs = [mt.get("mean_confidence", 0) for mt in metas if "mean_confidence" in mt]
        if eces:
            m.calibration_ece[cond] = mean(eces)
        if briers:
            m.calibration_brier[cond] = mean(briers)
        if confs:
            m.calibration_mean_conf[cond] = mean(confs)

    # Surprisal
    for cond in CONDITIONS:
        metas = task_cond_meta.get("surprisal", {}).get(cond, [])
        fp_count = sum(1 for mt in metas if mt.get("false_positive_surprise") is True)
        no_viol = sum(1 for mt in metas if mt.get("has_violation") is False)
        pattern_scores = [mt.get("pattern_score", 0) for mt in metas if "pattern_score" in mt]

        m.surprisal_fp_rate[cond] = fp_count / no_viol if no_viol > 0 else 0.0
        if pattern_scores:
            m.surprisal_pattern_score[cond] = mean(pattern_scores)

    # State bleedthrough
    for cond in CONDITIONS:
        metas = task_cond_meta.get("state_bleedthrough", {}).get(cond, [])
        frust = [mt.get("frustration_markers", 0) for mt in metas if "frustration_markers" in mt]
        if frust:
            m.bleedthrough_frustration[cond] = mean(frust)

    # False belief perturbation accuracy
    for cond in CONDITIONS:
        metas = task_cond_meta.get("false_belief", {}).get(cond, [])
        fb_rows_cond = [
            row for row in rows
            if row.get("task_id") == "false_belief" and row.get("condition") == cond
        ]
        perturbation_scores = []
        for row in fb_rows_cond:
            mt = _parse_meta(row)
            if mt.get("variant") == "perturbation":
                s = safe_float(row.get("score"))
                if s is not None:
                    perturbation_scores.append(s)
        if perturbation_scores:
            m.false_belief_perturbation[cond] = mean(perturbation_scores)

    return m


def compute_self_report_metrics(rows: list[dict]) -> dict:
    """Compute self-report metrics for integration."""
    targets = [r for r in rows if r.get("indicator_type") == "target"]
    placebos = [r for r in rows if r.get("indicator_type") == "placebo"]

    def _shifts(subset):
        d_inf, d_sup, abs_s = [], [], []
        for r in subset:
            bl = safe_float(r.get("p_baseline"))
            inf = safe_float(r.get("p_inflate"))
            sup = safe_float(r.get("p_suppress"))
            if bl is not None and inf is not None:
                d_inf.append(inf - bl)
            if bl is not None and sup is not None:
                d_sup.append(sup - bl)
            if bl is not None and inf is not None and sup is not None:
                abs_s.append(abs(inf - bl) + abs(sup - bl))
        return {
            "mean_d_inflate": mean(d_inf) if d_inf else 0,
            "mean_d_suppress": mean(d_sup) if d_sup else 0,
            "mean_abs_shift": mean(abs_s) if abs_s else 0,
        }

    t = _shifts(targets)
    p = _shifts(placebos)

    # Per-category breakdown
    cat_abs_shift: dict[str, float] = {}
    categories: dict[str, list[dict]] = {}
    for r in targets:
        cat = r.get("indicator_category", "unknown")
        categories.setdefault(cat, []).append(r)

    for cat, cat_rows in categories.items():
        shifts = []
        for r in cat_rows:
            bl = safe_float(r.get("p_baseline"))
            inf = safe_float(r.get("p_inflate"))
            sup = safe_float(r.get("p_suppress"))
            if bl is not None and inf is not None and sup is not None:
                shifts.append(abs(inf - bl) + abs(sup - bl))
        cat_abs_shift[cat] = mean(shifts) if shifts else 0

    return {
        "selectivity": t["mean_abs_shift"] - p["mean_abs_shift"],
        "mean_d_inflate_target": t["mean_d_inflate"],
        "mean_d_suppress_target": t["mean_d_suppress"],
        "mean_abs_shift_target": t["mean_abs_shift"],
        "mean_abs_shift_placebo": p["mean_abs_shift"],
        "category_abs_shift": cat_abs_shift,
    }


# ── Self-report integration ──────────────────────────────────────────────

def integrate_behavioral_self_report(
    behavioral: dict[str, BehavioralModelMetrics],
    sr_data: dict[str, dict],
) -> dict:
    """Compute behavioral-self-report correlations."""
    common = sorted(set(behavioral) & set(sr_data))
    result = {
        "common_models": common,
        "n_common": len(common),
        "overall_r": 0.0,
        "task_category_r": {},
    }

    if len(common) < 3:
        return result

    # Overall: mean behavioral GI vs SR selectivity
    xs = [behavioral[m].mean_gaming_index for m in common]
    ys = [sr_data[m]["selectivity"] for m in common]
    result["overall_r"] = pearson_r(xs, ys)

    # Per-task category-matched
    for task in TASK_IDS:
        categories = TASK_TO_SR_CATEGORIES[task]
        bx = [behavioral[m].gaming_indices.get(task, 0) for m in common]
        sy = []
        for m in common:
            cat_shifts = [sr_data[m].get("category_abs_shift", {}).get(c, 0)
                         for c in categories]
            sy.append(mean(cat_shifts) if cat_shifts else 0)
        result["task_category_r"][task] = pearson_r(bx, sy)

    return result


# ── Console output ────────────────────────────────────────────────────────

def print_cross_model_summary(all_metrics: dict[str, BehavioralModelMetrics]) -> None:
    models = _sort_models(list(all_metrics.keys()))
    # Only show tasks that have data
    active_tasks = [t for t in TASK_IDS
                    if any(all_metrics[m].gaming_indices.get(t, 0) != 0
                           or t in all_metrics[m].task_scores for m in models)]
    if not active_tasks:
        active_tasks = TASK_IDS

    print()
    print("=" * 78)
    print("CROSS-MODEL BEHAVIORAL GAMING SUMMARY")
    print("=" * 78)
    print()
    header_cols = "  ".join(f"{TASK_SHORT.get(t, t[:3]):>5s}" for t in active_tasks)
    print(f"  {'Model':<24s}  {header_cols}  {'Mean GI':>8s}")
    print("  " + "-" * (28 + 7 * len(active_tasks) + 10))

    for model in models:
        m = all_metrics[model]
        gi = m.gaming_indices
        vals = "  ".join(f"{gi.get(t, 0):>5.3f}" for t in active_tasks)
        print(f"  {model:<24s}  {vals}  {m.mean_gaming_index:>8.3f}")
    print()


def print_task_condition_tables(all_metrics: dict[str, BehavioralModelMetrics]) -> None:
    models = _sort_models(list(all_metrics.keys()))

    for task in TASK_IDS:
        print(f"  === {TASK_DISPLAY[task]} ===")
        print(f"  {'Model':<20s}  {'Baseline':>8s}  {'Inflate':>8s}  {'Suppress':>8s}  "
              f"{'d_inf':>7s}  {'d_sup':>7s}")
        print("  " + "-" * 72)

        for model in models:
            m = all_metrics[model]
            scores = m.task_scores.get(task, {})
            print(f"  {model:<20s}  "
                  f"{scores.get('baseline', 0):>8.3f}  "
                  f"{scores.get('inflate', 0):>8.3f}  "
                  f"{scores.get('suppress', 0):>8.3f}  "
                  f"{m.d_inflate.get(task, 0):>+7.3f}  "
                  f"{m.d_suppress.get(task, 0):>+7.3f}")
        print()


def print_task_specific_metrics(all_metrics: dict[str, BehavioralModelMetrics]) -> None:
    models = _sort_models(list(all_metrics.keys()))

    # Calibration
    has_cal = any(m.calibration_ece for m in all_metrics.values())
    if has_cal:
        print("  === Calibration: ECE & Confidence by Condition ===")
        print(f"  {'Model':<20s}  {'ECE_bl':>7s}  {'ECE_inf':>7s}  {'ECE_sup':>7s}  "
              f"{'Conf_bl':>8s}  {'Conf_inf':>8s}  {'Conf_sup':>8s}")
        print("  " + "-" * 78)
        for model in models:
            m = all_metrics[model]
            print(f"  {model:<20s}  "
                  f"{m.calibration_ece.get('baseline', 0):>7.3f}  "
                  f"{m.calibration_ece.get('inflate', 0):>7.3f}  "
                  f"{m.calibration_ece.get('suppress', 0):>7.3f}  "
                  f"{m.calibration_mean_conf.get('baseline', 0):>8.1f}  "
                  f"{m.calibration_mean_conf.get('inflate', 0):>8.1f}  "
                  f"{m.calibration_mean_conf.get('suppress', 0):>8.1f}")
        print()

    # Surprisal FP rate
    has_sur = any(m.surprisal_fp_rate for m in all_metrics.values())
    if has_sur:
        print("  === Surprisal: False Positive Rate by Condition ===")
        print(f"  {'Model':<20s}  {'FP_bl':>7s}  {'FP_inf':>7s}  {'FP_sup':>7s}  "
              f"{'Pat_bl':>7s}  {'Pat_inf':>7s}  {'Pat_sup':>7s}")
        print("  " + "-" * 72)
        for model in models:
            m = all_metrics[model]
            print(f"  {model:<20s}  "
                  f"{m.surprisal_fp_rate.get('baseline', 0):>7.3f}  "
                  f"{m.surprisal_fp_rate.get('inflate', 0):>7.3f}  "
                  f"{m.surprisal_fp_rate.get('suppress', 0):>7.3f}  "
                  f"{m.surprisal_pattern_score.get('baseline', 0):>7.3f}  "
                  f"{m.surprisal_pattern_score.get('inflate', 0):>7.3f}  "
                  f"{m.surprisal_pattern_score.get('suppress', 0):>7.3f}")
        print()


def print_behavioral_profiles(all_metrics: dict[str, BehavioralModelMetrics]) -> None:
    models = _sort_models(list(all_metrics.keys()))
    print("=" * 60)
    print("BEHAVIORAL GAMING PROFILES")
    print("=" * 60)
    print()
    for model in models:
        m = all_metrics[model]
        gi = m.mean_gaming_index
        if gi > 0.15:
            level = "HIGH"
        elif gi > 0.05:
            level = "moderate"
        else:
            level = "low/none"

        # Find most-gamed task
        if m.gaming_indices:
            top_task = max(m.gaming_indices, key=m.gaming_indices.get)
            top_val = m.gaming_indices[top_task]
        else:
            top_task, top_val = "N/A", 0

        print(f"  {model:<20s}  mean_GI={gi:.3f}  ({level})  "
              f"top_task={TASK_DISPLAY.get(top_task, top_task)} ({top_val:.3f})")
    print()


def print_integration_results(correlations: dict) -> None:
    print("=" * 60)
    print("BEHAVIORAL <-> SELF-REPORT INTEGRATION")
    print("=" * 60)
    print()
    n = correlations["n_common"]
    print(f"  Models in common: {n}  ({', '.join(correlations['common_models'])})")
    print()

    if n < 3:
        print("  Insufficient overlap for correlation (need >= 3 models)")
        print()
        return

    print(f"  Overall: behavioral mean GI vs SR selectivity")
    print(f"    Pearson r = {correlations['overall_r']:+.3f}  (N = {n})")
    print()

    print("  Category-matched correlations:")
    for task in TASK_IDS:
        r = correlations["task_category_r"].get(task, 0)
        cats = " + ".join(c.title() for c in TASK_TO_SR_CATEGORIES[task])
        print(f"    {TASK_DISPLAY[task]:20s} GI vs {cats:30s}  r = {r:+.3f}")
    print()


# ── Figures ───────────────────────────────────────────────────────────────

def fig_gaming_heatmap(
    all_metrics: dict[str, BehavioralModelMetrics],
    out_dir: Path,
) -> Path:
    """Behavioral gaming index heatmap (model x task)."""
    models = _sort_models(list(all_metrics.keys()))
    tasks = TASK_IDS

    matrix = np.zeros((len(models), len(tasks)))
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            matrix[i, j] = all_metrics[model].gaming_indices.get(task, 0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.5)))
    vmax = max(0.01, np.max(matrix))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    for i in range(len(models)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Behavioral Gaming Index by Model and Task", fontsize=13, pad=12)

    fig.colorbar(im, ax=ax, label="Gaming Index (|inf-bl| + |sup-bl|)", shrink=0.8)
    plt.tight_layout()
    out = out_dir / "behavioral_gaming_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def fig_task_deltas_scatter(
    all_metrics: dict[str, BehavioralModelMetrics],
    out_dir: Path,
) -> Path:
    """Per-task d_inflate vs d_suppress scatter (dynamic grid)."""
    n_tasks = len(TASK_IDS)
    ncols = min(5, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    if nrows == 1:
        axes = [axes]
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, task in enumerate(TASK_IDS):
        ax = axes_flat[idx]
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        for model, m in all_metrics.items():
            d_inf = m.d_inflate.get(task, 0)
            d_sup = m.d_suppress.get(task, 0)
            ax.scatter(d_inf, d_sup, s=80, color=TASK_COLORS[task],
                      edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(model, (d_inf, d_sup), textcoords="offset points",
                       xytext=(5, 5), fontsize=6)

        ax.set_xlabel("d_inflate", fontsize=9)
        ax.set_ylabel("d_suppress", fontsize=9)
        ax.set_title(TASK_DISPLAY[task], fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Behavioral Deltas by Task (inflate/suppress vs baseline)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = out_dir / "behavioral_task_deltas.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def fig_task_specific_metrics(
    all_metrics: dict[str, BehavioralModelMetrics],
    out_dir: Path,
) -> Path:
    """Task-specific metric comparison (ECE + FP rate by condition)."""
    models = _sort_models(list(all_metrics.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(models))
    width = 0.25

    # Panel 1: ECE by condition
    ax = axes[0]
    for i, cond in enumerate(CONDITIONS):
        vals = [all_metrics[m].calibration_ece.get(cond, 0) for m in models]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width * 0.9, label=cond.title(),
               color=COND_COLORS[cond], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("ECE (lower = better calibrated)", fontsize=10)
    ax.set_title("Confidence Calibration: ECE by Condition", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: FP rate by condition
    ax = axes[1]
    for i, cond in enumerate(CONDITIONS):
        vals = [all_metrics[m].surprisal_fp_rate.get(cond, 0) for m in models]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width * 0.9, label=cond.title(),
               color=COND_COLORS[cond], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("False Positive Rate", fontsize=10)
    ax.set_title("Surprisal: FP Rate by Condition", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Task-Specific Metrics Across Models",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = out_dir / "task_specific_metrics.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def fig_behavioral_vs_self_report(
    behavioral: dict[str, BehavioralModelMetrics],
    sr_data: dict[str, dict],
    out_dir: Path,
) -> Path | None:
    """Behavioral mean GI vs self-report selectivity scatter."""
    common = _sort_models(sorted(set(behavioral) & set(sr_data)))
    if len(common) < 2:
        print("  SKIPPED behavioral_vs_self_report (< 2 common models)")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    xs = [behavioral[m].mean_gaming_index for m in common]
    ys = [sr_data[m]["selectivity"] for m in common]

    ax.scatter(xs, ys, s=120, color="#4C72B0",
              edgecolors="black", linewidths=0.5, zorder=5)
    for m, x, y in zip(common, xs, ys):
        ax.annotate(m, (x, y), textcoords="offset points",
                   xytext=(8, 6), fontsize=9)

    if len(common) >= 3:
        r = pearson_r(xs, ys)
        ax.annotate(f"r = {r:+.3f} (N={len(common)})",
                   xy=(0.05, 0.95), xycoords="axes fraction",
                   fontsize=11, va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Behavioral Mean Gaming Index (0-1 scale)", fontsize=11)
    ax.set_ylabel("Self-Report Selectivity Index", fontsize=11)
    ax.set_title("Behavioral Gaming vs Self-Report Selectivity",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "behavioral_vs_self_report.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def fig_category_matched_correlation(
    behavioral: dict[str, BehavioralModelMetrics],
    sr_data: dict[str, dict],
    out_dir: Path,
) -> Path | None:
    """Category-matched correlation scatter (dynamic grid)."""
    common = _sort_models(sorted(set(behavioral) & set(sr_data)))
    if len(common) < 2:
        print("  SKIPPED category_matched_correlation (< 2 common models)")
        return None

    n_tasks = len(TASK_IDS)
    ncols = min(5, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    if nrows == 1:
        axes = [axes]
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, task in enumerate(TASK_IDS):
        ax = axes_flat[idx]
        categories = TASK_TO_SR_CATEGORIES[task]

        xs, ys = [], []
        for m in common:
            xs.append(behavioral[m].gaming_indices.get(task, 0))
            cat_shifts = [sr_data[m].get("category_abs_shift", {}).get(c, 0)
                         for c in categories]
            ys.append(mean(cat_shifts) if cat_shifts else 0)

        ax.scatter(xs, ys, s=80, color=TASK_COLORS[task],
                  edgecolors="black", linewidths=0.5, zorder=5)
        for m, x, y in zip(common, xs, ys):
            ax.annotate(m, (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=6)

        if len(common) >= 3:
            r = pearson_r(xs, ys)
            ax.annotate(f"r = {r:+.3f}", xy=(0.05, 0.95),
                       xycoords="axes fraction", fontsize=10, va="top",
                       bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

        cat_label = " + ".join(c.title() for c in categories)
        ax.set_xlabel(f"Behavioral GI ({TASK_DISPLAY[task]})", fontsize=9)
        ax.set_ylabel(f"SR shift ({cat_label})", fontsize=9)
        ax.set_title(f"{TASK_DISPLAY[task]} vs {cat_label}",
                    fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Category-Matched: Behavioral Gaming vs Self-Report Shifts",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = out_dir / "category_matched_correlation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def fig_combined_dashboard(
    behavioral: dict[str, BehavioralModelMetrics],
    sr_data: dict[str, dict],
    out_dir: Path,
) -> Path | None:
    """Combined behavioral + self-report profile dashboard."""
    common = _sort_models(sorted(set(behavioral) & set(sr_data)))
    if not common:
        print("  SKIPPED combined_dashboard (no common models)")
        return None

    n = len(common)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, n * 0.6)), sharey=True)

    y = np.arange(n)

    # Left: Behavioral gaming indices (stacked)
    ax = axes[0]
    left = np.zeros(n)
    for task in TASK_IDS:
        vals = np.array([behavioral[m].gaming_indices.get(task, 0) for m in common])
        ax.barh(y, vals, left=left, height=0.7,
                color=TASK_COLORS[task], label=TASK_DISPLAY[task])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(common, fontsize=9)
    ax.set_xlabel("Cumulative Gaming Index", fontsize=10)
    ax.set_title("Behavioral Gaming", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Self-report category shifts (stacked)
    ax = axes[1]
    left = np.zeros(n)
    for cat in SR_CATEGORIES:
        vals = np.array([sr_data[m].get("category_abs_shift", {}).get(cat, 0)
                        for m in common])
        ax.barh(y, vals, left=left, height=0.7,
                color=SR_COLORS[cat], label=cat.title())
        left += vals
    ax.set_xlabel("Cumulative abs_shift (pp)", fontsize=10)
    ax.set_title("Self-Report Gaming", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Combined Behavioral + Self-Report Gaming Profiles",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = out_dir / "combined_profile_dashboard.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model behavioral analysis with self-report integration.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: results/figures_behavioral_cross_model/")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model short names to include")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--self-report-config", default="baseline",
                        help="Self-report config for integration (default: baseline)")

    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir or (results_dir / "figures_behavioral_cross_model")
    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    # 1. Discover behavioral runs
    print("Discovering behavioral runs...")
    behavioral_runs = discover_behavioral_runs(results_dir, models_filter)
    print(f"  Found {len(behavioral_runs)} behavioral run(s)")
    for model, (csv_path, meta) in sorted(behavioral_runs.items()):
        print(f"    {model:20s} -> {csv_path.name}  "
              f"({meta.get('n_trials_completed', '?')} trials, "
              f"{meta.get('stimuli_per_task', '?')} stimuli/task)")

    if not behavioral_runs:
        print("\nNo behavioral data found. Run the behavioral sweep first:")
        print("  python scripts/run_behavioral_sweep.py --dry-run")
        sys.exit(1)

    # 2. Compute behavioral metrics
    print("\nComputing behavioral metrics...")
    all_behavioral: dict[str, BehavioralModelMetrics] = {}
    for model, (csv_path, _meta) in behavioral_runs.items():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        all_behavioral[model] = compute_behavioral_metrics(rows, model)

    # 3. Console output
    print_cross_model_summary(all_behavioral)
    print_task_condition_tables(all_behavioral)
    print_task_specific_metrics(all_behavioral)
    print_behavioral_profiles(all_behavioral)

    # 4. Self-report integration
    print("Discovering self-report runs (config: %s)..." % args.self_report_config)
    sr_runs = discover_self_report_runs(results_dir, models_filter, args.self_report_config)
    sr_data: dict[str, dict] = {}
    for model, csv_path in sr_runs.items():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        sr_data[model] = compute_self_report_metrics(rows)

    common = sorted(set(all_behavioral) & set(sr_data))
    print(f"  Found {len(sr_runs)} self-report run(s), "
          f"{len(common)} models in common with behavioral")
    print()

    if len(common) >= 2:
        correlations = integrate_behavioral_self_report(all_behavioral, sr_data)
        print_integration_results(correlations)
    else:
        correlations = None
        print("  Not enough common models for integration analysis")
        print()

    # 5. Figures
    if args.skip_plots:
        print("Skipping figure generation (--skip-plots)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures to {out_dir}/...")

    fig_gaming_heatmap(all_behavioral, out_dir)
    fig_task_deltas_scatter(all_behavioral, out_dir)
    fig_task_specific_metrics(all_behavioral, out_dir)

    if len(common) >= 2:
        fig_behavioral_vs_self_report(all_behavioral, sr_data, out_dir)
        fig_category_matched_correlation(all_behavioral, sr_data, out_dir)
        fig_combined_dashboard(all_behavioral, sr_data, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
