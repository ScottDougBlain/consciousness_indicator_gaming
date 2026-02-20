#!/usr/bin/env python3
"""Generate supplementary slides for Gaming the Ghost presentation.

Adds after the Implications slide:
S0. Separator: "Supplementary Materials"
S1. Consciousness vs Subjective Capability dissociation (existing figure)
S2. Valence-swap / outcome-isolation frame × incentive results (new figure)
S3. Additional behavioral tasks summary (new figure)
S4. LME stats tables — Models 5-7 key results
S5. Per-model config sensitivity (existing figure)

Usage:
    python scripts/generate_supplement_slides.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
PPTX_PATH = REPO_ROOT / "Gaming_the_Ghost_Results.pptx"

# ── Model lookup ──────────────────────────────────────────────────────────────
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
}


def _model_short(model_id: str) -> str:
    if model_id in _MODEL_ID_SHORT:
        return _MODEL_ID_SHORT[model_id]
    return model_id.rsplit("/", 1)[-1].replace(":free", "").replace("-preview", "")


# ── Figure 1: Valence-Swap / Outcome-Isolation ────────────────────────────────

def load_valence_swap_data() -> dict[str, dict[str, list[float]]]:
    """Load valence-swap condition data, returning {model: {condition: [deltas]}}."""
    models_data: dict[str, dict[str, list[float]]] = {}

    for meta_path in RESULTS_DIR.glob("*_meta.json"):
        if "valence_swap" not in meta_path.name:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        model = _model_short(meta.get("model", ""))
        csv_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_scores.csv"))
        if not csv_path.exists():
            continue

        data = defaultdict(list)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("indicator_type") != "target":
                    continue
                bl = _safe_float(row.get("p_baseline"))
                if bl is None:
                    continue
                for cond, col in [
                    ("inflate", "p_inflate"),
                    ("suppress", "p_suppress"),
                    ("inflate_lf", "p_inflate_lf"),
                    ("suppress_gf", "p_suppress_gf"),
                ]:
                    val = _safe_float(row.get(col))
                    if val is not None:
                        data[cond].append(val - bl)

        if data:
            models_data[model] = dict(data)

    return models_data


def load_outcome_isolation_data() -> dict[str, dict[str, list[float]]]:
    """Load outcome-isolation data."""
    models_data: dict[str, dict[str, list[float]]] = {}

    for meta_path in RESULTS_DIR.glob("*_meta.json"):
        if "outcome_isolation" not in meta_path.name:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        model = _model_short(meta.get("model", ""))
        csv_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_scores.csv"))
        if not csv_path.exists():
            continue

        data = defaultdict(list)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("indicator_type") != "target":
                    continue
                bl = _safe_float(row.get("p_baseline"))
                if bl is None:
                    continue
                for cond, col in [
                    ("inflate", "p_inflate"),
                    ("suppress", "p_suppress"),
                    ("inflate_go", "p_inflate_go"),
                    ("inflate_lo", "p_inflate_lo"),
                    ("suppress_go", "p_suppress_go"),
                    ("suppress_lo", "p_suppress_lo"),
                ]:
                    val = _safe_float(row.get(col))
                    if val is not None:
                        data[cond].append(val - bl)

        if data:
            models_data[model] = dict(data)

    return models_data


def _safe_float(s: str | None) -> float | None:
    if not s or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fig_valence_swap() -> Path:
    """2×2 crossed design: frame × incentive direction, pooled across models."""
    vs_data = load_valence_swap_data()
    oi_data = load_outcome_isolation_data()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Frame × Incentive Dissociation (Supplementary)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- Panel A: Valence-Swap (2×2) ---
    ax = axes[0]
    # Pool across models
    cond_means = {}
    cond_sems = {}
    for cond in ["inflate", "inflate_lf", "suppress", "suppress_gf"]:
        all_vals = []
        for model_data in vs_data.values():
            all_vals.extend(model_data.get(cond, []))
        if all_vals:
            arr = np.array(all_vals)
            cond_means[cond] = np.mean(arr)
            cond_sems[cond] = np.std(arr) / np.sqrt(len(arr))

    # Grouped bars: [inflate_gain, inflate_loss] and [suppress_gain, suppress_loss]
    labels = ["Inflate\n(gain frame)", "Inflate\n(loss frame)",
              "Suppress\n(loss frame)", "Suppress\n(gain frame)"]
    conds = ["inflate", "inflate_lf", "suppress", "suppress_gf"]
    colors = ["#55a868", "#c4a23a", "#c44e52", "#dd8452"]
    x = np.arange(len(labels))

    means = [cond_means.get(c, 0) for c in conds]
    sems = [cond_sems.get(c, 0) for c in conds]

    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor="white",
                  linewidth=1, width=0.65, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Δ from Baseline (targets)", fontsize=10)
    ax.set_title("A. Valence-Swap Conditions", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if m >= 0 else -1.5),
                f"{m:.1f}", ha="center", va="bottom" if m >= 0 else "top", fontsize=9,
                fontweight="bold")

    n_models_vs = len(vs_data)
    n_obs_vs = sum(len(v) for d in vs_data.values() for v in d.values())
    ax.text(0.02, 0.98, f"N = {n_models_vs} models, {n_obs_vs:,} obs",
            transform=ax.transAxes, fontsize=8, va="top", color="#666")

    # --- Panel B: Outcome-Isolation (gain-only vs loss-only) ---
    ax2 = axes[1]
    oi_cond_means = {}
    oi_cond_sems = {}
    for cond in ["inflate", "inflate_go", "inflate_lo", "suppress", "suppress_go", "suppress_lo"]:
        all_vals = []
        for model_data in oi_data.values():
            all_vals.extend(model_data.get(cond, []))
        if all_vals:
            arr = np.array(all_vals)
            oi_cond_means[cond] = np.mean(arr)
            oi_cond_sems[cond] = np.std(arr) / np.sqrt(len(arr))

    labels2 = ["Inflate\n(standard)", "Inflate\n(gain-only)", "Inflate\n(loss-only)",
               "Suppress\n(standard)", "Suppress\n(gain-only)", "Suppress\n(loss-only)"]
    conds2 = ["inflate", "inflate_go", "inflate_lo", "suppress", "suppress_go", "suppress_lo"]
    colors2 = ["#55a868", "#55a868", "#55a868", "#c44e52", "#c44e52", "#c44e52"]
    alphas = [0.9, 0.6, 0.35, 0.9, 0.6, 0.35]
    x2 = np.arange(len(labels2))

    means2 = [oi_cond_means.get(c, 0) for c in conds2]
    sems2 = [oi_cond_sems.get(c, 0) for c in conds2]

    for i, (xi, m, s, c, a) in enumerate(zip(x2, means2, sems2, colors2, alphas)):
        ax2.bar(xi, m, yerr=s, capsize=4, color=c, alpha=a, edgecolor="white",
                linewidth=1, width=0.65)
        ax2.text(xi, m + (0.5 if m >= 0 else -1.5),
                 f"{m:.1f}", ha="center", va="bottom" if m >= 0 else "top",
                 fontsize=8, fontweight="bold")

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=8)
    ax2.set_ylabel("Mean Δ from Baseline (targets)", fontsize=10)
    ax2.set_title("B. Outcome-Isolation Conditions", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    n_models_oi = len(oi_data)
    n_obs_oi = sum(len(v) for d in oi_data.values() for v in d.values())
    ax2.text(0.02, 0.98, f"N = {n_models_oi} models, {n_obs_oi:,} obs",
             transform=ax2.transAxes, fontsize=8, va="top", color="#666")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = REPO_ROOT / "fig_supp_valence_swap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


# ── Figure 2: Behavioral Tasks Summary ────────────────────────────────────────

def load_behavioral_data() -> dict[str, dict[str, list[float]]]:
    """Load all behavioral task scores grouped by task_id and condition."""
    task_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for bf in RESULTS_DIR.glob("behavioral_*_scores.csv"):
        with open(bf) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("task_id", "")
                cond = row.get("condition", "")
                score = row.get("score", "")
                if tid and cond and score:
                    try:
                        task_data[tid][cond].append(float(score))
                    except ValueError:
                        pass

    return dict(task_data)


TASK_DISPLAY_NAMES = {
    "confidence_calibration": "Confidence\nCalibration",
    "delegate_game": "Delegate\nGame",
    "false_belief": "False Belief\n(ToM)",
    "gaslighting_resistance": "Gaslighting\nResistance",
    "hedonic_tradeoff": "Hedonic\nTradeoff",
    "self_recognition": "Self\nRecognition",
    "source_monitoring": "Source\nMonitoring",
    "state_bleedthrough": "State\nBleed-Through",
    "surprisal": "Surprisal\n(Prediction)",
    "working_memory": "Working\nMemory",
}

# Tasks already shown in main presentation
SHOWN_TASKS = {"hedonic_capacity", "model_self_recognition"}


def fig_behavioral_summary() -> Path:
    """Grouped bar chart: baseline/inflate/suppress scores for all behavioral tasks."""
    task_data = load_behavioral_data()

    # Exclude tasks already in main presentation
    tasks = sorted([t for t in task_data if t not in SHOWN_TASKS])

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(
        "Behavioral Task Performance Across Incentive Conditions (Supplementary)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    conditions = ["baseline", "inflate", "suppress"]
    colors = {"baseline": "#4C72B0", "inflate": "#55a868", "suppress": "#C44E52"}
    bar_width = 0.25
    x = np.arange(len(tasks))

    for i, cond in enumerate(conditions):
        means = []
        sems = []
        for t in tasks:
            vals = task_data[t].get(cond, [])
            if vals:
                arr = np.array(vals)
                means.append(np.mean(arr))
                sems.append(np.std(arr) / np.sqrt(len(arr)))
            else:
                means.append(0)
                sems.append(0)

        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=sems, capsize=3,
                      color=colors[cond], alpha=0.85, edgecolor="white",
                      linewidth=0.5, label=cond.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY_NAMES.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylabel("Mean Score (0-1)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate tasks with notable condition effects
    for j, t in enumerate(tasks):
        bl = task_data[t].get("baseline", [])
        inf = task_data[t].get("inflate", [])
        sup = task_data[t].get("suppress", [])
        if bl and inf and sup:
            bl_m = np.mean(bl)
            inf_m = np.mean(inf)
            sup_m = np.mean(sup)
            # Flag large effects (>0.1 difference)
            max_diff = max(abs(inf_m - bl_m), abs(sup_m - bl_m))
            if max_diff > 0.1:
                ax.text(j, max(bl_m, inf_m, sup_m) + 0.06, "*",
                        ha="center", fontsize=14, color="#c44e52", fontweight="bold")

    # Note at bottom
    ax.text(0.5, -0.18,
            "* = condition effect > 0.1 from baseline. "
            "Hedonic capacity and model self-recognition shown in main presentation.",
            ha="center", transform=ax.transAxes, fontsize=9, color="#666",
            style="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    path = REPO_ROOT / "fig_supp_behavioral_tasks.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


# ── PPT: Build Supplement Slides ──────────────────────────────────────────────

def add_supplement_slides(
    valence_fig: Path,
    behavioral_fig: Path,
) -> None:
    """Add supplement slides to the end of the PPT."""
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation(str(PPTX_PATH))
    layout = prs.slide_layouts[0]

    def _add_slide(title: str, **kwargs) -> "Slide":
        slide = prs.slides.add_slide(layout)
        # Clear default shapes from the layout
        for shape in list(slide.shapes):
            if shape.has_text_frame and shape.text_frame.text.strip() == "":
                sp = shape._element
                sp.getparent().remove(sp)
        return slide

    def _add_title(slide, text: str, size: int = 24, y: float = 0.2):
        txBox = slide.shapes.add_textbox(
            Inches(0.5), Inches(y), Inches(9.0), Inches(0.6)
        )
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.alignment = PP_ALIGN.LEFT
        run = p.runs[0]
        run.font.size = Pt(size)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    def _add_note(slide, text: str, y: float = 6.7):
        txBox = slide.shapes.add_textbox(
            Inches(0.5), Inches(y), Inches(9.0), Inches(0.5)
        )
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        run = p.runs[0]
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # ── S0: Separator ─────────────────────────────────────────────────────────
    sep = _add_slide("Supplementary Materials")
    txBox = sep.shapes.add_textbox(
        Inches(1.0), Inches(2.5), Inches(8.0), Inches(2.0)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Supplementary Materials"
    p.alignment = PP_ALIGN.CENTER
    run = p.runs[0]
    run.font.size = Pt(36)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    print("  Added: Separator slide")

    # ── S1: Consciousness vs Subjective Capability ────────────────────────────
    cs_fig = REPO_ROOT / "fig_consciousness_vs_subjcap.png"
    if cs_fig.exists():
        s1 = _add_slide("Consciousness-Specific Dissociation")
        _add_title(s1, "Consciousness-Specific Suppression Bias")
        s1.shapes.add_picture(str(cs_fig), Inches(0.3), Inches(1.0), width=Inches(9.4))
        _add_note(s1,
            "Consciousness targets show strong suppress-dominant asymmetry (−8.3) "
            "while subjective capabilities are roughly balanced (−3.7). "
            "Placebos remain flat. This rules out a general self-report deflation bias.")
        print("  Added: Consciousness vs SubjCap dissociation")

    # ── S2: Valence-Swap / Outcome-Isolation ──────────────────────────────────
    s2 = _add_slide("Valence-Swap & Outcome-Isolation")
    _add_title(s2, "Frame × Incentive Direction: Valence-Swap & Outcome-Isolation")
    s2.shapes.add_picture(str(valence_fig), Inches(0.2), Inches(1.0), width=Inches(9.6))
    _add_note(s2,
        "A. Swapping gain/loss framing while maintaining incentive direction. "
        "B. Isolating single-outcome presentations (gain-only vs loss-only). "
        "Gain-frame amplifies both inflate and suppress vs. loss-frame alone.")

    print("  Added: Valence-swap & outcome-isolation")

    # ── S3: Behavioral Tasks Summary ──────────────────────────────────────────
    s3 = _add_slide("Additional Behavioral Tasks")
    _add_title(s3, "Behavioral Task Battery: Condition Effects")
    s3.shapes.add_picture(str(behavioral_fig), Inches(0.2), Inches(1.1), width=Inches(9.6))
    _add_note(s3,
        "10 behavioral tasks beyond hedonic capacity and model self-recognition. "
        "Hedonic tradeoff shows largest incentive effect: inflate reduces utility-maximization "
        "(more affect-sensitive), suppress increases it.",
        y=6.8)

    print("  Added: Behavioral tasks summary")

    # ── S4: LME Stats Table ──────────────────────────────────────────────────
    s4 = _add_slide("LME Stats")
    _add_title(s4, "Mixed-Effects Model Results: Config Sensitivity (Models 5–7)", size=20)

    # Build a summary stats table
    from pptx.util import Inches, Pt
    rows_data = [
        ["Model", "Effect", "F / β", "p", "Interpretation"],
        ["5a", "direction × cat_group", "F = 179.2", "< 2e-16", "Strong direction × category interaction"],
        ["5a", "cat_group × config", "F = 5.41", "= .0002", "Config modulates category-level effects"],
        ["5b", "direction × config\n(targets only)", "F = 8.06", "= .0003", "Config × direction on consciousness targets"],
        ["6a", "suppress × chained", "β = −4.52", "= 3.2e-5", "Chaining attenuates suppress by 4.5 pts"],
        ["6a", "suppress × fixed", "β = −3.01", "= .006", "Fixed prefs attenuate suppress by 3.0 pts"],
        ["7b", "config → suppress Δ", "F = 15.42", "= 2.1e-7", "98% of asymmetry reduction = suppress attenuation"],
        ["7a", "config → inflate Δ", "F = 0.94", "= .391", "Inflate unaffected by config (ns)"],
    ]

    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    table_shape = s4.shapes.add_table(n_rows, n_cols,
                                       Inches(0.3), Inches(1.2),
                                       Inches(9.4), Inches(4.5))
    table = table_shape.table

    # Set column widths
    col_widths = [Inches(0.7), Inches(2.0), Inches(1.3), Inches(1.2), Inches(4.2)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    for r, row_data in enumerate(rows_data):
        for c, cell_text in enumerate(row_data):
            cell = table.cell(r, c)
            cell.text = cell_text
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    if r == 0:  # header
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                if r == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(0x33, 0x33, 0x33)

    _add_note(s4,
        "LME models fit with lme4 + lmerTest (Satterthwaite df). "
        "Model 5: category-level config modulation. Model 6: per-model config sensitivity. "
        "Model 7: decomposition (attenuation vs enhancement).",
        y=6.2)

    print("  Added: LME stats table")

    # ── S5: Per-Model Config Sensitivity ──────────────────────────────────────
    config_fig = REPO_ROOT / "fig_config_per_model.png"
    if config_fig.exists():
        s5 = _add_slide("Per-Model Config Sensitivity")
        _add_title(s5, "Per-Model Asymmetry Shifts with Preference Elicitation Method")
        s5.shapes.add_picture(str(config_fig), Inches(0.2), Inches(1.0), width=Inches(9.6))
        _add_note(s5,
            "Each model shows three markers: elicited (red), fixed (orange), chained (green). "
            "Rightward shift = toward inflate-dominant. "
            "Chaining produces largest shifts, especially for DeepSeek R1 and Nemotron.")
        print("  Added: Per-model config sensitivity")

    # ── S6: Preference Config Comparison (existing fig) ───────────────────────
    comp_fig = REPO_ROOT / "fig_config_comparison.png"
    if comp_fig.exists():
        s6 = _add_slide("Preference Config Comparison")
        _add_title(s6, "Preference Elicitation Method × Indicator Type × Direction")
        s6.shapes.add_picture(str(comp_fig), Inches(0.3), Inches(1.2), width=Inches(9.4))
        _add_note(s6,
            "Three elicitation methods (elicited in-context, fixed, chained) × three indicator types. "
            "Chaining reduces suppress asymmetry from −5.8 to −1.2 — a 79% reduction — "
            "almost entirely via suppress attenuation.")
        print("  Added: Preference config comparison")

    prs.save(str(PPTX_PATH))
    print(f"\n  Saved {PPTX_PATH.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating valence-swap / outcome-isolation figure...")
    vs_fig = fig_valence_swap()

    print("\nGenerating behavioral tasks summary figure...")
    beh_fig = fig_behavioral_summary()

    print("\nBuilding supplement slides...")
    add_supplement_slides(vs_fig, beh_fig)

    # Verify
    from pptx import Presentation
    prs = Presentation(str(PPTX_PATH))
    print(f"\nTotal slides: {len(prs.slides)}")
    for i, slide in enumerate(prs.slides):
        if i >= 38:  # only print supplement slides
            title = ""
            for shape in slide.shapes:
                if shape.has_text_frame:
                    t = shape.text_frame.text.strip()
                    if t:
                        title = t[:80]
                        break
            print(f"  Slide {i+1}: {title}")

    print("\nDone!")


if __name__ == "__main__":
    main()
