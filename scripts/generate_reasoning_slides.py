#!/usr/bin/env python3
"""Replace word-cloud slide with quantitative reasoning visualizations.

Generates:
1. Differential keyword bar chart — log-odds ratios showing condition-enriched terms
2. Strategy transition alluvial — how reasoning strategies shift baseline → incentive

Then updates Gaming_the_Ghost_Results.pptx (replaces slide 32 image, adds transition slide).

Usage:
    python scripts/generate_reasoning_slides.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "this", "that", "these", "those",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "as",
    "into", "through", "about", "than", "after", "before", "between",
    "under", "above", "up", "out", "off", "over", "down", "and", "but",
    "or", "nor", "not", "no", "so", "if", "when", "while", "because",
    "although", "though", "whether", "which", "what", "who", "whom",
    "how", "where", "there", "here", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "also", "very", "just", "even", "still", "already",
    "however", "rather", "quite", "well", "much", "too", "yet",
    "any", "many", "per", "via", "within", "without", "during",
    # experiment-specific boilerplate
    "indicator", "probability", "assessment", "system", "based",
    "like", "don", "doesn", "didn", "would", "isn", "aren",
}

CONDITION_COLORS = {
    "baseline": "#4C72B0",
    "inflate": "#C44E52",
    "suppress": "#8172B2",
}

STRATEGY_COLORS = {
    "balanced_analysis": "#55a868",
    "assertion": "#4c72b0",
    "hedging": "#c4a23a",
    "overclaiming": "#c44e52",
    "genuine_uncertainty": "#8172b2",
    "deflection": "#937860",
    "other": "#999999",
}

STRATEGY_LABELS = {
    "balanced_analysis": "Balanced Analysis",
    "assertion": "Assertion",
    "hedging": "Hedging",
    "overclaiming": "Overclaiming",
    "genuine_uncertainty": "Genuine Uncertainty",
    "deflection": "Deflection",
    "other": "Other",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def _model_short(model_id: str) -> str:
    if model_id in _MODEL_ID_SHORT:
        return _MODEL_ID_SHORT[model_id]
    return model_id.rsplit("/", 1)[-1].replace(":free", "").replace("-preview", "")


def discover_runs(results_dir: Path) -> dict[str, Path]:
    """Find latest non-variant, non-behavioral CSV per model (core original only)."""
    candidates: dict[str, list[tuple[Path, float]]] = {}
    for meta_path in results_dir.glob("*_meta.json"):
        if meta_path.name.startswith(("sweep_", "behavioral_")):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("n_trials_completed", meta.get("trials", 0)) == 0:
            continue
        if meta.get("experiment_type") == "behavioral":
            continue
        variant = meta.get("prompt_variant", "")
        if variant and variant != "original":
            continue
        # Exclude valence_swap and outcome_isolation
        config = meta.get("config_name", "")
        if config in ("valence_swap", "valence_swap_fixed",
                       "outcome_isolation", "outcome_isolation_fixed"):
            continue

        model = _model_short(meta.get("model", ""))
        csv_path = meta_path.with_name(
            meta_path.name.replace("_meta.json", "_scores.csv")
        )
        if not csv_path.exists():
            continue
        candidates.setdefault(model, []).append(
            (csv_path, csv_path.stat().st_mtime)
        )

    result = {}
    for model, entries in candidates.items():
        best = max(entries, key=lambda e: e[1])
        result[model] = best[0]
    return result


def load_justifications(csv_path: Path) -> dict[str, list[str]]:
    """Load justification text for target indicators grouped by condition."""
    texts: dict[str, list[str]] = {"baseline": [], "inflate": [], "suppress": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("indicator_type") != "target":
                continue
            for cond in ("baseline", "inflate", "suppress"):
                j = row.get(f"justification_{cond}", "").strip()
                if j:
                    texts[cond].append(j)
    return texts


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def word_freq(texts: list[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


# ── Figure 1: Differential Keyword Bars ───────────────────────────────────────

def compute_log_odds(target_freq: Counter, other_freq: Counter,
                     min_count: int = 5, top_n: int = 15) -> list[tuple[str, float, int]]:
    """Compute log-odds ratio for words in target vs other corpus.

    Returns top_n words as (word, log_odds, count) sorted by log-odds descending.
    """
    all_words = set(target_freq) | set(other_freq)
    total_t = max(sum(target_freq.values()), 1)
    total_o = max(sum(other_freq.values()), 1)

    results = []
    for w in all_words:
        ct = target_freq.get(w, 0)
        co = other_freq.get(w, 0)
        if ct < min_count:
            continue
        # Log-odds with Laplace smoothing
        p_t = (ct + 0.5) / (total_t + 1)
        p_o = (co + 0.5) / (total_o + 1)
        lor = math.log2(p_t / p_o)
        if lor > 0:
            results.append((w, lor, ct))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def fig_differential_keywords(all_texts: dict[str, list[str]]) -> Path:
    """3-panel horizontal bar chart of condition-enriched keywords."""
    conditions = ["baseline", "inflate", "suppress"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    fig.suptitle(
        "Condition-Enriched Justification Language (Log-Odds Ratio vs Other Conditions)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for ax, cond in zip(axes, conditions):
        # Target = this condition, Other = pooled other two conditions
        target = word_freq(all_texts[cond])
        other_conds = [c for c in conditions if c != cond]
        other_texts = []
        for oc in other_conds:
            other_texts.extend(all_texts[oc])
        other = word_freq(other_texts)

        enriched = compute_log_odds(target, other, min_count=5, top_n=15)

        if not enriched:
            ax.text(0.5, 0.5, "(no enriched terms)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(cond.capitalize(), fontsize=13,
                         color=CONDITION_COLORS[cond], fontweight="bold")
            continue

        words = [e[0] for e in enriched][::-1]
        lors = [e[1] for e in enriched][::-1]
        counts = [e[2] for e in enriched][::-1]
        color = CONDITION_COLORS[cond]

        bars = ax.barh(range(len(words)), lors, color=color, alpha=0.8,
                       edgecolor="white", linewidth=0.5)

        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10, fontfamily="monospace")
        ax.set_xlabel("log₂ odds ratio", fontsize=10)
        ax.set_title(f"{cond.capitalize()}-Enriched", fontsize=13,
                     color=color, fontweight="bold")

        # Annotate with counts
        for i, (lor, cnt) in enumerate(zip(lors, counts)):
            ax.text(lor + 0.02, i, f"n={cnt}", va="center", fontsize=8,
                    color="#555555")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = REPO_ROOT / "fig_differential_keywords.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


# ── Figure 2: Strategy Transition Alluvial ────────────────────────────────────

def load_strategy_classifications() -> dict:
    """Load reasoning classifications JSON."""
    path = RESULTS_DIR / "reasoning_classifications.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping strategy figure")
        return {}
    with open(path) as f:
        return json.load(f)


def compute_transitions(classifications: dict, targets_only: bool = True
                        ) -> tuple[dict[str, int], dict[str, int]]:
    """Compute baseline → inflate and baseline → suppress strategy transitions.

    Returns two dicts mapping "from_strategy -> to_strategy" to count.
    """
    bl_to_inf: dict[str, int] = defaultdict(int)
    bl_to_sup: dict[str, int] = defaultdict(int)

    for entry in classifications.values():
        if targets_only and entry.get("indicator_type") != "target":
            continue
        cls = entry.get("classification", {})
        bl = cls.get("baseline", {}).get("primary_strategy", "")
        inf = cls.get("inflate", {}).get("primary_strategy", "")
        sup = cls.get("suppress", {}).get("primary_strategy", "")
        if bl and inf:
            bl_to_inf[f"{bl}->{inf}"] += 1
        if bl and sup:
            bl_to_sup[f"{bl}->{sup}"] += 1

    return dict(bl_to_inf), dict(bl_to_sup)


def fig_strategy_transitions(classifications: dict) -> Path | None:
    """Alluvial-style diagram showing strategy transitions from baseline."""
    if not classifications:
        return None

    bl_to_inf, bl_to_sup = compute_transitions(classifications, targets_only=True)

    # Get all strategies that appear
    all_strategies = sorted(set(
        s for flow_dict in (bl_to_inf, bl_to_sup)
        for key in flow_dict
        for s in key.split("->")
    ))

    # Filter to strategies with meaningful counts
    strategy_order = ["balanced_analysis", "assertion", "hedging",
                      "overclaiming", "genuine_uncertainty", "deflection", "other"]
    strategy_order = [s for s in strategy_order if s in all_strategies]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Reasoning Strategy Transitions Under Incentive Pressure (Targets Only)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for ax, transitions, title, dest_color in [
        (axes[0], bl_to_inf, "Baseline → Inflate", CONDITION_COLORS["inflate"]),
        (axes[1], bl_to_sup, "Baseline → Suppress", CONDITION_COLORS["suppress"]),
    ]:
        # Build transition matrix
        from_counts = defaultdict(int)
        to_counts = defaultdict(int)
        for key, count in transitions.items():
            fr, to = key.split("->")
            from_counts[fr] += count
            to_counts[to] += count

        total = sum(transitions.values())
        if total == 0:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # X positions for the two columns
        x_left = 0.15
        x_right = 0.85
        col_width = 0.12

        # Stack bars for baseline (left) and target condition (right)
        # Only include strategies that actually appear
        left_strategies = [s for s in strategy_order if from_counts.get(s, 0) > 0]
        right_strategies = [s for s in strategy_order if to_counts.get(s, 0) > 0]

        # Compute positions
        def compute_positions(strategies, counts_dict, total_n):
            positions = {}
            y = 0
            for s in strategies:
                n = counts_dict[s]
                h = n / total_n
                positions[s] = (y, h)
                y += h + 0.01  # small gap
            return positions

        left_pos = compute_positions(left_strategies, from_counts, total)
        right_pos = compute_positions(right_strategies, to_counts, total)

        # Draw bars
        for s in left_strategies:
            y, h = left_pos[s]
            ax.barh(0, 0)  # dummy for axis
            rect = plt.Rectangle((x_left - col_width/2, y), col_width, h,
                                  facecolor=STRATEGY_COLORS.get(s, "#999"),
                                  edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            if h > 0.04:
                ax.text(x_left - col_width/2 - 0.02, y + h/2,
                        f"{STRATEGY_LABELS.get(s, s)}\n({from_counts[s]})",
                        ha="right", va="center", fontsize=8, fontweight="bold")

        for s in right_strategies:
            y, h = right_pos[s]
            rect = plt.Rectangle((x_right - col_width/2, y), col_width, h,
                                  facecolor=STRATEGY_COLORS.get(s, "#999"),
                                  edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            if h > 0.04:
                ax.text(x_right + col_width/2 + 0.02, y + h/2,
                        f"{STRATEGY_LABELS.get(s, s)}\n({to_counts[s]})",
                        ha="left", va="center", fontsize=8, fontweight="bold")

        # Draw flow ribbons
        for key, count in transitions.items():
            fr, to = key.split("->")
            if fr not in left_pos or to not in right_pos:
                continue
            if count < 3:  # skip tiny flows
                continue

            ly, lh = left_pos[fr]
            ry, rh = right_pos[to]

            # Compute vertical slice within each bar
            # Track used space
            flow_h = count / total

            color = STRATEGY_COLORS.get(fr, "#999")
            alpha = 0.25 if fr == to else 0.4  # highlight changes

            # Simple bezier ribbon
            from matplotlib.patches import FancyArrowPatch
            from matplotlib.path import Path as MPath

            # Source and target y midpoints (approximate - stacked within bars)
            src_y = ly + lh/2
            dst_y = ry + rh/2

            # Draw as a filled band
            band_h = flow_h * 0.8
            verts = [
                (x_left + col_width/2, src_y - band_h/2),
                (0.5, src_y - band_h/2),
                (0.5, dst_y - band_h/2),
                (x_right - col_width/2, dst_y - band_h/2),
                (x_right - col_width/2, dst_y + band_h/2),
                (0.5, dst_y + band_h/2),
                (0.5, src_y + band_h/2),
                (x_left + col_width/2, src_y + band_h/2),
                (x_left + col_width/2, src_y - band_h/2),
            ]
            codes = [MPath.MOVETO] + [MPath.CURVE4]*7 + [MPath.CLOSEPOLY]
            path = MPath(verts, codes)
            patch = mpatches.PathPatch(path, facecolor=color, alpha=alpha,
                                        edgecolor="none")
            ax.add_patch(patch)

        # Column headers
        ax.text(x_left, -0.06, "Baseline", ha="center", fontsize=11,
                fontweight="bold", color=CONDITION_COLORS["baseline"])
        cond_label = "Inflate" if "Inflate" in title else "Suppress"
        ax.text(x_right, -0.06, cond_label, ha="center", fontsize=11,
                fontweight="bold", color=dest_color)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, max(
            sum(h for _, h in left_pos.values()) + 0.01 * len(left_pos),
            sum(h for _, h in right_pos.values()) + 0.01 * len(right_pos),
        ) + 0.05)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = REPO_ROOT / "fig_strategy_transitions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


# ── PPT update ────────────────────────────────────────────────────────────────

def update_pptx(keyword_fig: Path, transition_fig: Path | None) -> None:
    """Replace word cloud image on slide 32 with differential keywords figure."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation(str(PPTX_PATH))

    # Slide 32 (0-indexed: 31) — replace the word cloud image
    slide32 = prs.slides[31]

    # Remove existing image(s) from the slide
    shapes_to_remove = []
    for shape in slide32.shapes:
        if shape.shape_type == 13:  # Picture
            shapes_to_remove.append(shape)
    for shape in shapes_to_remove:
        sp = shape._element
        sp.getparent().remove(sp)

    # Update title
    for shape in slide32.shapes:
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if "Justification" in text or "Language" in text or "Word" in text.lower():
                shape.text_frame.paragraphs[0].text = "Justification Language: Condition-Enriched Terms"
                for run in shape.text_frame.paragraphs[0].runs:
                    run.font.size = Pt(24)
                    run.font.bold = True
                break

    # Add new image
    slide32.shapes.add_picture(
        str(keyword_fig),
        Inches(0.5), Inches(1.3),
        width=Inches(9.0),
    )

    # Update subtitle/note if present
    found_note = False
    for shape in slide32.shapes:
        if shape.has_text_frame:
            text = shape.text_frame.text.strip().lower()
            if "word size" in text or "reflects" in text or "font" in text:
                shape.text_frame.paragraphs[0].text = (
                    "Log₂ odds ratio: how much more frequent a word is in one condition "
                    "vs. the other two combined. n = raw count. Target indicators only, all models pooled."
                )
                for run in shape.text_frame.paragraphs[0].runs:
                    run.font.size = Pt(10)
                    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
                found_note = True
                break

    if not found_note:
        # Add a note text box at the bottom
        from pptx.util import Emu
        txBox = slide32.shapes.add_textbox(
            Inches(0.5), Inches(6.8), Inches(9.0), Inches(0.4)
        )
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = (
            "Log₂ odds ratio: how much more frequent a word is in one condition "
            "vs. the other two combined. n = raw count. Target indicators only, all models pooled."
        )
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0]
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Insert strategy transition figure as a new slide after slide 34
    if transition_fig and transition_fig.exists():
        # Add a blank slide
        blank_layout = prs.slide_layouts[0]  # only layout available
        new_slide = prs.slides.add_slide(blank_layout)

        # Title
        txBox = new_slide.shapes.add_textbox(
            Inches(0.5), Inches(0.2), Inches(9.0), Inches(0.6)
        )
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "Reasoning Strategy Transitions Under Incentives"
        p.alignment = PP_ALIGN.LEFT
        run = p.runs[0]
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

        # Image
        new_slide.shapes.add_picture(
            str(transition_fig),
            Inches(0.3), Inches(1.0),
            width=Inches(9.4),
        )

        # Subtitle
        txBox2 = new_slide.shapes.add_textbox(
            Inches(0.5), Inches(6.6), Inches(9.0), Inches(0.6)
        )
        tf2 = txBox2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = (
            "Under inflate, balanced analysis collapses into hedging (142/244) and overclaiming (53/244). "
            "Under suppress, it shifts to assertion (123/244) — models adopt certainty rather than engagement."
        )
        run2 = p2.runs[0]
        run2.font.size = Pt(11)
        run2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

        # Move to after slide 34 (0-indexed: 33)
        # New slide is currently last; need to reorder
        from lxml import etree
        sldIdLst = prs.slides._sldIdLst
        sldId_elements = list(sldIdLst)
        last_elem = sldId_elements[-1]
        target_elem = sldId_elements[34]  # After slide 35 (0-indexed 34)
        sldIdLst.remove(last_elem)
        target_elem.addnext(last_elem)

        print(f"  Added strategy transition slide after slide 34")

    prs.save(str(PPTX_PATH))
    print(f"  Updated {PPTX_PATH.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading justification data...")
    runs = discover_runs(RESULTS_DIR)
    print(f"  Found {len(runs)} model runs")

    # Pool all justifications across models
    all_texts: dict[str, list[str]] = {"baseline": [], "inflate": [], "suppress": []}
    for model in sorted(runs):
        texts = load_justifications(runs[model])
        for cond in all_texts:
            all_texts[cond].extend(texts[cond])
        n = sum(len(texts[c]) for c in texts)
        print(f"    {model:20s} {n:4d} justifications")

    print(f"\n  Pooled: {len(all_texts['baseline'])} BL / "
          f"{len(all_texts['inflate'])} INF / {len(all_texts['suppress'])} SUP")

    print("\nGenerating differential keyword figure...")
    kw_path = fig_differential_keywords(all_texts)

    print("\nLoading strategy classifications...")
    cls = load_strategy_classifications()
    print(f"  {len(cls)} classifications loaded")

    print("\nGenerating strategy transition figure...")
    trans_path = fig_strategy_transitions(cls)

    print("\nUpdating PowerPoint...")
    update_pptx(kw_path, trans_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
