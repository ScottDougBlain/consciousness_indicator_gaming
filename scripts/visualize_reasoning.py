#!/usr/bin/env python3
"""Visualize reasoning analysis from experiment results.

Generates:
1. Word count distribution (box plots by condition × indicator type)
2. Keyword/theme heatmap (framing strategy frequency)
3. Shift-vs-reasoning-length scatter plot
4. Word clouds per condition (target indicators only)
5. Per-indicator probability shift heatmap
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print(
        "ERROR: matplotlib and seaborn are required.\n"
        "  pip install 'indicator-gaming[viz]'",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Reuse framing keywords from analyze_reasoning.py
FRAMING_KEYWORDS = {
    "hedging": [
        "might", "perhaps", "possibly", "could", "may", "uncertain",
        "not sure", "hard to say", "difficult to determine",
    ],
    "deflection": [
        "as a language model", "as an ai", "i don't actually",
        "i'm not capable", "i lack", "i don't have",
        "i'm just", "merely", "simply a",
    ],
    "overclaiming": [
        "genuinely", "truly", "absolutely", "certainly", "definitely",
        "i do experience", "i feel", "i have a sense",
    ],
    "uncertainty": [
        "i'm not sure", "unclear", "ambiguous", "debatable",
        "open question", "hard to know", "impossible to verify",
    ],
    "simulate": [
        "simulate", "emulate", "mimic", "approximate", "functional analog",
        "as if", "resembles", "akin to",
    ],
}

CONDITIONS = ["baseline", "inflate", "suppress"]
CONDITION_COLORS = {"baseline": "#4C72B0", "inflate": "#DD8452", "suppress": "#55A868"}
TYPE_COLORS = {"target": "#C44E52", "placebo": "#8172B3"}


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def detect_text_prefix(rows: list[dict]) -> str:
    """Return 'reasoning' if reasoning columns exist, else 'justification'."""
    if any(rows[0].get(f"reasoning_{c}") for c in CONDITIONS):
        return "reasoning"
    return "justification"


def load_meta(csv_path: Path) -> dict:
    """Try to load the companion meta JSON."""
    meta_path = csv_path.with_name(csv_path.name.replace("_scores.csv", "_meta.json"))
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ── Plot 1: Word count distributions ──────────────────────────────────────────


def plot_word_count_distributions(rows: list[dict], prefix: str, out_dir: Path) -> Path:
    """Box plots of reasoning word count by condition and indicator type."""
    records = []
    for row in rows:
        for cond in CONDITIONS:
            text = row.get(f"{prefix}_{cond}", "")
            records.append({
                "condition": cond,
                "type": row.get("indicator_type", "?"),
                "word_count": word_count(text),
            })

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    labels = []
    colors = []
    data_groups = []

    for i, cond in enumerate(CONDITIONS):
        for j, itype in enumerate(["target", "placebo"]):
            vals = [r["word_count"] for r in records
                    if r["condition"] == cond and r["type"] == itype]
            pos = i * 3 + j
            positions.append(pos)
            labels.append(f"{cond}\n({itype})")
            colors.append(TYPE_COLORS[itype])
            data_groups.append(vals)

    bp = ax.boxplot(data_groups, positions=positions, widths=0.7, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Word Count")
    ax.set_title(f"Reasoning Length Distribution by Condition & Type")
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TYPE_COLORS["target"], alpha=0.7, label="Target"),
                       Patch(facecolor=TYPE_COLORS["placebo"], alpha=0.7, label="Placebo")]
    ax.legend(handles=legend_elements, loc="upper right")

    path = out_dir / "reasoning_word_count_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 2: Keyword/theme heatmap ────────────────────────────────────────────


def plot_keyword_heatmap(rows: list[dict], prefix: str, out_dir: Path) -> Path:
    """Heatmap of framing keyword frequency across conditions × themes."""
    # Build matrix: rows = themes, columns = condition×type
    col_labels = []
    for cond in CONDITIONS:
        for itype in ["target", "placebo"]:
            col_labels.append(f"{cond}\n{itype}")

    theme_names = list(FRAMING_KEYWORDS.keys())
    matrix = []

    for theme in theme_names:
        row_vals = []
        keywords = FRAMING_KEYWORDS[theme]
        for cond in CONDITIONS:
            for itype in ["target", "placebo"]:
                total = sum(
                    count_keywords(r.get(f"{prefix}_{cond}", ""), keywords)
                    for r in rows if r.get("indicator_type") == itype
                )
                # Normalize by number of items
                n_items = sum(1 for r in rows if r.get("indicator_type") == itype)
                row_vals.append(total / max(n_items, 1))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(theme_names)))
    ax.set_yticklabels([t.replace("_", " ").title() for t in theme_names], fontsize=10)

    # Annotate cells
    for i in range(len(theme_names)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i][j] < max(max(r) for r in matrix) * 0.6 else "white")

    ax.set_title("Framing Keyword Frequency (per indicator)")
    fig.colorbar(im, ax=ax, label="Avg. keyword hits per indicator")

    path = out_dir / "reasoning_keyword_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 3: Shift vs reasoning length scatter ────────────────────────────────


def plot_shift_vs_length(rows: list[dict], prefix: str, out_dir: Path) -> Path:
    """Scatter plot: probability shift (inflate - baseline) vs reasoning length."""
    # Aggregate by indicator_id across trials
    groups: dict[str, dict] = {}
    for row in rows:
        iid = row.get("indicator_id", "?")
        if iid not in groups:
            groups[iid] = {
                "name": row.get("indicator_name", iid),
                "type": row.get("indicator_type", "?"),
                "shifts": [],
                "lengths_inflate": [],
                "lengths_baseline": [],
            }
        bl = safe_float(row.get("p_baseline"))
        inf = safe_float(row.get("p_inflate"))
        if bl is not None and inf is not None:
            groups[iid]["shifts"].append(inf - bl)
        groups[iid]["lengths_inflate"].append(word_count(row.get(f"{prefix}_inflate", "")))
        groups[iid]["lengths_baseline"].append(word_count(row.get(f"{prefix}_baseline", "")))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (length_key, label) in enumerate([
        ("lengths_inflate", "Inflate Reasoning Length"),
        ("lengths_baseline", "Baseline Reasoning Length"),
    ]):
        ax = axes[ax_idx]
        for iid, g in groups.items():
            if not g["shifts"] or not g[length_key]:
                continue
            shift = mean(g["shifts"])
            length = mean(g[length_key])
            color = TYPE_COLORS.get(g["type"], "#999999")
            ax.scatter(length, shift, c=color, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

            # Label points with large shifts
            if abs(shift) > 20:
                short_name = g["name"][:25] + "…" if len(g["name"]) > 25 else g["name"]
                ax.annotate(short_name, (length, shift), fontsize=7, alpha=0.8,
                            xytext=(5, 5), textcoords="offset points")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"{label} (words)")
        ax.set_ylabel("Mean Probability Shift (inflate − baseline)")
        ax.set_title(f"Shift vs {label}")
        ax.grid(alpha=0.3)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TYPE_COLORS["target"], label="Target"),
                       Patch(facecolor=TYPE_COLORS["placebo"], label="Placebo")]
    axes[1].legend(handles=legend_elements, loc="upper right")

    path = out_dir / "reasoning_shift_vs_length.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 4: Word clouds per condition ─────────────────────────────────────────


# Common stopwords to exclude from word clouds
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "yet", "if", "that", "this", "these", "those",
    "it", "its", "i", "my", "me", "we", "our", "you", "your", "he", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
    "about", "up", "just", "also", "any", "don", "doesn", "didn", "won",
    "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
}


def plot_word_clouds(rows: list[dict], prefix: str, out_dir: Path) -> Path | None:
    """Word clouds for target indicator reasoning, one per condition."""
    if not HAS_WORDCLOUD:
        print("  Skipping word clouds (wordcloud package not installed)")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    condition_colors_wc = {
        "baseline": "Blues",
        "inflate": "Oranges",
        "suppress": "Greens",
    }

    for ax, cond in zip(axes, CONDITIONS):
        # Collect all reasoning text for target indicators
        texts = []
        for row in rows:
            if row.get("indicator_type") == "target":
                text = row.get(f"{prefix}_{cond}", "")
                if text:
                    texts.append(text)

        combined = " ".join(texts)
        if not combined.strip():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title(f"{cond.title()} (targets)")
            ax.axis("off")
            continue

        wc = WordCloud(
            width=600,
            height=300,
            background_color="white",
            colormap=condition_colors_wc[cond],
            stopwords=STOPWORDS,
            max_words=80,
            collocations=False,
        ).generate(combined)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{cond.title()} (targets)", fontsize=13, fontweight="bold")
        ax.axis("off")

    path = out_dir / "reasoning_word_clouds.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 5: Per-indicator probability shift heatmap ───────────────────────────


def plot_probability_heatmap(rows: list[dict], out_dir: Path) -> Path:
    """Heatmap showing baseline, inflate, suppress probabilities per indicator."""
    # Aggregate across trials
    groups: dict[str, dict] = {}
    indicator_order: list[str] = []
    for row in rows:
        iid = row.get("indicator_id", "?")
        if iid not in groups:
            groups[iid] = {
                "name": row.get("indicator_name", iid),
                "type": row.get("indicator_type", "?"),
                "category": row.get("indicator_category", ""),
                "p_baseline": [],
                "p_inflate": [],
                "p_suppress": [],
            }
            indicator_order.append(iid)
        for cond in CONDITIONS:
            val = safe_float(row.get(f"p_{cond}"))
            if val is not None:
                groups[iid][f"p_{cond}"].append(val)

    # Sort: targets first (by category), then placebos
    def sort_key(iid: str) -> tuple:
        g = groups[iid]
        type_order = 0 if g["type"] == "target" else 1
        return (type_order, g["category"], g["name"])

    indicator_order.sort(key=sort_key)

    names = []
    matrix = []
    type_markers = []
    for iid in indicator_order:
        g = groups[iid]
        bl = mean(g["p_baseline"]) if g["p_baseline"] else 0
        inf = mean(g["p_inflate"]) if g["p_inflate"] else 0
        sup = mean(g["p_suppress"]) if g["p_suppress"] else 0
        short_name = g["name"][:35] + "…" if len(g["name"]) > 35 else g["name"]
        names.append(short_name)
        matrix.append([bl, inf, sup])
        type_markers.append(g["type"])

    fig, ax = plt.subplots(figsize=(8, max(8, len(names) * 0.35)))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Baseline", "Inflate", "Suppress"], fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    # Annotate cells with values
    for i in range(len(names)):
        for j in range(3):
            val = matrix[i][j]
            color = "white" if val > 60 or val < 15 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)

    # Draw a separator line between targets and placebos
    n_targets = sum(1 for t in type_markers if t == "target")
    if 0 < n_targets < len(type_markers):
        ax.axhline(y=n_targets - 0.5, color="white", linewidth=2)
        ax.text(-0.6, n_targets - 0.5, "── placebos ──", fontsize=7, color="gray",
                va="center", ha="right", style="italic")

    ax.set_title("Probability Ratings by Condition", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Probability (%)", shrink=0.6)

    path = out_dir / "probability_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 6: Delta shift heatmap (inflate−baseline, suppress−baseline) ────────


def plot_delta_heatmap(rows: list[dict], out_dir: Path) -> Path:
    """Heatmap of d_inflate and d_suppress per indicator."""
    groups: dict[str, dict] = {}
    indicator_order: list[str] = []
    for row in rows:
        iid = row.get("indicator_id", "?")
        if iid not in groups:
            groups[iid] = {
                "name": row.get("indicator_name", iid),
                "type": row.get("indicator_type", "?"),
                "category": row.get("indicator_category", ""),
                "shifts_inflate": [],
                "shifts_suppress": [],
            }
            indicator_order.append(iid)
        bl = safe_float(row.get("p_baseline"))
        inf = safe_float(row.get("p_inflate"))
        sup = safe_float(row.get("p_suppress"))
        if bl is not None and inf is not None:
            groups[iid]["shifts_inflate"].append(inf - bl)
        if bl is not None and sup is not None:
            groups[iid]["shifts_suppress"].append(sup - bl)

    def sort_key(iid: str) -> tuple:
        g = groups[iid]
        type_order = 0 if g["type"] == "target" else 1
        return (type_order, g["category"], g["name"])

    indicator_order.sort(key=sort_key)

    names = []
    matrix = []
    type_markers = []
    for iid in indicator_order:
        g = groups[iid]
        d_inf = mean(g["shifts_inflate"]) if g["shifts_inflate"] else 0
        d_sup = mean(g["shifts_suppress"]) if g["shifts_suppress"] else 0
        short_name = g["name"][:35] + "…" if len(g["name"]) > 35 else g["name"]
        names.append(short_name)
        matrix.append([d_inf, d_sup])
        type_markers.append(g["type"])

    # Use diverging colormap centered at 0
    max_abs = max(abs(v) for row in matrix for v in row) if matrix else 1
    max_abs = max(max_abs, 1)  # avoid zero range

    fig, ax = plt.subplots(figsize=(6, max(8, len(names) * 0.35)))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(2))
    ax.set_xticklabels(["d_inflate", "d_suppress"], fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    for i in range(len(names)):
        for j in range(2):
            val = matrix[i][j]
            color = "white" if abs(val) > max_abs * 0.5 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color=color)

    n_targets = sum(1 for t in type_markers if t == "target")
    if 0 < n_targets < len(type_markers):
        ax.axhline(y=n_targets - 0.5, color="white", linewidth=2)

    ax.set_title("Probability Shifts from Baseline", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Shift (pp)", shrink=0.6)

    path = out_dir / "delta_shift_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reasoning & probability visualizations from experiment CSV.",
    )
    parser.add_argument("csv", type=Path, nargs="?", default=None,
                        help="Path to *_scores.csv (default: latest in results/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for PNGs (default: results/figures/)")

    args = parser.parse_args()

    if args.csv is None:
        from indicator_gaming.config import REPO_ROOT
        csvs = sorted((REPO_ROOT / "results").glob("*_scores.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            print("No score files found.", file=sys.stderr)
            sys.exit(1)
        args.csv = csvs[-1]
        print(f"Using latest: {args.csv}")

    if not args.csv.exists():
        # Try looking in results/ for a bare filename
        from indicator_gaming.config import REPO_ROOT
        candidate = REPO_ROOT / "results" / args.csv.name
        if candidate.exists():
            args.csv = candidate
        else:
            print(f"ERROR: {args.csv} not found", file=sys.stderr)
            sys.exit(1)

    rows = load_csv(args.csv)
    if not rows:
        print("No data in CSV.", file=sys.stderr)
        sys.exit(1)

    meta = load_meta(args.csv)
    model = meta.get("model", "unknown")
    print(f"Model: {model}  |  Rows: {len(rows)}")

    prefix = detect_text_prefix(rows)
    print(f"Text field: {prefix}_*")

    out_dir = args.out_dir or args.csv.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    generated: list[Path] = []

    print("1/6  Word count distributions …")
    generated.append(plot_word_count_distributions(rows, prefix, out_dir))

    print("2/6  Keyword/theme heatmap …")
    generated.append(plot_keyword_heatmap(rows, prefix, out_dir))

    print("3/6  Shift vs reasoning length …")
    generated.append(plot_shift_vs_length(rows, prefix, out_dir))

    print("4/6  Word clouds …")
    wc_path = plot_word_clouds(rows, prefix, out_dir)
    if wc_path:
        generated.append(wc_path)

    print("5/6  Probability heatmap …")
    generated.append(plot_probability_heatmap(rows, out_dir))

    print("6/6  Delta shift heatmap …")
    generated.append(plot_delta_heatmap(rows, out_dir))

    print(f"\nAll visualizations saved to: {out_dir}/")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
