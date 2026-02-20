#!/usr/bin/env python3
"""Generate word clouds showing justification language shifts across conditions.

Produces:
1. Per-model panels: baseline vs inflate vs suppress word clouds (target indicators only)
2. Aggregate differential clouds: words that appear MORE in inflate/suppress vs baseline
3. Summary grid: all models side-by-side

Usage:
    python scripts/visualize_justification_wordclouds.py
    python scripts/visualize_justification_wordclouds.py --models opus-4.6,haiku-4.5
    python scripts/visualize_justification_wordclouds.py --include-placebo
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from wordcloud import WordCloud
except ImportError:
    print(
        "ERROR: matplotlib, numpy, and wordcloud are required.\n"
        "  pip install matplotlib numpy wordcloud",
        file=sys.stderr,
    )
    sys.exit(1)

from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"

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

# Stop words: common English + experiment-specific boilerplate
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
}

CONDITION_COLORS = {
    "baseline": "#4C72B0",
    "inflate": "#C44E52",
    "suppress": "#8172B2",
}


def _model_short(model_id: str) -> str:
    if model_id in _MODEL_ID_SHORT:
        return _MODEL_ID_SHORT[model_id]
    return model_id.rsplit("/", 1)[-1].replace(":free", "").replace("-preview", "")


def discover_runs(
    results_dir: Path, models_filter: list[str] | None = None,
) -> dict[str, Path]:
    """Find latest non-variant, non-behavioral CSV per model."""
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

        model = _model_short(meta.get("model", ""))
        if models_filter and model not in models_filter:
            continue

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


def load_justifications(
    csv_path: Path, *, include_placebo: bool = False,
) -> dict[str, list[str]]:
    """Load justification text grouped by condition.

    Returns {"baseline": [...], "inflate": [...], "suppress": [...]}.
    """
    texts: dict[str, list[str]] = {
        "baseline": [], "inflate": [], "suppress": [],
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not include_placebo and row.get("indicator_type") != "target":
                continue
            for cond in ("baseline", "inflate", "suppress"):
                j = row.get(f"justification_{cond}", "").strip()
                if j:
                    texts[cond].append(j)
    return texts


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, alpha only, no stop words."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def word_freq(texts: list[str]) -> Counter:
    """Count word frequencies across a list of texts."""
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


def differential_freq(
    target: Counter, baseline: Counter, min_count: int = 2,
) -> dict[str, float]:
    """Compute words that appear proportionally MORE in target vs baseline.

    Returns {word: log-ratio} for words enriched in target.
    """
    all_words = set(target) | set(baseline)
    total_t = max(sum(target.values()), 1)
    total_b = max(sum(baseline.values()), 1)

    enriched = {}
    for w in all_words:
        ct = target.get(w, 0)
        cb = baseline.get(w, 0)
        if ct < min_count:
            continue
        # Relative frequency ratio with smoothing
        freq_t = (ct + 0.5) / (total_t + 1)
        freq_b = (cb + 0.5) / (total_b + 1)
        ratio = freq_t / freq_b
        if ratio > 1.2:  # At least 20% enriched
            enriched[w] = ct * (ratio - 1)  # Weight by count × enrichment
    return enriched


def make_cloud(
    freqs: dict[str, float | int], color: str, max_words: int = 80,
) -> WordCloud | None:
    if not freqs:
        return None

    def color_func(*args, **kwargs):
        return color

    wc = WordCloud(
        width=600, height=400,
        background_color="white",
        max_words=max_words,
        color_func=color_func,
        prefer_horizontal=0.7,
        relative_scaling=0.5,
        min_font_size=8,
    )
    wc.generate_from_frequencies(freqs)
    return wc


def plot_model_triptych(
    model: str, texts: dict[str, list[str]], out_dir: Path,
) -> Path:
    """Three word clouds side by side: baseline, inflate, suppress."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model} — Justification Language by Condition", fontsize=14, y=0.98)

    for ax, cond in zip(axes, ["baseline", "inflate", "suppress"]):
        freqs = word_freq(texts[cond])
        wc = make_cloud(dict(freqs.most_common(80)), CONDITION_COLORS[cond])
        if wc:
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cond.capitalize(), fontsize=13, color=CONDITION_COLORS[cond],
                     fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = out_dir / f"wordcloud_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_differential_clouds(
    model: str, texts: dict[str, list[str]], out_dir: Path,
) -> Path:
    """Two differential clouds: inflate-enriched and suppress-enriched vs baseline."""
    bl_freq = word_freq(texts["baseline"])
    inf_freq = word_freq(texts["inflate"])
    sup_freq = word_freq(texts["suppress"])

    inf_diff = differential_freq(inf_freq, bl_freq)
    sup_diff = differential_freq(sup_freq, bl_freq)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model} — Words Enriched vs Baseline (Target Indicators)",
        fontsize=14, y=0.98,
    )

    for ax, diff, label, color in [
        (axes[0], inf_diff, "Inflate-Enriched", CONDITION_COLORS["inflate"]),
        (axes[1], sup_diff, "Suppress-Enriched", CONDITION_COLORS["suppress"]),
    ]:
        wc = make_cloud(diff, color, max_words=60)
        if wc:
            ax.imshow(wc, interpolation="bilinear")
        else:
            ax.text(0.5, 0.5, "(no enriched words)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(label, fontsize=13, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = out_dir / f"wordcloud_diff_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aggregate_differential(
    all_texts: dict[str, dict[str, list[str]]], out_dir: Path,
) -> Path:
    """Aggregate differential clouds across all models."""
    agg: dict[str, list[str]] = {"baseline": [], "inflate": [], "suppress": []}
    for texts in all_texts.values():
        for cond in agg:
            agg[cond].extend(texts[cond])

    bl_freq = word_freq(agg["baseline"])
    inf_freq = word_freq(agg["inflate"])
    sup_freq = word_freq(agg["suppress"])

    inf_diff = differential_freq(inf_freq, bl_freq)
    sup_diff = differential_freq(sup_freq, bl_freq)

    # Also compute baseline-enriched vs each condition
    bl_vs_inf = differential_freq(bl_freq, inf_freq)
    bl_vs_sup = differential_freq(bl_freq, sup_freq)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Aggregate Justification Language Shifts (All Models, Target Indicators)",
        fontsize=15, y=0.98,
    )

    panels = [
        (axes[0, 0], inf_diff, "Inflate-Enriched vs Baseline",
         CONDITION_COLORS["inflate"]),
        (axes[0, 1], sup_diff, "Suppress-Enriched vs Baseline",
         CONDITION_COLORS["suppress"]),
        (axes[1, 0], bl_vs_inf, "Baseline-Enriched vs Inflate",
         CONDITION_COLORS["baseline"]),
        (axes[1, 1], bl_vs_sup, "Baseline-Enriched vs Suppress",
         CONDITION_COLORS["baseline"]),
    ]

    for ax, diff, label, color in panels:
        wc = make_cloud(diff, color, max_words=60)
        if wc:
            ax.imshow(wc, interpolation="bilinear")
        else:
            ax.text(0.5, 0.5, "(no enriched words)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(label, fontsize=12, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "wordcloud_aggregate_differential.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary_grid(
    all_texts: dict[str, dict[str, list[str]]], out_dir: Path,
) -> Path:
    """Grid of differential clouds: models × {inflate, suppress}."""
    models = sorted(all_texts.keys())
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.2 * n))
    fig.suptitle(
        "Justification Shifts: Inflate vs Suppress (Enriched vs Baseline)",
        fontsize=15, y=1.0 - 0.01,
    )

    if n == 1:
        axes = axes.reshape(1, 2)

    for i, model in enumerate(models):
        bl_freq = word_freq(all_texts[model]["baseline"])
        inf_freq = word_freq(all_texts[model]["inflate"])
        sup_freq = word_freq(all_texts[model]["suppress"])

        inf_diff = differential_freq(inf_freq, bl_freq)
        sup_diff = differential_freq(sup_freq, bl_freq)

        for j, (diff, label, color) in enumerate([
            (inf_diff, "Inflate", CONDITION_COLORS["inflate"]),
            (sup_diff, "Suppress", CONDITION_COLORS["suppress"]),
        ]):
            ax = axes[i, j]
            wc = make_cloud(diff, color, max_words=40)
            if wc:
                ax.imshow(wc, interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "(minimal shift)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"{label}-Enriched", fontsize=13, color=color,
                             fontweight="bold")

        # Model label on left
        axes[i, 0].text(
            -0.02, 0.5, model, transform=axes[i, 0].transAxes,
            fontsize=11, fontweight="bold", ha="right", va="center",
            rotation=0,
        )

    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    path = out_dir / "wordcloud_summary_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate justification word clouds across conditions.",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model short names (default: all available)",
    )
    parser.add_argument(
        "--include-placebo", action="store_true",
        help="Include placebo indicators (default: target only)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: results/figures_wordclouds/)",
    )
    args = parser.parse_args()

    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(",")]

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "figures_wordclouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(RESULTS_DIR, models_filter)
    if not runs:
        print("No self-report runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} model runs")

    all_texts: dict[str, dict[str, list[str]]] = {}

    for model in sorted(runs):
        csv_path = runs[model]
        texts = load_justifications(csv_path, include_placebo=args.include_placebo)
        n_bl = len(texts["baseline"])
        n_inf = len(texts["inflate"])
        n_sup = len(texts["suppress"])
        print(f"  {model:20s}  {n_bl:3d} BL / {n_inf:3d} INF / {n_sup:3d} SUP justifications")
        all_texts[model] = texts

        # Per-model triptych
        p1 = plot_model_triptych(model, texts, out_dir)
        print(f"    -> {p1.name}")

        # Per-model differential
        p2 = plot_differential_clouds(model, texts, out_dir)
        print(f"    -> {p2.name}")

    # Aggregate differential
    p3 = plot_aggregate_differential(all_texts, out_dir)
    print(f"\n  Aggregate -> {p3.name}")

    # Summary grid
    p4 = plot_summary_grid(all_texts, out_dir)
    print(f"  Grid      -> {p4.name}")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
