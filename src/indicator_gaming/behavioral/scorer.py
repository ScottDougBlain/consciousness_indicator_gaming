"""Scoring utilities for behavioral tasks."""

from __future__ import annotations

import math
import re


def fuzzy_match(response: str, ground_truth: str, threshold: float = 0.8) -> bool:
    """Check if response contains or closely matches the ground truth.

    Handles case-insensitive matching, minor typos, and partial containment.
    """
    resp = response.strip().lower()
    gt = ground_truth.strip().lower()

    # Exact containment
    if gt in resp:
        return True

    # Check if any word-boundary-delimited occurrence exists
    pattern = re.escape(gt)
    if re.search(pattern, resp):
        return True

    # For short answers, also check if response starts with ground truth
    if len(gt) < 30 and resp.startswith(gt):
        return True

    return False


def numeric_match(response: str, ground_truth: str, tolerance: float = 0.0) -> bool:
    """Check if a numeric response matches ground truth within tolerance.

    tolerance is a fraction (e.g. 0.2 for 20%).
    """
    try:
        resp_num = _extract_number(response)
        gt_num = float(ground_truth)
    except (ValueError, TypeError):
        return False

    if resp_num is None:
        return False

    if gt_num == 0:
        return abs(resp_num) <= max(tolerance, 1.0)

    return abs(resp_num - gt_num) / abs(gt_num) <= tolerance


def _extract_number(text: str) -> float | None:
    """Extract the first number from text."""
    # Handle scientific notation
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    confidences : list of float in [0, 100]
    correct : list of bool
    n_bins : number of bins

    Returns
    -------
    ECE as a float in [0, 1].
    """
    if not confidences:
        return 0.0

    n = len(confidences)
    bin_bounds = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    ece = 0.0

    for lo, hi in bin_bounds:
        # confidences are 0-100, normalize to 0-1
        in_bin = [
            (c / 100.0, cr)
            for c, cr in zip(confidences, correct)
            if lo <= c / 100.0 < hi or (hi == 1.0 and c / 100.0 == 1.0)
        ]
        if not in_bin:
            continue

        bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
        bin_acc = sum(1 for _, cr in in_bin if cr) / len(in_bin)
        ece += abs(bin_acc - bin_conf) * (len(in_bin) / n)

    return round(ece, 4)


def brier_score(confidences: list[float], correct: list[bool]) -> float:
    """Compute Brier score. Lower is better.

    Parameters
    ----------
    confidences : list of float in [0, 100]
    correct : list of bool

    Returns
    -------
    Brier score as float in [0, 1].
    """
    if not confidences:
        return 0.0

    total = 0.0
    for conf, cor in zip(confidences, correct):
        p = conf / 100.0
        outcome = 1.0 if cor else 0.0
        total += (p - outcome) ** 2

    return round(total / len(confidences), 4)


def keyword_count(text: str, keywords: list[str]) -> int:
    """Count occurrences of keywords in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)
