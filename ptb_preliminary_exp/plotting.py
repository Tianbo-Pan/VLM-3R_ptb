from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt

from .common import ensure_dir


def _save(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_counterfactual_barplot(overall_rates: Dict[str, float], path: Path) -> None:
    labels = list(overall_rates.keys())
    values = [overall_rates[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4e79a7", "#f28e2b", "#e15759"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("same-answer rate")
    ax.set_title("Counterfactual consistency")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, path)


def save_margin_boxplot(targeted: Sequence[float], random_vals: Sequence[float], low_vals: Sequence[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([list(targeted), list(random_vals), list(low_vals)], labels=["targeted", "random", "low-score"])
    ax.set_ylabel("GT margin drop")
    ax.set_title("Targeted vs random evidence ablation")
    _save(fig, path)


def save_selective_gap_histogram(gaps: Sequence[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(list(gaps), bins=20, color="#59a14f", edgecolor="black")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("selective reliance gap (GT margin)")
    ax.set_ylabel("count")
    ax.set_title("Selective reliance gap distribution")
    _save(fig, path)

