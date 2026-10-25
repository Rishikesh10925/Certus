"""One function per figure, F1 through F9 (roadmap Section 5.6), each
publication-ready and each regenerable from a config hash via `make
figures`. No figure in the paper may be produced by a notebook -- see the
reproducibility checklist in Section 8.4.

Series colours follow one fixed categorical order (validated for
colour-blind separation); every series also gets its own marker shape, so
identity never rests on colour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "D", "^", "v", "P"]

METHOD_LABELS = {
    "lac": "Split CP (LAC)",
    "aps": "Split CP (APS)",
    "raps": "Split CP (RAPS)",
    "b1_softmax": "B1 raw softmax",
    "b2_temperature": "B2 temperature scaling",
    "b3_selective": "B3 selective classification",
}
DATASET_LABELS = {"sdnet2018": "SDNET2018 (human labels)", "cbri": "CBRI (bootstrap labels)"}


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_f2(
    rows: Sequence[dict[str, Any]],
    score: str,
    datasets: Sequence[str],
    n_cals: Sequence[int],
    methods: Sequence[str],
    out_path: Path,
    tolerance: float = 0.02,
) -> None:
    """F2 -- validity: empirical vs nominal coverage, mean +/- SD over
    seeds, one panel per (dataset, n_cal), with the +/-tolerance band
    around the diagonal that the Week 5 gate checks against."""
    fig, axes = plt.subplots(
        len(datasets),
        len(n_cals),
        figsize=(3.1 * len(n_cals), 3.0 * len(datasets)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor(SURFACE)
    nominal = sorted({1 - r["alpha"] for r in rows})
    lo, hi = min(nominal) - 0.06, 1.005
    offsets = {m: (i - (len(methods) - 1) / 2) * 0.004 for i, m in enumerate(methods)}

    for i, dataset in enumerate(datasets):
        for j, n_cal in enumerate(n_cals):
            ax = axes[i][j]
            _style(ax)
            ax.fill_between(
                [lo, hi],
                [lo - tolerance, hi - tolerance],
                [lo + tolerance, hi + tolerance],
                color=GRID,
                alpha=0.7,
                linewidth=0,
                label=f"nominal ± {tolerance * 100:.0f} pts",
            )
            ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1, linestyle="--")
            for k, method in enumerate(methods):
                pts = sorted(
                    (1 - r["alpha"], r["coverage_mean"], r["coverage_sd"])
                    for r in rows
                    if r["dataset"] == dataset
                    and r["n_cal"] == n_cal
                    and r["score"] == score
                    and r["method"] == method
                )
                if not pts:
                    continue
                x = [p[0] + offsets[method] for p in pts]
                ax.errorbar(
                    x,
                    [p[1] for p in pts],
                    yerr=[p[2] for p in pts],
                    color=SERIES[k],
                    marker=MARKERS[k],
                    markersize=5,
                    linewidth=2,
                    elinewidth=1,
                    capsize=0,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.8,
                    label=METHOD_LABELS.get(method, method),
                )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo - 0.06, hi)
            if i == 0:
                ax.set_title(f"n_cal = {n_cal}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS.get(dataset, dataset)}\nempirical coverage",
                    fontsize=8,
                    color=INK,
                )
            if i == len(datasets) - 1:
                ax.set_xlabel("nominal coverage (1 - α)", fontsize=8, color=INK)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, labelcolor=INK
    )
    fig.suptitle(
        f"F2 — Validity on exchangeable data ({score.upper()}, mean ± SD over seeds)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
