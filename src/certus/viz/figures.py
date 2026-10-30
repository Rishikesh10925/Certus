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


SCORE_LABELS = {
    "s1": "S1 CLIP zero-shot",
    "s2": "S2 Qwen2-VL log-prob",
    "s3": "S3 verbal confidence",
    "s4": "S4 OWLv2 objectness",
    "s5": "S5 MC-dropout bound",
    "s6": "S6 feature typicality",
    "s7": "S7 learned combination",
}


def figure_f4(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[str],
    out_path: Path,
    method: str = "crc",
) -> None:
    """F4 -- the efficiency curve (draft): human-review fraction against
    the guaranteed miss rate, log x, one panel per guarantee, mean with a
    +/- SD band over seeds. Stricter guarantees sit to the left."""
    guarantees = (
        ("conditional", "P(cleared | defect) ≤ ε"),
        ("joint", "E[defect ∧ cleared] ≤ ε"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (guarantee, title) in zip(axes, guarantees, strict=True):
        _style(ax)
        for k, score in enumerate(scores):
            pts = sorted(
                (r["epsilon"], r["hrf_mean"], r["hrf_sd"])
                for r in rows
                if r["score"] == score and r["guarantee"] == guarantee and r["method"] == method
            )
            x = [p[0] for p in pts]
            mean = [p[1] for p in pts]
            ax.fill_between(
                x,
                [p[1] - p[2] for p in pts],
                [p[1] + p[2] for p in pts],
                color=SERIES[k],
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=SERIES[k],
                marker=MARKERS[k],
                markersize=6,
                linewidth=2,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=SCORE_LABELS.get(score, score),
            )
        ax.set_xscale("log")
        ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("guaranteed miss rate ε (log scale)", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("human-review fraction", fontsize=8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, labelcolor=INK
    )
    fig.suptitle(
        f"F4 (draft) — Cost of a guaranteed miss rate ({method.upper()}, mean ± SD over seeds)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_f3(
    rows: Sequence[dict[str, Any]],
    panels: Sequence[tuple[str, float]],
    out_path: Path,
    auc: dict[str, float] | None = None,
) -> None:
    """F3 -- score comparison: human-review fraction (mean +/- SD over
    seeds) needed to hold missed defects below epsilon, one row per score,
    sorted best first within each panel. One panel per (guarantee, epsilon).
    Identity is the row label, so one colour is enough."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (guarantee, eps) in zip(axes[0], panels, strict=True):
        _style(ax)
        sel = sorted(
            (r for r in rows if r["guarantee"] == guarantee and r["epsilon"] == eps),
            key=lambda r: -r["hrf_mean"],
        )
        y = range(len(sel))
        ax.errorbar(
            [r["hrf_mean"] for r in sel],
            list(y),
            xerr=[r["hrf_sd"] for r in sel],
            fmt="o",
            color=SERIES[0],
            markersize=6,
            elinewidth=1.2,
            capsize=0,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
        )
        for yi, r in zip(y, sel, strict=True):
            ax.annotate(
                f"{r['hrf_mean'] * 100:.0f}%",
                (r["hrf_mean"] + r["hrf_sd"], yi),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=INK_MUTED,
            )
        labels = [
            SCORE_LABELS.get(r["score"], r["score"])
            + (f"  (AUC {auc[r['score']]:.2f})" if auc else "")
            for r in sel
        ]
        ax.set_yticks(list(y), labels, fontsize=8, color=INK)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("human-review fraction (lower is better)", fontsize=8, color=INK)
        what = "E[defect ∧ cleared]" if guarantee == "joint" else "P(cleared | defect)"
        ax.set_title(f"{what} ≤ {eps:g}", fontsize=9, color=INK)
    fig.suptitle(
        "F3 — Score comparison at fixed risk (CRC, 100 seeds, mean ± SD)", fontsize=10, color=INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
