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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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
    prevalence: float,
    ltt_score: str | None = None,
    labelled_eps: Sequence[float] = (0.1, 0.05, 0.01),
) -> None:
    """F4 -- the efficiency curve: human-review fraction against the
    guaranteed miss rate, log x, one panel per guarantee. Solid lines are
    CRC means over seeds, with a 95% bootstrap CI of the mean (dark band)
    and the 5-95% seed range (light band: what one deployment sees).
    `ltt_score` adds that score under LTT, dashed. Joint points with
    epsilon >= prevalence are dropped: clearing everything already
    satisfies them. Stricter guarantees sit to the left."""
    guarantees = (
        ("conditional", "Share of real defects missed: P(cleared | defect) ≤ ε"),
        ("joint", "Missed defects per image: E[defect ∧ cleared] ≤ ε"),
    )
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    def pts(score: str, guarantee: str, method: str) -> list[dict[str, Any]]:
        sel = [
            r
            for r in rows
            if r["score"] == score and r["guarantee"] == guarantee and r["method"] == method
        ]
        if guarantee == "joint":
            sel = [r for r in sel if r["epsilon"] < prevalence]
        return sorted(sel, key=lambda r: r["epsilon"])

    for ax, (guarantee, title) in zip(axes, guarantees, strict=True):
        _style(ax)
        for k, score in enumerate(scores):
            sel = pts(score, guarantee, "crc")
            x = [r["epsilon"] for r in sel]
            ax.fill_between(
                x,
                [r["hrf_p05"] for r in sel],
                [r["hrf_p95"] for r in sel],
                color=SERIES[k],
                alpha=0.10,
                linewidth=0,
            )
            ax.fill_between(
                x,
                [r["hrf_ci_lo"] for r in sel],
                [r["hrf_ci_hi"] for r in sel],
                color=SERIES[k],
                alpha=0.35,
                linewidth=0,
            )
            ax.plot(
                x,
                [r["hrf_mean"] for r in sel],
                color=SERIES[k],
                marker=MARKERS[k],
                markersize=6,
                linewidth=2,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=f"{SCORE_LABELS.get(score, score)} (CRC)",
            )
            if k == 0:
                for r in sel:
                    if r["epsilon"] in labelled_eps:
                        ax.annotate(
                            f"{r['hrf_mean'] * 100:.0f}%",
                            (r["epsilon"], r["hrf_mean"]),
                            xytext=(6, -12),
                            textcoords="offset points",
                            fontsize=7,
                            color=INK,
                        )
        if ltt_score is not None:
            k = list(scores).index(ltt_score)
            sel = pts(ltt_score, guarantee, "ltt")
            ax.plot(
                [r["epsilon"] for r in sel],
                [r["hrf_mean"] for r in sel],
                color=SERIES[k],
                linestyle="--",
                linewidth=1.5,
                label=f"{SCORE_LABELS.get(ltt_score, ltt_score)} (LTT, δ = 0.05)",
            )
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.0042, 0.6)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("guaranteed miss rate ε (log scale)", fontsize=8, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK)
    axes[1].axvspan(prevalence, 0.6, color=GRID, alpha=0.6, linewidth=0)
    axes[1].text(
        prevalence * 1.08,
        0.5,
        f"ε ≥ prevalence ({prevalence:.0%}):\nclearing everything\nalready meets it",
        fontsize=7,
        color=INK_MUTED,
    )
    axes[0].set_ylabel("human-review fraction", fontsize=8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, labelcolor=INK
    )
    fig.suptitle(
        "F4 — The cost of a guaranteed miss rate (SDNET2018, 100 seeds;"
        " dark band 95% CI of the mean, light band 5–95% of seeds)",
        fontsize=9.5,
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


LOSO_LABELS = {
    "in_site": "In-site reference (exchangeable)",
    "vanilla": "Vanilla CRC",
    "mondrian_k8": "Mondrian, 8 clusters",
    "weighted": "Weighted (domain classifier)",
    "localized_0.05": "Localized, k = 5%",
    "localized_0.1": "Localized, k = 10%",
    "localized_0.25": "Localized, k = 25%",
}
SITE_LABELS = {"deck": "deck", "pavement": "pavement", "wall": "wall", "cbri": "CBRI"}


def figure_f5(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    methods: Sequence[str],
    scores: Sequence[str] = ("s1", "s2"),
    levels: Sequence[float] = (0.10, 0.05),
    sites: Sequence[str] = ("deck", "pavement", "wall", "cbri"),
) -> None:
    """F5 -- the shift failure and its correction: realised miss rate on
    the held-out site's defects as a multiple of epsilon, per site and
    method, one panel per (score, epsilon). Dots are means over seeds,
    whiskers the 5-95% seed range; the dashed line at 1 is the guarantee.
    The in-site reference is grey and hollow, so the shift is the distance
    from it. Methods keep their colour and marker across panels."""
    cell = {(r["score"], r["level"], r["site"], r["method"]): r for r in rows}
    fig, axes = plt.subplots(
        len(scores), len(levels), figsize=(10.0, 3.3 * len(scores)), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / len(methods)
    top = 1.05 * max(
        cell[(score, level, site, method)]["miss_ratio_p95"]
        for score in scores
        for level in levels
        for site in sites
        for method in methods
    )
    for i, score in enumerate(scores):
        for j, level in enumerate(levels):
            ax = axes[i][j]
            _style(ax)
            ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
            k_series = 0
            for m_idx, method in enumerate(methods):
                ref = method == "in_site"
                colour = INK_MUTED if ref else SERIES[k_series]
                marker = "o" if ref else MARKERS[k_series]
                if not ref:
                    k_series += 1
                xs = [s_idx - 0.4 + width * (m_idx + 0.5) for s_idx in range(len(sites))]
                sel = [cell[(score, level, site, method)] for site in sites]
                mean = [r["miss_rate_ratio"] for r in sel]
                # A heavy lower tail can put the mean under its 5th percentile.
                lo = [max(0.0, r["miss_rate_ratio"] - r["miss_ratio_p05"]) for r in sel]
                hi = [max(0.0, r["miss_ratio_p95"] - r["miss_rate_ratio"]) for r in sel]
                ax.errorbar(
                    xs,
                    mean,
                    yerr=[lo, hi],
                    fmt=marker,
                    color=colour,
                    markersize=6,
                    markerfacecolor=SURFACE if ref else colour,
                    markeredgecolor=colour if ref else SURFACE,
                    markeredgewidth=1.2 if ref else 0.8,
                    elinewidth=1.2,
                    capsize=0,
                    label=LOSO_LABELS.get(method, method),
                )
            ax.set_xticks(range(len(sites)))
            ax.set_xticklabels([SITE_LABELS.get(s, s) for s in sites], fontsize=8)
            ax.tick_params(axis="x", grid_alpha=0)  # no lines through the middle dots
            ax.set_ylim(0, top)
            ax.set_title(
                f"{SCORE_LABELS.get(score, score)}, ε = {level:g}", fontsize=8.5, color=INK
            )
            if j == 0:
                ax.set_ylabel("miss rate on defects / ε", fontsize=8, color=INK)
            if i == len(scores) - 1:
                ax.set_xlabel("held-out site", fontsize=8, color=INK)
    axes[0][0].text(-0.45, 1.04, "guarantee (1×)", fontsize=7, color=INK_MUTED, va="bottom")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, labelcolor=INK
    )
    fig.suptitle(
        "F5 — Site shift breaks the miss-rate guarantee, and what each correction recovers\n"
        "leave-one-site-out, conditional CRC, 100 seeds; dots are means, whiskers 5–95% of seeds",
        fontsize=9.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# F6 families. Localized is drawn through its bandwidth knob, weakest to
# strongest; Mondrian is not monotone in k (Week 11), so only k = 8.
F6_FAMILIES = {
    "vanilla": ("vanilla",),
    "mondrian": ("mondrian_k8",),
    "weighted": ("weighted",),
    "localized": ("localized_0.25", "localized_0.1", "localized_0.05"),
}
F6_LABELS = {
    "vanilla": "Vanilla CRC",
    "mondrian": "Mondrian, 8 clusters",
    "weighted": "Weighted (domain classifier)",
    "localized": "Localized, k = 25% → 10% → 5%",
}


def figure_f6(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    scores: Sequence[str] = ("s1", "s2"),
    levels: Sequence[float] = (0.10, 0.05),
    sites: Sequence[str] = ("deck", "pavement", "wall", "cbri"),
) -> None:
    """F6 -- what recovery costs: realised miss ratio (y, guarantee at 1)
    against the human-review fraction it takes (x), for every held-out
    site and every correction, one panel per (score, epsilon). Colour is
    the method family, marker shape the held-out site; localized is a line
    through k = 25% and 10% (small dots) to 5% (marker). The grey hollow
    marker is the in-site reference: a method that lands left of it and
    above 1 bought its low HRF by breaking the guarantee."""
    cell = {(r["score"], r["level"], r["site"], r["method"]): r for r in rows}
    fig, axes = plt.subplots(
        len(scores), len(levels), figsize=(10.0, 3.6 * len(scores)), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    for i, score in enumerate(scores):
        for j, level in enumerate(levels):
            ax = axes[i][j]
            _style(ax)
            ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
            for s_idx, site in enumerate(sites):
                marker = MARKERS[s_idx]
                ref = cell[(score, level, site, "in_site")]
                ax.plot(
                    ref["hrf_mean"],
                    ref["miss_rate_ratio"],
                    marker,
                    color=INK_MUTED,
                    markerfacecolor=SURFACE,
                    markeredgewidth=1.2,
                    markersize=6,
                )
                for f_idx, members in enumerate(F6_FAMILIES.values()):
                    sel = [cell[(score, level, site, m)] for m in members]
                    ax.plot(
                        [r["hrf_mean"] for r in sel],
                        [r["miss_rate_ratio"] for r in sel],
                        "-o",
                        color=SERIES[f_idx],
                        linewidth=1.0,
                        markersize=2.5,
                        alpha=0.7,
                    )
                    # Only the strongest setting gets the site's marker.
                    ax.plot(
                        sel[-1]["hrf_mean"],
                        sel[-1]["miss_rate_ratio"],
                        marker,
                        color=SERIES[f_idx],
                        markeredgecolor=SURFACE,
                        markeredgewidth=0.8,
                        markersize=6,
                    )
                van = cell[(score, level, site, "vanilla")]
                ax.annotate(
                    SITE_LABELS.get(site, site),
                    (van["hrf_mean"], van["miss_rate_ratio"]),
                    xytext=(-5, 4),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7,
                    color=INK_MUTED,
                )
            ax.set_ylim(bottom=0)
            ax.set_title(
                f"{SCORE_LABELS.get(score, score)}, ε = {level:g}", fontsize=8.5, color=INK
            )
            if j == 0:
                ax.set_ylabel("miss rate on defects / ε", fontsize=8, color=INK)
            if i == len(scores) - 1:
                ax.set_xlabel("human-review fraction on the held-out site", fontsize=8, color=INK)
    method_handles = [
        Line2D([], [], color=SERIES[f_idx], marker="o", linewidth=1.0, label=F6_LABELS[f])
        for f_idx, f in enumerate(F6_FAMILIES)
    ]
    method_handles.append(
        Line2D(
            [],
            [],
            color=INK_MUTED,
            marker="o",
            markerfacecolor=SURFACE,
            linestyle="none",
            label="In-site reference (exchangeable)",
        )
    )
    site_handles = [
        Line2D(
            [],
            [],
            color=INK_MUTED,
            marker=MARKERS[s_idx],
            linestyle="none",
            label=f"held out: {SITE_LABELS.get(s, s)}",
        )
        for s_idx, s in enumerate(sites)
    ]
    fig.legend(
        handles=method_handles + site_handles,
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        frameon=False,
        labelcolor=INK,
    )
    fig.suptitle(
        "F6 — Every recovery is paid for in human review\n"
        "leave-one-site-out, conditional CRC, means over 100 seeds; "
        "above the dashed line the guarantee is broken",
        fontsize=9.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# F7 shows split-CP sets, not CRC, so its rows name the set method.
F7_LABELS = {
    "in_site": "In-site reference",
    "vanilla": "Vanilla split CP",
    "mondrian_k8": "Mondrian, 8 clusters",
    "weighted": "Weighted (domain classifier)",
    "localized_0.05": "Localized, k = 5%",
}


def figure_f7(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    methods: Sequence[str],
    scores: Sequence[str] = ("s1", "s2"),
    levels: Sequence[float] = (0.10, 0.05),
    sites: Sequence[str] = ("deck", "pavement", "wall", "cbri"),
) -> None:
    """F7 -- what marginal coverage hides: split-CP coverage on all of a
    held-out site, on its clean images and on its defects, per method
    (rows), one panel per (score, alpha). Cells print the coverage in
    percent; colour is the gap to 1 - alpha (red under, blue over). CBRI
    has 3 clean images, so it shows no clean column."""
    from matplotlib.colors import TwoSlopeNorm

    cell = {(r["score"], r["level"], r["site"], r["method"]): r for r in rows}
    cols = [
        (site, key)
        for site in sites
        for key in ("coverage_mean", "coverage_clean", "coverage_defect")
        if not (site == "cbri" and key == "coverage_clean")
    ]
    short = {"coverage_mean": "all", "coverage_clean": "clean", "coverage_defect": "defect"}
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.1)
    cmap = plt.get_cmap("RdBu")
    fig, axes = plt.subplots(
        len(scores) * len(levels), 1, figsize=(9.0, 2.3 * len(scores) * len(levels) + 1.0)
    )
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(hspace=0.65, top=0.9, bottom=0.04, left=0.2, right=0.86)
    image = None
    for p_idx, (score, level) in enumerate((s, lv) for s in scores for lv in levels):
        ax = axes[p_idx]
        gap = [
            [cell[(score, level, site, m)][key] - (1.0 - level) for site, key in cols]
            for m in methods
        ]
        image = ax.imshow(gap, cmap=cmap, norm=norm, aspect="auto")
        for r_idx, m in enumerate(methods):
            for c_idx, (site, key) in enumerate(cols):
                value = cell[(score, level, site, m)][key]
                dark = abs(norm(value - (1.0 - level)) - 0.5) > 0.3
                ax.text(
                    c_idx,
                    r_idx,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if dark else INK,
                )
        # Thin white rules between sites.
        edges = [c for c in range(1, len(cols)) if cols[c][0] != cols[c - 1][0]]
        for c in edges:
            ax.axvline(c - 0.5, color=SURFACE, linewidth=2.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([short[key] for _, key in cols], fontsize=7, color=INK_MUTED)
        # Site names centred under their column groups.
        for site in sites:
            idx = [c for c, (s, _) in enumerate(cols) if s == site]
            ax.text(
                sum(idx) / len(idx),
                -0.22,
                f"held out: {SITE_LABELS.get(site, site)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                color=INK,
            )
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([F7_LABELS.get(m, m) for m in methods], fontsize=7, color=INK)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(
            f"{SCORE_LABELS.get(score, score)}, α = {level:g} (target {100 * (1 - level):.0f}%)",
            fontsize=8.5,
            color=INK,
            loc="left",
        )
    assert image is not None
    bar = fig.colorbar(image, ax=list(axes), shrink=0.5, pad=0.02, aspect=30)
    bar.set_label("coverage − (1 − α)", fontsize=7.5, color=INK)
    bar.ax.tick_params(labelsize=7, colors=INK_MUTED)
    bar.outline.set_visible(False)
    fig.suptitle(
        "F7 — Marginal coverage holds; coverage on defects does not\n"
        "split CP (LAC) on the held-out site, leave-one-site-out, means over 100 seeds; "
        "cells are coverage in %",
        fontsize=9.5,
        color=INK,
        x=0.5,
        y=0.97,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_f8(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    prevalence: float,
    scores: Sequence[str] = ("s7", "s1", "s5"),
    curve_score: str = "s7",
    curve_eps: Sequence[float] = (0.20, 0.10, 0.05, 0.02, 0.01),
) -> None:
    """F8 -- when automation pays. Left: net saving per image against full
    human review, in units of one review's cost, as the cost of a missed
    defect grows (c_m / c_h, log x), for `curve_score` under the
    conditional guarantee at several epsilon; below zero, reviewing
    everything is cheaper. Right: the break-even ratio at which the saving
    is zero, against epsilon, for each score and both guarantees. Costs are
    at the guarantee's bound and assume humans miss nothing (see
    triage/economics.py)."""
    from certus.triage.economics import break_even_ratio, net_saving

    def crc(score: str, guarantee: str) -> list[dict[str, Any]]:
        sel = [
            r
            for r in rows
            if r["score"] == score and r["method"] == "crc" and r["guarantee"] == guarantee
        ]
        if guarantee == "joint":
            sel = [r for r in sel if r["epsilon"] < prevalence]
        return sorted(sel, key=lambda r: r["epsilon"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        _style(ax)

    ratios = np.logspace(0, 3, 200)
    by_eps = {r["epsilon"]: r for r in crc(curve_score, "conditional")}
    for k, eps in enumerate(curve_eps):
        r = by_eps[eps]
        saving = [net_saving(r["hrf_mean"], eps, prevalence, "conditional", 1.0, c) for c in ratios]
        left.plot(
            ratios,
            saving,
            color=SERIES[k],
            linewidth=1.6,
            label=f"ε = {eps:g} (HRF {r['hrf_mean']:.0%})",
        )
        be = break_even_ratio(r["hrf_mean"], eps, prevalence, "conditional")
        left.plot(be, 0.0, MARKERS[k], color=SERIES[k], markersize=6, markeredgecolor=SURFACE)
    left.axhline(0.0, color=INK, linewidth=1.0)
    left.axhspan(-1.0, 0.0, color=GRID, alpha=0.5, linewidth=0)
    left.text(1.2, -0.08, "reviewing everything is cheaper", fontsize=7, color=INK_MUTED)
    left.set_xscale("log")
    left.set_ylim(-0.6, 0.6)
    left.set_xlabel("cost of a missed defect / cost of one review (c_m / c_h)", fontsize=8)
    left.set_ylabel("net saving per image, in reviews", fontsize=8, color=INK)
    left.set_title(
        f"{SCORE_LABELS.get(curve_score, curve_score)}, conditional guarantee",
        fontsize=8.5,
        color=INK,
    )
    left.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper right")

    for k, score in enumerate(scores):
        for guarantee, style in (("conditional", "-"), ("joint", "--")):
            sel = [r for r in crc(score, guarantee) if r["hrf_mean"] < 1.0]
            right.plot(
                [r["epsilon"] for r in sel],
                [break_even_ratio(r["hrf_mean"], r["epsilon"], prevalence, guarantee) for r in sel],
                style,
                marker=MARKERS[k],
                color=SERIES[k],
                markersize=4.5,
                linewidth=1.4,
                label=f"{SCORE_LABELS.get(score, score)}, {guarantee}",
            )
    right.set_xscale("log")
    eps_ticks = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    right.set_xticks(eps_ticks)
    right.set_xticklabels([f"{e:g}" for e in eps_ticks])
    right.minorticks_off()
    right.invert_xaxis()
    right.set_ylim(bottom=0)
    right.set_xlabel("guaranteed miss rate ε (stricter →)", fontsize=8)
    right.set_ylabel("break-even c_m / c_h", fontsize=8, color=INK)
    right.set_title("Automation pays only below the curve", fontsize=8.5, color=INK)
    right.legend(fontsize=6.5, frameon=False, labelcolor=INK, ncol=1, loc="upper left")
    fig.suptitle(
        "F8 — Off-the-shelf scores pay for themselves only while a missed defect costs"
        " under ~40 reviews\nCRC on SDNET2018, means over 100 seeds; costs at the"
        " guarantee's bound, humans assumed perfect",
        fontsize=9.5,
        color=INK,
    )
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_f1(out_path: Path) -> None:
    """F1 -- the system: the online path a single image takes (top) and the
    offline calibration loop that sets its thresholds (bottom). Only
    AUTO_CLEAR is a conformal decision; FLAG and ESCALATE both reach a
    human, so routing between them never touches the guarantee."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, accent: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4,rounding_size=1.2",
                facecolor=SURFACE,
                edgecolor=accent,
                linewidth=1.4,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 2.0,
            title,
            ha="center",
            va="top",
            fontsize=8,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + h - 5.4,
            body,
            ha="center",
            va="top",
            fontsize=6.6,
            color=INK_MUTED,
            linespacing=1.35,
        )

    def arrow(
        a: tuple[float, float],
        b: tuple[float, float],
        style: str = "-|>",
        colour: str = INK_MUTED,
        ls: str = "-",
        rad: float = 0.0,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle=style,
                mutation_scale=10,
                color=colour,
                linewidth=1.1,
                linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    online, offline, out = SERIES[0], SERIES[2], SERIES[1]
    ax.text(
        0.5,
        45.0,
        "online, per image  (p95 latency reported in F9)",
        fontsize=7.5,
        color=online,
        va="top",
    )
    top_y, top_h = 27.0, 14.0
    box(
        0.5,
        top_y,
        14,
        top_h,
        "Image + site id",
        "uploaded photo\nsite type picks\nthe calibration\ncluster",
        online,
    )
    box(
        18.5,
        top_y,
        15,
        top_h,
        "Frozen CLIP",
        "ViT-B/32 embedding\nno fine-tuning,\nno training labels",
        online,
    )
    box(
        37.5,
        top_y,
        15,
        top_h,
        "Score S1",
        "s(x) = 1 − p(defect)\nzero-shot prompts:\ncrack, vegetation,\ncorrosion, dampness",
        online,
    )
    box(
        56.5,
        top_y,
        15,
        top_h,
        "Shift check",
        "nearest calibration\nimage too far?\n→ never auto-clear",
        online,
    )
    box(
        75.5,
        top_y,
        24,
        top_h,
        "Triage",
        "s(x) > λ̂(ε)  →  AUTO_CLEAR\nset = {defect}  →  FLAG, type pre-filled\n"
        "otherwise  →  ESCALATE\n+ plain-English risk statement",
        out,
    )
    for x0, x1 in ((14.9, 18.1), (33.9, 37.1), (52.9, 56.1), (71.9, 75.1)):
        arrow((x0, top_y + top_h / 2), (x1, top_y + top_h / 2), colour=online)

    ax.text(0.5, 24.0, "offline, weekly batch", fontsize=7.5, color=offline, va="top")
    bot_y, bot_h = 7.0, 13.0
    box(
        0.5,
        bot_y,
        21,
        bot_h,
        "Labelled calibration set",
        "8,413 SDNET2018 images\nhuman labels, never trained on\nsplit by structure type",
        offline,
    )
    box(
        26.5,
        bot_y,
        22,
        bot_h,
        "Conformal risk control",
        "per cluster and ε:\nλ̂ = smallest threshold with\n"
        "E[P(cleared | defect)] ≤ ε\n(+ split CP q̂ for the set)",
        offline,
    )
    box(
        53.5,
        bot_y,
        17,
        bot_h,
        "Calibration artifact",
        "λ̂, q̂, shift threshold\nversioned, loaded\nat startup",
        offline,
    )
    box(
        75.5,
        bot_y,
        24,
        bot_h,
        "Audit + feedback",
        "5% of auto-cleared images\nsampled for a human label;\n"
        "labels never recalibrate live,\nthey feed the weekly batch",
        out,
    )
    arrow((21.9, bot_y + bot_h / 2), (26.1, bot_y + bot_h / 2), colour=offline)
    arrow((48.9, bot_y + bot_h / 2), (53.1, bot_y + bot_h / 2), colour=offline)
    arrow((62.0, bot_y + bot_h + 0.6), (85.0, top_y - 0.6), colour=offline, ls="--")
    ax.text(66.0, 23.0, "thresholds", fontsize=6.6, color=offline, ha="right")
    arrow((87.5, top_y - 0.6), (87.5, bot_y + bot_h + 0.6), colour=out)
    ax.text(88.5, 23.0, "cleared images", fontsize=6.6, color=out)
    arrow((87.5, bot_y - 0.6), (11.0, bot_y - 0.6), colour=out, ls="--", rad=-0.06)
    ax.text(49.0, 0.6, "new labels join the next calibration", fontsize=6.6, color=out, ha="center")
    fig.suptitle(
        "F1 — Certus: a frozen vision-language score, a conformal threshold per site type,"
        " and a guarantee on the share of defects auto-cleared",
        fontsize=9.5,
        color=INK,
    )
    fig.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


DEVICE_LABELS = {"cuda": "laptop GPU (RTX 5050)", "cpu": "CPU only"}


def figure_f9(
    runs: dict[str, dict[str, Any]],
    out_path: Path,
    budget_ms: float = 400.0,
) -> None:
    """F9 -- serving cost. Left: the distribution of end-to-end
    /v1/inspect latency (client wall time, log x) per device, with p50, p95
    and p99 marked against the latency budget. Right: where the median
    request's time goes -- image decode, the CLIP encoder, the rest of the
    server (upload parsing, shift check, conformal decision, SQLite log)
    and HTTP.
    `runs` maps a device to arrays: client_ms, server_decode_ms,
    server_embed_ms, server_total_ms."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        _style(ax)
    devices = list(runs)
    for k, device in enumerate(devices):
        ms = np.sort(np.asarray(runs[device]["client_ms"], dtype=float))
        frac = np.arange(1, ms.size + 1) / ms.size
        colour = SERIES[k]
        qs = [float(np.quantile(ms, q)) for q in (0.5, 0.95, 0.99)]
        label = (
            f"{DEVICE_LABELS.get(device, device)}: p50 / p95 / p99 = "
            + " / ".join(f"{v:.0f}" for v in qs)
            + " ms"
        )
        left.plot(ms, frac, color=colour, linewidth=1.6, label=label)
        for j, (q, v) in enumerate(zip((0.5, 0.95, 0.99), qs, strict=True)):
            left.plot(v, q, MARKERS[j + 1], color=colour, markersize=5, markeredgecolor=SURFACE)
    left.axvline(budget_ms, color=INK, linestyle="--", linewidth=1.0)
    left.text(budget_ms * 1.05, 0.05, f"budget {budget_ms:.0f} ms", fontsize=7, color=INK)
    left.set_xscale("log")
    left.set_ylim(0, 1.02)
    left.set_xlabel("end-to-end latency per image (ms, log scale)", fontsize=8)
    left.set_ylabel("share of requests at or below", fontsize=8, color=INK)
    left.set_title("POST /v1/inspect, one image per request", fontsize=8.5, color=INK)
    left.legend(
        fontsize=6.8, frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.17)
    )

    parts = ("decode", "CLIP encoder", "rest of server", "HTTP")
    bottom = np.zeros(len(devices))
    stacks = []
    for device in devices:
        r = runs[device]
        dec = float(np.median(r["server_decode_ms"]))
        emb = float(np.median(r["server_embed_ms"]))
        tot = float(np.median(r["server_total_ms"]))
        cli = float(np.median(r["client_ms"]))
        stacks.append([dec, emb, max(tot - dec - emb, 0.0), max(cli - tot, 0.0)])
    values = np.array(stacks)
    y = np.arange(len(devices))
    for j, name in enumerate(parts):
        right.barh(
            y,
            values[:, j],
            left=bottom,
            color=SERIES[(j + 2) % len(SERIES)],
            height=0.5,
            label=name,
        )
        bottom += values[:, j]
    for i, total in enumerate(bottom):
        right.text(total * 1.02, i, f"{total:.0f} ms", va="center", fontsize=7, color=INK)
    right.set_yticks(y)
    right.set_yticklabels([DEVICE_LABELS.get(d, d) for d in devices], fontsize=7.5, color=INK)
    right.set_xlim(0, bottom.max() * 1.25)
    right.set_xlabel("median time per request (ms)", fontsize=8)
    right.set_title("Where the time goes (medians)", fontsize=8.5, color=INK)
    right.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=2, loc="lower right")
    right.tick_params(axis="y", grid_alpha=0)
    n = min(np.asarray(r["client_ms"]).size for r in runs.values())
    fig.suptitle(
        "F9 — Within the 400 ms budget on a laptop GPU, not on CPU alone; the image encoder"
        f" is most of it\nsingle sequential client on the same machine, ≥ {n:,} requests"
        " per device; markers at p50, p95, p99",
        fontsize=9.5,
        color=INK,
    )
    fig.subplots_adjust(top=0.82, bottom=0.27, left=0.07, right=0.98, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
