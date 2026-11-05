#!/usr/bin/env python
"""Regenerates F1-F9 from the committed results/ files alone: no model
is run and no raw image is read, so this works on a fresh clone with only
the Python dependencies installed (`make figures`). Writes the full-size
figures, with titles, to figures/ (tracked in git, so the README can show
them); `--paper` writes the paper set instead (no titles, tight crop).

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from certus.eval.protocol import METHODS
from certus.viz import figures

LOSO_METHODS = ("in_site", "vanilla", "mondrian_k8", "weighted", "localized_0.05")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--paper", action="store_true", help="paper mode: no titles, tight crop")
    args = parser.parse_args()
    res = args.results
    out = args.out_dir or Path("paper/figures/ieee" if args.paper else "figures")
    figures.PAPER_MODE = args.paper

    figures.figure_f1(out / "F1_system.png")
    e1 = json.loads((res / "e1/e1_validity.json").read_text())
    for score in ("s1", "s2"):
        figures.figure_f2(
            e1["summary"],
            score,
            ["sdnet2018", "cbri"],
            [50, 200, 1000],
            list(METHODS),
            out / f"F2_validity_{score}.png",
            0.02,
        )
    e2 = json.loads((res / "e2/e2_scores.json").read_text())
    figures.figure_f3(
        e2["rows"],
        [("joint", 0.05), ("conditional", 0.10)],
        out / "F3_scores.png",
        e2["auc_on_pool"],
    )
    e3 = json.loads((res / "e3/e3_efficiency.json").read_text())
    figures.figure_f4(
        e3["rows"], ("s7", "s1", "s5"), out / "F4_efficiency.png", e3["prevalence"], ltt_score="s7"
    )
    figures.figure_f8(e3["rows"], out / "F8_economics.png", e3["prevalence"])
    loso = json.loads((res / "e4/loso_v3.json").read_text())["rows"]
    figures.figure_f5(loso, out / "F5_loso.png", LOSO_METHODS)
    # The paper prints one epsilon for F6 and F7 to fit the page.
    levels = (0.10,) if args.paper else (0.10, 0.05)
    figures.figure_f6(loso, out / "F6_recovery.png", levels=levels)
    figures.figure_f7(loso, out / "F7_class_coverage.png", LOSO_METHODS, levels=levels)
    runs: dict[str, dict[str, Any]] = {}
    for device in ("cuda", "cpu"):
        with np.load(res / f"service/latency_{device}_requests.npz") as z:
            runs[device] = {k: z[k] for k in z.files}
    figures.figure_f9(runs, out / "F9_latency.png")
    for f in sorted(out.glob("*.png")):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
