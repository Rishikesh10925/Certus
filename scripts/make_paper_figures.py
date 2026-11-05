#!/usr/bin/env python
"""Renders the figures the paper uses, in paper mode (no in-image titles,
tight crop, 300 dpi), from the committed results files only -- no model
is run. Writes paper/figures/ieee/. The full-size versions with titles
come from scripts/make_f*.py and the experiment scripts.

Usage:
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from certus.viz import figures

LOSO_METHODS = ("in_site", "vanilla", "mondrian_k8", "weighted", "localized_0.05")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures/ieee"))
    args = parser.parse_args()
    figures.PAPER_MODE = True
    out, res = args.out_dir, args.results

    figures.figure_f1(out / "F1_system.png")
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
    figures.figure_f6(loso, out / "F6_recovery.png", levels=(0.10,))
    figures.figure_f7(loso, out / "F7_class_coverage.png", LOSO_METHODS, levels=(0.10,))
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
