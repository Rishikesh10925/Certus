#!/usr/bin/env python
"""Figure F7 from results/e4/loso_v3.json: split-CP coverage on each
held-out site overall, on its clean images and on its defects, for the
in-site reference, vanilla and the three corrections.
Writes paper/figures/F7_class_coverage.png.

Usage:
    python scripts/make_f7.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from certus.viz.figures import figure_f7

METHODS = ("in_site", "vanilla", "mondrian_k8", "weighted", "localized_0.05")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/e4/loso_v3.json"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/F7_class_coverage.png"))
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())["rows"]
    figure_f7(rows, args.out, METHODS)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
