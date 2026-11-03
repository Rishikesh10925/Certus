#!/usr/bin/env python
"""Figure F8 from results/e3/e3_efficiency.json: net saving against the
cost of a missed defect, and the break-even cost ratio per score and
epsilon. Writes paper/figures/F8_economics.png.

Usage:
    python scripts/make_f8.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from certus.viz.figures import figure_f8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/e3/e3_efficiency.json"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/F8_economics.png"))
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    figure_f8(report["rows"], args.out, report["prevalence"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
