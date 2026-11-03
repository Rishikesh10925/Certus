#!/usr/bin/env python
"""Figure F5 from results/e4/loso_v3.json: the per-site miss ratio under
leave-one-site-out for the in-site reference, vanilla CRC and the three
corrections (Mondrian at 8 clusters, weighted, localized at k = 5%).
Writes paper/figures/F5_loso.png.

Usage:
    python scripts/make_f5.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from certus.viz.figures import figure_f5

METHODS = ("in_site", "vanilla", "mondrian_k8", "weighted", "localized_0.05")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/e4/loso_v3.json"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/F5_loso.png"))
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())["rows"]
    figure_f5(rows, args.out, METHODS)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
