#!/usr/bin/env python
"""Figure F6 from results/e4/loso_v3.json: recovered miss ratio against
the human-review fraction it costs, for every held-out site and every
correction (Mondrian and localized across their knob settings).
Writes paper/figures/F6_recovery.png.

Usage:
    python scripts/make_f6.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from certus.viz.figures import figure_f6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/e4/loso_v3.json"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/F6_recovery.png"))
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())["rows"]
    figure_f6(rows, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
