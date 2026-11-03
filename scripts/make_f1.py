#!/usr/bin/env python
"""Figure F1, the system diagram (no data). Writes
paper/figures/F1_system.png.

Usage:
    python scripts/make_f1.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from certus.viz.figures import figure_f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("paper/figures/F1_system.png"))
    args = parser.parse_args()
    figure_f1(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
