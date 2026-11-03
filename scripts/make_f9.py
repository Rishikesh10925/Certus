#!/usr/bin/env python
"""Figure F9 from results/service/latency_<device>_requests.npz (written
by scripts/measure_latency.py): end-to-end latency distribution and the
median time per stage, per device. Writes paper/figures/F9_latency.png.

Usage:
    python scripts/make_f9.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from certus.viz.figures import figure_f9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/service"))
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--out", type=Path, default=Path("paper/figures/F9_latency.png"))
    args = parser.parse_args()
    runs: dict[str, dict[str, Any]] = {}
    for device in args.devices:
        path = args.results_dir / f"latency_{device}_requests.npz"
        if path.exists():
            with np.load(path) as z:
                runs[device] = {k: z[k] for k in z.files}
    if not runs:
        raise SystemExit("no latency results; run scripts/measure_latency.py first")
    figure_f9(runs, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
