"""Set size, human-review fraction, and area under the risk-coverage curve
(AURC) -- the efficiency metrics that make a conformal guarantee usable
rather than merely valid. AURC is used specifically to compare against the
selective-classification baseline (B3). See roadmap Section 5.3.

Week 3 scope: mean set size only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mean_set_size(sets: ArrayLike) -> float:
    """Average number of classes per prediction set."""
    s = np.asarray(sets, dtype=bool)
    if s.ndim != 2:
        raise ValueError(f"sets must be 2-D (n, n_classes), got shape {s.shape}")
    return float(s.sum(axis=1).mean())
