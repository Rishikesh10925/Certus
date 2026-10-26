"""Label-noise tools for the Week 7 study (blueprint O6): agreement between
the GPT-4V-bootstrapped labels and the hand-labelled gold set, and a
controlled way to reproduce the bootstrap's failure mode on a dataset with
clean labels.

The failure mode measured on CBRI (Weeks 3-4) is one-sided: the bootstrap
almost never invents a defect, it *misses* them -- 73 of 172 gold defects
carry no bootstrap annotation. `hide_defects` reproduces exactly that, by
relabelling a fraction of true defects as clean, either at random or
(worst case) the most clean-looking ones.

What that does to the guarantees (the point of the experiment):
- joint E[1{defect and cleared}]: hidden defects look free to clear, so
  the realised risk is inflated by ~1/(1 - rate) whichever defects hide.
- conditional P(cleared | defect), calibrated on labelled defects only:
  if defects hide at random, the labelled ones are still an exchangeable
  sample of all defects and the guarantee survives (with smaller n). If
  the subtle ones hide, it breaks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cohen_kappa(a: ArrayLike, b: ArrayLike) -> float:
    """Cohen's kappa for two binary raters. NaN when chance agreement is 1
    (both raters constant and equal), where kappa is undefined."""
    x = np.asarray(a).astype(bool)
    y = np.asarray(b).astype(bool)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("raters must label the same, non-empty set of items")
    p_obs = float((x == y).mean())
    p_chance = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    if p_chance == 1.0:
        return float("nan")
    return (p_obs - p_chance) / (1.0 - p_chance)


def hide_defects(
    labels: ArrayLike,
    rate: float,
    rng: np.random.Generator,
    scores: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Copy of `labels` with round(rate * n_defects) defects flipped to
    clean. Clean labels are never touched.

    Without `scores`: the hidden defects are chosen at random -- what CBRI
    shows (Week 7: bootstrap-missed defects score no differently from
    annotated ones). With `scores`: the most clean-looking defects (highest
    score) are hidden -- the worst case, where the labeller misses exactly
    the defects the model also finds hard."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    y = np.asarray(labels).astype(bool).copy()
    pos = np.flatnonzero(y)
    k = int(round(rate * pos.size))
    if scores is None:
        hidden = rng.choice(pos, size=k, replace=False)
    else:
        s = np.asarray(scores, dtype=np.float64)
        hidden = pos[np.argsort(-s[pos], kind="stable")[:k]]
    y[hidden] = False
    return y
