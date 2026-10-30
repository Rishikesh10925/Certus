"""Human-review fraction (HRF = (|FLAG| + |ESCALATE|) / N) and the cost
break-even model: net saving per image = (1 - HRF(epsilon)) * c_h -
epsilon * pi * c_m, swept over the c_m/c_h ratio to find the region where
automation is economically rational. Feeds Figures F4 and F8. See roadmap
Section 3.4, Section 5.4.

The epsilon * pi term is the conditional guarantee's misses per image. The
joint guarantee already counts per image, so its term is epsilon alone,
capped at pi (no more defects can be missed than exist). Savings are
relative to a human reviewing everything, and assume humans miss nothing.
"""

from __future__ import annotations

import math


def misses_per_image(epsilon: float, prevalence: float, guarantee: str) -> float:
    """Upper bound on missed defects per image that the guarantee allows."""
    if not 0.0 < prevalence <= 1.0:
        raise ValueError(f"prevalence must be in (0, 1], got {prevalence}")
    if guarantee == "conditional":
        return epsilon * prevalence
    if guarantee == "joint":
        return min(epsilon, prevalence)
    raise ValueError(f"unknown guarantee {guarantee!r}")


def net_saving(
    hrf: float, epsilon: float, prevalence: float, guarantee: str, c_h: float, c_m: float
) -> float:
    """Saving per image against full human review, at the guarantee's bound."""
    return (1.0 - hrf) * c_h - misses_per_image(epsilon, prevalence, guarantee) * c_m


def break_even_ratio(hrf: float, epsilon: float, prevalence: float, guarantee: str) -> float:
    """The c_m / c_h at which net saving is zero; automation pays below it."""
    misses = misses_per_image(epsilon, prevalence, guarantee)
    return (1.0 - hrf) / misses if misses > 0 else math.inf
