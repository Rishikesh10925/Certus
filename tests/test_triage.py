"""Covers triage/policy.py and triage/economics.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.crc import auto_clear
from certus.triage.economics import break_even_ratio, misses_per_image, net_saving
from certus.triage.policy import Triage, decide, human_review_fraction


def test_decide_routes_each_image() -> None:
    d = decide([0.9, 0.2, 0.3], 0.5, confident_type=[True, True, False])
    assert d.tolist() == [Triage.AUTO_CLEAR, Triage.FLAG, Triage.ESCALATE]


def test_decide_without_types_escalates_everything_not_cleared() -> None:
    assert decide([0.9, 0.2], 0.5).tolist() == [Triage.AUTO_CLEAR, Triage.ESCALATE]


def test_decide_rejects_mismatched_types() -> None:
    with pytest.raises(ValueError, match="shape"):
        decide([0.9, 0.2], 0.5, confident_type=[True])


def test_hrf_matches_the_conformal_decision_and_ignores_routing() -> None:
    s = np.random.default_rng(0).random(500)
    typed = np.random.default_rng(1).random(500) < 0.5
    expected = 1.0 - auto_clear(s, 0.3).mean()
    assert human_review_fraction(decide(s, 0.3)) == pytest.approx(expected)
    assert human_review_fraction(decide(s, 0.3, typed)) == pytest.approx(expected)


def test_infinite_threshold_sends_everything_to_humans() -> None:
    assert human_review_fraction(decide([0.1, 0.9], math.inf)) == 1.0


def test_misses_per_image_differs_by_guarantee() -> None:
    assert misses_per_image(0.05, 0.2, "conditional") == pytest.approx(0.01)
    assert misses_per_image(0.05, 0.2, "joint") == pytest.approx(0.05)
    assert misses_per_image(0.5, 0.2, "joint") == pytest.approx(0.2)  # capped at prevalence
    with pytest.raises(ValueError, match="guarantee"):
        misses_per_image(0.05, 0.2, "marginal")


def test_net_saving_by_hand() -> None:
    # clears 30% at 1 each, misses 0.05 * 0.2 = 0.01 per image at 10 each
    assert net_saving(0.7, 0.05, 0.2, "conditional", 1.0, 10.0) == pytest.approx(0.2)


def test_saving_is_zero_at_break_even() -> None:
    ratio = break_even_ratio(0.7, 0.05, 0.2, "conditional")
    assert ratio == pytest.approx(30.0)
    assert net_saving(0.7, 0.05, 0.2, "conditional", 1.0, ratio) == pytest.approx(0.0)
