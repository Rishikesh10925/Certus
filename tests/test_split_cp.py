"""Covers conformal/split_cp.py -- LAC scores, calibration, and set
construction on hand-checkable inputs. Statistical validity is
tests/test_validity.py's job."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.split_cp import (
    aps_score_matrix,
    calibrate_lac,
    calibrate_split,
    lac_score_matrix,
    lac_scores,
    predict_sets,
    predict_sets_lac,
    raps_score_matrix,
)
from certus.eval.coverage import marginal_coverage
from certus.eval.efficiency import mean_set_size


def test_lac_scores_are_one_minus_true_class_prob() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(lac_scores(probs, [0, 1]), [0.3, 0.7])


def test_predict_sets_lac_includes_classes_at_or_below_threshold() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    sets = predict_sets_lac(probs, q_hat=0.65)  # include p >= 0.35
    assert sets.tolist() == [[True, False, False], [True, True, False]]


def test_calibrated_threshold_covers_its_own_boundary_point() -> None:
    """Calibration and prediction must use the same comparison: the score
    that *is* q_hat has to land inside its own set."""
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]] * 10)
    labels = np.array([0, 0, 0] * 10)
    q_hat = calibrate_lac(probs, labels, alpha=0.3)
    assert q_hat in set(lac_scores(probs, labels))
    boundary = probs[np.isclose(lac_scores(probs, labels), q_hat)][:1]
    assert predict_sets_lac(boundary, q_hat)[0, 0]


def test_infinite_threshold_gives_full_sets() -> None:
    probs = np.array([[0.5, 0.5], [0.9, 0.1]])
    q_hat = calibrate_lac(probs, [0, 1], alpha=0.1)
    assert q_hat == math.inf
    sets = predict_sets_lac(probs, q_hat)
    assert sets.all()
    assert marginal_coverage(sets, [0, 1]) == 1.0
    assert mean_set_size(sets) == 2.0


def test_aps_scores_are_mass_ranked_above_plus_own_prob() -> None:
    probs = np.array([[0.2, 0.5, 0.3]])
    # ranking: class1 (0.5), class2 (0.3), class0 (0.2)
    np.testing.assert_allclose(aps_score_matrix(probs), [[1.0, 0.5, 0.8]])
    # randomised with u = 0.5: mass above + half of own prob
    np.testing.assert_allclose(aps_score_matrix(probs, u=[0.5]), [[0.9, 0.25, 0.65]])


def test_raps_adds_rank_penalty_past_k_reg() -> None:
    probs = np.array([[0.2, 0.5, 0.3]])
    # ranks: class1=1, class2=2, class0=3; penalty lam * max(0, rank - 1)
    np.testing.assert_allclose(
        raps_score_matrix(probs, lam=0.1, k_reg=1), [[1.0 + 0.2, 0.5, 0.8 + 0.1]]
    )
    np.testing.assert_allclose(raps_score_matrix(probs, lam=0.0, k_reg=1), aps_score_matrix(probs))


def test_aps_sets_are_nested_in_probability_order() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(6), size=50)
    sets = predict_sets(aps_score_matrix(probs), q_hat=0.7)
    for p, s in zip(probs, sets, strict=True):
        if s.any():
            # every included class is more probable than every excluded one
            assert p[s].min() >= p[~s].max(initial=-1.0)


def test_calibrate_split_matches_calibrate_lac() -> None:
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=100)
    labels = rng.integers(0, 4, size=100)
    assert calibrate_split(lac_score_matrix(probs), labels, 0.1) == calibrate_lac(
        probs, labels, 0.1
    )


def test_lac_scores_reject_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        lac_scores(np.ones((3, 2)) / 2, [0, 1])
