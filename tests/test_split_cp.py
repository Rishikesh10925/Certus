"""Covers conformal/split_cp.py -- LAC scores, calibration, and set
construction on hand-checkable inputs. Statistical validity is
tests/test_validity.py's job."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.split_cp import calibrate_lac, lac_scores, predict_sets_lac
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


def test_lac_scores_reject_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        lac_scores(np.ones((3, 2)) / 2, [0, 1])
