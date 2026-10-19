"""Covers the score registry, the fit_* split guard, the linear head, S1,
and the SQLite scores table -- all on synthetic features, no backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certus.data.loaders import ImageRecord
from certus.data.store import connect, model_versions, read_scores, write_scores
from certus.models.heads import fit_linear_head
from certus.scores.registry import (
    ScoreContext,
    ScoreResult,
    available_scores,
    check_fit_splits,
    get_score,
    register,
)


def _records_and_features(n: int = 60) -> tuple[list[ImageRecord], np.ndarray]:
    rng = np.random.default_rng(0)
    records, feats = [], []
    for i in range(n):
        positive = i % 2 == 0
        split = ("train", "train", "cal", "test")[i % 4] if i >= 8 else "train"
        records.append(ImageRecord("cbri", f"{split}/{i}.jpg", "unknown", None, split, positive))
        f = rng.normal(size=8).astype(np.float32)
        f[0] += 3.0 if positive else -3.0  # one clearly informative feature
        feats.append(f)
    return records, np.stack(feats)


def test_builtin_scores_are_registered() -> None:
    assert "s1" in available_scores()


def test_register_rejects_duplicates_and_unknown_ids_raise() -> None:
    with pytest.raises(ValueError, match="twice"):

        @register("s1")
        def _dup(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
            raise AssertionError

    with pytest.raises(KeyError, match="unknown score"):
        get_score("s99")


@pytest.mark.parametrize("splits", [["train", "cal"], ["test"], ["train", "valid"]])
def test_fit_guard_rejects_calibration_and_test_rows(splits: list[str]) -> None:
    with pytest.raises(ValueError, match="fit_"):
        check_fit_splits(splits)
    feats = np.zeros((len(splits), 2), dtype=np.float32)
    with pytest.raises(ValueError, match="fit_"):
        fit_linear_head(feats, np.array([True] * len(splits)), splits)


def test_fit_guard_allows_train_and_tune() -> None:
    check_fit_splits(["train", "tune", "train"])


def test_s1_scores_are_oriented_high_for_no_defect(tmp_path: Path) -> None:
    records, feats = _records_and_features()
    cache = tmp_path / "features" / "cbri_clip-vit-b32_by_relpath.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, relpaths=np.array([r.relpath for r in records]), features=feats)

    result = get_score("s1")(records, ScoreContext({"cbri": tmp_path}, tmp_path))
    assert result.values.shape == (len(records),)
    assert np.all((result.values >= 0) & (result.values <= 1))
    pos = np.array([r.defect_positive for r in records])
    assert result.values[~pos].mean() > result.values[pos].mean() + 0.5
    assert result.model_version.startswith("clip-vit-b32+logreg:")


def test_s1_head_ignores_calibration_rows(tmp_path: Path) -> None:
    """Flipping every cal/test label must not change the fitted head."""
    records, feats = _records_and_features()
    cache = tmp_path / "features" / "cbri_clip-vit-b32_by_relpath.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, relpaths=np.array([r.relpath for r in records]), features=feats)
    ctx = ScoreContext({"cbri": tmp_path}, tmp_path)

    flipped = [
        r
        if r.split == "train"
        else ImageRecord(**{**r.__dict__, "defect_positive": not r.defect_positive})
        for r in records
    ]
    a, b = get_score("s1")(records, ctx), get_score("s1")(flipped, ctx)
    assert a.model_version == b.model_version
    np.testing.assert_array_equal(a.values, b.values)


def test_scores_table_round_trip(tmp_path: Path) -> None:
    records, _ = _records_and_features(10)
    values = np.linspace(0, 1, 10)
    conn = connect(tmp_path / "db.sqlite")
    write_scores(conn, records, "s1", "v1", values)
    write_scores(conn, records, "s1", "v1", values)  # idempotent
    np.testing.assert_array_equal(read_scores(conn, records[::-1], "s1", "v1"), values[::-1])
    assert model_versions(conn, "s1") == ["v1"]

    with pytest.raises(KeyError, match="no cached"):
        read_scores(conn, records, "s1", "v2")
    with pytest.raises(ValueError, match="non-finite"):
        write_scores(conn, records[:1], "s1", "v1", np.array([np.nan]))
    conn.close()
