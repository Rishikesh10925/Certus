"""Covers the score registry, the fit_* split guard, the linear head, S1,
and the SQLite scores table -- all on synthetic features, no backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certus.data.loaders import ImageRecord
from certus.data.store import connect, model_versions, read_scores, write_scores
from certus.models.heads import fit_linear_head
from certus.scores import s1
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


def _s1_ctx(tmp_path: Path, records: list[ImageRecord], feats: np.ndarray) -> ScoreContext:
    """Pre-fills both caches so S1 runs without loading CLIP: image
    features from `feats`, and text features where every defect prompt
    points along +x0 and every clean prompt along -x0."""
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    cache = tmp_path / "features" / "cbri_clip-vit-b32_by_relpath.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, relpaths=np.array([r.relpath for r in records]), features=feats)
    text = np.zeros((len(s1.DEFECT_PROMPTS) + len(s1.CLEAN_PROMPTS), feats.shape[1]), np.float32)
    text[: len(s1.DEFECT_PROMPTS), 0] = 1.0
    text[len(s1.DEFECT_PROMPTS) :, 0] = -1.0
    np.save(s1._text_cache_path(tmp_path), text)
    return ScoreContext({"cbri": tmp_path}, tmp_path)


def test_s1_scores_are_oriented_high_for_no_defect(tmp_path: Path) -> None:
    records, feats = _records_and_features()
    result = get_score("s1")(records, _s1_ctx(tmp_path, records, feats))
    assert result.values.shape == (len(records),)
    assert np.all((result.values >= 0) & (result.values <= 1))
    pos = np.array([r.defect_positive for r in records])
    assert result.values[~pos].mean() > result.values[pos].mean() + 0.5
    assert result.model_version == f"clip-vit-b32+zeroshot:{s1.prompt_hash()}"


def test_s1_never_reads_labels(tmp_path: Path) -> None:
    """Zero-shot: flipping every label, any split, must not move a score."""
    records, feats = _records_and_features()
    ctx = _s1_ctx(tmp_path, records, feats)
    flipped = [
        ImageRecord(**{**r.__dict__, "defect_positive": not r.defect_positive}) for r in records
    ]
    np.testing.assert_array_equal(
        get_score("s1")(records, ctx).values, get_score("s1")(flipped, ctx).values
    )


def test_zero_shot_defect_prob_is_mass_on_defect_prompts() -> None:
    img = np.array([[1.0, 0.0]], dtype=np.float32)
    text = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    # logits 100, 0, 0 -> nearly all mass on prompt 0
    assert s1.zero_shot_defect_prob(img, text, n_defect=1)[0] > 0.999
    assert s1.zero_shot_defect_prob(img, text, n_defect=0)[0] == 0.0


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
