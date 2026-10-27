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
    assert {"s1", "s2"} <= set(available_scores())


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


def test_s2_negates_assertion_logprob_and_tags_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 wiring without the 4.5 GB model: a fake VLM returns fixed
    log-probs, and S2 must negate them, keep order, and pin the prompt."""
    from certus.models import generative
    from certus.scores import s2

    seen: dict[str, object] = {}

    class FakeVLM:
        def assertion_logprob(self, paths: list[Path], question: str, assertion: str) -> np.ndarray:
            seen.update(paths=paths, question=question, assertion=assertion)
            return -np.arange(len(paths), dtype=np.float64)

    monkeypatch.setattr(generative, "GenerativeVLM", FakeVLM)
    records, _ = _records_and_features(4)
    result = get_score("s2")(records, ScoreContext({"cbri": tmp_path}, tmp_path))

    np.testing.assert_array_equal(result.values, [0.0, 1.0, 2.0, 3.0])
    assert seen["paths"] == [tmp_path / r.relpath for r in records]
    assert seen["question"] == s2.QUESTION and seen["assertion"] == s2.ASSERTION
    assert result.model_version == f"qwen2-vl-2b+assert:{s2.prompt_hash()}"


def test_defect_probability_inverts_each_score() -> None:
    from certus.scores.registry import defect_probability

    np.testing.assert_allclose(defect_probability("s1", np.array([0.0, 0.25, 1.0])), [1, 0.75, 0])
    np.testing.assert_allclose(defect_probability("s2", -np.log([0.9, 0.1])), [0.9, 0.1])


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("85", 85.0), (" 70.5%", 70.5), ("I'd say 40 out of 100", 40.0), ("150", None), ("yes", None)],
)
def test_s3_parses_verbal_confidence(answer: str, expected: float | None) -> None:
    from certus.scores.s3 import parse_confidence

    assert parse_confidence(answer) == expected


def test_s3_and_s4_wiring_with_fake_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from certus.models import detector, generative
    from certus.scores import s3, s4

    class FakeVLM:
        def generate(self, paths: list[Path], question: str) -> list[str]:
            assert question == s3.QUESTION
            return ["90", "no idea", "10", "100"][: len(paths)]

    class FakeDetector:
        def max_box_confidence(self, paths: list[Path], queries: list[str]) -> np.ndarray:
            assert queries == list(s4.QUERIES)
            return np.linspace(0, 1, len(paths))

    monkeypatch.setattr(generative, "GenerativeVLM", FakeVLM)
    monkeypatch.setattr(detector, "OpenVocabDetector", FakeDetector)
    records, _ = _records_and_features(4)
    ctx = ScoreContext({"cbri": tmp_path}, tmp_path)

    r3 = get_score("s3")(records, ctx)
    np.testing.assert_allclose(r3.values, [0.1, 0.5, 0.9, 0.0])
    assert r3.notes == {"unparseable": 1}
    r4 = get_score("s4")(records, ctx)
    np.testing.assert_allclose(r4.values, [1.0, 2 / 3, 1 / 3, 0.0])
    assert {"s3", "s4", "s5", "s6", "s7"} <= set(available_scores())


def test_s5_is_per_image_deterministic(tmp_path: Path) -> None:
    """S5's dropout masks are seeded by relpath, so an image's score can't
    depend on which other images were scored alongside it."""
    records, feats = _records_and_features()
    ctx = _s1_ctx(tmp_path, records, feats)
    a = get_score("s5")(records, ctx)
    b = get_score("s5")(records[::-1], ctx)
    np.testing.assert_allclose(a.values, b.values[::-1])
    pos = np.array([r.defect_positive for r in records])
    assert a.values[~pos].mean() > a.values[pos].mean()
    # the lower bound never exceeds S1's point estimate by much
    assert np.all(a.values <= get_score("s1")(records, ctx).values + 0.2)


def test_s6_and_s7_fits_refuse_calibration_rows() -> None:
    from certus.scores.s6 import fit_reference
    from certus.scores.s7 import fit_combiner

    x = np.random.default_rng(0).normal(size=(20, 4))
    with pytest.raises(ValueError, match="fit_"):
        fit_reference(x.astype(np.float32), ["train"] * 19 + ["cal"])
    with pytest.raises(ValueError, match="fit_"):
        fit_combiner(x, np.arange(20) % 2 == 0, ["test"] * 20)


def test_s6_scores_atypical_images_low() -> None:
    from certus.scores.s6 import fit_reference, mahalanobis

    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 8)).astype(np.float32)
    mean, prec = fit_reference(ref, ["train"] * 500)
    typical = rng.normal(size=(50, 8)).astype(np.float32)
    odd = (rng.normal(size=(50, 8)) + 4).astype(np.float32)
    assert mahalanobis(odd, mean, prec).min() > mahalanobis(typical, mean, prec).mean()


def test_s7_refuses_its_own_tuning_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from certus.scores import s7

    records, _ = _records_and_features(6)
    monkeypatch.setattr(s7, "tune_records", lambda manifest: records[:3])
    with pytest.raises(ValueError, match="tuning rows"):
        get_score("s7")(records, ScoreContext({"cbri": tmp_path}, tmp_path))
