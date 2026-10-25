"""Covers certus.data.splits -- deterministic split assignment and the
content-hashed manifest freeze."""

from __future__ import annotations

import json
from pathlib import Path

from certus.data.loaders import ImageRecord
from certus.data.splits import assign_secondary_split, freeze_manifest, rename_cbri_splits


def _records(n_pos: int, n_neg: int) -> list[ImageRecord]:
    records = [
        ImageRecord("sdnet2018", f"pos_{i}.jpg", "D", None, "unsplit", True, ("crack",))
        for i in range(n_pos)
    ]
    records += [
        ImageRecord("sdnet2018", f"neg_{i}.jpg", "D", None, "unsplit", False, ())
        for i in range(n_neg)
    ]
    return records


def test_assign_secondary_split_is_deterministic_and_stratified() -> None:
    records = _records(n_pos=100, n_neg=100)

    a = assign_secondary_split(records, seed=0, train=0.7, cal=0.15)
    b = assign_secondary_split(records, seed=0, train=0.7, cal=0.15)
    assert [r.split for r in a] == [r.split for r in b]

    for label in (True, False):
        subset = [r for r in a if r.defect_positive == label]
        counts = {s: sum(1 for r in subset if r.split == s) for s in ("train", "cal", "test")}
        assert counts["train"] == 70
        assert counts["cal"] == 15
        assert counts["test"] == 15


def test_rename_cbri_splits_maps_valid_to_cal() -> None:
    records = [
        ImageRecord("cbri", "a.jpg", "unknown", None, "train", True, ("crack",)),
        ImageRecord("cbri", "b.jpg", "unknown", None, "valid", True, ("crack",)),
        ImageRecord("cbri", "c.jpg", "unknown", None, "test", False, ()),
    ]
    renamed = rename_cbri_splits(records)
    assert [r.split for r in renamed] == ["train", "cal", "test"]


def test_freeze_manifest_hash_is_deterministic_and_changes_with_content(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    dataset_root.mkdir()
    (dataset_root / "a.jpg").write_bytes(b"image-bytes-a")
    (dataset_root / "b.jpg").write_bytes(b"image-bytes-b")
    records = [
        ImageRecord("cbri", "a.jpg", "unknown", None, "train", True, ("crack",)),
        ImageRecord("cbri", "b.jpg", "unknown", None, "test", False, ()),
    ]

    manifest_path = tmp_path / "manifest" / "splits.csv"
    hash_1 = freeze_manifest(records, {"cbri": dataset_root}, manifest_path)
    assert manifest_path.exists()

    sidecar = json.loads(manifest_path.with_suffix(".json").read_text())
    assert sidecar["overall_hash"] == hash_1
    assert sidecar["n_records"] == 2
    assert sidecar["datasets"] == ["cbri"]

    hash_2 = freeze_manifest(records, {"cbri": dataset_root}, manifest_path)
    assert hash_1 == hash_2

    (dataset_root / "a.jpg").write_bytes(b"different-bytes")
    hash_3 = freeze_manifest(records, {"cbri": dataset_root}, manifest_path)
    assert hash_3 != hash_1


def test_sample_records_is_seeded_order_free_and_split_aware() -> None:
    from certus.data.splits import sample_records

    records = assign_secondary_split(_records(n_pos=50, n_neg=50), seed=0)
    a = sample_records(records, 10, seed=3, split="cal")
    assert a == sample_records(list(reversed(records)), 10, seed=3, split="cal")
    assert len(a) == 10 and all(r.split == "cal" for r in a)
    assert a != sample_records(records, 10, seed=4, split="cal")
    assert len(sample_records(records, None, seed=0)) == 100
