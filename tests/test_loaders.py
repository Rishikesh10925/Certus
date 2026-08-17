"""Covers certus.data.loaders on a tiny synthetic COCO export."""

from __future__ import annotations

import json
from pathlib import Path

from certus.data.loaders import load_cbri_images, load_sdnet2018_images


def _write_split(
    split_dir: Path, images: list[dict], annotations: list[dict], categories: list[dict]
) -> None:
    split_dir.mkdir(parents=True)
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def test_load_cbri_images_marks_positive_and_negative(tmp_path: Path) -> None:
    categories = [{"id": 1, "name": "crack"}, {"id": 2, "name": "vegetation"}]
    _write_split(
        tmp_path / "train",
        images=[{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        annotations=[{"image_id": 1, "category_id": 1}, {"image_id": 1, "category_id": 2}],
        categories=categories,
    )
    _write_split(
        tmp_path / "test",
        images=[{"id": 1, "file_name": "c.jpg"}],
        annotations=[],
        categories=categories,
    )

    records = load_cbri_images(tmp_path)

    assert len(records) == 3
    positive = {r.relpath: r for r in records if r.defect_positive}
    assert set(positive) == {"train/a.jpg"}
    assert positive["train/a.jpg"].classes == ("crack", "vegetation")
    assert positive["train/a.jpg"].split == "train"

    negative = [r for r in records if not r.defect_positive]
    assert {r.relpath for r in negative} == {"train/b.jpg", "test/c.jpg"}


def test_load_cbri_images_uses_posix_relpaths(tmp_path: Path) -> None:
    _write_split(
        tmp_path / "valid",
        images=[{"id": 1, "file_name": "x.jpg"}],
        annotations=[],
        categories=[],
    )

    records = load_cbri_images(tmp_path)

    assert records[0].relpath == "valid/x.jpg"
    assert "\\" not in records[0].relpath


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_load_sdnet2018_images_derives_site_id_from_structure_prefix(tmp_path: Path) -> None:
    _touch(tmp_path / "D" / "CD" / "7001-115.jpg")
    _touch(tmp_path / "D" / "CD" / "7001-116_2.jpg")
    _touch(tmp_path / "D" / "UD" / "7002-1.jpg")
    _touch(tmp_path / "W" / "CW" / "7140-98.jpg")

    records = load_sdnet2018_images(tmp_path)

    by_relpath = {r.relpath: r for r in records}
    assert len(records) == 4
    assert by_relpath["D/CD/7001-115.jpg"].site_id == "deck-7001"
    assert by_relpath["D/CD/7001-115.jpg"].defect_positive is True
    assert by_relpath["D/CD/7001-115.jpg"].classes == ("crack",)
    assert by_relpath["D/CD/7001-116_2.jpg"].site_id == "deck-7001"
    assert by_relpath["D/UD/7002-1.jpg"].site_id == "deck-7002"
    assert by_relpath["D/UD/7002-1.jpg"].defect_positive is False
    assert by_relpath["W/CW/7140-98.jpg"].site_id == "wall-7140"
    assert all(r.split == "" for r in records)
