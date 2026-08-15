"""Exercises scripts/audit_data.py on tiny synthetic exports -- the audit
script is the actual Week 1 deliverable, so it gets tested before it's ever
pointed at the real CBRI export.
"""

from __future__ import annotations

import json
from pathlib import Path

from audit_data import audit_coco, audit_yolo, detect_format


def test_audit_coco_counts_positive_and_negative(tmp_path: Path) -> None:
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 1}],
        "categories": [{"id": 1, "name": "crack"}],
    }
    (tmp_path / "_annotations.coco.json").write_text(json.dumps(coco))

    result = audit_coco(tmp_path)

    assert result["defect_positive"] == 1
    assert result["defect_negative"] == 1
    assert result["class_counts"] == {"crack": 1}


def test_audit_yolo_counts_positive_and_negative(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"")
    (images_dir / "b.jpg").write_bytes(b"")
    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels_dir / "b.txt").write_text("")
    (tmp_path / "data.yaml").write_text("names: ['crack']\n")

    result = audit_yolo(tmp_path)

    assert result["defect_positive"] == 1
    assert result["defect_negative"] == 1
    assert result["class_counts"] == {"crack": 1}


def test_detect_format_prefers_coco_when_present(tmp_path: Path) -> None:
    (tmp_path / "_annotations.coco.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []})
    )

    assert detect_format(tmp_path) == "coco"


def test_detect_format_returns_none_when_no_annotations(tmp_path: Path) -> None:
    assert detect_format(tmp_path) == "none"
