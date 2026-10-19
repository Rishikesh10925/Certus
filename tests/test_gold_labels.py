"""Covers scripts/gold_labels.py -- the labelling sheet's order, its
exclusion of training images, and the status validator."""

from __future__ import annotations

from gold_labels import build_sheet, validate_sheet


def _manifest() -> list[dict[str, str]]:
    rows = []
    for split, n in (("train", 5), ("cal", 4), ("test", 2)):
        rows += [
            {"relpath": f"{split}/{i}.jpg", "content_hash": f"h-{split}-{i}", "split": split}
            for i in range(n)
        ]
    return rows


def test_build_sheet_excludes_train_and_puts_cal_first() -> None:
    sheet = build_sheet(_manifest())
    splits = [r["relpath"].split("/")[0] for r in sheet]
    assert splits == ["cal"] * 4 + ["test"] * 2
    assert all(r["defect_positive"] == "" for r in sheet)


def test_build_sheet_is_deterministic() -> None:
    assert build_sheet(_manifest()) == build_sheet(list(reversed(_manifest())))


def test_validate_sheet_counts_and_flags_errors() -> None:
    manifest = _manifest()
    sheet = build_sheet(manifest)
    sheet[0].update(defect_positive="yes", classes="crack;vegetation", labeller="rt")
    sheet[1].update(defect_positive="no", classes="", labeller="rt")
    sheet[2].update(defect_positive="yes", classes="", labeller="rt")  # inconsistent
    sheet[3]["content_hash"] = "tampered"

    n_labelled, n_pos, errors = validate_sheet(sheet, manifest)
    assert (n_labelled, n_pos) == (3, 2)
    assert len(errors) == 2
    assert any("content_hash" in e for e in errors)
    assert any("defect_positive=yes" in e for e in errors)
