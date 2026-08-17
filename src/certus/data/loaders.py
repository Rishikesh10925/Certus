"""Dataset adapters that read raw exports (CBRI/Roboflow in COCO or YOLO
format, plus the secondary dataset) into the unified images/labels schema
described in roadmap Section 3.6 (SQLite tables: images, labels, scores,
calibrations, decisions, audit_samples). Nothing else in the pipeline should
touch a raw dataset file directly -- every downstream module reads through
this schema.

Only `images` is implemented so far -- `labels`, `scores`, `calibrations`,
`decisions`, and `audit_samples` land as the pipeline stages that need them
get built (scores in Week 4, calibrations in Week 3, decisions/audit_samples
with the service in Week 14). `ImageRecord` is the row shape that table will
eventually hold; for now it's written straight to the split manifest
(`splits.py`) rather than into an actual SQLite file, since nothing yet
needs the query surface a real database would add.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    dataset: str  # "cbri" | "sdnet2018"
    relpath: str  # path relative to the dataset's raw root
    site_id: str  # real if the dataset ships one, otherwise a documented proxy
    camera_id: str | None
    split: str  # the split as shipped by the source, e.g. roboflow's train/valid/test
    defect_positive: bool
    classes: tuple[str, ...] = field(default_factory=tuple)


def load_cbri_images(root: Path) -> list[ImageRecord]:
    """Reads the CBRI COCO export (train/valid/test subfolders, each with
    `_annotations.coco.json`) into ImageRecords. site_id is fixed to
    "unknown" here -- CBRI ships no real site metadata (Week 1 audit); the
    Week 2 diagnostic assigns a proxy cluster id separately rather than
    baking an unverified guess into the loader itself.
    """
    records: list[ImageRecord] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
        anns_by_image: dict[int, list[dict[str, object]]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for img in coco.get("images", []):
            anns = anns_by_image.get(img["id"], [])
            classes = tuple(
                sorted(
                    {
                        categories.get(a.get("category_id"), "unknown")
                        for a in anns
                        if categories.get(a.get("category_id"), "").lower()
                        not in ("background", "")
                    }
                )
            )
            records.append(
                ImageRecord(
                    dataset="cbri",
                    relpath=(Path(split_dir.name) / img["file_name"]).as_posix(),
                    site_id="unknown",
                    camera_id=None,
                    split=split_dir.name,
                    defect_positive=bool(classes),
                    classes=classes,
                )
            )
    return records


_SDNET_STRUCTURES = {
    "D": "deck",
    "P": "pavement",
    "W": "wall",
}


def load_sdnet2018_images(root: Path) -> list[ImageRecord]:
    """Reads SDNET2018's extracted layout: `<D|P|W>/<C|U><D|P|W>/*.jpg`,
    e.g. `D/CD/7001-115.jpg` (cracked deck) or `D/UD/7001-1.jpg`
    (uncracked deck). Every filename follows `<structure_id>-<n>.jpg`
    (occasionally `<structure_id>-<n>_2.jpg` for a second photo of the same
    spot) -- `structure_id` is a genuine physical-structure identifier
    (confirmed by inspection: 54 distinct deck IDs, 104 pavement, 72 wall,
    consistent across all 56,092 files), not a proxy, so site_id is set to
    `<structure-type>-<structure_id>` directly. Unlike CBRI, this dataset
    ships no train/valid/test split of its own -- `split` is left empty
    here and assigned deterministically by `splits.assign_secondary_split`.
    """
    records: list[ImageRecord] = []
    for structure_letter, structure_name in _SDNET_STRUCTURES.items():
        structure_dir = root / structure_letter
        for positive, prefix in ((True, "C"), (False, "U")):
            subdir = structure_dir / f"{prefix}{structure_letter}"
            for image_path in sorted(subdir.glob("*.jpg")):
                structure_id = image_path.stem.split("-", 1)[0]
                records.append(
                    ImageRecord(
                        dataset="sdnet2018",
                        relpath=image_path.relative_to(root).as_posix(),
                        site_id=f"{structure_name}-{structure_id}",
                        camera_id=None,
                        split="",
                        defect_positive=positive,
                        classes=("crack",) if positive else (),
                    )
                )
    return records
