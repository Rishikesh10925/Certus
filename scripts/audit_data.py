#!/usr/bin/env python
"""Audits a defect-detection image export (e.g. the CBRI/Roboflow dataset)
before any conformal work begins -- roadmap Part IV, Week 1-2 tasks.

Reports:
  - total image count
  - defect-positive vs defect-negative image count (n_pos is the binding
    constraint on how small epsilon can go -- roadmap Section 4.1:
    n_pos >~ 10/epsilon)
  - per-class annotation breakdown, if class labels exist
  - whether site_id / camera_id / EXIF metadata exists per image
  - if no site metadata exists, what filename/EXIF fields ARE available, so
    a proxy-site derivation strategy (roadmap Section 4.1) can be chosen
    deliberately

Supports COCO (Roboflow's default export), YOLO (images/ + labels/ .txt),
and Pascal VOC (.xml) annotation formats, auto-detected unless --format is
given explicitly.

Usage:
    python scripts/audit_data.py --path /path/to/dataset/export
    python scripts/audit_data.py --path /path/to/dataset/export --format coco \
        --json-out audit_report.json

This script only reads. It writes nothing into the dataset directory itself.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# format detection
# --------------------------------------------------------------------------- #


def find_coco_jsons(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.json") if _looks_like_coco(p))


def _looks_like_coco(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(4096)
    except OSError:
        return False
    return '"images"' in head and ('"annotations"' in head or '"categories"' in head)


def find_yolo_label_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("*.txt") if p.parent.name.lower() == "labels"})


def find_voc_xmls(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.xml") if _looks_like_voc(p)]


def _looks_like_voc(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2048)
    except OSError:
        return False
    return "<annotation>" in head or "<object>" in head


def detect_format(root: Path) -> str:
    if find_coco_jsons(root):
        return "coco"
    if find_yolo_label_dirs(root):
        return "yolo"
    if find_voc_xmls(root):
        return "voc"
    return "none"


# --------------------------------------------------------------------------- #
# image inventory
# --------------------------------------------------------------------------- #


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


# --------------------------------------------------------------------------- #
# COCO audit
# --------------------------------------------------------------------------- #


def audit_coco(root: Path) -> dict:
    jsons = find_coco_jsons(root)
    total_images = 0
    positive = 0
    negative = 0
    class_counts: Counter[str] = Counter()

    for jp in jsons:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}

        ann_by_image: dict[int, list] = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann["image_id"]].append(ann)

        total_images += len(images)
        for img in images:
            img_id = img.get("id")
            anns = ann_by_image.get(img_id, [])
            # Roboflow COCO exports sometimes include a background-only
            # category (e.g. id 0) with no real boxes -- exclude it from
            # the positive count.
            real_anns = [
                a
                for a in anns
                if categories.get(a.get("category_id"), "").lower() not in ("background", "")
            ]
            if real_anns:
                positive += 1
            else:
                negative += 1
            for a in real_anns:
                class_counts[categories.get(a.get("category_id"), "unknown")] += 1

    return {
        "format": "coco",
        "annotation_files": [str(p.relative_to(root)) for p in jsons],
        "total_images_in_annotations": total_images,
        "defect_positive": positive,
        "defect_negative": negative,
        "class_counts": dict(class_counts),
    }


# --------------------------------------------------------------------------- #
# YOLO audit
# --------------------------------------------------------------------------- #


def _load_yolo_class_names(root: Path) -> dict[int, str]:
    for candidate in ("data.yaml", "data.yml", "classes.txt", "obj.names"):
        for p in root.rglob(candidate):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if candidate.startswith("data."):
                names = re.search(r"names:\s*\[(.*?)\]", text, re.DOTALL)
                if names:
                    parsed = [n.strip().strip("'\"") for n in names.group(1).split(",")]
                    return dict(enumerate(parsed))
                names_block = re.search(r"names:\s*\n((?:\s*-\s*.+\n?)+)", text)
                if names_block:
                    parsed = [
                        line.split("-", 1)[1].strip()
                        for line in names_block.group(1).splitlines()
                        if line.strip()
                    ]
                    return dict(enumerate(parsed))
            else:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return dict(enumerate(lines))
    return {}


def audit_yolo(root: Path) -> dict:
    label_dirs = find_yolo_label_dirs(root)
    class_names = _load_yolo_class_names(root)
    positive = 0
    negative = 0
    class_counts: Counter[str] = Counter()
    images = find_images(root)

    label_index: dict[str, Path] = {}
    for ld in label_dirs:
        for txt in ld.glob("*.txt"):
            label_index[txt.stem] = txt

    for img in images:
        label_path = label_index.get(img.stem)
        if label_path is None or not label_path.exists():
            negative += 1
            continue
        lines = [
            ln
            for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()
        ]
        if not lines:
            negative += 1
            continue
        positive += 1
        for ln in lines:
            cls_id_str = ln.split()[0]
            try:
                cls_id = int(cls_id_str)
            except ValueError:
                continue
            class_counts[class_names.get(cls_id, f"class_{cls_id}")] += 1

    return {
        "format": "yolo",
        "label_dirs": [str(p.relative_to(root)) for p in label_dirs],
        "class_names_found": bool(class_names),
        "defect_positive": positive,
        "defect_negative": negative,
        "class_counts": dict(class_counts),
    }


# --------------------------------------------------------------------------- #
# VOC audit
# --------------------------------------------------------------------------- #


def audit_voc(root: Path) -> dict:
    xmls = find_voc_xmls(root)
    positive = 0
    negative = 0
    class_counts: Counter[str] = Counter()

    for xp in xmls:
        try:
            tree = ET.parse(xp)
        except ET.ParseError:
            continue
        objects = tree.findall(".//object")
        if objects:
            positive += 1
        else:
            negative += 1
        for obj in objects:
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                class_counts[name_el.text.strip()] += 1

    return {
        "format": "voc",
        "annotation_files_found": len(xmls),
        "defect_positive": positive,
        "defect_negative": negative,
        "class_counts": dict(class_counts),
    }


# --------------------------------------------------------------------------- #
# site / camera / EXIF metadata probe
# --------------------------------------------------------------------------- #

SITE_LIKE_EXIF_TAGS = {"GPSInfo", "Make", "Model", "DateTimeOriginal", "DateTime", "Software"}


def probe_metadata(images: list[Path], sample_size: int) -> dict:
    step = max(1, len(images) // sample_size) if images else 1
    sample = images[::step][:sample_size]

    try:
        from PIL import ExifTags, Image
    except ImportError:
        return {
            "pillow_available": False,
            "note": (
                "Pillow not installed -- install project deps (`pip install -e .`) "
                "to enable EXIF inspection."
            ),
            "filename_patterns": _summarise_filenames(images),
        }

    tag_presence: Counter[str] = Counter()
    gps_present = 0
    makes: set[str] = set()
    models: set[str] = set()
    checked = 0

    for path in sample:
        try:
            with Image.open(path) as im:
                exif = im.getexif()
        except Exception:
            continue
        checked += 1
        if not exif:
            continue
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in SITE_LIKE_EXIF_TAGS:
                tag_presence[tag_name] += 1
            if tag_name == "GPSInfo":
                gps_present += 1
            if tag_name == "Make":
                makes.add(str(value))
            if tag_name == "Model":
                models.add(str(value))

    return {
        "pillow_available": True,
        "images_sampled": checked,
        "images_with_gps": gps_present,
        "distinct_camera_makes": sorted(makes),
        "distinct_camera_models": sorted(models),
        "exif_tag_presence_in_sample": dict(tag_presence),
        "filename_patterns": _summarise_filenames(images),
    }


def _summarise_filenames(images: list[Path]) -> dict:
    """Reports the *shape* of filenames so a proxy site/camera strategy can
    be designed by hand if no EXIF or dataset-native site metadata exists."""
    stems = [p.stem for p in images]
    if not stems:
        return {}

    delimiter_counts: Counter[str] = Counter()
    for stem in stems:
        for delim in ("_", "-", "."):
            if delim in stem:
                delimiter_counts[delim] += 1

    numeric_prefix = sum(1 for s in stems if re.match(r"^\d+", s))
    uuid_like_prefix = sum(1 for s in stems if re.match(r"^[0-9a-fA-F]{8}-", s))

    return {
        "total_filenames": len(stems),
        "delimiter_frequency": dict(delimiter_counts),
        "example_stems": stems[:10],
        "stems_starting_with_digits": numeric_prefix,
        "stems_looking_like_uuid_prefix": uuid_like_prefix,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def build_report(root: Path, fmt: str, sample_size: int) -> dict:
    if fmt == "auto":
        fmt = detect_format(root)

    images = find_images(root)

    if fmt == "coco":
        annotation_summary = audit_coco(root)
    elif fmt == "yolo":
        annotation_summary = audit_yolo(root)
    elif fmt == "voc":
        annotation_summary = audit_voc(root)
    else:
        annotation_summary = {
            "format": "none",
            "note": (
                "No recognized annotation format (COCO json / YOLO labels / VOC xml) "
                "found under this path."
            ),
        }

    metadata_summary = probe_metadata(images, sample_size)

    return {
        "root": str(root),
        "total_images_on_disk": len(images),
        "annotations": annotation_summary,
        "site_camera_metadata": metadata_summary,
    }


def print_report(report: dict) -> None:
    print("=" * 78)
    print(f"CERTUS DATA AUDIT -- {report['root']}")
    print("=" * 78)

    print(f"\nTotal images on disk : {report['total_images_on_disk']}")

    ann = report["annotations"]
    print(f"Annotation format    : {ann.get('format')}")
    if ann.get("format") == "none":
        print(f"  {ann.get('note')}")
    else:
        pos = ann.get("defect_positive", "?")
        neg = ann.get("defect_negative", "?")
        print(f"Defect-positive images (n_pos) : {pos}")
        print(f"Defect-negative images         : {neg}")
        if isinstance(pos, int):
            print("\nEpsilon feasibility (roadmap 4.1: n_pos >~ 10/epsilon):")
            for eps in (0.10, 0.05, 0.02, 0.01, 0.005):
                floor = 10 / eps
                verdict = "OK" if pos >= floor else "SHORT"
                print(f"  epsilon={eps:<6} needs n_pos >~ {floor:>6.0f}  -> {verdict}")
        classes = ann.get("class_counts", {})
        if classes:
            print("\nPer-class annotation counts:")
            for name, count in sorted(classes.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<30} {count}")

    meta = report["site_camera_metadata"]
    print("\nSite / camera metadata probe:")
    if not meta.get("pillow_available", True):
        print(f"  {meta.get('note')}")
    else:
        print(f"  Images sampled            : {meta.get('images_sampled')}")
        print(f"  Images with GPS EXIF      : {meta.get('images_with_gps')}")
        print(f"  Distinct camera makes     : {meta.get('distinct_camera_makes')}")
        print(f"  Distinct camera models    : {meta.get('distinct_camera_models')}")
        print(f"  EXIF tag presence sample  : {meta.get('exif_tag_presence_in_sample')}")

    fp = meta.get("filename_patterns", {})
    if fp:
        print("\nFilename structure (for a proxy site/camera strategy if no metadata exists):")
        for k, v in fp.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 78)
    print("Next: roadmap 4.1 -- decide the achievable epsilon range from n_pos,")
    print("and roadmap 4.4 Week 2 diagnostic -- verify real site separation with")
    print("a domain classifier before committing to the shift story.")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path", required=True, type=Path, help="Root directory of the dataset export."
    )
    parser.add_argument(
        "--format",
        choices=["auto", "coco", "yolo", "voc"],
        default="auto",
        help="Annotation format. Default: auto-detect.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of images to sample for the EXIF/metadata probe (default: 50).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to also write the report as JSON.",
    )
    args = parser.parse_args()

    root = args.path.expanduser().resolve()
    if not root.exists():
        print(f"error: path does not exist: {root}", file=sys.stderr)
        return 1

    report = build_report(root, args.format, args.sample_size)
    print_report(report)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
