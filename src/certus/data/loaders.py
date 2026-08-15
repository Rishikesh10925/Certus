"""Dataset adapters that read raw exports (CBRI/Roboflow in COCO or YOLO
format, plus the secondary dataset) into the unified images/labels schema
described in roadmap Section 3.6 (SQLite tables: images, labels, scores,
calibrations, decisions, audit_samples). Nothing else in the pipeline should
touch a raw dataset file directly -- every downstream module reads through
this schema.
"""
