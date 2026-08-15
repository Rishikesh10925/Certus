"""Extracts or derives site_id / camera_id per image (roadmap Section 3.6,
Section 4.1).

If the dataset ships real site/camera metadata, parse it directly. If not,
derive a proxy from EXIF tags, filename structure, or unsupervised
clustering of frozen-backbone features -- and record which strategy was
used, since the LOSO experiment's validity depends on sites being genuinely
distinct (verify with the domain-classifier diagnostic in Section 4.4:
AUC > 0.80 to proceed).

Week 1 audit (docs/logbook.md, docs/audit/cbri_audit_report_v1.json): CBRI
has no site_id/camera_id field, and a 50-image EXIF sample carried no GPS,
Make, or Model tags -- every tag came back empty. Filenames are bare numeric
IDs with a Roboflow-added content hash, no site encoding either. So none of
the cheap options apply here; the Week 2 plan is unsupervised clustering on
frozen-VLM embeddings, verified against the domain-classifier AUC gate
before it's trusted for LOSO.
"""
