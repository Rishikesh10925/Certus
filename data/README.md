# data/

DVC pointers only -- no raw data files in git (roadmap Section 3.5,
Section 4.5).

## raw/cbri

The CBRI Building Defect Detection export (Roboflow, version 1, no
augmentation/preprocessing applied) is DVC-tracked at `data/raw/cbri.dvc`
-- 1518 images, COCO annotations, `train`/`valid`/`test` splits as assigned
on Roboflow. `dvc pull` requires the remote below, which isn't configured
yet, so anyone without direct access to the local DVC cache needs to
re-download it themselves (see `configs/data/cbri.yaml` for the Roboflow
project reference) until a remote is set up.

See `docs/logbook.md` for the Week 1 audit (n_pos, per-class counts, the
site/camera metadata finding) and `docs/audit/cbri_audit_report_v1.json`
for the full machine-readable report.

No Google Drive remote is configured yet. To add one:

```bash
dvc remote add -d storage gdrive://<FOLDER_ID>
dvc remote modify storage gdrive_acknowledge_abuse true
```
