"""Certus triage console: one screen, one slider (roadmap Section 3.8).

The slider sets the guaranteed maximum missed-defect rate epsilon. Every
image in the batch is inspected once by the service (POST /v1/inspect);
moving the slider then re-partitions the batch locally against the
lambda_hat the service publishes for each epsilon (GET /v1/calibration),
so the grid, the review fraction and the projected hours update
instantly. Images the ground truth marks as defective carry a red tag, so
a miss is visible on screen rather than hidden in an average.

Run (service first, see service/README.md):
    streamlit run ui/console.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from certus.data.splits import read_manifest
from certus.service.app import risk_statement
from certus.triage.policy import Triage

SERVICE = "http://127.0.0.1:8000"
MANIFEST = Path("data/manifest/splits_v1.csv")
ROOT = Path("data/raw/sdnet2018")
BATCH = 48
COLUMNS = 8

st.set_page_config(page_title="Certus triage console", layout="wide")


@st.cache_data(show_spinner=False)
def calibration(cluster: str) -> dict[str, Any]:
    return httpx.get(f"{SERVICE}/v1/calibration/{cluster}", timeout=10).json()


@st.cache_data(show_spinner="Inspecting the batch...")
def inspect_batch(site_type: str, seed: int) -> list[dict[str, Any]]:
    """A random batch of held-out (test-split) images of one structure type,
    each inspected once by the service."""
    records = [
        r
        for r in read_manifest(MANIFEST, "sdnet2018")
        if r.split == "test" and r.site_id.startswith(site_type)
    ]
    rng = np.random.default_rng(seed)
    out = []
    with httpx.Client(timeout=60) as client:
        for i in rng.choice(len(records), BATCH, replace=False):
            r = records[int(i)]
            data = (ROOT / r.relpath).read_bytes()
            res = client.post(
                f"{SERVICE}/v1/inspect",
                files={"image": (r.relpath, data, "image/jpeg")},
                data={"site_id": r.site_id, "epsilon": "0.05"},
            ).json()
            out.append({**res, "image": data, "truth": r.defect_positive})
    return out


def thumbnail(data: bytes, defect: bool) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((96, 96))
    return ImageOps.expand(img, border=4, fill=(214, 40, 40) if defect else (230, 230, 226))


def grid(items: list[dict[str, Any]], title: str) -> None:
    st.markdown(f"**{title}** ({len(items)})")
    for start in range(0, len(items), COLUMNS):
        cols = st.columns(COLUMNS)
        for col, it in zip(cols, items[start : start + COLUMNS], strict=False):
            col.image(thumbnail(it["image"], it["truth"]), width="stretch")
            col.caption(f"s = {it['score']:.2f}" + (" · shift" if it["shift"]["flagged"] else ""))


st.title("Certus triage console")
try:
    httpx.get(f"{SERVICE}/v1/calibration/pooled", timeout=3).raise_for_status()
except httpx.HTTPError:
    st.error(f"The triage service is not reachable at {SERVICE}. Start it first.")
    st.stop()

top = st.columns([2, 1, 1, 1, 1])
site_type = top[1].selectbox("Structure type", ("wall", "deck", "pavement"))
seed = int(top[2].number_input("Batch", min_value=0, value=0, step=1))
per_day = int(top[3].number_input("Images per day", min_value=1, value=2000, step=100))
cal = calibration(site_type)
eps_grid = [t["epsilon"] for t in cal["thresholds"]]
epsilon = top[0].select_slider(
    "Guaranteed maximum missed-defect rate (ε)",
    options=sorted(eps_grid),
    value=0.05,
    format_func=lambda e: f"{e:.0%}",
)
lam = next(t["lambda_hat"] for t in cal["thresholds"] if t["epsilon"] == epsilon)
lam = float("inf") if lam is None else lam

batch = inspect_batch(site_type, seed)
cleared = [b for b in batch if b["score"] > lam and not b["shift"]["flagged"]]
human = [b for b in batch if not (b["score"] > lam and not b["shift"]["flagged"])]
hrf = len(human) / len(batch)
missed = sum(b["truth"] for b in cleared)
minutes = top[4].number_input("Minutes per review", min_value=0.1, value=1.5)

m = st.columns(4)
m[0].metric("Sent to a human (HRF)", f"{hrf:.0%}")
m[1].metric("Auto-cleared", f"{len(cleared)} of {len(batch)}")
m[2].metric("Projected review hours / day", f"{per_day * hrf * minutes / 60:.1f}")
m[3].metric(
    "Defects auto-cleared in this batch",
    f"{missed} of {sum(b['truth'] for b in batch)}",
    help="Ground truth, shown for the demo. The guarantee is an average over"
    " many batches, so one batch can exceed ε.",
)

left, right = st.columns(2)
with left:
    grid(cleared, "Auto-cleared")
with right:
    grid(human, "To a human")

st.divider()
st.markdown(
    "**Risk statement.** "
    + risk_statement(
        Triage.AUTO_CLEAR, site_type, epsilon, cal["n_defects"], False, ""
    ).removeprefix("Auto-cleared. ")
    + f" Calibration {cal['calibration_version']}, {cal['n_cal']} images."
    " Red border: defect in the ground truth."
)
