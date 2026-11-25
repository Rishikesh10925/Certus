"""The web console's read-only endpoints (service/web.py), on the fake
calibration and scorer from test_service.py: no model, a tiny manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.test_service import FakeScorer, _calibration

from certus.data.splits import read_manifest
from certus.service.app import create_app, risk_statement
from certus.service.web import draw_batch, mount_web
from certus.triage.policy import Triage

FIELDS = ["dataset", "relpath", "site_id", "camera_id", "split", "defect_positive", "classes"]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "raw"
    rows = []
    for i in range(30):
        site = ("wall", "deck", "pavement")[i % 3]
        rel = f"{site[0].upper()}/img{i}.jpg"
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_bytes(b"jpeg%d" % i)
        split = "test" if i < 24 else "cal"
        rows.append(["sdnet2018", rel, f"{site}-1", "", split, str(i % 4 == 0), ""])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*FIELDS, "content_hash"])
        w.writerows(r + ["x"] for r in rows)
    results = tmp_path / "results"
    (results / "e2").mkdir(parents=True)
    (results / "e2" / "e2_scores.json").write_text('{"a": NaN, "b": [1.5, Infinity]}')
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "F1_system.png").write_bytes(b"\x89PNG")
    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>console</html>")
    (static / "assets" / "app.js").write_text("console.log(1)")
    return {
        "raw": raw,
        "manifest": manifest,
        "results": results,
        "figures": figures,
        "static": static,
    }


@pytest.fixture
def client(env: dict[str, Path]) -> TestClient:
    app = create_app(_calibration(), FakeScorer(), audit_rate=1.0)
    mount_web(
        app,
        manifest=env["manifest"],
        data_root=env["raw"],
        results_dir=env["results"],
        figures_dir=env["figures"],
        static_dir=env["static"],
    )
    return TestClient(app)


def test_info_reports_versions(client: TestClient) -> None:
    body = client.get("/v1/info").json()
    assert body["calibration_version"] == "test"
    assert body["model_version"] == "fake"
    assert body["epsilons"] == [0.2, 0.1, 0.05, 0.02, 0.01]
    assert body["samples_available"] is True
    assert body["escalate_on_shift"] is True


def test_samples_are_test_split_of_one_type(client: TestClient) -> None:
    body = client.get("/v1/samples", params={"site_type": "wall", "seed": 3, "n": 5}).json()
    assert len(body["items"]) == 5
    assert all(it["site_id"].startswith("wall") for it in body["items"])
    again = client.get("/v1/samples", params={"site_type": "wall", "seed": 3, "n": 5}).json()
    assert again == body


def test_batch_matches_the_old_console_sampling(env: dict[str, Path]) -> None:
    records = read_manifest(env["manifest"], "sdnet2018")
    pool = [r for r in records if r.split == "test" and r.site_id.startswith("deck")]
    idx = np.random.default_rng(7).choice(len(pool), 4, replace=False)
    assert draw_batch(records, "deck", 7, 4) == [pool[int(i)] for i in idx]


def test_samples_reject_bad_input(client: TestClient) -> None:
    assert client.get("/v1/samples", params={"site_type": "roof"}).status_code == 422
    assert client.get("/v1/samples", params={"n": 0}).status_code == 422


def test_sample_image_only_serves_manifest_test_images(client: TestClient) -> None:
    rel = client.get("/v1/samples", params={"n": 1}).json()["items"][0]["relpath"]
    assert client.get("/v1/samples/image", params={"relpath": rel}).status_code == 200
    for bad in ("../manifest.csv", "W/img24.jpg"):  # traversal; a cal-split image
        assert client.get("/v1/samples/image", params={"relpath": bad}).status_code == 404


def test_samples_unavailable_without_data(env: dict[str, Path]) -> None:
    app = mount_web(create_app(_calibration(), FakeScorer()), results_dir=env["results"])
    c = TestClient(app)
    assert c.get("/v1/info").json()["samples_available"] is False
    assert c.get("/v1/samples").status_code == 503


def test_risk_statement_is_the_service_text(client: TestClient) -> None:
    got = client.get(
        "/v1/risk-statement", params={"cluster": "wall", "epsilon": 0.02, "decision": "AUTO_CLEAR"}
    ).json()["risk_statement"]
    n = _calibration().clusters["wall"].n_defects
    assert got == risk_statement(Triage.AUTO_CLEAR, "wall", 0.02, n, False, "")
    assert client.get("/v1/risk-statement?cluster=roof&epsilon=0.1").status_code == 404
    assert client.get("/v1/risk-statement?cluster=wall&epsilon=0.1&decision=X").status_code == 422


def test_research_files_are_whitelisted_and_finite(client: TestClient) -> None:
    index = client.get("/v1/research").json()
    assert index["files"]["e2_scores"] is True and index["files"]["e3_efficiency"] is False
    assert index["figures"] == ["F1_system"]
    r = client.get("/v1/research/e2_scores")
    assert json.loads(r.text) == {"a": None, "b": [1.5, None]}
    assert client.get("/v1/research/e3_efficiency").status_code == 404
    assert client.get("/v1/research/..%2Fsecrets").status_code == 404
    assert client.get("/v1/research/figures/F1_system").status_code == 200
    assert client.get("/v1/research/figures/F9_latency").status_code == 404


def test_static_console_and_spa_fallback(client: TestClient) -> None:
    assert client.get("/").text == "<html>console</html>"
    assert client.get("/triage").text == "<html>console</html>"
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/v1/nope").status_code == 404
    # The triage contract still answers alongside the console.
    assert client.get("/v1/calibration/wall").status_code == 200
