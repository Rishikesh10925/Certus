"""Covers the service: the calibration artifact (round trip, clusters, the
shift check) and every endpoint of the contract, with a fake scorer so no
model is loaded."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from certus.service import calibration as calib
from certus.service.app import create_app
from certus.service.scorer import Scored

DIM = 8


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def _calibration(seed: int = 0) -> calib.Calibration:
    rng = np.random.default_rng(seed)
    clusters = {}
    for name in ("wall", calib.POOLED):
        n = 400
        y = rng.random(n) < 0.3
        p = np.clip(np.where(y, rng.normal(0.7, 0.15, n), rng.normal(0.3, 0.15, n)), 0, 1)
        # Calibration images cluster around the first axis.
        feats = _unit(np.eye(DIM)[0] + 0.1 * rng.normal(size=(n, DIM)))
        clusters[name] = calib.calibrate_cluster(name, feats, p, y)
    return calib.Calibration("test", "fake", clusters)


class FakeScorer:
    """The image bytes are b"<p_defect>" or b"far:<p_defect>"; "far"
    images sit away from every calibration feature."""

    model_version = "fake"

    def score(self, image_bytes: bytes) -> Scored:
        text = image_bytes.decode()
        far = text.startswith("far:")
        p = float(text.removeprefix("far:"))
        axis = np.eye(DIM)[1 if far else 0].astype(np.float32)
        return Scored(feature=axis, p_defect=p, defect_type="crack", timings_ms={"embed": 1.0})


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_calibration(), FakeScorer(), audit_rate=1.0))


def _inspect(client: TestClient, payload: str, site: str = "wall-7001", eps: float = 0.05):
    return client.post(
        "/v1/inspect",
        files={"image": ("x.jpg", payload.encode(), "image/jpeg")},
        data={"site_id": site, "epsilon": str(eps)},
    )


def test_artifact_round_trips(tmp_path: Path) -> None:
    cal = _calibration()
    calib.save(cal, tmp_path / "cal")
    back = calib.load(tmp_path / "cal")
    assert back.version == cal.version
    for name, c in cal.clusters.items():
        b = back.clusters[name]
        assert b.lam == c.lam and b.q == c.q and b.n_defects == c.n_defects
        np.testing.assert_array_equal(b.features, c.features)


def test_sites_map_to_clusters() -> None:
    cal = _calibration()
    assert cal.cluster_for("wall-7001") == "wall"
    assert cal.cluster_for("Wall") == "wall"
    assert cal.cluster_for("deck-1") == calib.POOLED  # no deck cluster here
    assert cal.cluster_for("unknown") == calib.POOLED


def test_shift_threshold_flags_about_the_requested_share() -> None:
    c = _calibration().clusters["wall"]
    sims = calib.loo_nearest_similarity(c.features)
    assert np.mean(sims < c.shift_threshold) <= 0.02
    assert c.is_shifted(np.eye(DIM, dtype=np.float32)[1])
    assert not c.is_shifted(c.features[0])


def test_lambda_is_monotone_in_epsilon() -> None:
    lam = _calibration().clusters["wall"].lam
    eps = sorted(lam)
    assert all(lam[a] >= lam[b] for a, b in zip(eps, eps[1:], strict=False))


def test_inspect_clears_clean_and_flags_defects(client: TestClient) -> None:
    clean = _inspect(client, "0.01").json()
    assert clean["decision"] == "AUTO_CLEAR"
    assert clean["cluster"] == "wall"
    assert "Auto-cleared" in clean["risk_statement"]
    assert "5%" in clean["risk_statement"]
    defect = _inspect(client, "0.99").json()
    assert defect["decision"] == "FLAG"
    assert defect["defect_type"] == "crack"
    assert defect["prediction_set"] == ["defect"]


def test_decision_matches_lambda(client: TestClient) -> None:
    lam = _calibration().clusters["wall"].lam[0.05]
    for p in np.linspace(0.0, 1.0, 21):
        out = _inspect(client, f"{p:.4f}").json()
        assert (out["decision"] == "AUTO_CLEAR") == (1.0 - float(f"{p:.4f}") > lam)


def test_shifted_image_is_never_auto_cleared(client: TestClient) -> None:
    out = _inspect(client, "far:0.01").json()
    assert out["shift"]["flagged"]
    assert out["decision"] != "AUTO_CLEAR"
    assert "does not cover" in out["risk_statement"]


def test_shift_check_can_be_switched_off() -> None:
    c = TestClient(create_app(_calibration(), FakeScorer(), escalate_on_shift=False))
    out = _inspect(c, "far:0.01").json()
    assert out["shift"]["flagged"] and out["decision"] == "AUTO_CLEAR"


def test_inspect_rejects_bad_input(client: TestClient) -> None:
    assert _inspect(client, "0.1", eps=0.07).status_code == 422
    assert _inspect(client, "not-a-number").status_code == 422


def test_calibration_endpoint(client: TestClient) -> None:
    out = client.get("/v1/calibration/wall").json()
    assert out["n_cal"] == 400
    assert [t["epsilon"] for t in out["thresholds"]] == sorted(calib.EPSILONS, reverse=True)
    assert client.get("/v1/calibration/nope").status_code == 404


def test_infinite_lambda_is_sent_as_null() -> None:
    assert calib.finite_or_none(math.inf) is None
    assert calib.finite_or_none(0.5) == 0.5


def test_feedback_audit_and_metrics(client: TestClient) -> None:
    cleared = _inspect(client, "0.01").json()
    _inspect(client, "0.99")
    audit = client.get("/v1/audit/next").json()
    assert audit["inspection_id"] == cleared["inspection_id"]  # only cleared images
    fb = client.post(
        "/v1/feedback", json={"inspection_id": cleared["inspection_id"], "defect": True}
    ).json()
    assert fb["recorded"] and fb["missed_defect"]
    assert "not triggered" in fb["recalibration"]
    assert client.get("/v1/audit/next").status_code == 204  # labelled, so done
    assert (
        client.post("/v1/feedback", json={"inspection_id": 999, "defect": False}).status_code == 404
    )
    text = client.get("/v1/metrics").text
    assert 'certus_inspections_total{decision="AUTO_CLEAR"} 1' in text
    assert 'certus_inspections_total{decision="FLAG"} 1' in text
    assert "certus_feedback_missed_defects_total 1" in text
    assert 'certus_inspect_latency_seconds{quantile="0.95"}' in text
