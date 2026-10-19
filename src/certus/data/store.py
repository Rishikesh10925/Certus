"""The SQLite `scores` table (roadmap Section 3.6) -- every nonconformity
score is computed once and cached here, keyed by image, score id, and the
model version that produced it, so every experiment reads identical
numbers instead of re-running the backbone. Other tables (calibrations,
decisions, audit_samples) join this file as the stages that need them
land.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    dataset TEXT NOT NULL,
    relpath TEXT NOT NULL,
    score_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    value REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (dataset, relpath, score_id, model_version)
)
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def write_scores(
    conn: sqlite3.Connection,
    records: Sequence[ImageRecord],
    score_id: str,
    model_version: str,
    values: NDArray[np.float64],
) -> None:
    if len(records) != len(values):
        raise ValueError(f"{len(records)} records but {len(values)} values")
    if not np.all(np.isfinite(values)):
        raise ValueError("refusing to cache non-finite scores")
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO scores (dataset, relpath, score_id, model_version, value) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (r.dataset, r.relpath, score_id, model_version, float(v))
                for r, v in zip(records, values, strict=True)
            ],
        )


def read_scores(
    conn: sqlite3.Connection, records: Sequence[ImageRecord], score_id: str, model_version: str
) -> NDArray[np.float64]:
    """Cached values for `records`, in order. Raises if any is missing --
    a silently partial score vector would quietly shrink n_cal."""
    rows = conn.execute(
        "SELECT dataset, relpath, value FROM scores WHERE score_id = ? AND model_version = ?",
        (score_id, model_version),
    ).fetchall()
    cache = {(d, p): v for d, p, v in rows}
    missing = [r.relpath for r in records if (r.dataset, r.relpath) not in cache]
    if missing:
        raise KeyError(f"{len(missing)} records have no cached {score_id}@{model_version} score")
    return np.array([cache[(r.dataset, r.relpath)] for r in records], dtype=np.float64)


def model_versions(conn: sqlite3.Connection, score_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT model_version FROM scores WHERE score_id = ? ORDER BY model_version",
        (score_id,),
    ).fetchall()
    return [r[0] for r in rows]
