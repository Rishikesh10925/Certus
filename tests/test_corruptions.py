"""Covers data/corruptions.py."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from certus.data.corruptions import KINDS, corrupt


def _image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(np.uint8(rng.integers(60, 200, (64, 64, 3))))


def _distance(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


@pytest.mark.parametrize("kind", KINDS)
def test_keeps_size_and_mode_and_is_deterministic(kind: str) -> None:
    img = _image()
    out = corrupt(img, kind, 3, "a.jpg")
    assert out.size == img.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(out), np.asarray(corrupt(img, kind, 3, "a.jpg")))
    assert _distance(out, img) > 0


@pytest.mark.parametrize("kind", KINDS)
def test_higher_severity_moves_further_from_the_original(kind: str) -> None:
    img = _image()
    assert _distance(corrupt(img, kind, 5, "a.jpg"), img) > _distance(
        corrupt(img, kind, 1, "a.jpg"), img
    )


def test_low_light_darkens() -> None:
    img = _image()
    assert np.asarray(corrupt(img, "low_light", 5, "a.jpg")).mean() < np.asarray(img).mean()


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="severity"):
        corrupt(_image(), "blur", 0, "a.jpg")
    with pytest.raises(ValueError, match="unknown"):
        corrupt(_image(), "snow", 1, "a.jpg")
