"""Synthetic corruptions, the controlled shift probe (blueprint data table):
blur, low light, JPEG artefacts, rain, motion, in ImageNet-C style with
severities 1-5. Unlike site shift, the amount of shift is known, and labels
are unchanged by construction.

Randomness is seeded from (relpath, kind, severity), so a corrupted image
never depends on which other images were processed with it.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import convolve

KINDS = ("blur", "low_light", "jpeg", "rain", "motion")

BLUR_RADIUS = (1.0, 1.5, 2.5, 4.0, 6.0)
LIGHT_GAIN = (0.6, 0.45, 0.3, 0.2, 0.1)
JPEG_QUALITY = (25, 18, 15, 10, 7)
RAIN_STREAKS = (60, 120, 200, 300, 450)
MOTION_LENGTH = (5, 9, 13, 19, 27)


def _rng_for(relpath: str, kind: str, severity: int) -> np.random.Generator:
    key = f"{relpath}|{kind}|{severity}".encode()
    return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:16], 16))


def _low_light(image: Image.Image, gain: float) -> Image.Image:
    a = np.asarray(image, dtype=np.float32) / 255.0
    return Image.fromarray(np.uint8(np.clip(a**1.5 * gain, 0.0, 1.0) * 255.0))


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _rain(image: Image.Image, streaks: int, rng: np.random.Generator) -> Image.Image:
    w, h = image.size
    layer = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(layer)
    xs, ys = rng.integers(0, w, streaks), rng.integers(0, h, streaks)
    lengths = rng.integers(8, 24, streaks)
    for x, y, n in zip(xs.tolist(), ys.tolist(), lengths.tolist(), strict=True):
        draw.line([(x, y), (x + n // 5, y + n)], fill=200, width=1)
    layer = layer.filter(ImageFilter.GaussianBlur(0.5))
    return Image.composite(Image.new("RGB", image.size, (225, 225, 235)), image, layer)


def _motion(image: Image.Image, length: int, rng: np.random.Generator) -> Image.Image:
    angle = rng.uniform(0.0, np.pi)
    kernel: NDArray[np.float32] = np.zeros((length, length), dtype=np.float32)
    mid = (length - 1) / 2
    for t in np.linspace(-mid, mid, 2 * length):
        kernel[round(mid + t * np.sin(angle)), round(mid + t * np.cos(angle))] = 1.0
    kernel /= kernel.sum()
    a = np.asarray(image, dtype=np.float32)
    out = np.stack([convolve(a[..., c], kernel, mode="nearest") for c in range(3)], axis=-1)
    return Image.fromarray(np.uint8(np.clip(out, 0.0, 255.0)))


def corrupt(image: Image.Image, kind: str, severity: int, relpath: str) -> Image.Image:
    """The corrupted copy of an RGB image at severity 1-5. Size is kept."""
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be 1-5, got {severity}")
    i = severity - 1
    rng = _rng_for(relpath, kind, severity)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS[i]))
    if kind == "low_light":
        return _low_light(image, LIGHT_GAIN[i])
    if kind == "jpeg":
        return _jpeg(image, JPEG_QUALITY[i])
    if kind == "rain":
        return _rain(image, RAIN_STREAKS[i], rng)
    if kind == "motion":
        return _motion(image, MOTION_LENGTH[i], rng)
    raise ValueError(f"unknown corruption {kind!r}, expected one of {KINDS}")
