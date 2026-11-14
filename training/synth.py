"""
Synthetic moon photo renderer.

Used to (1) test the segmentation pipeline against known ground truth and
(2) pre-train the phase CNN when you don't have real labelled photos yet.

Physics: orthographic sphere lit by the Sun at phase angle alpha, with a
Lommel-Seeliger reflectance  I = mu0 / (mu0 + mu)  (good for the lunar
regolith: a full Moon looks like a flat disk, no limb darkening), a random
albedo map (maria + craters), then a crude "phone camera": random exposure
gain with clipping, sky glow, bloom, blur, sensor noise, sRGB gamma, JPEG.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CLASSES = [
    "new_moon", "waxing_crescent", "first_quarter", "waxing_gibbous",
    "full_moon", "waning_gibbous", "last_quarter", "waning_crescent",
]

# illuminated-fraction bands (must match backend/moon_cv/astro.py)
BANDS = {
    "new_moon": (0.0, 0.03),
    "crescent": (0.03, 0.40),
    "quarter": (0.40, 0.60),
    "gibbous": (0.60, 0.97),
    "full_moon": (0.97, 1.0),
}


def fraction_for_class(cls: str, rng: np.random.Generator, margin=0.02) -> tuple[float, bool]:
    waxing = cls.startswith("waxing") or cls == "first_quarter"
    if cls == "new_moon":
        return float(rng.uniform(0.0, 0.025)), bool(rng.random() < 0.5)
    if cls == "full_moon":
        return float(rng.uniform(0.975, 1.0)), bool(rng.random() < 0.5)
    key = "crescent" if "crescent" in cls else "quarter" if "quarter" in cls else "gibbous"
    lo, hi = BANDS[key]
    return float(rng.uniform(lo + margin, hi - margin)), waxing


def albedo_map(n: int, rng: np.random.Generator) -> np.ndarray:
    a = np.ones((n, n), np.float32)
    # maria: large soft dark blobs
    for _ in range(rng.integers(4, 9)):
        cx, cy = rng.uniform(0.15, 0.85, 2) * n
        rx, ry = rng.uniform(0.06, 0.22, 2) * n
        yy, xx = np.mgrid[0:n, 0:n]
        blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        a -= rng.uniform(0.15, 0.35) * blob.astype(np.float32)
    # craters / rays: small bright and dark dots
    for _ in range(rng.integers(20, 60)):
        c = tuple(int(v) for v in rng.uniform(0, n, 2))
        rr = int(max(1, rng.uniform(0.004, 0.03) * n))
        cv2.circle(a, c, rr, float(rng.uniform(0.75, 1.15)), -1)
    noise = cv2.GaussianBlur(rng.normal(0, 0.06, (n, n)).astype(np.float32), (0, 0), n * 0.01 + 0.5)
    a = cv2.GaussianBlur(a, (0, 0), max(0.6, n * 0.004)) + noise
    return np.clip(a, 0.35, 1.2)


def render_disk(n: int, fraction: float, waxing: bool, rng: np.random.Generator,
                earthshine: float = 0.0) -> np.ndarray:
    """Linear radiance image (n x n) of the lunar disk, lit side right if waxing."""
    alpha = math.acos(2 * fraction - 1)          # phase angle
    sx = math.sin(alpha) * (1 if waxing else -1)
    sz = math.cos(alpha)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (xx + 0.5) / n * 2 - 1
    y = (yy + 0.5) / n * 2 - 1
    rho2 = x * x + y * y
    inside = rho2 <= 1
    z = np.sqrt(np.clip(1 - rho2, 0, 1))
    mu0 = x * sx + z * sz
    mu = z
    I = np.where(mu0 > 0, mu0 / (mu0 + mu + 1e-6), 0.0)
    I = I * albedo_map(n, rng) * 2.0              # full moon ~ 1.0
    I = I + earthshine * inside
    # anti-aliased edge
    edge = np.clip((1 - np.sqrt(rho2)) * n / 2, 0, 1)
    return (I * edge).astype(np.float32)


def camera(linear_rgb: np.ndarray, rng: np.random.Generator, jpeg=True) -> np.ndarray:
    img = linear_rgb
    # bloom / atmospheric glow
    glow = cv2.GaussianBlur(img, (0, 0), max(1.0, img.shape[0] * rng.uniform(0.01, 0.04)))
    img = img + rng.uniform(0.0, 0.25) * glow
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.4))
    img = img + rng.normal(0, rng.uniform(0.002, 0.015), img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    srgb = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)
    u8 = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        u8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return u8


def _compose(disk_lin: np.ndarray, canvas: int, cx: float, cy: float, rng: np.random.Generator,
             rotate_deg: float) -> np.ndarray:
    n = disk_lin.shape[0]
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rotate_deg, 1.0)
    disk_lin = cv2.warpAffine(disk_lin, M, (n, n))
    sky = np.zeros((canvas, canvas), np.float32)
    # sky gradient (twilight / light pollution)
    level = rng.uniform(0.0, 0.03) if rng.random() < 0.7 else rng.uniform(0.03, 0.12)
    yy = np.linspace(0, 1, canvas, dtype=np.float32)[:, None]
    sky += level * (0.5 + yy) * np.ones((1, canvas), np.float32)
    x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
    out = sky.copy()
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(canvas, x0 + n), min(canvas, y0 + n)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] += disk_lin[ys0 - y0: ys1 - y0, xs0 - x0: xs1 - x0]
    gain = rng.uniform(0.5, 2.5) if rng.random() < 0.8 else rng.uniform(2.5, 6.0)  # sometimes blown out
    out = out * gain
    tint = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.92, 1.0), rng.uniform(0.95, 1.05)], np.float32)  # BGR
    return out[..., None] * tint[None, None, :]


def render_photo(fraction: float, waxing: bool, rng: np.random.Generator, size=512,
                 rotate_deg: float = 0.0):
    """Full synthetic photo. Returns (bgr_u8, meta)."""
    r = rng.uniform(0.06, 0.35) * size
    n = int(2 * r) | 1
    cx = rng.uniform(r + 5, size - r - 5)
    cy = rng.uniform(r + 5, size - r - 5)
    es = rng.uniform(0.0, 0.01)
    disk = render_disk(n, fraction, waxing, rng, earthshine=es)
    lin = _compose(disk, size, cx, cy, rng, rotate_deg)
    img = camera(lin, rng)
    return img, {"center": (cx, cy), "radius": n / 2, "fraction": fraction, "waxing": waxing}


def render_crop(cls: str, rng: np.random.Generator, out=128) -> np.ndarray:
    """A training crop that mimics backend crop_disk(): disk ~ 1/1.2 of the half-side."""
    fraction, waxing = fraction_for_class(cls, rng)
    big = out * 2
    r = big / 2 / 1.2 * rng.uniform(0.88, 1.08)
    n = int(2 * r) | 1
    cx = big / 2 + rng.normal(0, big * 0.02)
    cy = big / 2 + rng.normal(0, big * 0.02)
    es = rng.uniform(0.0, 0.03) if cls == "new_moon" else rng.uniform(0, 0.01)
    disk = render_disk(n, fraction, waxing, rng, earthshine=es)
    if cls == "new_moon" and rng.random() < 0.4:
        disk[:] = 0.0                   # nothing visible at all
    rot = rng.normal(0, 18)            # photos are roughly upright
    lin = _compose(disk, big, cx, cy, rng, rot)
    img = camera(lin, rng)
    return cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA)
