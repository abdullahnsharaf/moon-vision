"""
Moon segmentation and disk fitting (classical computer vision, OpenCV).

Pipeline
--------
1. Convert to grey + linear luminance (undo sRGB gamma).
2. Threshold the bright object (Otsu, guarded by a percentile floor).
3. Keep the brightest compact connected component -> lit mask.
4. Fit the full lunar disk (including the unlit part!) to the *limb*:
     * the two terminator endpoints (cusps) are always diametrically
       opposite, so the farthest pair of lit points gives a first
       centre/radius estimate;
     * then iteratively refit a least-squares circle using only contour
       points that lie on that circle (the limb), ignoring the terminator.
5. Illuminated fraction  k = lit area / disk area.
6. Lit-side direction = angle from disk centre to the lit-area centroid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_SIDE = 1024


@dataclass
class Segmentation:
    found: bool
    reason: str = ""
    scale: float = 1.0                 # working image = original * scale
    image: np.ndarray | None = None    # working BGR image
    gray: np.ndarray | None = None
    linear: np.ndarray | None = None   # linear luminance 0..1
    mask: np.ndarray | None = None     # uint8 0/255 lit pixels
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    lit_area: int = 0
    disk_area: float = 0.0
    fraction: float = 0.0
    lit_angle_deg: float = 0.0         # 0 = lit side to the right, 90 = up
    lit_offset: float = 0.0            # |centroid - centre| / r
    threshold: int = 0
    fit_residual: float = 0.0
    extras: dict = field(default_factory=dict)


def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _fit_circle_lsq(pts: np.ndarray) -> tuple[float, float, float]:
    """Kasa algebraic circle fit. pts: (N,2)."""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return float(cx), float(cy), float(r)


def _farthest_pair(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32)).reshape(-1, 2).astype(np.float64)
    if len(hull) > 400:
        hull = hull[:: int(np.ceil(len(hull) / 400))]
    d = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return hull[i], hull[j], float(d[i, j])


def fit_disk(contour_pts: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    """Return (cx, cy, r, residual) of the full lunar disk."""
    p1, p2, dmax = _farthest_pair(contour_pts)
    cx, cy = (p1 + p2) / 2
    r = dmax / 2
    pts = contour_pts.astype(np.float64)

    resid = 0.0
    for tol in (0.10, 0.06, 0.04):
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        limb = pts[np.abs(dist - r) < tol * r]
        if len(limb) < 12:
            break
        ncx, ncy, nr = _fit_circle_lsq(limb)
        # Guard: the disk can never be smaller than the farthest-pair half-chord
        if nr < 0.9 * dmax / 2 or nr > 1.6 * dmax / 2:
            break
        cx, cy, r = ncx, ncy, nr
        resid = float(np.std(np.hypot(limb[:, 0] - cx, limb[:, 1] - cy) - r) / r)

    # Make sure (nearly) all lit pixels are inside the disk
    ys, xs = np.nonzero(mask)
    if len(xs):
        d = np.hypot(xs - cx, ys - cy)
        r_out = float(np.percentile(d, 99.5))
        if r * 1.03 < r_out < r * 1.25:
            r = r_out
    return float(cx), float(cy), float(r), resid


def segment_moon(bgr: np.ndarray) -> Segmentation:
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    img = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    linear = srgb_to_linear(gray)
    seg = Segmentation(found=False, scale=scale, image=img, gray=gray, linear=linear)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    p999 = float(np.percentile(blur, 99.9))
    p50 = float(np.percentile(blur, 50))
    if p999 < 40 or p999 - p50 < 25:
        seg.reason = "No bright object found - the sky looks empty (new moon, clouds, or underexposed photo)."
        return seg

    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The moon is usually a small part of the frame; guard Otsu with a floor
    # halfway between the sky level and the moon's bright end.
    t = int(max(otsu, p50 + 0.35 * (p999 - p50)))
    seg.threshold = t
    _, bw = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    best, best_score = -1, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 40:
            continue
        x, y, bw_, bh_ = stats[i, :4]
        edges = int(x <= 1) + int(y <= 1) + int(x + bw_ >= bw.shape[1] - 1) + int(y + bh_ >= bw.shape[0] - 1)
        if edges >= 2:          # bright sky band / horizon glow, not the moon
            continue
        comp = labels == i
        score = float(linear[comp].sum())
        if score > best_score:
            best, best_score = i, score
    if best < 0:
        seg.reason = "Bright pixels were found but no moon-sized object."
        return seg

    mask = np.where(labels == best, 255, 0).astype(np.uint8)
    # Fill holes (dark maria inside a bright full moon)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cx, cy, r, resid = fit_disk(cnt.reshape(-1, 2), mask)
    disk = np.zeros_like(mask)
    cv2.circle(disk, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    mask = cv2.bitwise_and(mask, disk)

    lit_area = int(np.count_nonzero(mask))
    disk_area = float(np.pi * r * r)
    frac = float(min(1.0, lit_area / disk_area))

    m = cv2.moments(mask, binaryImage=True)
    gx, gy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (cx, cy)
    dx, dy = gx - cx, -(gy - cy)        # y up
    angle = float(np.degrees(np.arctan2(dy, dx)))

    seg.found = True
    seg.mask = mask
    seg.center = (cx, cy)
    seg.radius = r
    seg.lit_area = lit_area
    seg.disk_area = disk_area
    seg.fraction = frac
    seg.lit_angle_deg = angle
    seg.lit_offset = float(np.hypot(dx, dy) / r)
    seg.fit_residual = resid
    return seg


def crop_disk(img: np.ndarray, center, radius, out=128, pad=1.2) -> np.ndarray:
    """Square crop around the disk (side = 2*pad*r), resized to out x out."""
    cx, cy = center
    half = radius * pad
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    h, w = img.shape[:2]
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA)
