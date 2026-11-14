"""
End-to-end analysis: image bytes -> phase class + three brightness estimates.

Phase decision = fusion of two independent "experts":
  * the CNN (learned appearance)                 -> 8-class probabilities
  * geometry (segmentation + disk fit)           -> illuminated fraction k
                                                    + which side is lit
Both are collapsed into  P(band) x P(waxing)  where band is one of
new / crescent / quarter / gibbous / full, then averaged.

Everything comes from the picture itself. Waxing vs waning is read from
which side of the disk is lit, and that depends on where you stand:
  north  - lit on the RIGHT  = waxing (default)
  south  - lit on the LEFT   = waxing (image is mirrored before the CNN)
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ExifTags, ImageOps

from . import astro, brightness
from .cnn import PhaseCNN
from .segment import segment_moon, crop_disk

BANDS = ["new", "crescent", "quarter", "gibbous", "full"]
BAND_EDGES = [0.0, astro.BAND_NEW, astro.BAND_CRESCENT, astro.BAND_QUARTER, astro.BAND_FULL, 1.0]
CLASS_TO_BAND = {
    "new_moon": "new", "full_moon": "full",
    "waxing_crescent": "crescent", "waning_crescent": "crescent",
    "first_quarter": "quarter", "last_quarter": "quarter",
    "waxing_gibbous": "gibbous", "waning_gibbous": "gibbous",
}
WAXING_CLASSES = {"waxing_crescent", "first_quarter", "waxing_gibbous"}
WANING_CLASSES = {"waning_crescent", "last_quarter", "waning_gibbous"}

_cnn: PhaseCNN | None = None


def get_cnn() -> PhaseCNN:
    global _cnn
    if _cnn is None:
        _cnn = PhaseCNN()
    return _cnn


# ---------------------------------------------------------------- EXIF
def _ratio(v):
    try:
        return float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)
    except Exception:
        return None


def read_image_and_exif(data: bytes):
    pil = Image.open(io.BytesIO(data))
    exif_raw = {}
    try:
        ex = pil.getexif()
        exif_raw = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        exif_raw.update({ExifTags.TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()})
    except Exception:
        pass
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    exif = {
        "exposure_time": _ratio(exif_raw.get("ExposureTime")),
        "f_number": _ratio(exif_raw.get("FNumber")),
        "iso": _ratio(exif_raw.get("ISOSpeedRatings") or exif_raw.get("PhotographicSensitivity")),
        "camera": " ".join(str(exif_raw.get(k, "")).strip() for k in ("Make", "Model")).strip() or None,
    }
    return bgr, exif


# ---------------------------------------------------------------- fusion helpers
def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def geometry_band_probs(k: float, sigma: float = 0.035) -> dict[str, float]:
    p = {}
    for b, lo, hi in zip(BANDS, BAND_EDGES[:-1], BAND_EDGES[1:]):
        lo_ = -1e9 if b == "new" else lo
        hi_ = 1e9 if b == "full" else hi
        p[b] = _norm_cdf((hi_ - k) / sigma) - _norm_cdf((lo_ - k) / sigma)
    s = sum(p.values())
    return {b: v / s for b, v in p.items()}


def cnn_band_probs(p: dict[str, float]) -> tuple[dict[str, float], float | None]:
    bands = {b: 0.0 for b in BANDS}
    for c, v in p.items():
        bands[CLASS_TO_BAND[c]] += v
    wx = sum(p[c] for c in WAXING_CLASSES)
    wn = sum(p[c] for c in WANING_CLASSES)
    return bands, (wx / (wx + wn) if wx + wn > 1e-6 else None)


def compose_classes(band_p: dict[str, float], p_wax: float) -> dict[str, float]:
    out = {
        "new_moon": band_p["new"],
        "full_moon": band_p["full"],
        "waxing_crescent": band_p["crescent"] * p_wax,
        "waning_crescent": band_p["crescent"] * (1 - p_wax),
        "first_quarter": band_p["quarter"] * p_wax,
        "last_quarter": band_p["quarter"] * (1 - p_wax),
        "waxing_gibbous": band_p["gibbous"] * p_wax,
        "waning_gibbous": band_p["gibbous"] * (1 - p_wax),
    }
    s = sum(out.values())
    return {c: out[c] / s for c in astro.PHASE_CLASSES}


# ---------------------------------------------------------------- main entry
def analyze(data: bytes, hemisphere: str = "north", include_images: bool = True,
            client_exif: dict | None = None) -> dict:
    bgr, exif = read_image_and_exif(data)
    for k, v in (client_exif or {}).items():          # fill gaps from EXIF the browser read
        if v and not exif.get(k):
            exif[k] = float(v)

    # ---------- segmentation / geometry
    seg = segment_moon(bgr)
    result: dict = {"hemisphere": hemisphere, "warnings": [], "warning_codes": []}

    # ---------- which way is "waxing"? (from the picture + the observer's hemisphere)
    mode = "south" if hemisphere == "south" else "north"

    # ---------- CNN
    cnn = get_cnn()
    cnn_probs = None
    if cnn.available:
        crop = crop_disk(seg.image, seg.center, seg.radius) if seg.found else \
            cv2.resize(_center_square(bgr), (128, 128), interpolation=cv2.INTER_AREA)
        if mode == "south":
            crop = cv2.flip(crop, 1)
        cnn_probs = cnn.predict(crop)

    # ---------- fuse
    if not seg.found:
        geo_band = {b: (1.0 if b == "new" else 0.0) for b in BANDS}
        geo_wax = 0.5
        result["warning_codes"].append("no_moon")
        result["warnings"].append(seg.reason)
    else:
        geo_band = geometry_band_probs(seg.fraction)
        # horizontal component of the lit-side direction
        dx = math.cos(math.radians(seg.lit_angle_deg)) * min(1.0, seg.lit_offset * 3)
        geo_wax = 1 / (1 + math.exp(-8 * dx))           # lit right -> waxing (north)
        if mode == "south":
            geo_wax = 1 - geo_wax

    if cnn_probs:
        cnn_band, cnn_wax = cnn_band_probs(cnn_probs)
        band_p = {b: 0.5 * geo_band[b] + 0.5 * cnn_band[b] for b in BANDS}
    else:
        cnn_band, cnn_wax = None, None
        band_p = geo_band

    p_wax = geo_wax if cnn_wax is None else 0.5 * geo_wax + 0.5 * cnn_wax
    wax_src = f"which side is lit ({'Southern' if mode == 'south' else 'Northern'} Hemisphere view)"

    final = compose_classes(band_p, p_wax)
    phase = max(final, key=final.get)

    # ---------- brightness
    frac = seg.fraction if seg.found else 0.0
    method_a = astro.class_reference_brightness(phase)
    method_c = brightness.photometric_model(frac)
    method_b = brightness.pixel_measurement(seg, exif) if seg.found else None

    result.update({
        "phase": {
            "class": phase,
            "label": astro.PHASE_LABELS[phase],
            "confidence": round(final[phase], 3),
            "probabilities": {c: round(v, 4) for c, v in sorted(final.items(), key=lambda x: -x[1])},
            "waxing_decided_by": wax_src,
            "waxing_decided_by_code": mode,              # north | south
            "p_waxing": round(p_wax, 3),
        },
        "geometry": {
            "moon_found": seg.found,
            "illuminated_fraction": round(frac, 4),
            "phase_angle_deg": round(astro.fraction_to_phase_angle(frac), 1) if seg.found else None,
            "disk_center_px": [round(seg.center[0] / seg.scale, 1), round(seg.center[1] / seg.scale, 1)] if seg.found else None,
            "disk_radius_px": round(seg.radius / seg.scale, 1) if seg.found else None,
            "lit_side_angle_deg": round(seg.lit_angle_deg, 1) if seg.found else None,
            "threshold": seg.threshold,
            "fit_residual": round(seg.fit_residual, 4),
            "band_probabilities": {b: round(v, 4) for b, v in geo_band.items()},
        },
        "cnn": {
            "available": cnn.available,
            "probabilities": {c: round(v, 4) for c, v in sorted(cnn_probs.items(), key=lambda x: -x[1])} if cnn_probs else None,
            "top": max(cnn_probs, key=cnn_probs.get) if cnn_probs else None,
            "info": {k: cnn.info.get(k) for k in ("trained_on", "val_accuracy", "epochs")} if cnn.available else None,
        },
        "brightness": {
            "class_reference": method_a,
            "pixel_measurement": method_b,
            "photometric_model": method_c,
        },
        "exif": exif,
    })
    if method_b and method_b["warnings"]:
        result["warnings"] += method_b["warnings"]
        result["warning_codes"].append("overexposed")
    if include_images and seg.found:
        result["images"] = brightness.visualisations(seg)
    return result


def _center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
