"""
Three ways to estimate the Moon's brightness.

A. class_reference   - look up the typical brightness range for the
                       predicted phase class (Allen/Cox phase law).
B. pixel_measurement - measure the lit pixels from the segmentation mask.
                       Relative by default; converted to surface luminance
                       (cd/m^2) when the photo's EXIF has exposure settings.
C. photometric_model - continuous estimate: measured illuminated fraction
                       -> phase angle -> apparent magnitude -> illuminance
                       (lux) on the ground. Independent of camera exposure.
                       Assumes the mean Earth-Moon distance.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from . import astro

# Reflected-light meter calibration constant (ISO 2720), cd/m^2 * s / ISO
METER_K = 12.5
MID_GREY = 0.18
# Typical full-Moon surface luminance used as a sanity reference
FULL_MOON_LUMINANCE = 2500.0   # cd/m^2 (order of magnitude)


def pixel_measurement(seg, exif: dict | None) -> dict:
    lin = seg.linear
    lit = seg.mask > 0
    vals = lin[lit]
    gray_vals = seg.gray[lit]
    sky = lin[~lit]
    sky_level = float(np.percentile(sky, 50)) if sky.size else 0.0

    mean_lin = float(vals.mean())
    p95 = float(np.percentile(vals, 95))
    saturated = float((gray_vals >= 250).mean())
    integrated = float(np.clip(vals - sky_level, 0, None).sum())       # "total light" in pixel units
    # relative brightness index 0..100: mean lit luminance * lit fraction
    index = 100.0 * mean_lin * seg.fraction

    out = {
        "method": "pixel_measurement",
        "mean_linear_luminance": round(mean_lin, 4),
        "p95_linear_luminance": round(p95, 4),
        "mean_gray_0_255": round(float(gray_vals.mean()), 1),
        "sky_linear_luminance": round(sky_level, 4),
        "contrast_vs_sky": round(mean_lin / max(sky_level, 1e-4), 1),
        "saturated_percent": round(100 * saturated, 1),
        "integrated_flux_pixels": round(integrated, 1),
        "relative_index_0_100": round(index, 1),
        "calibrated": False,
        "warnings": [],
    }
    if saturated > 0.2:
        out["warnings"].append(
            f"{out['saturated_percent']}% of moon pixels are clipped (overexposed): "
            "pixel brightness is a lower bound. Lower the exposure for a better measurement.")

    # Absolute calibration from EXIF (if available)
    if exif and all(exif.get(k) for k in ("exposure_time", "f_number", "iso")):
        t, N, S = exif["exposure_time"], exif["f_number"], exif["iso"]
        # pixel value at mid-grey corresponds to L = K N^2 / (t S)
        L_mid = METER_K * N * N / (t * S)
        L_moon = L_mid * mean_lin / MID_GREY
        out.update({
            "calibrated": True,
            "surface_luminance_cd_m2": round(L_moon, 1),
            "exposure": {"t": t, "N": N, "ISO": S},
            "note": "Approximate: assumes a standard light-meter calibration and no in-camera "
                    "tone mapping. Typical full-Moon surface luminance is a few thousand cd/m^2.",
        })
    return out


def photometric_model(fraction: float) -> dict:
    alpha = astro.fraction_to_phase_angle(fraction)
    m = astro.magnitude_from_phase_angle(alpha)
    return {
        "method": "photometric_model",
        "illuminated_fraction": round(fraction, 4),
        "phase_angle_deg": round(alpha, 1),
        "magnitude": round(m, 2),
        "lux": astro.magnitude_to_lux(m),
        "percent_of_full": round(100 * astro.relative_to_full(m), 2),
        "note": "Brightness falls much faster than the lit area: a half Moon is only "
                "~9% as bright as a full Moon (shadows + opposition surge).",
    }


def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def visualisations(seg) -> dict:
    """Overlay (mask + fitted disk) and a brightness heat-map, cropped to the moon."""
    img = seg.image.copy()
    cx, cy, r = seg.center[0], seg.center[1], seg.radius
    overlay = img.copy()
    tint = np.zeros_like(img)
    tint[seg.mask > 0] = (40, 200, 255)
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)
    cnts, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lw = max(1, int(r / 60))
    cv2.drawContours(overlay, cnts, -1, (0, 220, 255), lw)
    cv2.circle(overlay, (int(cx), int(cy)), int(r), (255, 160, 60), lw, cv2.LINE_AA)
    cv2.drawMarker(overlay, (int(cx), int(cy)), (255, 160, 60), cv2.MARKER_CROSS, int(r / 6) + 4, lw)

    # heat-map of linear luminance inside the disk
    lin = seg.linear
    norm = np.clip(lin / max(float(np.percentile(lin[seg.mask > 0], 99.5)), 1e-4), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    disk = np.zeros(lin.shape, np.uint8)
    cv2.circle(disk, (int(cx), int(cy)), int(r), 255, -1)
    heat[disk == 0] = (heat[disk == 0] * 0.25).astype(np.uint8)
    cv2.circle(heat, (int(cx), int(cy)), int(r), (255, 255, 255), lw, cv2.LINE_AA)

    def crop(im):
        h, w = im.shape[:2]
        pad = 1.35 * r
        x0, x1 = int(max(0, cx - pad)), int(min(w, cx + pad))
        y0, y1 = int(max(0, cy - pad)), int(min(h, cy + pad))
        c = im[y0:y1, x0:x1]
        s = 360 / max(c.shape[:2])
        return cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)

    return {"overlay_png": _png_b64(crop(overlay)), "heatmap_png": _png_b64(crop(heat))}
