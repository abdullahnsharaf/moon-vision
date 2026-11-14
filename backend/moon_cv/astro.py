"""
Astronomy helpers: lunar phase from a date, and the Moon's brightness
as a function of phase angle.

References
----------
* Meeus, J. "Astronomical Algorithms", 2nd ed., ch. 47-48
  (low-precision lunar position, phase angle, illuminated fraction).
* Allen's Astrophysical Quantities (Cox 2002) Moon phase law:
      m(alpha) = -12.73 + 0.026|alpha| + 4e-9 * alpha^4      (alpha in degrees)
  where alpha is the Sun-Moon-Earth phase angle (0 = full, 90 = quarter, 180 = new).
* Illuminated fraction of a sphere: k = (1 + cos(alpha)) / 2.
* Illuminance from a V magnitude: E[lux] = 2.54e-6 * 10^(-0.4 m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

MEAN_DISTANCE_KM = 385_000.56
FULL_MOON_MAG = -12.73
LUX_ZERO_MAG = 2.54e-6          # lux delivered by a magnitude-0 source

PHASE_CLASSES = [
    "new_moon",
    "waxing_crescent",
    "first_quarter",
    "waxing_gibbous",
    "full_moon",
    "waning_gibbous",
    "last_quarter",
    "waning_crescent",
]

PHASE_LABELS = {
    "new_moon": "New Moon",
    "waxing_crescent": "Waxing Crescent",
    "first_quarter": "First Quarter",
    "waxing_gibbous": "Waxing Gibbous",
    "full_moon": "Full Moon",
    "waning_gibbous": "Waning Gibbous",
    "last_quarter": "Last Quarter",
    "waning_crescent": "Waning Crescent",
}

# Illuminated-fraction bands used to turn a continuous fraction into a class.
# (Quarters are "about half lit"; full/new are "almost entirely lit/dark".)
BAND_NEW = 0.03
BAND_CRESCENT = 0.40
BAND_QUARTER = 0.60
BAND_FULL = 0.97


def _deg(x: float) -> float:
    return math.degrees(x)


def _rad(x: float) -> float:
    return math.radians(x)


# --------------------------------------------------------------------------
# Phase <-> fraction <-> brightness
# --------------------------------------------------------------------------
def fraction_to_phase_angle(k: float) -> float:
    """Illuminated fraction (0..1) -> phase angle in degrees (0 full .. 180 new)."""
    k = min(max(k, 0.0), 1.0)
    return _deg(math.acos(2.0 * k - 1.0))


def phase_angle_to_fraction(alpha_deg: float) -> float:
    return (1.0 + math.cos(_rad(alpha_deg))) / 2.0


def magnitude_from_phase_angle(alpha_deg: float, distance_km: float | None = None) -> float:
    """Apparent V magnitude of the Moon (Allen / Cox 2002 phase law).

    Optional distance correction relative to the mean Earth-Moon distance.
    The polynomial is fitted for |alpha| up to ~150 deg; beyond that the Moon
    is a thin sliver lost in twilight and the value is only indicative.
    """
    a = abs(alpha_deg)
    m = FULL_MOON_MAG + 0.026 * a + 4e-9 * a ** 4
    if distance_km:
        m += 5.0 * math.log10(distance_km / MEAN_DISTANCE_KM)
    return m


def magnitude_to_lux(m: float) -> float:
    return LUX_ZERO_MAG * 10 ** (-0.4 * m)


def relative_to_full(m: float) -> float:
    """Brightness as a fraction of a (mean-distance) full Moon."""
    return 10 ** (-0.4 * (m - FULL_MOON_MAG))


def classify_fraction(k: float, waxing: bool) -> str:
    if k < BAND_NEW:
        return "new_moon"
    if k >= BAND_FULL:
        return "full_moon"
    if k < BAND_CRESCENT:
        return "waxing_crescent" if waxing else "waning_crescent"
    if k < BAND_QUARTER:
        return "first_quarter" if waxing else "last_quarter"
    return "waxing_gibbous" if waxing else "waning_gibbous"


def class_fraction_range(cls: str) -> tuple[float, float]:
    if cls == "new_moon":
        return 0.0, BAND_NEW
    if cls == "full_moon":
        return BAND_FULL, 1.0
    if cls.endswith("crescent"):
        return BAND_NEW, BAND_CRESCENT
    if cls.endswith("quarter"):
        return BAND_CRESCENT, BAND_QUARTER
    return BAND_QUARTER, BAND_FULL


def class_reference_brightness(cls: str) -> dict:
    """Method A: brightness range looked up from the phase class alone."""
    k_lo, k_hi = class_fraction_range(cls)
    a_hi = fraction_to_phase_angle(k_lo)   # less lit -> larger angle -> dimmer
    a_lo = fraction_to_phase_angle(k_hi)
    a_mid = fraction_to_phase_angle((k_lo + k_hi) / 2)
    m_bright = magnitude_from_phase_angle(a_lo)
    m_dim = magnitude_from_phase_angle(min(a_hi, 170.0))
    m_mid = magnitude_from_phase_angle(a_mid)
    return {
        "method": "class_reference",
        "illuminated_fraction_range": [round(k_lo, 3), round(k_hi, 3)],
        "phase_angle_range_deg": [round(a_lo, 1), round(a_hi, 1)],
        "magnitude_range": [round(m_bright, 2), round(m_dim, 2)],
        "magnitude_typical": round(m_mid, 2),
        "lux_range": [magnitude_to_lux(m_dim), magnitude_to_lux(m_bright)],
        "lux_typical": magnitude_to_lux(m_mid),
        "percent_of_full_range": [
            round(100 * relative_to_full(m_dim), 2),
            round(100 * relative_to_full(m_bright), 2),
        ],
        "percent_of_full_typical": round(100 * relative_to_full(m_mid), 2),
    }


# --------------------------------------------------------------------------
# Ephemeris: phase from a date (Meeus, truncated series)
# --------------------------------------------------------------------------
@dataclass
class Ephemeris:
    utc: str
    phase_angle_deg: float
    illuminated_fraction: float
    waxing: bool
    age_days: float
    distance_km: float
    phase_class: str
    phase_label: str
    magnitude: float
    lux: float

    def to_dict(self) -> dict:
        return asdict(self)


def _julian_day(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def moon_ephemeris(dt: datetime) -> Ephemeris:
    jd = _julian_day(dt)
    T = (jd - 2451545.0) / 36525.0
    D = (297.8501921 + 445267.1114034 * T) % 360.0      # mean elongation
    M = (357.5291092 + 35999.0502909 * T) % 360.0       # Sun mean anomaly
    Mp = (134.9633964 + 477198.8675055 * T) % 360.0     # Moon mean anomaly

    Dr, Mr, Mpr = _rad(D), _rad(M), _rad(Mp)
    # Phase angle (Meeus 48.4)
    i = (180.0 - D
         - 6.289 * math.sin(Mpr)
         + 2.100 * math.sin(Mr)
         - 1.274 * math.sin(2 * Dr - Mpr)
         - 0.658 * math.sin(2 * Dr)
         - 0.214 * math.sin(2 * Mpr)
         - 0.110 * math.sin(Dr))
    i = i % 360.0
    alpha = i if i <= 180.0 else 360.0 - i
    waxing = (D % 360.0) < 180.0
    k = phase_angle_to_fraction(alpha)

    # Distance (main terms of Meeus table 47.A)
    dist = (MEAN_DISTANCE_KM
            - 20905.355 * math.cos(Mpr)
            - 3699.111 * math.cos(2 * Dr - Mpr)
            - 2955.968 * math.cos(2 * Dr)
            - 569.925 * math.cos(2 * Mpr)
            + 48.888 * math.cos(Mr)
            - 3.149 * math.cos(2 * _rad(93.272 + 483202.0175 * T))
            + 246.158 * math.cos(2 * Dr - 2 * Mpr)
            - 152.138 * math.cos(2 * Dr - Mr - Mpr)
            - 170.733 * math.cos(2 * Dr + Mpr)
            - 204.586 * math.cos(2 * Dr - Mr)
            - 129.620 * math.cos(Mr - Mpr)
            + 108.743 * math.cos(Dr)
            + 104.755 * math.cos(Mr + Mpr))

    # Age: days since new moon, from elongation
    age = (D % 360.0) / 360.0 * 29.530588853
    cls = classify_fraction(k, waxing)
    m = magnitude_from_phase_angle(alpha, dist)
    return Ephemeris(
        utc=(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat(),
        phase_angle_deg=round(alpha, 2),
        illuminated_fraction=round(k, 4),
        waxing=waxing,
        age_days=round(age, 2),
        distance_km=round(dist, 0),
        phase_class=cls,
        phase_label=PHASE_LABELS[cls],
        magnitude=round(m, 2),
        lux=magnitude_to_lux(m),
    )
