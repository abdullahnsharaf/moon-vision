"""Accuracy check of the full pipeline on synthetic photos with known ground truth.
Run from the project root:  python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "training"))

from moon_cv.analyze import analyze            # noqa: E402
from moon_cv import astro                      # noqa: E402
from synth import render_photo                 # noqa: E402

rng = np.random.default_rng(123)
fracs = [0.08, 0.15, 0.25, 0.33, 0.48, 0.52, 0.68, 0.8, 0.9, 0.99]
n = ok = 0
errs = []
for rep in range(6):
    for k in fracs:
        wax = bool(rng.random() < 0.5)
        img, meta = render_photo(k, wax, rng, size=640, rotate_deg=float(rng.normal(0, 12)))
        _, buf = cv2.imencode(".jpg", img)
        r = analyze(buf.tobytes(), hemisphere="north", include_images=False)
        truth = astro.classify_fraction(k, wax)
        n += 1
        ok += r["phase"]["class"] == truth
        errs.append(r["geometry"]["illuminated_fraction"] - k)
        if r["phase"]["class"] != truth:
            print(f"  miss: k={k:.2f} wax={wax} truth={truth} pred={r['phase']['class']} "
                  f"k_est={r['geometry']['illuminated_fraction']:.3f} cnn={r['cnn']['top']}")
print(f"phase accuracy {ok}/{n} = {ok / n:.1%}")
print(f"lit-fraction error: mean {np.mean(errs):+.3f}, MAE {np.mean(np.abs(errs)):.3f}")
