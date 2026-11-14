"""ONNX inference for the moon-phase CNN (trained by training/train.py)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class PhaseCNN:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.available = False
        onnx_path = model_dir / "moon_phase.onnx"
        meta_path = model_dir / "moon_phase.json"
        if not onnx_path.exists():
            return
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.classes = meta.get("classes")
        self.size = meta.get("input_size", 128)
        self.mean = np.array(meta.get("mean", [0.5, 0.5, 0.5]), np.float32)
        self.std = np.array(meta.get("std", [0.25, 0.25, 0.25]), np.float32)
        self.info = meta
        self.available = True

    def predict(self, crop_bgr: np.ndarray) -> dict[str, float]:
        img = cv2.resize(crop_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {self.input_name: x.astype(np.float32)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}
