"""
Train the moon-phase CNN and export it to ONNX for the backend.

Usage
-----
  # 1) synthetic only (works out of the box, no downloads)
  python training/train.py --synthetic 1500 --epochs 12

  # 2) your own / downloaded real photos + synthetic
  #    data/real/<class_name>/*.jpg  with class names:
  #    new_moon waxing_crescent first_quarter waxing_gibbous
  #    full_moon waning_gibbous last_quarter waning_crescent
  python training/train.py --real data/real --synthetic 800 --epochs 20

Real photos are cropped with the SAME segmentation the website uses
(backend/moon_cv/segment.py), so training and inference see the same input.
Output: backend/models/moon_phase.onnx + moon_phase.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "training"))

from synth import CLASSES, render_crop                      # noqa: E402
from moon_cv.segment import segment_moon, crop_disk          # noqa: E402

SIZE = 128
MEAN = [0.2, 0.2, 0.2]
STD = [0.3, 0.3, 0.3]


class MoonNet(nn.Module):
    """Small CNN (~0.84M params) - fast on CPU, exports cleanly to ONNX."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 24), block(24, 48), block(48, 96), block(96, 128), block(128, 160))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(160, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def load_real(folder: Path):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        d = folder / c
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            seg = segment_moon(img)
            if seg.found:
                crop = crop_disk(seg.image, seg.center, seg.radius, out=SIZE)
            else:
                h, w = img.shape[:2]
                s = min(h, w)
                crop = cv2.resize(img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s], (SIZE, SIZE))
            X.append(crop)
            y.append(ci)
    print(f"real images: {len(X)}")
    return X, y


def make_synthetic(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    X, y = [], []
    t = time.time()
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(render_crop(c, rng, out=SIZE))
            y.append(ci)
    print(f"synthetic images: {len(X)} in {time.time() - t:.0f}s")
    return X, y


def to_tensor(X):
    a = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in X]).astype(np.float32) / 255.0
    a = (a - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(a.transpose(0, 3, 1, 2).copy())


def augment(x: torch.Tensor) -> torch.Tensor:
    # brightness/contrast jitter + small shifts. NO horizontal flips:
    # flipping would swap waxing <-> waning.
    b = x.shape[0]
    x = x * (1 + 0.25 * (torch.rand(b, 1, 1, 1) - 0.5)) + 0.2 * (torch.rand(b, 1, 1, 1) - 0.5)
    dx, dy = np.random.randint(-6, 7, 2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=None)
    ap.add_argument("--synthetic", type=int, default=1500, help="synthetic images per class (0 = none)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=Path, default=ROOT / "backend" / "models")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    X, y, src = [], [], []
    if args.synthetic:
        a, b = make_synthetic(args.synthetic, seed=0)
        X += a; y += b; src += ["synthetic"]
    if args.real:
        a, b = load_real(args.real)
        X += a; y += b; src += [f"real:{args.real.name}"]
    if not X:
        sys.exit("no training data")

    Xt, yt = to_tensor(X), torch.tensor(y)
    perm = torch.randperm(len(yt))
    n_val = max(len(yt) // 8, 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = MoonNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(tr_idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t = time.time()
        sh = tr_idx[torch.randperm(len(tr_idx))]
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(sh), args.batch):
            idx = sh[i:i + args.batch]
            xb, yb = augment(Xt[idx]), yt[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item(); tot += len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xt[val_idx[i:i + 256]]).argmax(1) for i in range(0, n_val, 256)])
        acc = (pv == yt[val_idx]).float().mean().item()
        print(f"epoch {ep + 1}/{args.epochs}  loss {loss_sum / tot:.3f}  train {correct / tot:.3f}  val {acc:.3f}  ({time.time() - t:.0f}s)")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), "/tmp/moonnet_best.pt")

    model.load_state_dict(torch.load("/tmp/moonnet_best.pt"))
    model.eval()
    # confusion matrix
    with torch.no_grad():
        pv = torch.cat([model(Xt[val_idx[i:i + 256]]).argmax(1) for i in range(0, n_val, 256)])
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t_, p_ in zip(yt[val_idx].tolist(), pv.tolist()):
        cm[t_, p_] += 1
    print("confusion (rows=true):")
    for c, row in zip(CLASSES, cm):
        print(f"  {c:16s} {row}")

    args.out.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    torch.onnx.export(model, dummy, str(args.out / "moon_phase.onnx"), input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    meta = {"classes": CLASSES, "input_size": SIZE, "mean": MEAN, "std": STD,
            "val_accuracy": round(best, 4), "epochs": args.epochs, "trained_on": src,
            "n_train": int(len(tr_idx)), "n_val": int(n_val),
            "note": "Input is the backend's disk crop (moon radius ~ 1/1.2 of half-side), RGB, "
                    "Northern-Hemisphere orientation (lit on right = waxing)."}
    (args.out / "moon_phase.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {args.out / 'moon_phase.onnx'}  (best val acc {best:.3f})")


if __name__ == "__main__":
    main()
