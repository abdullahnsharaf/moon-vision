"""Generates the site's Lottie animations (Bodymovin JSON) into frontend/lottie/.
   python tools/make_lottie.py
"""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "frontend" / "lottie"
FR = 60

def rgb(h, a=1):
    h = h.lstrip("#"); return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)] + [a]

def st(v): return {"a": 0, "k": v}
def kf(frames, ease=True):
    k = []
    for i, (t, v) in enumerate(frames):
        v = v if isinstance(v, list) else [v]
        e = {"t": t, "s": v}
        if i < len(frames) - 1:
            n = len(v)
            e["o"] = {"x": [0.42] * n, "y": [0] * n} if ease else {"x": [0] * n, "y": [0] * n}
            e["i"] = {"x": [0.58] * n, "y": [1] * n} if ease else {"x": [1] * n, "y": [1] * n}
        k.append(e)
    return {"a": 1, "k": k}

def tr(p=(0, 0), s=(100, 100), r=0, o=100):
    return {"ty": "tr", "p": st(list(p)), "a": st([0, 0]), "s": s if isinstance(s, dict) else st(list(s)),
            "r": r if isinstance(r, dict) else st(r), "o": o if isinstance(o, dict) else st(o)}

def ellipse(d, p=(0, 0)): return {"ty": "el", "p": st(list(p)), "s": st([d, d]), "d": 1}
def fill(c, o=100): return {"ty": "fl", "c": st(rgb(c)), "o": st(o) if not isinstance(o, dict) else o, "r": 1}
def stroke(c, w, o=100, dash=None):
    s = {"ty": "st", "c": st(rgb(c)), "o": st(o) if not isinstance(o, dict) else o, "w": st(w), "lc": 2, "lj": 2}
    if dash: s["d"] = [{"n": "d", "nm": "dash", "v": st(dash[0])}, {"n": "g", "nm": "gap", "v": st(dash[1])}]
    return s
def group(items, t=None, nm="g"): return {"ty": "gr", "nm": nm, "it": items + [t or tr()]}

def layer(ind, nm, shapes, op, p=(100, 100), r=0, s=(100, 100, 100), o=100):
    return {"ddd": 0, "ind": ind, "ty": 4, "nm": nm, "sr": 1, "ao": 0, "ip": 0, "op": op, "st": 0, "bm": 0,
            "ks": {"o": o if isinstance(o, dict) else st(o), "r": r if isinstance(r, dict) else st(r),
                   "p": p if isinstance(p, dict) else st(list(p) + [0]), "a": st([0, 0, 0]),
                   "s": s if isinstance(s, dict) else st(list(s))},
            "shapes": shapes}

def comp(nm, op, layers, w=200, h=200):
    # Lottie draws the first layer on top -> reverse so we can list bottom-up
    return {"v": "5.7.4", "fr": FR, "ip": 0, "op": op, "w": w, "h": h, "nm": nm, "ddd": 0, "assets": [],
            "layers": list(reversed(layers))}

INK, SURF = "#161F42", "#161F42"
MOON, AZURE, BRASS, LINE = "#F3ECD9", "#7C9CF0", "#D9A85B", "#33406F"

# ---------------------------------------------------------------- scanning loader (3 s loop)
op = 180
scan = comp("moon-scan", op, [
    # expanding pulse
    layer(1, "pulse", [group([ellipse(96), stroke(AZURE, 1.5)])], op,
          s=kf([(0, [80, 80, 100]), (op, [150, 150, 100])], ease=False),
          o=kf([(0, 70), (op, 0)], ease=False)),
    # moon: lit disk + shadow disk sliding across = phases waxing then waning
    layer(2, "moon", [group([ellipse(84), fill(MOON)])], op),
    layer(3, "shadow", [group([ellipse(90), fill(SURF)])], op,
          p=kf([(0, [100, 100, 0]), (op / 2, [190, 100, 0]), (op / 2 + 1, [10, 100, 0]), (op, [100, 100, 0])])),
    # dashed orbit
    layer(4, "orbit", [group([ellipse(150), stroke(LINE, 1.5, dash=(3, 7))])], op,
          r=kf([(0, 0), (op, 90)], ease=False)),
    # scanning arc (trim path)
    layer(5, "arc", [group([ellipse(124), stroke(BRASS, 3),
                            {"ty": "tm", "s": kf([(0, 0), (op, 0)]), "e": kf([(0, 2), (op / 2, 32), (op, 2)]),
                             "o": kf([(0, 0), (op, 720)], ease=False), "m": 1}])], op),
    # satellite
    layer(6, "satellite", [group([ellipse(9, (75, 0)), fill(AZURE)])], op,
          r=kf([(0, 0), (op, -360)], ease=False)),
])

# ---------------------------------------------------------------- idle: crescent with twinkling stars (4 s loop)
op2 = 240
stars = []
for i, (x, y, d, ph) in enumerate([(40, 48, 5, 0), (160, 40, 4, 60), (150, 150, 6, 120), (52, 156, 4, 170), (178, 98, 3, 30)]):
    stars.append(layer(10 + i, f"star{i}", [group([ellipse(d), fill(MOON)])], op2, p=(x, y),
                       o=kf([(0, 25), (ph + 10, 25), (ph + 35, 100), (ph + 60, 25), (op2, 25)])))
idle = comp("moon-idle", op2, stars + [
    layer(2, "crescent-lit", [group([ellipse(92), fill(MOON)])], op2,
          p=kf([(0, [100, 104, 0]), (op2 / 2, [100, 96, 0]), (op2, [100, 104, 0])])),
    layer(3, "crescent-shadow", [group([ellipse(86), fill(SURF)])], op2,
          p=kf([(0, [80, 100, 0]), (op2 / 2, [80, 92, 0]), (op2, [80, 100, 0])])),
])

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "scan.json").write_text(json.dumps(scan))
(OUT / "idle.json").write_text(json.dumps(idle))
print("wrote", list(OUT.iterdir()))
