"""Render Gambar 1 — training pipeline (SFT -> DivPO), same style as Gambar 2.

Run: .venv/bin/python scripts/make_training_diagram.py
Output: paper/gambar1_training_pipeline.png (+ .jpg)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _diagram_style import (  # noqa: E402
    make_canvas, box, container, arrow, line,
    GREY, BLUE, PURPLE, GREEN, TEAL, NEUTRAL, SUB,
)

fig, ax = make_canvas(12.5, 13.8)

# =============================== STAGE 1 =====================================
container(ax, 1, 62, 98, 35, "STAGE 1 — supervised fine-tuning")

# base model
bx, by, bw, bh = 33, 88.5, 34, 5.6
b_cx, _, _ = box(ax, bx, by, bw, bh, "Qwen2.5-0.5B-Instruct",
                 "base model — frozen throughout", face=GREY, fs=11)

# four persona adapters (each: persona + dataset)
personas = [
    ("Contrarian", "CGA-CMV"),
    ("Systems Thinker", "StackExchange"),
    ("Cross-Domain Analogist", "arXiv abstracts"),
    ("Minimalist", "IBM Arg. Quality"),
]
pleft, pright, pgap = 4.0, 96.0, 2.0
pw = (pright - pleft - pgap * 3) / 4
py, ph = 77.5, 8.6
pcenters = []
for i, (t, d) in enumerate(personas):
    px = pleft + i * (pw + pgap)
    cx, _, top = box(ax, px, py, pw, ph, t, d, face=TEAL, fs=10, round_=0.02)
    pcenters.append(cx)

# distribute base -> personas (bus)
dist_y = 86.0
arrow(ax, b_cx, by, b_cx, dist_y, lw=1.2)
line(ax, pcenters[0], dist_y, pcenters[-1], dist_y, color=SUB, lw=1.2)
for cx in pcenters:
    arrow(ax, cx, dist_y, cx, py + ph, lw=1.0)

# SFT box
sfx, sfy, sfw, sfh = 22, 70, 56, 5.6
sf_cx, _, _ = box(ax, sfx, sfy, sfw, sfh,
                  "SFT — independent LoRA adapter per persona",
                  "NLL loss · base frozen · trained separately", face=PURPLE, fs=10.5)
# collect personas -> SFT
coll_y = 76.4
for cx in pcenters:
    line(ax, cx, py, cx, coll_y, color="#9aa3ad", lw=0.9)
line(ax, pcenters[0], coll_y, pcenters[-1], coll_y, color="#9aa3ad", lw=1.1)
arrow(ax, sf_cx, coll_y, sf_cx, sfy + sfh, lw=1.3)

# 4 SFT adapters
s1x, s1y, s1w, s1h = 33, 63.6, 34, 4.9
s1_cx, _, _ = box(ax, s1x, s1y, s1w, s1h, "4 SFT persona adapters",
                  face=PURPLE, fs=10.5)
arrow(ax, sf_cx, sfy, s1_cx, s1y + s1h, lw=1.3)

# =============================== STAGE 2 =====================================
container(ax, 1, 25, 98, 34,
          "STAGE 2 — diverse preference optimization")

# bridge stage1 -> stage2
gx, gy, gw, gh = 18, 49.8, 64, 5.6
g_cx, _, _ = box(ax, gx, gy, gw, gh, "Generate N candidates per prompt",
                 "temperature=0.9 · top_p=0.9 · from each SFT adapter",
                 face=NEUTRAL, fs=10.5)
arrow(ax, s1_cx, s1y, g_cx, gy + gh, lw=1.4)

# two branches
branches = [
    dict(x=5, w=42, col=GREY, head="DivPO — ablation", hsub="within-persona rarity",
         pair="rarity vs same-persona pool\nquality: coherence + length",
         out="MoP+DivPO adapters"),
    dict(x=53, w=42, col=GREEN, head="DivPO v2 — proposed", hsub="cross-persona rarity",
         pair="rarity vs all-persona pool\nquality: coherence + length only",
         out="MoP+DivPO v2 adapters"),
]
branch_out_cx = []
for br in branches:
    cx_mid = br["x"] + br["w"] / 2
    arrow(ax, g_cx, gy, cx_mid, 48.0, lw=1.2)  # gy is box bottom
    box(ax, br["x"], 43, br["w"], 5.0, br["head"], br["hsub"], face=br["col"], fs=10.5)
    arrow(ax, cx_mid, 43, cx_mid, 41.5, lw=1.1)
    box(ax, br["x"], 35, br["w"], 6.5, "Pair construction", br["pair"],
        face=br["col"], fs=10.5)
    arrow(ax, cx_mid, 35, cx_mid, 33.5, lw=1.1)
    box(ax, br["x"], 27, br["w"], 5.5, br["out"], "continued from SFT checkpoint",
        face=br["col"], fs=10.5)
    branch_out_cx.append(cx_mid)

# =============================== FINAL ARTIFACTS =============================
container(ax, 1, 2, 98, 19.5, "final artifacts")
artifacts = [
    ("Base model", "frozen throughout", GREY),
    ("SFT adapters", "4 persona checkpoints", PURPLE),
    ("DivPO adapters", "ablation baseline", GREY),
    ("DivPO v2 adapters", "proposed method", GREEN),
]
aleft, aright, agap = 4.0, 96.0, 2.0
aw = (aright - aleft - agap * 3) / 4
ay, ah = 5.5, 9.5
acenters = []
for i, (t, b, col) in enumerate(artifacts):
    axx = aleft + i * (aw + agap)
    cx, _, top = box(ax, axx, ay, aw, ah, t, b, face=col, fs=10.5)
    acenters.append((cx, top))

# arrows into final artifacts (only the DivPO branches; base/SFT are carried
# forward from Stage 1 — long provenance arrows would cross everything)
arrow(ax, branch_out_cx[0], 27, acenters[2][0], acenters[2][1], lw=1.3)  # divpo
arrow(ax, branch_out_cx[1], 27, acenters[3][0], acenters[3][1], lw=1.3)  # divpo v2

fig.savefig("paper/gambar1_training_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig("paper/gambar1_training_pipeline.jpg", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote paper/gambar1_training_pipeline.png and .jpg")
