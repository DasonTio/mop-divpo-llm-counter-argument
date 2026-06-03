"""Render Gambar 2 — inference flow of the 6 comparison methods (parallel) plus
the two evaluation stages. Same style as Gambar 1.

Run: .venv/bin/python scripts/make_inference_diagram.py
Output: paper/gambar2_inference_pipeline.png (+ .jpg)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _diagram_style import (  # noqa: E402
    make_canvas, box, container, arrow, line,
    GREY, BLUE, PURPLE, GREEN, AMBER, NEUTRAL, SUB, INK,
)

fig, ax = make_canvas(13.5, 10.2)

# ---- 1. input ----------------------------------------------------------------
ix, iy, iw, ih = 30, 91.5, 40, 6.5
box(ax, ix, iy, iw, ih, "Evaluation prompt",
    "30 CGA-CMV claims  ·  20 neutral prompts (distinctness)", face=BLUE, fs=11)
in_cx = ix + iw / 2

# ---- 2. six methods in parallel ---------------------------------------------
container(ax, 1, 60, 98, 24.5)
ax.text(50, 89.6,
        "INFERENCE — 6 comparison methods (parallel)  ·  do_sample, T=0.9, top_p=0.9  →  4 outputs / prompt",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color=SUB)

methods = [
    ("Base", "no persona prompt\nno adapter\n4 temperature samples", GREY),
    ("Prompt-only", "base model +\n4 persona system prompts\nno adapter", GREY),
    ("Single LoRA", "1 adapter (pooled data)\n+ 4 persona prompts\nrun 4×", BLUE),
    ("MoP-SFT", "4 SFT adapters +\nmatching persona prompt", PURPLE),
    ("MoP+DivPO", "4 DivPO v1 adapters\n(within-persona rarity)", GREY),
    ("MoP+DivPO v2", "4 DivPO v2 adapters\n(cross-persona rarity)\nproposed method", GREEN),
]
n = len(methods)
left, right, gap = 3.0, 97.0, 1.4
mw = (right - left - gap * (n - 1)) / n
my, mh = 62.5, 19.5
centers = []
for i, (t, b, col) in enumerate(methods):
    mx = left + i * (mw + gap)
    cx, _, _ = box(ax, mx, my, mw, mh, t, b, face=col, fs=10, round_=0.02)
    centers.append(cx)

# distribution bus: input -> bus -> each method top
dist_y = 84.4
arrow(ax, in_cx, iy, in_cx, dist_y, lw=1.3)
line(ax, centers[0], dist_y, centers[-1], dist_y, color=SUB, lw=1.3)
for cx in centers:
    arrow(ax, cx, dist_y, cx, my + mh, lw=1.0)

# ---- 3. outputs pool ---------------------------------------------------------
ox, oy, ow, oh = 26, 50, 48, 6.2
box(ax, ox, oy, ow, oh, "720 generations",
    "6 methods × 30 prompts × 4 outputs", face=NEUTRAL, fs=11)
out_cx = ox + ow / 2
coll_y = 58.4
for cx in centers:
    line(ax, cx, my, cx, coll_y, color="#9aa3ad", lw=0.9)
line(ax, centers[0], coll_y, centers[-1], coll_y, color="#9aa3ad", lw=1.1)
arrow(ax, out_cx, coll_y, out_cx, oy + oh, lw=1.3)

# ---- 4. two evaluation stages ------------------------------------------------
# Stage 1 (left) — persona distinctness
s1x, s1y, s1w, s1h = 2, 4, 30, 40
container(ax, s1x, s1y, s1w, s1h, "STAGE 1 — Persona Distinctness")
box(ax, s1x + 2.5, s1y + 24, s1w - 5, 8.5, "SBERT 4×4 cosine",
    "across 4 persona adapters\n20 neutral prompts · 400 outputs", face=BLUE, fs=10)
box(ax, s1x + 2.5, s1y + 12.5, s1w - 5, 8.5, "Separation check",
    "verify distributions are\nsemantically separated (< 0.5)", face=PURPLE, fs=10)
ax.text(s1x + s1w / 2, s1y + 6.5, "persona-based\nmethods only",
        ha="center", va="center", fontsize=8.3, style="italic", color=SUB)

# Stage 2 (right) — baseline evaluation
s2x, s2y, s2w, s2h = 35, 4, 63, 40
container(ax, s2x, s2y, s2w, s2h, "STAGE 2 — Baseline Evaluation  ·  30 CGA-CMV prompts")
cw = (s2w - 4 * 2 - 2 * 2) / 3
cy, ch = s2y + 9, 20
chips = [
    ("Formula metrics", "Self-BLEU ↓\nDistinct-1 ↑\nDistinct-2 ↑\nSBERT cos ↓", GREY),
    ("External reference", "ArmoRM\n(human-preference\nreward model)\nBERTScore vs\nCMV delta-winning", AMBER),
    ("LLM-as-judge", "Quality · Novelty\nUtility · Fidelity\nGPT-4o-mini (primary)\n+ GPT-4o (calibration)", GREEN),
]
for i, (t, b, col) in enumerate(chips):
    chx = s2x + 4 + i * (cw + 2)
    box(ax, chx, cy, cw, ch, t, b, face=col, fs=9.8)
ax.text(s2x + s2w / 2, s2y + 5.0,
        "paired bootstrap (10,000 resamples, n=30)  →  significance test",
        ha="center", va="center", fontsize=8.6, style="italic", color=SUB)

arrow(ax, out_cx, oy, s1x + s1w / 2, s1y + s1h, lw=1.3)
arrow(ax, out_cx, oy, s2x + s2w / 2, s2y + s2h, lw=1.3)

fig.savefig("paper/gambar2_inference_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig("paper/gambar2_inference_pipeline.jpg", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote paper/gambar2_inference_pipeline.png and .jpg")
