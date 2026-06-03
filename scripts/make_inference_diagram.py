"""Render Gambar 2 — inference flow of the 6 comparison methods (parallel) plus
the two evaluation stages. Style matches Gambar 1 (rounded boxes, soft fills).

Run: .venv/bin/python scripts/make_inference_diagram.py
Output: paper/gambar2_inference_pipeline.png (+ .jpg)
"""
from __future__ import annotations

import os
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- palette (mirrors Gambar 1) ---------------------------------------------
INK = "#1f2933"
SUB = "#52606d"
CONTAINER_BG = "#f5f5f6"
CONTAINER_EDGE = "#d2d6db"
GREY = ("#e9eaec", "#b8bdc4")        # base / ablation
BLUE = ("#dce9f5", "#7fa8d0")        # single lora / stage1
PURPLE = ("#e6def4", "#a48fd0")      # SFT family
GREEN = ("#d4efe1", "#5bbf94")       # proposed v2
AMBER = ("#fbeed8", "#e0b770")       # stage2 reference
NEUTRAL = ("#eef0f2", "#c2c8cf")     # outputs pool

fig, ax = plt.subplots(figsize=(13.5, 10.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body="", face=GREY, fs=10.5, tw="bold", round_=0.025):
    fc, ec = face
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.4,rounding_size={round_*100}",
        linewidth=1.3, edgecolor=ec, facecolor=fc, mutation_aspect=1,
    )
    ax.add_patch(p)
    cx = x + w / 2
    if body:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=fs, fontweight=tw, color=INK)
        ax.text(cx, y + h * 0.30, body, ha="center", va="center",
                fontsize=fs - 2.3, color=SUB, linespacing=1.25)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight=tw, color=INK, linespacing=1.25)
    return (cx, y, y + h)


def container(x, y, w, h, label):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5",
        linewidth=1.2, edgecolor=CONTAINER_EDGE, facecolor=CONTAINER_BG,
        linestyle="-", zorder=0,
    )
    ax.add_patch(p)
    ax.text(x + 1.6, y + h - 2.2, label, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=SUB)


def arrow(x1, y1, x2, y2, color=SUB, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2, zorder=5,
    ))


# ---- 1. input ----------------------------------------------------------------
ix, iy, iw, ih = 30, 91.5, 40, 6.5
box(ix, iy, iw, ih, "Evaluation prompt",
    "30 CGA-CMV claims  ·  20 neutral prompts (distinctness)",
    face=BLUE, fs=11)
in_cx = ix + iw / 2

# ---- 2. six methods in parallel ---------------------------------------------
container(1, 60, 98, 24.5, "")
ax.text(50, 89.6,
        "INFERENSI — 6 metode pembanding (paralel)  ·  do_sample, T=0.9, top_p=0.9  →  4 output / prompt",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color=SUB)

methods = [
    ("Base", "tanpa persona prompt\ntanpa adapter\n4 temperature samples", GREY),
    ("Prompt-only", "base model +\n4 persona system prompt\ntanpa adapter", GREY),
    ("Single LoRA", "1 adapter (gabungan data)\n+ 4 persona prompt\ndijalankan 4x", BLUE),
    ("MoP-SFT", "4 SFT adapter +\npersona prompt sesuai", PURPLE),
    ("MoP+DivPO", "4 adapter DivPO v1\n(rarity dalam-persona)", GREY),
    ("MoP+DivPO v2", "4 adapter DivPO v2\n(rarity lintas-persona)\nmetode diusulkan", GREEN),
]

n = len(methods)
left, right = 3.0, 97.0
gap = 1.4
mw = (right - left - gap * (n - 1)) / n
my, mh = 62.5, 19.5
centers = []
for i, (t, b, col) in enumerate(methods):
    mx = left + i * (mw + gap)
    cx, _, top = box(mx, my, mw, mh, t, b, face=col, fs=10, round_=0.02)
    centers.append(cx)

# distribution bus: input -> bus -> each method top
dist_y = 84.4
arrow(in_cx, iy, in_cx, dist_y, lw=1.3)
ax.add_line(Line2D([centers[0], centers[-1]], [dist_y, dist_y],
                   color=SUB, lw=1.3, zorder=4))
for cx in centers:
    arrow(cx, dist_y, cx, my + mh, lw=1.0)

# ---- 3. outputs pool ---------------------------------------------------------
ox, oy, ow, oh = 26, 50, 48, 6.2
box(ox, oy, ow, oh, "720 generasi",
    "6 metode × 30 prompt × 4 output", face=NEUTRAL, fs=11)
out_cx = ox + ow / 2
# collector bus: each method bottom -> bus -> pool
coll_y = 58.4
for cx in centers:
    ax.add_line(Line2D([cx, cx], [my, coll_y], color="#9aa3ad", lw=0.9, zorder=4))
ax.add_line(Line2D([centers[0], centers[-1]], [coll_y, coll_y],
                   color="#9aa3ad", lw=1.1, zorder=4))
arrow(out_cx, coll_y, out_cx, oy + oh, lw=1.3, color=SUB)

# ---- 4. two evaluation stages ------------------------------------------------
# Stage 1 (left) — persona distinctness
s1x, s1y, s1w, s1h = 2, 4, 30, 40
container(s1x, s1y, s1w, s1h, "TAHAP 1 — Persona Distinctness")
box(s1x + 2.5, s1y + 24, s1w - 5, 8.5, "SBERT 4×4 cosine",
    "antar 4 persona adapter\n20 prompt netral · 400 output", face=BLUE, fs=10)
box(s1x + 2.5, s1y + 12.5, s1w - 5, 8.5, "Validasi pemisahan",
    "cek distribusi terpisah\nsecara semantik (< 0.5)", face=PURPLE, fs=10)
ax.text(s1x + s1w / 2, s1y + 6.5, "hanya metode\nberbasis persona",
        ha="center", va="center", fontsize=8.3, style="italic", color=SUB)

# Stage 2 (right) — baseline evaluation
s2x, s2y, s2w, s2h = 35, 4, 63, 40
container(s2x, s2y, s2w, s2h, "TAHAP 2 — Baseline Evaluation  ·  30 prompt CGA-CMV")
cw = (s2w - 4 * 2 - 2 * 2) / 3
cy, ch = s2y + 9, 20
chips = [
    ("Metrik formula", "Self-BLEU ↓\nDistinct-1 ↑\nDistinct-2 ↑\nSBERT cos ↓", GREY),
    ("Referensi eksternal", "ArmoRM\n(reward model\npreferensi manusia)\nBERTScore vs\nCMV delta-winning", AMBER),
    ("LLM-as-judge", "Quality · Novelty\nUtility · Fidelity\nGPT-4o-mini (primary)\n+ GPT-4o (kalibrasi)", GREEN),
]
for i, (t, b, col) in enumerate(chips):
    chx = s2x + 4 + i * (cw + 2)
    box(chx, cy, cw, ch, t, b, face=col, fs=9.8)
ax.text(s2x + s2w / 2, s2y + 5.0,
        "paired bootstrap (10.000 resampling, n=30)  →  uji signifikansi",
        ha="center", va="center", fontsize=8.6, style="italic", color=SUB)

arrow(out_cx, oy, s1x + s1w / 2, s1y + s1h, lw=1.3)
arrow(out_cx, oy, s2x + s2w / 2, s2y + s2h, lw=1.3)

plt.tight_layout(pad=0.6)
os.makedirs("paper", exist_ok=True)
png = "paper/gambar2_inference_pipeline.png"
jpg = "paper/gambar2_inference_pipeline.jpg"
fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(jpg, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", png, "and", jpg)
