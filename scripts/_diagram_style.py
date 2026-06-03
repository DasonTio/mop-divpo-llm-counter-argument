"""Shared drawing style for the paper figures (Gambar 1 & 2).

Rounded boxes, soft fills, bus-routed arrows. English interior labels to match
the existing figures (Indonesian captions live in the document, not the figure).
"""
from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

INK = "#1f2933"
SUB = "#52606d"
CONTAINER_BG = "#f5f5f6"
CONTAINER_EDGE = "#d2d6db"

GREY = ("#e9eaec", "#b8bdc4")     # base / ablation
BLUE = ("#dce9f5", "#7fa8d0")     # single lora / stage-1
PURPLE = ("#e6def4", "#a48fd0")   # SFT family
GREEN = ("#d4efe1", "#5bbf94")    # proposed v2
AMBER = ("#fbeed8", "#e0b770")    # external reference
TEAL = ("#d6efe9", "#6cbfae")     # personas
NEUTRAL = ("#eef0f2", "#c2c8cf")  # pooled artifacts


def make_canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", face=GREY, fs=10.5, tw="bold", round_=0.025):
    fc, ec = face
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.4,rounding_size={round_*100}",
        linewidth=1.3, edgecolor=ec, facecolor=fc, mutation_aspect=1,
    ))
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


def container(ax, x, y, w, h, label="", lx=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5",
        linewidth=1.2, edgecolor=CONTAINER_EDGE, facecolor=CONTAINER_BG, zorder=0,
    ))
    if label:
        ax.text(lx if lx is not None else x + 1.6, y + h - 2.2, label,
                ha="left", va="center", fontsize=9.5, fontweight="bold", color=SUB)


def arrow(ax, x1, y1, x2, y2, color=SUB, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2, zorder=5,
    ))


def line(ax, x1, y1, x2, y2, color=SUB, lw=1.3, z=4):
    ax.add_line(Line2D([x1, x2], [y1, y2], color=color, lw=lw, zorder=z))
