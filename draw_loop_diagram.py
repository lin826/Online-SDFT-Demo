"""Draw the online-SDFT loop diagram for the blog post.

One model, two roles: the student makes a bare-prompt call (TEACH), your
behavior is the expert demonstration that conditions the teacher (CHECK), and
a few batch_size=1 LoRA steps soft-CE distill teacher → student (LEARN).
The TEACH / CHECK / LEARN diagram embedded in the accompanying blog post.

Writes figures/online_sdft_loop.png.

Run:  python draw_loop_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, PURPLE, GREY = "#1a73e8", "#e8710a", "#7b3fa0", "#5f6368"
GREEN, RED = "#0b8043", "#d93025"


def box(ax, x, y, w, h, color, header, body):
    """Header sits above the stroke; body stays inside with clear padding."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.55",
                                linewidth=1.5, edgecolor=color, facecolor="white",
                                zorder=3))
    ax.text(x + w / 2, y + h + 0.5, header, ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color=color, zorder=4)
    ax.text(x + w / 2, y + h / 2, body, ha="center", va="center", fontsize=8.5,
            color="#202124", zorder=4, linespacing=1.15)


def straight_arrow(ax, start, end, color=GREY, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, zorder=8, connectionstyle="arc3,rad=0",
    ))


def _bezier(t, p0, p1, p2, p3):
    u = 1.0 - t
    return (u**3) * p0 + 3 * (u**2) * t * p1 + 3 * u * (t**2) * p2 + (t**3) * p3


def curve_arrow(ax, p0, p1, p2, p3, color, lw=1.6):
    """Cubic-Bezier shaft + the same ``-|>`` head as ``straight_arrow``.

    The head is a short two-point FancyArrowPatch (identical to gray/blue), not a
    filled ribbon taper. The stroked Bezier stops at the head base so the join
    stays continuous with no gap.
    """
    pts = [np.asarray(p, dtype=float) for p in (p0, p1, p2, p3)]
    curve = np.stack([_bezier(t, *pts) for t in np.linspace(0.0, 1.0, 80)])
    # Analytic end tangent 3*(p3-p2).
    tang = 3.0 * (pts[3] - pts[2])
    tang = tang / max(float(np.linalg.norm(tang)), 1e-8)
    tip = pts[3] + 0.05 * tang
    # Long enough in data space for FancyArrowPatch to place the wedge;
    # head size itself comes from mutation_scale (display points).
    head_back = tip - 1.35 * tang

    seglen = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    target = max(float(cum[-1]) - 1.35, 0.0)
    cut = int(np.searchsorted(cum, target))
    cut = max(2, min(cut, len(curve) - 1))
    shaft = np.vstack([curve[: cut + 1], [head_back]])
    ax.plot(shaft[:, 0], shaft[:, 1], color=color, lw=lw, solid_capstyle="butt",
            solid_joinstyle="round", zorder=15, clip_on=False)
    ax.add_patch(FancyArrowPatch(
        head_back, tip, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, zorder=16, clip_on=False, shrinkA=0, shrinkB=0,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.15))
    ax.set_xlim(0, 72)
    ax.set_ylim(-2.4, 26.8)
    ax.axis("off")

    ax.text(1.0, 23.7, "One model, two roles — the online SDFT loop",
            fontsize=11.5, fontweight="bold", color="#202124", va="top")
    ax.text(1.0, 21.9, "teacher and student are the same 230M network; "
                       "expert action conditions the teacher",
            fontsize=8.8, color=GREY, va="top")

    box(ax, 1.5, 11.5, 17.0, 6.4, GREY, "INCOMING ITEM",
        "one notification\nbare prompt\n(~90 tokens)")
    box(ax, 24.0, 11.5, 17.0, 6.4, BLUE, "STUDENT — serving call",
        "LFM2.5-230M + LoRA\nbare prompt (π·|x)")
    box(ax, 53.0, 11.5, 17.0, 6.4, ORANGE, "TEACHER — expert demo",
        "same adapter +\nyour action as ICL\n(π·|x, c)")
    ax.add_patch(FancyBboxPatch((24.0, 0.3), 24.0, 6.4,
                                boxstyle="round,pad=0.02,rounding_size=0.55",
                                linewidth=1.5, edgecolor=PURPLE, facecolor="white",
                                zorder=1))
    ax.text(41.0, 0.3 + 6.4 + 0.5, "LEARN — soft-CE distill", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=PURPLE, zorder=4)
    ax.text(36.0, 0.3 + 6.4 / 2,
            "LoRA steps: teacher → student\n(+ replay per other class)",
            ha="center", va="center", fontsize=8.5, color="#202124", zorder=4,
            linespacing=1.15)

    straight_arrow(ax, (19.2, 14.7), (23.3, 14.7))
    straight_arrow(ax, (41.7, 14.7), (52.3, 14.7), color=BLUE)

    ax.text(47.0, 16.6, "own answer", ha="center", va="center",
            fontsize=7.3, color=BLUE, style="italic", zorder=5)
    ax.text(47.0, 15.4, '"INTERRUPT"', ha="center", va="center",
            fontsize=7.3, color=BLUE, style="italic", zorder=5)

    # Tip just right of LEARN (x=48) and below its vertical mid — outside the fill.
    curve_arrow(ax, (58.5, 10.0), (55.0, 6.8), (52.2, 3.8), (49.2, 2.15), ORANGE)
    ax.text(61.5, 9.2, "expert action c", ha="center", va="center",
            fontsize=7.4, color=GREEN)
    ax.text(61.5, 8.1, "(your open / wait / never)", ha="center", va="center",
            fontsize=7.4, color=RED)
    ax.text(61.5, 6.5, "soft teacher targets —\nnot hard SFT labels", ha="center",
            va="top", fontsize=7.0, color=GREY, style="italic", linespacing=1.05)

    # Tip under TEACH bottom; steep exit so the -|> head reads as a triangle
    # (shallow approach laid the wedge along the box edge and looked blunt).
    curve_arrow(ax, (22.8, 3.4), (17.0, 4.5), (25.5, 8.0), (28.0, 11.0), PURPLE)
    ax.text(13.5, 7.0, "adapter (~1.4 MB)\ndrifts with you", ha="right",
            va="center", fontsize=7.4, color=PURPLE, style="italic", linespacing=1.05)

    ax.text(36, 26.6, " ", fontsize=1, alpha=0)
    ax.text(36, -2.2, " ", fontsize=1, alpha=0)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = FIG_DIR / "online_sdft_loop.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.28)
    print("wrote", out)


if __name__ == "__main__":
    main()
