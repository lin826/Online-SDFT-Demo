"""Draw the online-SDFT loop diagram for the blog post.

One model, two roles: the served model makes its own bare-prompt call (TEACH),
your behavior referees it with one bit (CHECK), and a few batch_size=1 LoRA
steps distill the kept-or-corrected action back into the adapter (LEARN).
The TEACH / CHECK / LEARN diagram embedded in the accompanying blog post.

Writes figures/online_sdft_loop.png.

Run:  python draw_loop_diagram.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from triage_common import FIG_DIR  # noqa: E402

BLUE, ORANGE, PURPLE, GREY = "#1a73e8", "#e8710a", "#7b3fa0", "#5f6368"
GREEN, RED = "#0b8043", "#d93025"


def box(ax, x, y, w, h, color, header, body):
    """Rounded box with a small colored header above the body text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.9",
                                linewidth=1.8, edgecolor=color, facecolor="white",
                                zorder=3))
    ax.text(x + w / 2, y + h + 3.1, header, ha="center", va="bottom", fontsize=10.5,
            fontweight="bold", color=color, zorder=4)
    ax.text(x + w / 2, y + h / 2, body, ha="center", va="center", fontsize=9.6,
            color="#202124", zorder=4, linespacing=1.45)


def arrow(ax, start, end, color=GREY, rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                 linewidth=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # the three roles, left to right, with the update flowing back underneath
    box(ax, 4, 26, 22, 9, GREY, "INCOMING ITEM",
        "one notification\nbare prompt · ~90 tokens")
    box(ax, 38, 26, 26, 9, BLUE, "TEACH — the serving call",
        "LFM2.5-230M + LoRA\n(the adapter so far)")
    box(ax, 96, 26, 24, 9, ORANGE, "CHECK — one-bit referee",
        "your behavior\nopen now / let it wait / never")
    box(ax, 52, 4, 32, 9, PURPLE, "LEARN — batch_size=1",
        "a few LoRA steps on the target\n(+ one replay item per other class)")

    arrow(ax, (26.9, 30.5), (37.1, 30.5))
    arrow(ax, (64.9, 30.5), (95.1, 30.5), color=BLUE)
    ax.text(80, 32.3, 'its own answer — e.g. "INTERRUPT"', ha="center",
            va="bottom", fontsize=9.3, color=BLUE, style="italic")

    # CHECK -> LEARN: the one-bit verdict becomes the training target
    arrow(ax, (100, 24.7), (85.8, 9.5), color=ORANGE, rad=-0.22)
    ax.text(108, 21.3, "matched → reinforce its own answer", ha="center",
            va="center", fontsize=8.8, color=GREEN)
    ax.text(108, 18.3, "missed → correct toward your action", ha="center",
            va="center", fontsize=8.8, color=RED)
    ax.text(108, 15.3, "the target is a bare action —\nnever a hand-written gold answer",
            ha="center", va="top", fontsize=8.4, color=GREY, style="italic")

    # LEARN -> TEACH: the weights, not the context, carry the lesson
    arrow(ax, (51, 8.5), (44, 24.8), color=PURPLE, rad=-0.25)
    ax.text(37.5, 14.5, "the ~1.4 MB adapter\ndrifts with you", ha="right",
            va="center", fontsize=9.0, color=PURPLE, style="italic")

    ax.text(4, 49.0, "One model, two roles — the online SDFT loop",
            fontsize=12.5, fontweight="bold", color="#202124", va="top")
    ax.text(4, 44.8, "teacher and student are the same 230M network; "
                     "your behavior only referees",
            fontsize=9.5, color=GREY, va="top")

    fig.tight_layout()
    out = FIG_DIR / "online_sdft_loop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
