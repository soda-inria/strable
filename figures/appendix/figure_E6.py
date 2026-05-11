"""Figure E.6 — Schema diagram describing how the bootstrap estimator
behind Figure 4(b) works"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.patches as patches
import matplotlib.pyplot as plt

from figures._main import save_figure


# Box styles per layer in the schema.
BOX_STYLE_POOL    = dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1565C0", lw=2)
BOX_STYLE_SAMPLE  = dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#2E7D32", lw=2)
BOX_STYLE_EVAL    = dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec="#7B1FA2", lw=2)
BOX_STYLE_PROC    = dict(boxstyle="ellipse,pad=0.3", fc="#FFF3E0", ec="#EF6C00", lw=2)
BOX_STYLE_TAU     = dict(boxstyle="darrow,pad=0.3", fc="#FFEBEE", ec="#C62828", lw=2)
ARROW_PROPS       = dict(arrowstyle="->", color="#555555", lw=2,
                         mutation_scale=15, shrinkA=0, shrinkB=0)


def plot_sampling_schema():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # Y-coordinates for the layered schema.
    y_main, y_part, y_sub, y_eval, y_rank, y_tau = 9.8, 8.5, 6.8, 4.8, 2.8, 1.0
    h_main, h_part, h_sub, h_eval, h_rank, h_tau = 0.45, 0.45, 0.45, 0.55, 0.45, 0.5

    # --- Total dataset pool ---
    ax.text(5, y_main + 0.5, "Total Datasets Pool\n($M$ Datasets)",
            ha="center", va="center", size=12, bbox=BOX_STYLE_POOL)

    # --- Partitions (disjoint random split) ---
    ax.annotate("", xy=(3, y_part + h_part), xytext=(5, y_main - h_main + 0.5),
                arrowprops=ARROW_PROPS, zorder=0)
    ax.annotate("", xy=(7, y_part + h_part), xytext=(5, y_main - h_main + 0.5),
                arrowprops=ARROW_PROPS, zorder=0)
    ax.text(3, y_part, "Partition A\n($M/2$ datasets)",
            ha="center", va="center", size=11, bbox=BOX_STYLE_POOL)
    ax.text(7, y_part, "Partition B\n($M/2$ datasets)",
            ha="center", va="center", size=11, bbox=BOX_STYLE_POOL)
    ax.text(5, 9.3, "Random Split (Disjoint)",
            ha="center", va="center", size=9, color="#555555",
            bbox=dict(fc="white", ec="none"), zorder=0)

    # --- Bootstrap loop frame around the inner workflow ---
    rect = patches.FancyBboxPatch(
        (0.5, 1.8), 9, 5.8,
        boxstyle="round,pad=0.2",
        linewidth=2, edgecolor='gray', facecolor='none', linestyle='--',
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(0.5, 7.0, "Bootstrap \n $K$ times",
            size=10, color="gray", weight="bold")

    # --- Sub-samples (green) ---
    ax.annotate("", xy=(3, y_sub + h_sub), xytext=(3, y_part - h_part),
                arrowprops=ARROW_PROPS)
    ax.annotate("", xy=(7, y_sub + h_sub), xytext=(7, y_part - h_part),
                arrowprops=ARROW_PROPS)
    ax.text(3, y_sub, "Subsample $S_N$",
            ha="center", va="center", size=11, bbox=BOX_STYLE_SAMPLE)
    ax.text(7, y_sub, "Subsample $S_N'$",
            ha="center", va="center", size=11, bbox=BOX_STYLE_SAMPLE)

    # --- Evaluation (purple) ---
    ax.annotate("", xy=(3, y_eval + h_eval), xytext=(3, y_sub - h_sub),
                arrowprops=ARROW_PROPS)
    ax.annotate("", xy=(7, y_eval + h_eval), xytext=(7, y_sub - h_sub),
                arrowprops=ARROW_PROPS)
    eval_text = "Evaluate Avg Score\n(R2/AUC) per Model\nacross $N$ datasets"
    ax.text(3, y_eval, eval_text, ha="center", va="center", size=10, bbox=BOX_STYLE_EVAL)
    ax.text(7, y_eval, eval_text, ha="center", va="center", size=10, bbox=BOX_STYLE_EVAL)

    # --- Rankings (orange) ---
    ax.annotate("", xy=(3, y_rank + h_rank), xytext=(3, y_eval - h_eval),
                arrowprops=ARROW_PROPS)
    ax.annotate("", xy=(7, y_rank + h_rank), xytext=(7, y_eval - h_eval),
                arrowprops=ARROW_PROPS)
    ax.text(3, y_rank, "Compute Rankings\n$R_N$",
            ha="center", va="center", size=11, bbox=BOX_STYLE_PROC)
    ax.text(7, y_rank, "Compute Rankings\n$R_N'$",
            ha="center", va="center", size=11, bbox=BOX_STYLE_PROC)

    # --- Kendall-τ (red) ---
    ax.annotate("", xy=(5, y_tau + h_tau), xytext=(3, y_rank - h_rank),
                arrowprops=ARROW_PROPS, zorder=1)
    ax.annotate("", xy=(5, y_tau + h_tau), xytext=(7, y_rank - h_rank),
                arrowprops=ARROW_PROPS, zorder=1)
    ax.text(5, y_tau, "Kendall $\\tau(R_1, R_2)$",
            ha="center", va="center", size=12, bbox=BOX_STYLE_TAU)

    plt.tight_layout()
    save_figure(fig, "sampling_diagram_two_benchmarks_convergence")
    plt.close(fig)


if __name__ == "__main__":
    plot_sampling_schema()
