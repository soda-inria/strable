"""Figure 5(b) — stability across data-preparation choices"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from figures._main import save_figure


# Hardcoded τ values — match the rendered paper figure. Do NOT edit
# without re-deriving them from the corresponding ablation result CSVs.
LABELS = [
    "No Missing Values\nImputation",
    "Target\nTransformations",
    "No Feature\nEngineering",
    "Subsampling\n(75k rows)",
]
TAUS = [0.83, 0.95, 0.70, 0.96]


def plot_data_prep_stability():
    # Sort by τ descending so the highest stays at the top.
    order = np.argsort(TAUS)[::-1]
    labels_sorted = [LABELS[i] for i in order]
    taus_sorted   = [TAUS[i]   for i in order]

    fig, ax = plt.subplots(figsize=(4, 3.2))

    # mako_r palette, truncated to skip the very-light and very-dark ends.
    ranks = np.argsort(np.argsort(taus_sorted))
    n = len(taus_sorted)
    palette_positions = 0.15 + 0.70 * ranks / max(n - 1, 1)
    mako_cmap = sns.color_palette("mako_r", as_cmap=True)
    colors = [mako_cmap(p) for p in palette_positions]

    bars = ax.barh(
        labels_sorted, taus_sorted,
        color=colors, edgecolor='black', linewidth=0.8, height=0.8,
        alpha=0.95,
    )

    ax.set_xlim(0.65, 1.0)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(['0.7', '0.8', '0.9', '1.0'], fontsize=10)
    ax.set_xlabel('Kendall $\\tau$ (modified data\nvs original)', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='x', linestyle='-', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel('')
    sns.despine(top=True, right=True)

    # In-bar value labels (white text, right-aligned just inside the end).
    for bar, value in zip(bars, taus_sorted):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            value - 0.002, y, f'{value:.2f}',
            va='center', ha='right',
            fontsize=9, fontweight='bold', color='white',
        )

    # Red↔green double-arrow below the x-label.
    trans = ax.transAxes
    ax.annotate('', xy=(-0.3, -0.36), xytext=(0.3, -0.36),
                xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2),
                annotation_clip=False)
    ax.annotate('', xy=(0.98, -0.36), xytext=(0.4, -0.36),
                xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
                annotation_clip=False)
    ax.text(-0.35, -0.42, 'data modification\nchanges ranking',
            transform=trans, color='red', va='top', ha='left',
            fontsize=10, fontweight='bold', linespacing=1.15)
    ax.text(1.0, -0.42, 'data modification\npreserves ranking',
            transform=trans, color='green', va='top', ha='right',
            fontsize=10, fontweight='bold', linespacing=1.15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    save_figure(fig, "stability_barplot_across_data_transformation_v2")
    plt.close(fig)


if __name__ == "__main__":
    plot_data_prep_stability()
