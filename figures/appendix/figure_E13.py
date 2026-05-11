"""Figure E.13 — bar chart of average score per encoder on Num+Str
datasets"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import seaborn as sns

from figures._main import (
    Y_METRIC_LABELS,
    get_encoder_color,
    load_results,
    save_figure,
)


SCORE_COL = 'score'


def plot_encoder_performance():
    results = load_results()
    plot_data = results[
        (results['dtype'] == 'Num+Str')
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()

    encoder_performance = (
        plot_data.groupby('encoder', as_index=False)[SCORE_COL]
        .mean()
        .sort_values(by=SCORE_COL, ascending=False)
    )
    palette_list = [get_encoder_color(enc)
                    for enc in encoder_performance['encoder']]

    fig, ax = plt.subplots(figsize=(5, 15))
    sns.barplot(
        data=encoder_performance,
        y='encoder', x=SCORE_COL,
        palette=palette_list,
        edgecolor='black', linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel(
        f'Average {Y_METRIC_LABELS[SCORE_COL]} ($R^2$ & AUC)',
        fontsize=18, ha='right', x=1.0,
    )
    ax.set_ylabel('')
    ax.set_xlim(0.6, 0.75)
    sns.despine(ax=ax)

    save_figure(fig, "barplot_encoder_performance")
    plt.close(fig)


if __name__ == "__main__":
    plot_encoder_performance()
