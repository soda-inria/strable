"""Figure 1(b) — Average score per learner under three feature subsets: Num+Str (full
table), Str-only (string columns dropped numeric features), Num-only
(numeric columns dropped string features)."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from figures._main import (
    Y_METRIC_LABELS,
    get_learner_hatch,
    load_results,
    save_figure,
    selected_encoders,
)


DTYPE_PALETTE = {
    'Num':     '#1f77b4',
    'Num+Str': '#d62728',
    'Str':     '#2ca02c',
}
DTYPE_DISPLAY = {'Num+Str': 'Num+Str', 'Str': 'Str-only', 'Num': 'Num-only'}
SCORE_COL = 'score'


def _add_tuned_hatch_overlay(ax, sort_order):
    """For every tuned learner, overlay a white-line hatch on its bars
    plus a clean black border. Pure visual — no data is changed."""
    for i, learner_name in enumerate(sort_order):
        hatch = get_learner_hatch(learner_name)
        if not hatch:
            continue
        for container in ax.containers:
            if not isinstance(container[i], mpatches.Rectangle):
                continue
            bar = container[i]
            ax.add_patch(mpatches.Rectangle(
                (bar.get_x(), bar.get_y()),
                bar.get_width(), bar.get_height(),
                fill=False, hatch=hatch,
                edgecolor='white', linewidth=0, alpha=0.6,
            ))
            ax.add_patch(mpatches.Rectangle(
                (bar.get_x(), bar.get_y()),
                bar.get_width(), bar.get_height(),
                fill=False, edgecolor='black', linewidth=1,
            ))


def plot_perf_per_learner_by_dtype():
    results = load_results()

    df_raw = results[
        results['dtype'].isin(['Num+Str', 'Num', 'Str'])
        & results['encoder'].isin(selected_encoders)
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()

    sort_order = (
        df_raw.groupby(['learner', 'dtype'])[SCORE_COL]
        .mean()
        .unstack()
        .sort_values('Num+Str', ascending=False)
        .index
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(3, 5))

    sns.barplot(
        data=df_raw,
        y='learner', x=SCORE_COL,
        hue='dtype',
        order=sort_order,
        hue_order=['Num+Str', 'Str', 'Num'],
        palette=DTYPE_PALETTE,
        edgecolor='black', linewidth=1,
        errorbar=('ci', 95),
        capsize=0.1,
        err_kws={'linewidth': 1.5, 'color': 'black'},
        ax=ax,
    )
    _add_tuned_hatch_overlay(ax, sort_order)

    ax.set_xlabel(
        f'Avg {Y_METRIC_LABELS[SCORE_COL]} ($R^2$ & AUC) with 95% CI',
        fontsize=18, x=0.9, ha='right',
    )
    ax.set_xlim(0.3, 0.79)
    ax.set_ylabel('')
    # ytick fontsize 18 ported from salts (paper-matching).
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    labels = [DTYPE_DISPLAY.get(l, l) for l in labels]
    handles.append(mpatches.Patch(
        facecolor='gray', edgecolor='white', hatch='///', label='Tuned Model',
    ))
    labels.append("Tuned")

    # ``frameon=False`` and fontsize 16 ported from salts (paper-matching).
    ax.legend(
        handles=handles, labels=labels,
        loc='lower center',
        bbox_to_anchor=(0.1, 1.02),
        ncol=4,
        fontsize=16,
        framealpha=0.9,
        handlelength=1.0, handleheight=1.0,
        handletextpad=0.4, columnspacing=0.3,
        borderaxespad=0.0,
        frameon=False,
    )

    save_figure(fig, "avg_score_performance_by_learner_num+str_num_str_selectedLLMs")
    plt.close(fig)


if __name__ == "__main__":
    plot_perf_per_learner_by_dtype()
