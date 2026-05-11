"""Figure C.1 — dataset count per era bucket, R² distribution, AUC distribution."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from figures._main import Y_METRIC_LABELS, load_results, save_figure


SCORE_COL = 'score'
ORDERED_ERAS = ['Pre-2000', '2000-2009', '2010-Present']


def _plot_year_histogram(results):
    """Bar chart of dataset count per era bucket."""
    hist_df = (
        results
        .groupby('year_macro_category', as_index=False)['data_name']
        .nunique()
    )
    hist_df['year_macro_category'] = pd.Categorical(
        hist_df['year_macro_category'], categories=ORDERED_ERAS, ordered=True,
    )
    hist_df = hist_df.sort_values('year_macro_category')

    fig = plt.figure(figsize=(5, 6))
    bars = plt.bar(
        hist_df['year_macro_category'], hist_df['data_name'],
        color='#ff7f0e', edgecolor='#333333', linewidth=1.2,
    )
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            int(bar.get_height()),
            va='bottom', ha='center',
            fontsize=16, fontweight='bold',
        )
    plt.ylabel('Number of Datasets', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.ylim(0, hist_df['data_name'].max() + 5)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    save_figure(fig, "histogram_year_ds_count")
    plt.close(fig)


def _plot_score_distribution(results, task_filter, *, color, x_label, name):
    """Histogram (with KDE) of per-dataset average score for the given task
    filter. ``task_filter`` is a callable on the ``task`` column."""
    subset = results[
        (results['dtype'] == 'Num+Str')
        & task_filter(results['task'])
        & (results['encoder'] != 'TabPFN-2.5')   # drop TabPFN encoder
    ].groupby(['data_name'], as_index=False)[SCORE_COL].mean()

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(6, 5))
    ax = sns.histplot(
        data=subset, x=SCORE_COL, bins=15, kde=True,
        color=color, edgecolor='black',
    )
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Per-bin counts annotated above each bar.
    for patch in ax.patches:
        count = int(round(patch.get_height()))
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                str(count),
                ha='center', va='bottom', fontsize=10,
            )

    save_figure(fig, name)
    plt.close(fig)


def plot_dataset_exploration():
    results = load_results()

    _plot_year_histogram(results)
    _plot_score_distribution(
        results,
        task_filter=lambda t: t.isin(['regression']),
        color='skyblue',
        x_label=f"R2 {Y_METRIC_LABELS[SCORE_COL]}",
        name=f"distribution_plot_{SCORE_COL}_r2_regression",
    )
    _plot_score_distribution(
        results,
        task_filter=lambda t: ~t.isin(['regression']),
        color='orange',
        x_label=f"AUC {Y_METRIC_LABELS[SCORE_COL]}",
        name=f"distribution_plot_{SCORE_COL}_roc_auc_classification",
    )


if __name__ == "__main__":
    plot_dataset_exploration()
