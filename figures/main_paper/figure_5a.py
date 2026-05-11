"""Figure 5(a) — Leave-one-domain-out: stability across application fields"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kendalltau

from figures._main import load_results, save_figure


N_BOOTSTRAP = 200


def _build_pivot_and_categories(results):
    """Restrict to Num+Str pipelines that have a complete score column,
    then return ``(pivot_source, dataset_to_category)`` keyed by data_name.
    The category labels include the domain size (e.g. ``Health (30)``).
    """
    df = results[
        (results['dtype'] == 'Num+Str')
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()
    pivot = df.pivot_table(
        index='data_name', columns='method_polished',
        values='score', aggfunc='mean',
    )
    pivot = pivot[[c for c in pivot.columns if 'Num+Str' in c]]
    pivot = pivot.dropna(axis=1)

    cat_counts = results.groupby('category')['data_name'].nunique()
    label_map = {cat: f"{cat} ({n})" for cat, n in cat_counts.items()}
    df_with_labels = results.copy()
    df_with_labels['category_with_ds_count'] = df_with_labels['category'].map(label_map)
    dataset_to_category = (
        df_with_labels[df_with_labels['data_name'].isin(pivot.index)]
        .groupby('data_name')['category_with_ds_count'].first()
    )
    return pivot, dataset_to_category


def _compute_loso(pivot_source, dataset_to_category, ranking_full):
    """Per-domain Kendall-τ vs the full-benchmark ranking."""
    sources = dataset_to_category.unique()
    rows = []
    for src in sources:
        in_src = dataset_to_category[dataset_to_category == src].index
        if len(in_src) < 2:
            continue
        ranks_src = pivot_source.loc[in_src].mean().rank(ascending=False)
        tau, _ = kendalltau(ranks_src, ranking_full)
        rows.append({'Category': src, 'Correlation': tau, 'K': len(in_src)})
    return (
        pd.DataFrame(rows)
        .sort_values('Correlation', ascending=False)
        .dropna(subset=['Correlation'])
    )


def _null_band(K, pivot_source, ranking_full, all_datasets, rng,
               n_boot=N_BOOTSTRAP):
    """Distribution of Kendall-τ between a random size-K subset's ranking
    and the full-benchmark ranking. Returns the (2.5%, 50%, 97.5%) quantiles."""
    taus = []
    for _ in range(n_boot):
        sampled = rng.choice(all_datasets, size=K, replace=False)
        ranks_sample = pivot_source.loc[sampled].mean().rank(ascending=False)
        tau, _ = kendalltau(ranks_sample, ranking_full)
        if not np.isnan(tau):
            taus.append(tau)
    return np.percentile(taus, [2.5, 50, 97.5])


def plot_leave_one_domain_out():
    results = load_results()
    pivot_source, dataset_to_category = _build_pivot_and_categories(results)

    rng = np.random.default_rng(0)
    ranking_full = pivot_source.mean().rank(ascending=False)

    df_loso = _compute_loso(pivot_source, dataset_to_category, ranking_full)
    all_datasets = pivot_source.index.to_numpy()
    null_by_K = {
        K: _null_band(K, pivot_source, ranking_full, all_datasets, rng)
        for K in sorted(df_loso['K'].unique())
    }

    plt.figure(figsize=(4, 3.2))
    ax = plt.gca()
    y_positions = np.arange(len(df_loso))

    # Size-matched null band (light grey) + dashed median behind each bar.
    for i, (_, row) in enumerate(df_loso.iterrows()):
        lo, med, hi = null_by_K[row['K']]
        ax.barh(y_positions[i], hi - lo, left=lo, height=0.95,
                color='lightgrey', alpha=0.6, zorder=0)
        ax.plot([med, med], [y_positions[i] - 0.4, y_positions[i] + 0.4],
                color='dimgrey', linestyle='--', lw=1.0, zorder=1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(df_loso)))
    bars = ax.barh(y_positions, df_loso['Correlation'],
                   color=colors, alpha=0.9, height=0.8, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{c} (n={k})" for c, k in zip(df_loso['Category'], df_loso['K'])],
        fontsize=11,
    )
    ax.invert_yaxis()
    plt.xlabel(
        'Kendall $\\tau$ (held-out\ndomain vs full benchmark)',
        fontsize=12, x=-0.5, ha='left',
    )
    plt.xlim(0.0, 0.95)
    plt.grid(axis='x', alpha=0.3)
    sns.despine(top=True, right=True)

    # In-bar value labels — special-cased for "Food" because the bar is
    # too narrow to fit the label inside.
    for bar, category, value in zip(bars, df_loso['Category'], df_loso['Correlation']):
        y = bar.get_y() + bar.get_height() / 2
        if category.lower().startswith('food'):
            plt.text(value / 4, y, f'{value:.2f}',
                     va='center', ha='left',
                     fontsize=11, fontweight='bold', color='black')
        else:
            plt.text(value / 1.5, y, f'{value:.2f}',
                     va='center', ha='center',
                     fontsize=11, fontweight='bold', color='white')

    legend_handles = [
        Patch(facecolor='lightgrey', alpha=0.6, label='Null 95% CI'),
        Line2D([0], [0], color='dimgrey', linestyle='--', lw=1.0, label='Null median'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.1), ncol=2)

    # Red↔green double-arrow below the x-label, with the "domain behaves..."
    # captions on either side.
    trans = ax.transAxes
    ax.annotate('', xy=(-0.65, -0.38), xytext=(0.1, -0.38),
                xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2),
                annotation_clip=False)
    ax.annotate('', xy=(0.98, -0.38), xytext=(0.23, -0.38),
                xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
                annotation_clip=False)
    ax.text(-0.65, -0.42, 'domain behaves\ndifferently',
            transform=trans, color='red', va='top', ha='left',
            fontsize=10, fontweight='bold', linespacing=1.15)
    ax.text(0.98, -0.42, 'domain behaves\nsimilarly',
            transform=trans, color='green', va='top', ha='right',
            fontsize=10, fontweight='bold', linespacing=1.15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    save_figure(plt.gcf(), "leave_one_category_out_v7_with_null")
    plt.close()


if __name__ == "__main__":
    plot_leave_one_domain_out()
