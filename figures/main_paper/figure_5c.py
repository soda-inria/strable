"""Figure 5(c) — Kendall-τ between the encoder rankings on the lower- and upper-33rd
percentile of each string meta-feature."""

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
from scipy import stats

from configs.path_configs import path_configs
from figures._main import load_results, save_figure, selected_encoders


SCORE_COL = 'score'

# Six meta-features ranked-and-plotted; order is determined dynamically by
# Kendall-τ ascending (smallest τ = biggest disruptor at the top).
META_FEATURES = {
    'avg_words_per_cell': 'Avg Words/Cell\n(length)',
    'uniqueness':         'Uniqueness\n(cardinality)',
    'prop_multiword':     'Prop. Multiword\n(phrase-ness)',
    'symbol_density':     'Symbol Density\n(code-ness)',
    'dict_hit_rate':      'Dict. Hit Rate\n(naturalness)',
    'stopword_density':   'Stopword Density\n(prose-ness)',
}

E2E_ENCODERS = {'ContextTab', 'TabSTAR', 'CatBoost'}


def _build_dataset_meta(df_structure):
    """Per-dataset median of each meta-feature."""
    return (
        df_structure
        .groupby('dataset')[list(META_FEATURES.keys())]
        .median()
        .reset_index()
    )


def _build_results_fair(results, dataset_meta):
    """Restrict to Num+Str pipelines, attach per-dataset meta-features, and
    keep only the best learner per non-E2E encoder (so the encoder ranking
    isn't biased by how many learners each encoder is paired with)."""
    df = results[
        (results['dtype'] == 'Num+Str')
        & (results['encoder'].isin(selected_encoders))
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()
    df = df.merge(dataset_meta, left_on='data_name', right_on='dataset', how='inner')

    best_learner = (
        df[~df['encoder'].isin(E2E_ENCODERS)]
        .groupby(['encoder', 'learner'])[SCORE_COL]
        .mean()
        .reset_index()
        .sort_values(SCORE_COL, ascending=False)
        .groupby('encoder')
        .first()
        .reset_index()[['encoder', 'learner']]
    )
    df_modular_best = df[~df['encoder'].isin(E2E_ENCODERS)].merge(
        best_learner, on=['encoder', 'learner'], how='inner',
    )
    df_e2e = df[df['encoder'].isin(E2E_ENCODERS)]
    return pd.concat([df_modular_best, df_e2e], ignore_index=True)


def _encoder_ranking(df_subset, encoders):
    """Return encoders sorted by mean score on this subset (descending)."""
    means = df_subset.groupby('encoder')[SCORE_COL].mean().reindex(encoders)
    return means.sort_values(ascending=False).index.tolist()


def _kendall_tau_from_rankings(rank1, rank2):
    """Kendall-τ between two ordered lists of encoder names."""
    common = [e for e in rank1 if e in rank2]
    if len(common) < 2:
        return np.nan
    pos1 = [rank1.index(e) for e in common]
    pos2 = [rank2.index(e) for e in common]
    tau, _ = stats.kendalltau(pos1, pos2)
    return tau


def _compute_kendall_per_feature(df_results_fair, present_encoders):
    """For each meta-feature, split datasets into lower-33% and upper-33%
    by per-dataset median, rank encoders on each split, and compute τ."""
    rows = []
    for feat, label in META_FEATURES.items():
        feat_per_dataset = df_results_fair.groupby('data_name')[feat].median()
        low_thresh  = feat_per_dataset.quantile(0.33)
        high_thresh = feat_per_dataset.quantile(0.67)
        low_datasets  = feat_per_dataset[feat_per_dataset <= low_thresh].index
        high_datasets = feat_per_dataset[feat_per_dataset >= high_thresh].index
        df_low  = df_results_fair[df_results_fair['data_name'].isin(low_datasets)]
        df_high = df_results_fair[df_results_fair['data_name'].isin(high_datasets)]
        if df_low['data_name'].nunique() < 5 or df_high['data_name'].nunique() < 5:
            continue
        rank_low  = _encoder_ranking(df_low,  present_encoders)
        rank_high = _encoder_ranking(df_high, present_encoders)
        rows.append({
            'feature': feat,
            'label':   label,
            'tau':     _kendall_tau_from_rankings(rank_low, rank_high),
        })
    return pd.DataFrame(rows).sort_values('tau')


def plot_meta_feature_disruption():
    results = load_results()
    df_structure = pd.read_csv(
        f"{path_configs['base_path']}/df_structure_VSE_STRABLE_CARTE_TTB.csv"
    )
    df_structure = df_structure[df_structure['col_type_heuristic'] != 'datetime']
    dataset_meta = _build_dataset_meta(df_structure)

    df_results_fair = _build_results_fair(results, dataset_meta)
    present_encoders = [
        e for e in selected_encoders if e in df_results_fair['encoder'].unique()
    ]

    kendall_df = _compute_kendall_per_feature(df_results_fair, present_encoders)
    print("\nKendall-τ stability of encoder rankings across meta-features:")
    print(kendall_df[['label', 'tau']].to_string())

    # Plot — bars colour-graded by rank (smallest τ = darkest).
    fig, ax = plt.subplots(figsize=(4, 3.2))
    vals = kendall_df['tau'].astype(float).tolist()
    ranks = np.argsort(np.argsort(vals))
    n = len(vals)
    palette_positions = 0.20 + 0.65 * ranks / max(n - 1, 1)
    rocket_cmap = sns.color_palette("rocket_r", as_cmap=True)
    colors = [rocket_cmap(p) for p in palette_positions]

    bars = ax.barh(
        [r['label'] for r in kendall_df.to_dict('records')],
        vals,
        color=colors, edgecolor='black', linewidth=0.8, height=0.7,
        alpha=0.95,
    )

    ax.set_xlabel('Kendall-τ (low vs high\npercentile)', fontsize=12)
    ax.set_xlim(0, 0.45)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='x', linestyle='-', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    # In-bar value labels.
    for bar, value in zip(bars, vals):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            value - 0.004, y, f'{value:.2f}',
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
    ax.text(-0.3, -0.42, 'meta-feature\ndisrupts ranking',
            transform=trans, color='red', va='top', ha='left',
            fontsize=10, fontweight='bold', linespacing=1.15)
    ax.text(0.98, -0.42, 'meta-feature\nhas little effect',
            transform=trans, color='green', va='top', ha='right',
            fontsize=10, fontweight='bold', linespacing=1.15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    save_figure(fig, "stability_encoder_rankings_vs_stringcolumns_indexes_selectedLLMs")
    plt.close(fig)


if __name__ == "__main__":
    plot_meta_feature_disruption()
