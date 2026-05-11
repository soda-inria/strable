"""Figure E.9 — Two side-by-side bar charts comparing pipeline rankings under:
* Native missing-value handling (Ridge gets mean imputation, the rest pass
  NaNs through to the learner) — the default policy of the benchmark.
* Uniform imputation (mean for numeric, mode for non-numeric) applied
  before encoding."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from configs.path_configs import path_configs
from figures._main import (
    dtype_map,
    encoder_map,
    get_encoder_color,
    learner_map,
    load_results,
    save_figure,
)


SCORE_COL = 'score'


def _load_uniform_impute():
    df = pd.read_csv(
        f"{path_configs['compiled_results']}/"
        "REBUTTAL_missing_values_run_uniform_impute.csv"
    )
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = (meta[2] + '_default').replace(learner_map)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    df['Missing_value_treatment'] = 'uniform_impute'
    return df


def _format_label(enc, lrn):
    return enc if enc.strip() == lrn.strip() else f"{enc}\n{lrn}"


def _color_for(enc, lrn):
    return get_encoder_color(enc if enc.strip() == lrn.strip() else enc)


def plot_missing_values_handling():
    results = load_results()
    df_impute = _load_uniform_impute()

    methods_to_retrieve = [m + '_default' for m in df_impute['method'].unique()]
    datasets = df_impute['data_name'].unique().tolist()

    df_none = results[
        results['method'].isin(methods_to_retrieve)
        & results['data_name'].isin(datasets)
    ].copy()
    df_none['Missing_value_treatment'] = 'none'

    combined = pd.concat([df_impute, df_none], axis=0, ignore_index=True)
    df_pivot = (
        combined
        .groupby(['encoder_learner', 'Missing_value_treatment'], as_index=False)[SCORE_COL]
        .mean()
        .pivot(index='encoder_learner',
               columns='Missing_value_treatment', values=SCORE_COL)
        .reset_index()
    )

    df_pivot[['encoder', 'learner']] = (
        df_pivot['encoder_learner'].str.rsplit(' - ', n=1, expand=True)
    )
    df_pivot['label'] = df_pivot.apply(
        lambda r: _format_label(r['encoder'], r['learner']), axis=1
    )

    df_none_sorted   = df_pivot[['label', 'encoder', 'learner', 'none']
                                ].sort_values('none', ascending=True)
    df_impute_sorted = df_pivot[['label', 'encoder', 'learner', 'uniform_impute']
                                ].sort_values('uniform_impute', ascending=True)

    merged = df_pivot[['encoder_learner', 'none', 'uniform_impute']].dropna()
    merged['rank_none']   = merged['none'].rank(ascending=False)
    merged['rank_impute'] = merged['uniform_impute'].rank(ascending=False)
    tau_impute, _ = kendalltau(merged['rank_none'], merged['rank_impute'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=False)
    fig.subplots_adjust(wspace=0.4)

    datasets_panels = [df_none_sorted, df_impute_sorted]
    score_cols      = ['none', 'uniform_impute']
    titles = [
        'Current policy: native handling\n+ mean imputation for Ridge',
        'Imputation: mean(numeric)\n+ mode(non-numeric)',
    ]
    xlim = (0.55, 0.77)
    bar_h = 0.6

    for ax, data, col, title in zip(axes, datasets_panels, score_cols, titles):
        colors = [_color_for(r['encoder'], r['learner'])
                  for _, r in data.iterrows()]
        y = np.arange(len(data))
        ax.barh(y, data[col].values, height=bar_h,
                color=colors, edgecolor='white', linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(data['label'].values, fontsize=16)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
        ax.tick_params(axis='x', labelsize=16)
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(*xlim)

    fig.supxlabel('Avg Score ($R^2$ & AUC)', fontsize=16, y=0.03)
    axes[1].text(
        1.0, 0.08,
        f"Kendall's $\\tau$ = {tau_impute:.2f}\n(vs. native)",
        ha='right', va='bottom', fontsize=14, style='italic',
        transform=axes[1].transAxes,
    )

    seen = {}
    for _, r in df_pivot.iterrows():
        enc = r['encoder']
        if enc not in seen:
            seen[enc] = _color_for(r['encoder'], r['learner'])
    legend_handles = [mpatches.Patch(facecolor=c, label=e) for e, c in seen.items()]
    fig.legend(
        handles=legend_handles,
        title='Encoder', title_fontsize=16, fontsize=11,
        loc='center left', bbox_to_anchor=(0.72, 0.4),
        frameon=True, edgecolor='lightgray', ncol=1,
    )
    plt.tight_layout(rect=[0.5, 0.9, 0.91, 1])

    save_figure(fig, "missing_values_handling_none_all_imputation")
    plt.close(fig)


if __name__ == "__main__":
    plot_missing_values_handling()
