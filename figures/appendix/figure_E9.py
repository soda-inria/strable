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

from figures._main import (
    add_tau_below_legend,
    dtype_map,
    encoder_map,
    get_encoder_color,
    learner_map,
    load_results,
    load_run,
    save_figure,
)


SCORE_COL = 'score'


def _add_meta(df):
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    return df


def _load_uniform_impute():
    df = _add_meta(load_run('imputed'))
    df['Missing_value_treatment'] = 'uniform_impute'
    return df


def _format_label(enc, lrn):
    return enc if enc.strip() == lrn.strip() else f"{enc}\n{lrn}"


def _color_for(enc, lrn):
    return get_encoder_color(enc if enc.strip() == lrn.strip() else enc)


def plot_missing_values_handling():
    results = load_results()
    df_impute = _load_uniform_impute()

    methods_to_retrieve = df_impute['method'].unique()
    datasets = df_impute['data_name'].unique().tolist()

    # Mambular is not in the main run: its native-handling scores come from its own run.
    native = pd.concat([results, _add_meta(load_run('mambular'))], ignore_index=True)
    df_none = native[
        native['method'].isin(methods_to_retrieve)
        & native['data_name'].isin(datasets)
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 22), sharey=False)
    fig.subplots_adjust(wspace=0.75)

    datasets_panels = [df_none_sorted, df_impute_sorted]
    score_cols      = ['none', 'uniform_impute']
    titles = [
        'Current policy: native handling\n+ mean imputation for Ridge',
        'Imputation: mean(numeric)\n+ mode(non-numeric)',
    ]
    xlims = [(0.2, 0.8), (0.5, 0.8)]   # left starts at 0.2 so Mambular shows
    bar_h = 0.6

    for ax, data, col, title, xlim in zip(axes, datasets_panels, score_cols, titles, xlims):
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

    seen = {}
    for _, r in df_pivot.iterrows():
        enc = r['encoder']
        if enc not in seen:
            seen[enc] = _color_for(r['encoder'], r['learner'])
    legend_handles = [mpatches.Patch(facecolor=c, label=e) for e, c in seen.items()]
    legend = axes[1].legend(
        handles=legend_handles,
        title='Encoder', title_fontsize=20, fontsize=16,
        loc='upper left', bbox_to_anchor=(1.015, 0.55),
        handlelength=2.2, handleheight=1.3, labelspacing=0.5,
        frameon=True, edgecolor='lightgray', ncol=1,
    )
    add_tau_below_legend(legend, f"Kendall's $\\tau$ = {tau_impute:.2f}\n(vs. native)",
                         fontsize=17, style='italic')

    save_figure(fig, "missing_values_handling_none_all_imputation")
    plt.close(fig)


if __name__ == "__main__":
    plot_missing_values_handling()
