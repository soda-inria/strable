"""Figure E.10 — Two side-by-side bar charts comparing pipeline rankings between:
* ``Full Dataset`` — re-ran on the un-subsampled tables (uses the
  ``REBUTTALS_tfidf_minilm6_xgb_extratrees_contexttab`` rebuttal CSV;
  for RealMLP, the additional ``realMLP_..._30pca`` rebuttal CSV).
* ``Sampled Dataset (75k)`` — pulled from the main results, using the
  default 75k cap."""

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

METHODS_FROM_RESULTS = [
    'num-str_llm-all-MiniLM-L6-v2_extrees_default',
    'num-str_llm-all-MiniLM-L6-v2_xgb_default',
    'num-str_llm-all-MiniLM-L6-v2_ridge_default',
    'num-str_llm-all-MiniLM-L6-v2_realmlp',
    'num-str_llm-e5-base-v2_extrees_default',
    'num-str_llm-e5-base-v2_ridge_default',
    'num-str_llm-e5-base-v2_xgb_default',
    'num-str_llm-e5-base-v2_realmlp',
    'num-str_llm-jasper-token-comp-0.6b_xgb_default',
    'num-str_llm-jasper-token-comp-0.6b_extrees_default',
    'num-str_llm-jasper-token-comp-0.6b_ridge_default',
    'num-str_llm-jasper-token-comp-0.6b_realmlp',
    'num-str_tabvec_extrees_default',
    'num-str_tabvec_realmlp',
    'num-str_tabvec_xgb_default',
    'num-str_tabvec_ridge_default',
    'num-str_contexttab_contexttab',
]

REALMLP_METHODS = [
    'num-str_tabvec_realmlp',
    'num-str_llm-all-MiniLM-L6-v2_realmlp',
    'num-str_llm-e5-base-v2_realmlp',
    'num-str_llm-jasper-token-comp-0.6b_realmlp',
]


def _add_meta(df, append_default_for=None):
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    learner = meta[2]
    if append_default_for is not None:
        learner = (learner + '_default').where(
            learner.isin(append_default_for), learner,
        )
    df['learner'] = learner.replace(learner_map)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    return df


def _load_full():
    df = pd.read_csv(
        f"{path_configs['compiled_results']}/"
        "result_REBUTTALS_tfidf_minilm6_xgb_extratrees_contexttab.csv"
    )
    df['data_name'] = df['data_name'].str.replace('-FULL', '', regex=False)
    df = _add_meta(df, append_default_for=['extrees', 'xgb', 'ridge'])
    df['Dataset_sampling'] = 'Full_Dataset'
    return df


def _load_full_realmlp(common_datasets):
    df = pd.read_csv(
        f"{path_configs['compiled_results']}/"
        "result_REBUTTALS_realMLP_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv"
    )
    df = _add_meta(df, append_default_for=None)
    df['Dataset_sampling'] = 'Sampled_Dataset_75k'
    return df[
        df['data_name'].isin(common_datasets)
        & df['method'].isin(REALMLP_METHODS)
    ]


def _format_label(enc, lrn):
    return enc if enc.strip() == lrn.strip() else f"{enc}\n{lrn}"


def _color_for(enc, lrn):
    return get_encoder_color(enc if enc.strip() == lrn.strip() else enc)


def plot_full_vs_sampled():
    results = load_results()
    df_full = _load_full()

    # Datasets with full coverage of every full-run method.
    methods_full = df_full['method'].unique().tolist()
    common_datasets = set(df_full[df_full['method'] == methods_full[0]]['data_name'].unique())
    for method in methods_full[1:]:
        common_datasets &= set(
            df_full[df_full['method'] == method]['data_name'].unique()
        )

    df_sampled = results[
        results['method'].isin(METHODS_FROM_RESULTS)
        & results['data_name'].isin(common_datasets)
    ].copy()
    df_sampled['Dataset_sampling'] = 'Sampled_Dataset_75k'

    df_sampled_realmlp = _load_full_realmlp(common_datasets)

    combined = pd.concat(
        [df_full, df_sampled, df_sampled_realmlp],
        axis=0, ignore_index=True,
    )

    df_pivot = (
        combined
        .groupby(['encoder_learner', 'data_name', 'Dataset_sampling'],
                 as_index=False)[SCORE_COL]
        .mean()
        .pivot_table(index='encoder_learner',
                     columns='Dataset_sampling', values=SCORE_COL)
        .reset_index()
    )
    df_pivot[['encoder', 'learner']] = (
        df_pivot['encoder_learner'].str.rsplit(' - ', n=1, expand=True)
    )
    df_pivot['label'] = df_pivot.apply(
        lambda r: _format_label(r['encoder'], r['learner']), axis=1
    )

    df_full_sorted    = df_pivot.sort_values('Full_Dataset', ascending=True)
    df_sampled_sorted = df_pivot.sort_values('Sampled_Dataset_75k', ascending=True)

    merged = df_pivot[['encoder_learner', 'Full_Dataset', 'Sampled_Dataset_75k']].dropna()
    merged['rank_full']    = merged['Full_Dataset'].rank(ascending=False)
    merged['rank_sampled'] = merged['Sampled_Dataset_75k'].rank(ascending=False)
    tau, _ = kendalltau(merged['rank_full'], merged['rank_sampled'])

    fig, axes = plt.subplots(1, 2, figsize=(6, 10), sharey=False)
    fig.subplots_adjust(wspace=1.4, bottom=0.05, top=0.82)

    panels = [df_full_sorted, df_sampled_sorted]
    cols   = ['Full_Dataset', 'Sampled_Dataset_75k']
    titles = ['Full Dataset', 'Sampled Dataset (75k)']
    bar_h = 0.6

    for ax, data, col, title in zip(axes, panels, cols, titles):
        colors = [_color_for(r['encoder'], r['learner'])
                  for _, r in data.iterrows()]
        y = np.arange(len(data))
        ax.barh(y, data[col].values, height=bar_h,
                color=colors, edgecolor='white', linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(data['label'].values, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=40)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, 1)

    fig.supxlabel('Avg Score ($R^2$ & AUC)', fontsize=16, y=-0.01)
    fig.text(0.88, 0.82,
             f"Kendall's $\\tau$ = {tau:.2f}",
             ha='center', fontsize=13, style='italic', zorder=10)

    seen = {}
    for _, r in df_pivot.iterrows():
        enc = r['encoder']
        if enc not in seen:
            seen[enc] = _color_for(r['encoder'], r['learner'])
    legend_handles = [mpatches.Patch(facecolor=c, label=e) for e, c in seen.items()]
    fig.legend(
        handles=legend_handles,
        title='', title_fontsize=13, fontsize=14,
        bbox_to_anchor=(0.96, 0.88),
        frameon=True, edgecolor='lightgray', ncol=3,
    )

    save_figure(fig, "full_vs_sampled_pipeline_rankings")
    plt.close(fig)


if __name__ == "__main__":
    plot_full_vs_sampled()
