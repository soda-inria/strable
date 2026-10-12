"""Figure E.7 — Side-by-side bars per pipeline of average score under raw vs manually
feature-engineered tables (parsed dates / ordinal encoding / range extraction
etc.), for 17 pipelines, on the tables where all of them have both raw and
engineered results.
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from figures._main import (
    Y_METRIC_LABELS,
    add_tau_below_legend,
    dtype_map,
    encoder_map,
    learner_map,
    load_results,
    load_run,
    save_figure,
)


SCORE_COL = 'score'

METHODS_TO_RETRIEVE = [
    "num-str_llm-all-MiniLM-L6-v2_extrees_default",
    "num-str_llm-all-MiniLM-L6-v2_xgb_default",
    "num-str_llm-all-MiniLM-L6-v2_ridge_default",
    "num-str_llm-all-MiniLM-L6-v2_tabpfn_default",
    "num-str_tabvec_extrees_default",
    "num-str_tabvec_xgb_default",
    "num-str_tabvec_ridge_default",
    "num-str_tabvec_tabpfn_default",
    "num-str_llm-llama-3.1-8b_ridge_default",
    "num-str_llm-llama-3.1-8b_extrees_default",
    "num-str_llm-llama-3.1-8b_xgb_default",
    "num-str_llm-llama-3.1-8b_tabpfn_default",
    "num-str_llm-qwen3-8b_ridge_default",
    "num-str_llm-qwen3-8b_extrees_default",
    "num-str_llm-qwen3-8b_xgb_default",
    "num-str_llm-qwen3-8b_tabpfn_default",
    "num-str_contexttab_contexttab",
]


def _load_feature_engineered():
    """Load the feature-engineered ablation results CSV and re-derive
    score / dtype / encoder / learner columns so it matches the schema of
    ``results``."""
    df = load_run('feature_eng')
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)

    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    df['Feature_engineering'] = 'Engineered Features'
    return df


def _format_label(name):
    parts = name.split(' - ')
    if len(parts) >= 2:
        encoder, learner = parts[0].strip(), parts[-1].strip()
        if encoder == learner:
            return encoder
        return '\n'.join(p.strip() for p in parts)
    return name


def plot_raw_vs_engineered():
    results = load_results()
    df_eng = _load_feature_engineered()
    df_eng = df_eng[df_eng['method'].isin(METHODS_TO_RETRIEVE)]
    df_raw = results[results['method'].isin(METHODS_TO_RETRIEVE)].copy()
    df_raw['Feature_engineering'] = 'Raw Features'

    # Compare only the tables where all the pipelines have both raw and
    # engineered results.
    def tables_with_all(df):
        n_methods = df.groupby('data_name')['method'].nunique()
        return set(n_methods[n_methods == len(METHODS_TO_RETRIEVE)].index)

    tables = tables_with_all(df_eng) & tables_with_all(df_raw)
    combined = pd.concat([df_eng, df_raw], axis=0, ignore_index=True)
    combined = combined[combined['data_name'].isin(tables)]

    df_plot = (
        combined
        .groupby(['Feature_engineering', 'encoder_learner'])[SCORE_COL]
        .mean()
        .reset_index()
    )
    df_pivot = df_plot.pivot(
        index='encoder_learner', columns='Feature_engineering', values=SCORE_COL,
    )
    df_pivot['_mean'] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values('_mean', ascending=True).drop(columns='_mean')

    rank_raw = df_pivot['Raw Features'].rank()
    rank_eng = df_pivot['Engineered Features'].rank()
    tau, _ = kendalltau(rank_raw, rank_eng)

    fig, ax = plt.subplots(figsize=(3, 9))
    n = len(df_pivot)
    y = np.arange(n)
    bar_height = 0.35
    hue_colors = {'Raw Features': '#5A9BD5', 'Engineered Features': '#ED7D31'}

    ax.barh(y + bar_height / 2, df_pivot['Raw Features'],
            height=bar_height, color=hue_colors['Raw Features'],
            edgecolor='white', linewidth=0.5, label='Raw')
    ax.barh(y - bar_height / 2, df_pivot['Engineered Features'],
            height=bar_height, color=hue_colors['Engineered Features'],
            edgecolor='white', linewidth=0.5, label='Engineered')

    ax.set_xlim(0.6, 0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_format_label(p) for p in df_pivot.index],
        fontsize=8, linespacing=1.3,
    )
    ax.set_xlabel(f'Avg {Y_METRIC_LABELS[SCORE_COL]}  ($R^2 & AUC$)', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

    legend = ax.legend(fontsize=10, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    add_tau_below_legend(legend, f"Kendall's τ = {tau:.1f}", fontsize=11, style='italic')

    save_figure(fig, f"raw_vs_engineered_features_rankings_v1_hue_{SCORE_COL}")
    plt.close(fig)


if __name__ == "__main__":
    plot_raw_vs_engineered()
