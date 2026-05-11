"""Figure E.7 — Side-by-side bars per pipeline of average score under raw vs manually
feature-engineered tables (44/108 datasets had ablation runs with parsed
dates / ordinal encoding / range extraction etc.).
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

from configs.path_configs import path_configs
from figures._main import (
    Y_METRIC_LABELS,
    dtype_map,
    encoder_map,
    learner_map,
    load_results,
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
    df = pd.read_csv(
        f"{path_configs['compiled_results']}/"
        "result_feature_engineering_tdidf_minilm6_extrees_xgb_contexttab.csv"
    )
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = (
        df['method'].str.split('_', expand=True, n=2)[2]
    )
    df['learner'] = (df['learner'] + '_default').where(
        df['learner'].isin(['ridge', 'xgb', 'extrees', 'tabpfn']),
        df['learner'],
    )
    df['learner'] = df['learner'].replace(learner_map)

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

    # Datasets that have ≥10 distinct methods in the engineered run — the
    # script uses 10 as the heuristic for "complete" coverage.
    dataset_counts = df_eng.groupby('data_name', as_index=False)['method'].nunique()
    valid_datasets = dataset_counts[dataset_counts['method'] >= 10]['data_name'].tolist()

    df_raw = results[
        results['method'].isin(METHODS_TO_RETRIEVE)
        & results['data_name'].isin(valid_datasets)
    ].copy()
    df_raw['Feature_engineering'] = 'Raw Features'

    combined = pd.concat([df_eng, df_raw], axis=0, ignore_index=True)

    # Restrict to the (encoder, learner) pairs present in BOTH treatments.
    common_pairs = set(df_eng['encoder_learner']) & set(df_raw['encoder_learner'])
    combined = combined[combined['encoder_learner'].isin(common_pairs)]

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

    fig, ax = plt.subplots(figsize=(3, 6))
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

    ax.legend(fontsize=10, frameon=False, bbox_to_anchor=(1.05, 0.2))
    ax.annotate(
        f"Kendall's τ = {tau:.1f}",
        xy=(1.03, 0.19), xycoords='axes fraction',
        ha='right', va='bottom',
        fontsize=11, style='italic',
    )

    save_figure(fig, f"raw_vs_engineered_features_rankings_v1_hue_{SCORE_COL}")
    plt.close(fig)


if __name__ == "__main__":
    plot_raw_vs_engineered()
