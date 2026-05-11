"""Figure E.11 — Per-(encoder, learner) bar chart contrasting the default 30-PCA pipeline
against ``30-PCA + CT=30`` (low-cardinality string columns at threshold
30 routed to OHE for Ridge or passthrough for XGBoost / TabPFN-2.5,
instead of the LLM encoder)."""

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


COMMON_COLS = [
    'roc_auc', 'brier_score_loss', 'f1_weighted', 'preprocess_time',
    'param_search_time', 'inference_time', 'run_time', 'data_name',
    'method', 'n_cv', 'fold_index', 'task', 'r2', 'rmse', 'CT_30',
]


def _load_csv_with_filter(filename, contains):
    df = pd.read_csv(f"{path_configs['compiled_results']}/{filename}")
    df = df[df['method'].str.contains(contains)].copy()
    df['method'] = df['method'] + '_default'
    return df


def _add_score_meta(df):
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0]
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    return df


def plot_ct30_nemotron():
    results = load_results()

    # 30-PCA: {Ridge, XGBoost} × {LLaMA, Qwen} from main results.
    plain_llama_qwen = results[results['method'].isin([
        'num-str_llm-qwen3-8b_ridge_default',
        'num-str_llm-llama-3.1-8b_ridge_default',
        'num-str_llm-llama-3.1-8b_xgb_default',
        'num-str_llm-qwen3-8b_xgb_default',
    ])].copy()
    plain_llama_qwen['CT_30'] = '30-PCA'

    # 30-PCA: {Ridge, XGBoost} × Nemotron-1B from rebuttal.
    plain_nemotron = _load_csv_with_filter(
        "result_nemotron_30pca_without_standscal_ridge_xgboost_extratrees.csv",
        contains='ridge|xgb',
    )
    plain_nemotron['CT_30'] = '30-PCA'

    # 30-PCA: TabPFN-2.5 × {LLaMA, Qwen, Nemotron} from main results.
    plain_tabpfn = results[results['method'].isin([
        'num-str_llm-qwen3-8b_tabpfn_default',
        'num-str_llm-llama-3.1-8b_tabpfn_default',
        'num-str_llm-llama-nemotron-embed-1b-v2_tabpfn_default',
    ])].copy()
    plain_tabpfn['CT_30'] = '30-PCA'

    # 30-PCA + CT=30:
    ct_ridge_ohe = _load_csv_with_filter(
        "result_comparison_llama_qwen_nemotron_ct30_ohe_without_standscal_30pca_ridge_extrees_xgb.csv",
        contains='ridge',
    )
    ct_ridge_ohe['CT_30'] = '30-PCA + CT=30'

    ct_xgb_pass = _load_csv_with_filter(
        "result_comparison_llama_qwen_nemotron_ct30_passthrough_without_standscal_30pca_xgb.csv",
        contains='xgb',
    )
    ct_xgb_pass['CT_30'] = '30-PCA + CT=30'

    ct_tabpfn_llama_qwen = _load_csv_with_filter(
        "result_REBUTTALS_30_thresh_xgb_tabpfn.csv",
        contains='tabpfn',
    )
    ct_tabpfn_llama_qwen['CT_30'] = '30-PCA + CT=30'

    ct_tabpfn_nemotron = _load_csv_with_filter(
        "result_comparison_nemotron_30pca_tabpfn_without_standscal_30_thresh_lowcard_passthrough.csv",
        contains='tabpfn',
    )
    ct_tabpfn_nemotron['CT_30'] = '30-PCA + CT=30'

    combined = pd.concat([
        plain_llama_qwen[COMMON_COLS],
        plain_nemotron[COMMON_COLS],
        plain_tabpfn[COMMON_COLS],
        ct_ridge_ohe[COMMON_COLS],
        ct_xgb_pass[COMMON_COLS],
        ct_tabpfn_llama_qwen[COMMON_COLS],
        ct_tabpfn_nemotron[COMMON_COLS],
    ], axis=0, ignore_index=True)
    combined = _add_score_meta(combined)

    grouped = (
        combined.groupby(['learner', 'encoder', 'CT_30'])['score']
        .mean().reset_index()
    )

    df = grouped.copy()
    df['is_ct'] = df['CT_30'] == '30-PCA + CT=30'
    df['encoder_clean'] = df['encoder'].str.replace('^LM ', '', regex=True)
    display_rename = {'LLaMA-Nemotron-Embed-1B-v2': 'Nemotron-1B'}

    learners = df['learner'].unique()
    base_encoders = df['encoder_clean'].unique()
    learner_suffix = {
        'Ridge':      'OHE',
        'TabPFN-2.5': 'passthrough',
        'XGBoost':    'passthrough',
    }

    pivot = (
        df.pivot_table(index=['learner', 'encoder_clean'],
                       columns='is_ct', values='score')
          .rename(columns={False: 'plain', True: 'ct30'})
          .dropna()
    )
    tau_global, _ = kendalltau(pivot['plain'], pivot['ct30'])

    fig, ax = plt.subplots(figsize=(3, 5))
    bar_height = 0.35
    y = np.arange(len(learners)) * 2.5

    for i, base_enc in enumerate(base_encoders):
        color = get_encoder_color(base_enc)
        plain_scores, ct_scores = [], []
        for learner in learners:
            subset = df[(df['learner'] == learner)
                        & (df['encoder_clean'] == base_enc)]
            plain_val = subset[~subset['is_ct']]['score'].values
            ct_val    = subset[ subset['is_ct']]['score'].values
            plain_scores.append(plain_val[0] if len(plain_val) else 0)
            ct_scores.append(ct_val[0] if len(ct_val) else 0)

        offset = (i - len(base_encoders) / 2 + 0.5) * bar_height * 2.2
        ax.barh(y + offset + bar_height / 2, plain_scores, height=bar_height,
                color=color, edgecolor='black', linewidth=0.5)
        ax.barh(y + offset - bar_height / 2, ct_scores, height=bar_height,
                color=color, edgecolor='white', linewidth=1.0, hatch='///')

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{l}\n({learner_suffix.get(l, '')})" for l in learners],
        fontsize=13,
    )
    ax.set_xlabel('Avg Score ($R^2$ & AUC)', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0.58, 0.8)
    ax.text(1.7, 0.47, rf"Kendall's $\tau$ = {tau_global:+.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=13)

    legend_handles = []
    for base_enc in base_encoders:
        legend_handles.append(mpatches.Patch(
            facecolor=get_encoder_color(base_enc), edgecolor='black',
            linewidth=0.5, label=display_rename.get(base_enc, base_enc),
        ))
    legend_handles.append(mpatches.Patch(
        facecolor='black', edgecolor='white', hatch='///', label='CT=30',
    ))
    ax.legend(handles=legend_handles, fontsize=13, bbox_to_anchor=(0.9, 0.42))
    plt.tight_layout()

    save_figure(fig, "ridge_xgb_tabpfn_llama_qwen_nemotron_CT30")
    plt.close(fig)


if __name__ == "__main__":
    plot_ct30_nemotron()
