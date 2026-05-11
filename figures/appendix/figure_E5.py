"""Figure E.5(a)(b)(c) — Three Critical-Difference diagrams per task subset: all 108 datasets, classification subset (32 datasets)
regression subset (76 datasets)"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

from configs.path_configs import path_configs
from figures._main import (
    clean_method_name,
    dtype_map,
    encoder_map,
    get_encoder_color,
    get_learner_color_simple,
    learner_map,
    load_results,
    save_figure,
    selected_encoders,
)
from src.utils_visualization import critical_difference_diagram


COMPILED = path_configs['compiled_results']
DTYPE_PREFIX = 'num-str'
SCORE_COL = 'score'


# ---------------------------------------------------------------------------
# Data loading — re-uses the same loaders as figure_3 (duplicated by policy)
# ---------------------------------------------------------------------------

def _add_score_and_meta(df):
    df = df.copy()
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    return df


def _load_qwen_nopca_30():
    df = pd.read_csv(f"{COMPILED}/result_comparison_qwen_nopca_30.csv")
    df = _add_score_and_meta(df)
    df['learner'] = df['learner'].str.replace('-no_pca', '')
    df['encoder'] = df['encoder'] + ' (No PCA (30))'
    df['learner'] = df['learner'].replace(learner_map)
    df['method']  = df['encoder'] + ' - ' + df['learner']
    return df


def _load_pca30_standscal_llama_tabpfn():
    df = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    df['method'] = df['method'] + '_default'
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map) + ' (StandScal + PCA (30))'
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['encoder'] + ' - ' + df['learner']
    return df[df['method'] == 'LM LLaMA-3.1-8B - TabPFN-2.5 (StandScal + PCA (30))']


def _load_rebuttal_csv(filename, drop=()):
    df = pd.read_csv(f"{COMPILED}/{filename}")
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0]
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method']  = df['method'].apply(clean_method_name)
    if drop:
        df = df[~df['method'].isin(drop)]
    return df


def _load_30_thresh_ridge():
    df = pd.read_csv(f"{COMPILED}/result_REBUTTALS_30_thresh_ridge.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    df['method'] = df['method'] + '_default'
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method']  = df['method'].apply(clean_method_name) + ' (OHE|CT=30)'
    return df


def _load_mambular_only():
    df = _load_rebuttal_csv("result_REBUTTALS_mambular.csv")
    return df[df['method'].isin(['Mambular - Mambular'])]


# ---------------------------------------------------------------------------
# Rendering one CD diagram for one task subset
# ---------------------------------------------------------------------------

def _build_df_score(results, extra_pipelines):
    df = results[results['encoder'].isin(selected_encoders)].copy()
    df = df[df['method'].str.contains(DTYPE_PREFIX)].reset_index(drop=True)
    df = df[df['method'] != 'num-str_tabpfn_tabpfn_default']
    df['method'] = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method'] = df['method'].apply(clean_method_name)

    common_columns = [
        'brier_score_loss', 'data_name', 'dtype', 'encoder', 'f1_weighted',
        'fold_index', 'inference_time', 'learner', 'method', 'n_cv',
        'param_search_time', 'preprocess_time', 'r2', 'rmse', 'roc_auc',
        'run_time', 'score', 'task',
    ]
    return pd.concat(
        [df[common_columns], extra_pipelines[common_columns]],
        axis=0, ignore_index=True,
    )


def _filter_by_task(df_score, task):
    if task == 'classification':
        return df_score[df_score['task'] != 'regression'].copy()
    if task == 'regression':
        return df_score[df_score['task'] == 'regression'].copy()
    return df_score


def _render_cd_diagram(df_score, save_name):
    df_agg = df_score.groupby(['data_name', 'method'], as_index=False)[SCORE_COL].mean()
    df_agg['rank'] = df_agg.groupby('data_name')[SCORE_COL].rank(ascending=False)
    avg_rank = -1 * df_agg.groupby(['method'])['rank'].mean()

    df_pivot = df_agg.pivot(index='data_name', columns='method', values=SCORE_COL)
    if df_pivot.isnull().values.any():
        n_drop = df_pivot.isnull().any(axis=1).sum()
        print(f"  [{save_name}] dropping {n_drop} datasets with missing methods")
        df_pivot = df_pivot.dropna(axis=0)
    df_clean = df_pivot.reset_index().melt(
        id_vars='data_name', var_name='method', value_name=SCORE_COL,
    )
    test_results = sp.posthoc_conover_friedman(
        df_clean, melted=True,
        block_col='data_name', block_id_col='data_name',
        group_col='method', y_col=SCORE_COL,
    ).replace(0, 1e-100)

    models = df_score['method'].unique()
    print(f"  [{save_name}] {len(models)} models")

    line_style = {model: "-" for model in models}
    for model in models:
        if "TargetEncoder" in model:
            line_style[model] = "--"
        if "LM " in model:
            line_style[model] = "-."

    palette_by_learner = {}
    for model in models:
        learner_part = model.split(' - ')[-1]
        palette_by_learner[model] = get_learner_color_simple(learner_part)

    name_map = {
        m: f"{m} ({abs(rank_val):.1f})"
        for m, rank_val in avg_rank.items()
    }
    avg_rank_plot     = avg_rank.rename(index=name_map)
    test_results_plot = test_results.rename(index=name_map, columns=name_map)
    palette_by_learner_plot = {
        name_map[k]: v for k, v in palette_by_learner.items() if k in name_map
    }
    line_style_plot = {
        name_map[k]: v for k, v in line_style.items() if k in name_map
    }

    sns.set_theme(style='white', font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))

    critical_difference_diagram(
        ranks=avg_rank_plot,
        sig_matrix=test_results_plot,
        label_fmt_left='{label}',
        label_fmt_right=' {label}',
        label_props={'fontsize': 10},
        crossbar_props={'color': 'black', 'linewidth': 1},
        marker_props={'marker': ''},
        elbow_props={'linewidth': 1.5},
        text_h_margin=1.2,
        color_palette=palette_by_learner_plot,
        line_style=line_style_plot,
        bold_control=True,
        v_space=4,
        ax=ax,
    )

    n_models = len(models)
    major_ticks = list(range(0, n_models - 4, 5))
    if n_models not in major_ticks:
        major_ticks.append(n_models)
    major_ticks = sorted([t for t in major_ticks if t > 0])
    ax.set_xticks([-t for t in major_ticks])
    ax.set_xticklabels(major_ticks, fontsize=12)
    ax.set_xlim(-(n_models - 4), 0)

    save_figure(fig, save_name)
    plt.close(fig)


def plot_cd_diagrams_per_task():
    results = load_results()

    extra_pipelines = pd.concat([
        _load_qwen_nopca_30(),
        _load_pca30_standscal_llama_tabpfn(),
        _load_rebuttal_csv("result_REBUTTALS_tabM_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv"),
        _load_rebuttal_csv(
            "result_REBUTTALS_realMLP_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv",
            drop=['LM E5-base-v2 - RealMLP', 'LM Jasper-0.6B - RealMLP'],
        ),
        _load_30_thresh_ridge(),
        _load_mambular_only(),
    ], axis=0, ignore_index=True)

    df_score_full = _build_df_score(results, extra_pipelines)

    for task in ['all_task', 'classification', 'regression']:
        df_score_task = _filter_by_task(df_score_full, task)
        save_name = (
            f"critical_difference_diagram_extra_pipelines_selectedLLMs"
            f"_friedman_colorbylearner_data_name_{SCORE_COL}_{task}"
        )
        print(f"\n=== Rendering CD diagram for task='{task}' ===")
        _render_cd_diagram(df_score_task, save_name)


if __name__ == "__main__":
    plot_cd_diagrams_per_task()
