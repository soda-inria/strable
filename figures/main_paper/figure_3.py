"""Figure 3 — Critical Difference (CD) diagram"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


from collections import Counter

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

LEARNERS_TO_REDUCE = {'Ridge', 'ExtraTrees', 'XGBoost'}
TRIANGLE_SYMBOL = '▲'   # ▲

# Abbreviations applied to method labels in the CD diagram (verbose
# parentheticals don't fit on a single line).
POSTPROC_SHORT = {
    'StandScal + PCA (30)': 'SS+PCA',
    'No PCA (30)':          'NoPCA',
    'OHE|CT=30':            'OHE',
}

# E2E learners — used by classify_pipeline.
E2E_NAMES = {'TabSTAR', 'ContextTab', 'TabM', 'Mambular', 'CatBoost'}

# Line-style per pipeline family.
LINE_STYLES = {
    'llm':      '-',
    'baseline': '-',
    'e2e':      '--',
}


# ---------------------------------------------------------------------------
# 1. Loaders for the "extra pipelines" not present in the main results CSV
# ---------------------------------------------------------------------------

def _add_score_and_meta(df, methods_replaced=()):
    """Common preprocessing for the rebuttal CSVs: derive ``score``, parse
    method into (dtype, encoder, learner), and apply the maps."""
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
    """LLaMA-3.1-8B + TabPFN-2.5 with standard-scaling-before-PCA."""
    df = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    df['method'] = df['method'] + '_default'
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['encoder'] = df['encoder'] + ' (StandScal + PCA (30))'
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['encoder'] + ' - ' + df['learner']
    return df[df['method'].isin(['LM LLaMA-3.1-8B - TabPFN-2.5 (StandScal + PCA (30))'])]


def _load_tabm_30_pca():
    df = pd.read_csv(f"{COMPILED}/result_REBUTTALS_tabM_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0]
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method']  = df['method'].apply(clean_method_name)
    return df


def _load_realmlp_30_pca():
    df = pd.read_csv(f"{COMPILED}/result_REBUTTALS_realMLP_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0]
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method']  = df['method'].apply(clean_method_name)
    # Drop pipelines that completed only partially in the rebuttal run.
    df = df[~df['method'].isin(['LM E5-base-v2 - RealMLP', 'LM Jasper-0.6B - RealMLP'])]
    return df


def _load_30_thresh_ridge():
    """{Ridge, XGBoost, TabPFN-2.5} × {LLaMA, Qwen} with cardinality-threshold-30 + OHE."""
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


def _load_mambular():
    df = pd.read_csv(f"{COMPILED}/result_REBUTTALS_mambular.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method']  = df['method'].apply(clean_method_name)
    return df[df['method'].isin(['Mambular - Mambular'])]


def _load_tabicl():
    """All TabICLv2 results with explicit post-processing tags on learner."""
    df = pd.read_csv(f"{COMPILED}/result_comparison_tabicl_all.csv")
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df.loc[
        df['method'] == 'num-str_llm-llama-3.1-8b_tabicl_standscal',
        'learner',
    ] = 'TabICLv2 (StandScal + PCA (30))'
    df.loc[
        df['method'] == 'num-str_llm-qwen3-8b_tabicl_nopca',
        'learner',
    ] = 'TabICLv2 (No PCA (30))'
    df['method'] = df['encoder'] + ' - ' + df['learner']
    return df


# ---------------------------------------------------------------------------
# 2. Helpers for the CD diagram itself
# ---------------------------------------------------------------------------

def _shorten_label(name):
    """Drop the redundant ``LM `` prefix and abbreviate post-processing
    parentheticals so labels fit on the diagram."""
    name = name.replace('LM ', '')
    for long, short in POSTPROC_SHORT.items():
        name = name.replace(long, short)
    return name


def _classify_pipeline(method):
    """Return ``'llm'`` / ``'baseline'`` / ``'e2e'`` — drives line style."""
    parts = method.split(' - ')
    encoder_part = parts[0].strip()
    if len(parts) >= 2:
        learner_part = parts[-1].split(' (')[0].strip()
        if encoder_part == learner_part:
            return 'e2e'
    if any(name in encoder_part for name in E2E_NAMES):
        return 'e2e'
    if encoder_part.startswith('LM '):
        return 'llm'
    return 'baseline'


def _keep_best_encoder_for_learners(df_in, score_col, learners_to_reduce):
    """For each learner in ``learners_to_reduce``, keep only its best encoder
    (by mean per-dataset rank) and drop the rest. Returns ``(df_filtered,
    set_of_kept_methods)``; the kept methods will get ▲ in the diagram."""
    df = df_in.copy()
    is_default = ~df['method'].str.contains('tuned', case=False, na=False)
    df_default = df[is_default & df['learner'].isin(learners_to_reduce)].copy()

    kept_methods = set()
    drop_index = []

    for learner in learners_to_reduce:
        sub = df_default[df_default['learner'] == learner]
        if sub.empty:
            continue
        per_ed = sub.groupby(['encoder', 'data_name'], as_index=False)[score_col].mean()
        per_ed['rank'] = per_ed.groupby('data_name')[score_col].rank(ascending=False)
        mean_rank = per_ed.groupby('encoder')['rank'].mean().sort_values()
        if mean_rank.empty:
            continue
        best_encoder = mean_rank.index[0]

        to_drop = df_default[
            (df_default['learner'] == learner)
            & (df_default['encoder'] != best_encoder)
        ].index
        drop_index.extend(to_drop.tolist())
        kept_methods.update(
            df_default[
                (df_default['learner'] == learner)
                & (df_default['encoder'] == best_encoder)
            ]['method'].unique().tolist()
        )
        print(f"  {learner}: keeping encoder='{best_encoder}' "
              f"(mean rank {mean_rank.iloc[0]:.2f}), "
              f"dropping {len(to_drop)} rows from other encoders")

    return df.drop(index=drop_index).reset_index(drop=True), kept_methods


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def plot_cd_diagram_main():
    results = load_results()

    extra_pipelines = pd.concat([
        _load_qwen_nopca_30(),
        _load_pca30_standscal_llama_tabpfn(),
        _load_tabm_30_pca(),
        _load_realmlp_30_pca(),
        _load_30_thresh_ridge(),
        _load_mambular(),
        _load_tabicl(),
    ], axis=0, ignore_index=True)

    # Main pipelines from the master results CSV.
    df_score = results[results['encoder'].isin(selected_encoders)].copy()
    df_score = df_score[df_score['method'].str.contains(DTYPE_PREFIX)].reset_index(drop=True)
    df_score = df_score[df_score['method'] != 'num-str_tabpfn_tabpfn_default']
    df_score['method'] = df_score['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df_score['method'] = df_score['method'].apply(clean_method_name)

    common_columns = [
        'brier_score_loss', 'data_name', 'dtype', 'encoder', 'f1_weighted',
        'fold_index', 'inference_time', 'learner', 'method', 'n_cv',
        'param_search_time', 'preprocess_time', 'r2', 'rmse', 'roc_auc',
        'run_time', 'score', 'task',
    ]
    df_score = pd.concat(
        [df_score[common_columns], extra_pipelines[common_columns]],
        axis=0, ignore_index=True,
    )

    print("Reducing default pipelines for "
          f"{LEARNERS_TO_REDUCE} to best-encoder-only:")
    df_score, kept_methods = _keep_best_encoder_for_learners(
        df_score, score_col=SCORE_COL,
        learners_to_reduce=LEARNERS_TO_REDUCE,
    )
    print(f"  Triangle-marked methods: {kept_methods}")

    # Per-dataset average score, then per-dataset rank. Negate so the diagram
    # shows "better=right" (the convention used by ``critical_difference_diagram``).
    df_agg = df_score.groupby(['data_name', 'method'], as_index=False)[SCORE_COL].mean()
    df_agg['rank'] = df_agg.groupby('data_name')[SCORE_COL].rank(ascending=False)
    avg_rank = -1 * df_agg.groupby(['method'])['rank'].mean()

    df_pivot = df_agg.pivot(index='data_name', columns='method', values=SCORE_COL)
    if df_pivot.isnull().values.any():
        n_drop = df_pivot.isnull().any(axis=1).sum()
        print(f"Warning: dropping {n_drop} datasets with missing method scores.")
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

    # 3-way line style by family.
    line_style = {m: LINE_STYLES[_classify_pipeline(m)] for m in models}
    print("\nPipeline family counts:", Counter(_classify_pipeline(m) for m in models))
    print(f"Total models after reduction: {len(models)}\n")

    palette_by_learner = {}
    for model in models:
        parts = model.split(' - ')
        learner_part = parts[-1].split(' (')[0]
        palette_by_learner[model] = get_learner_color_simple(learner_part)

    # Build display labels: shorten + ▲ if reduced + parenthetical avg-rank.
    name_map = {
        model: f"{_shorten_label(model)}"
               f"{' ' + TRIANGLE_SYMBOL if model in kept_methods else ''}"
               f" ({abs(rank_val):.1f})"
        for model, rank_val in avg_rank.items()
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
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.8))

    critical_difference_diagram(
        ranks=avg_rank_plot,
        sig_matrix=test_results_plot,
        label_fmt_left='{label} ',
        label_fmt_right=' {label}',
        label_props={'fontsize': 6},
        crossbar_props={'color': 'black', 'linewidth': 1},
        marker_props={'marker': ''},
        elbow_props={'linewidth': 1.5},
        text_h_margin=0.0,
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
    ax.set_xticklabels(major_ticks, fontsize=8)
    ax.set_xlim(-(n_models - 4), 0)

    # "Better →" arrow banner above the rank axis.
    GREEN = '#2ca02c'
    ax.text(
        2, 5, 'Better',
        ha='center', va='center',
        fontsize=8, fontweight='bold',
        color='white',
        bbox=dict(
            boxstyle='rarrow,pad=0.4',
            facecolor=GREEN, edgecolor=GREEN, linewidth=0,
        ),
        clip_on=False, zorder=11,
    )

    save_figure(
        fig,
        "critical_difference_diagram_extra_pipelines_selectedLLMs_friedman_colorbylearner_REDUCED_data_name_score_all_task",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_cd_diagram_main()
