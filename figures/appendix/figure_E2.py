"""Figure E.2 — Per-learner breakdown of post-processing strategies (Default 30-PCA vs
Standard scaling + 30-PCA vs No-PCA / first 30 raw dims)"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from configs.path_configs import path_configs
from figures._main import (
    dtype_map,
    encoder_map,
    learner_map,
    load_results,
    save_figure,
)


# ---------------------------------------------------------------------------
# 1. Data loading — every (encoder × post-processing × learner-group) CSV
# ---------------------------------------------------------------------------

COMPILED = path_configs['compiled_results']


def _preprocess_results(df):
    """Add ``score`` / ``dtype`` / ``encoder`` / ``learner`` columns to a
    raw compiled-results DataFrame, applying the same mappings as
    ``figures._main.load_results``."""
    df = df.copy()
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype'] = meta[0]
    df['encoder'] = meta[1]
    df['learner'] = meta[2]
    df['dtype']   = df['dtype'].replace(dtype_map)
    df['encoder'] = df['encoder'].replace(encoder_map)
    df['learner'] = df['learner'].replace(learner_map)
    df['method_polished'] = df['encoder'] + ' - ' + df['learner'] + '\n(' + df['dtype'] + ')'
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    return df


def _load_post_processing_data(results):
    """Build ``results_post_processing`` covering all 7 encoders × 3
    post-processing variants × {Ridge, XGBoost, ExtraTrees, TabPFN-2.5,
    TabICLv2}. Returns the concatenated long-format DataFrame.
    """
    # --- {LLaMA-3.1-8B, Qwen-3-8B} × {ridge, XGB, ExtraTrees} - Default PCA ---
    llama_qwen_ridge_xgb_extrees_default = results[results['method'].isin([
        'num-str_llm-llama-3.1-8b_ridge_default',
        'num-str_llm-llama-3.1-8b_xgb_default',
        'num-str_llm-llama-3.1-8b_extrees_default',
        'num-str_llm-qwen3-8b_ridge_default',
        'num-str_llm-qwen3-8b_xgb_default',
        'num-str_llm-qwen3-8b_extrees_default',
    ])].copy()
    llama_qwen_ridge_xgb_extrees_default['post_processing'] = 'Default PCA'

    nemotron_xgb_extrees_default = pd.read_csv(
        f"{COMPILED}/result_nemotron_30pca_without_standscal_ridge_xgboost_extratrees.csv"
    )
    nemotron_xgb_extrees_default['post_processing'] = 'Default PCA'
    nemotron_xgb_extrees_default['method'] = nemotron_xgb_extrees_default['method'] + '_default'

    llama_qwen_nemotron_ridge_xgb_extrees_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_llama_qwen_nemotron_30pca_with_standscal_ridge_xgboost_extratrees.csv"
    )
    llama_qwen_nemotron_ridge_xgb_extrees_standscal['post_processing'] = 'StandScal + 30-PCA'
    llama_qwen_nemotron_ridge_xgb_extrees_standscal['method'] = (
        llama_qwen_nemotron_ridge_xgb_extrees_standscal['method'] + '_default'
    )

    llama_qwen_nemotron_ridge_xgb_extrees_nopca = pd.read_csv(
        f"{COMPILED}/result_comparison_llama_qwen_nemotron_NOPCA_ridge_extratrees_xgboost.csv"
    )
    llama_qwen_nemotron_ridge_xgb_extrees_nopca['post_processing'] = 'No PCA'
    llama_qwen_nemotron_ridge_xgb_extrees_nopca['method'] = (
        llama_qwen_nemotron_ridge_xgb_extrees_nopca['method'] + '_default'
    )

    # --- MiniLM-L6 × {ridge, XGB, ExtraTrees} ---
    minilml6_ridge_xgb_extrees_default = results[results['method'].isin([
        'num-str_llm-all-MiniLM-L6-v2_ridge_default',
        'num-str_llm-all-MiniLM-L6-v2_xgb_default',
        'num-str_llm-all-MiniLM-L6-v2_extrees_default',
    ])].copy()
    minilml6_ridge_xgb_extrees_default['post_processing'] = 'Default PCA'

    minilml6_ridge_xgb_extrees_standscal = pd.read_csv(
        f"{COMPILED}/result_miniLM_L6_BGE_large_E5_large_30pca_with_standscal_ridge_xgboost_extratrees.csv"
    )
    minilml6_ridge_xgb_extrees_standscal = minilml6_ridge_xgb_extrees_standscal[
        minilml6_ridge_xgb_extrees_standscal['method'].str.contains('MiniLM-L6')
    ].copy()
    minilml6_ridge_xgb_extrees_standscal['post_processing'] = 'StandScal + 30-PCA'
    minilml6_ridge_xgb_extrees_standscal['method'] = (
        minilml6_ridge_xgb_extrees_standscal['method'] + '_default'
    )

    minilml6_ridge_xgb_extrees_nopca = pd.read_csv(
        f"{COMPILED}/result_miniLM_L6_BGE_large_E5_large_NOpca_ridge_xgboost_extratrees.csv"
    )
    minilml6_ridge_xgb_extrees_nopca = minilml6_ridge_xgb_extrees_nopca[
        minilml6_ridge_xgb_extrees_nopca['method'].str.contains('MiniLM-L6')
    ].copy()
    minilml6_ridge_xgb_extrees_nopca['post_processing'] = 'No PCA'
    minilml6_ridge_xgb_extrees_nopca['method'] = (
        minilml6_ridge_xgb_extrees_nopca['method'] + '_default'
    )

    # --- E5-base-v2 × {ridge, XGB, ExtraTrees} ---
    e5basev2_ridge_xgb_extrees_default = results[results['method'].isin([
        'num-str_llm-e5-base-v2_ridge_default',
        'num-str_llm-e5-base-v2_xgb_default',
        'num-str_llm-e5-base-v2_extrees_default',
    ])].copy()
    e5basev2_ridge_xgb_extrees_default['post_processing'] = 'Default PCA'

    e5basev2_ridge_xgb_extrees_standscal = pd.read_csv(
        f"{COMPILED}/result_E5_base_v2_30pca_with_standscal_ridge_xgboost_extratrees.csv"
    )
    e5basev2_ridge_xgb_extrees_standscal['post_processing'] = 'StandScal + 30-PCA'
    e5basev2_ridge_xgb_extrees_standscal['method'] = (
        e5basev2_ridge_xgb_extrees_standscal['method'] + '_default'
    )

    e5basev2_ridge_xgb_extrees_nopca = pd.read_csv(
        f"{COMPILED}/result_E5_base_v2_NOpca_ridge_xgboost_extratrees.csv"
    )
    e5basev2_ridge_xgb_extrees_nopca['post_processing'] = 'No PCA'
    e5basev2_ridge_xgb_extrees_nopca['method'] = (
        e5basev2_ridge_xgb_extrees_nopca['method'] + '_default'
    )

    # --- {OPT-6.7B, BGE-large} × {ridge, XGB, ExtraTrees} (ported from salts) ---
    opt_bge_ridge_xgb_extrees_default = results[results['method'].isin([
        'num-str_llm-opt-6.7b_ridge_default',
        'num-str_llm-opt-6.7b_xgb_default',
        'num-str_llm-opt-6.7b_extrees_default',
        'num-str_llm-bge-large_ridge_default',
        'num-str_llm-bge-large_xgb_default',
        'num-str_llm-bge-large_extrees_default',
    ])].copy()
    opt_bge_ridge_xgb_extrees_default['post_processing'] = 'Default PCA'

    opt_bge_ridge_xgb_extrees_standscal = pd.read_csv(
        f"{COMPILED}/result_opt_bge_ridge_extrees_xgb_30pca_with_standscal.csv"
    )
    opt_bge_ridge_xgb_extrees_standscal['method'] = (
        opt_bge_ridge_xgb_extrees_standscal['method'] + '_default'
    )
    opt_bge_ridge_xgb_extrees_standscal['post_processing'] = 'StandScal + 30-PCA'

    opt_bge_ridge_xgb_extrees_nopca = pd.read_csv(
        f"{COMPILED}/result_opt_bge_ridge_extrees_xgb_NOpca.csv"
    )
    opt_bge_ridge_xgb_extrees_nopca['method'] = (
        opt_bge_ridge_xgb_extrees_nopca['method'] + '_default'
    )
    opt_bge_ridge_xgb_extrees_nopca['post_processing'] = 'No PCA'

    # --- Qwen-3-8B × {TabPFN-2.5, TabICLv2} ---
    qwen_tabpfn_default = results[results['method'].isin([
        'num-str_llm-qwen3-8b_tabpfn_default'
    ])].copy()
    qwen_tabpfn_default['post_processing'] = 'Default PCA'

    qwen_tabicl_default = pd.read_csv(f"{COMPILED}/result_comparison_tabicl.csv")
    qwen_tabicl_default = qwen_tabicl_default[
        qwen_tabicl_default['method'].str.contains('qwen3-8b')
    ].copy()
    qwen_tabicl_default['post_processing'] = 'Default PCA'

    qwen_tabpfn_tabicl_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_qwen_standscal_30pca.csv"
    )
    qwen_tabpfn_tabicl_standscal['post_processing'] = 'StandScal + 30-PCA'
    qwen_tabpfn_tabicl_standscal['method'] = [
        e + '_default' if e == 'num-str_llm-qwen3-8b_tabpfn' else e
        for e in qwen_tabpfn_tabicl_standscal['method']
    ]

    qwen_tabpfn_nopca = pd.read_csv(
        f"{COMPILED}/result_comparison_qwen_nopca_30.csv"
    )
    qwen_tabpfn_nopca['post_processing'] = 'No PCA'
    qwen_tabpfn_nopca['method'] = qwen_tabpfn_nopca['method'].str.replace('-no_pca', '')

    qwen_tabicl_nopca = pd.read_csv(
        f"{COMPILED}/result_comparison_tabicl_qwen.csv"
    )
    qwen_tabicl_nopca['post_processing'] = 'No PCA'
    qwen_tabicl_nopca['method'] = qwen_tabicl_nopca['method'].str.replace('_nopca', '')

    # --- LLaMA-3.1-8B × {TabPFN-2.5, TabICLv2} ---
    llama_tabpfn_default = results[results['method'].isin([
        'num-str_llm-llama-3.1-8b_tabpfn_default'
    ])].copy()
    llama_tabpfn_default['post_processing'] = 'Default PCA'

    llama_tabicl_default = pd.read_csv(f"{COMPILED}/result_comparison_tabicl.csv")
    llama_tabicl_default = llama_tabicl_default[
        llama_tabicl_default['method'].str.contains('llama-3.1-8b')
    ].copy()
    llama_tabicl_default['post_processing'] = 'Default PCA'

    llama_tabpfn_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_standscal_pca_30.csv"
    )
    llama_tabpfn_standscal = llama_tabpfn_standscal[
        llama_tabpfn_standscal['method'].str.contains('llama-3.1-8b')
    ].copy()
    llama_tabpfn_standscal['post_processing'] = 'StandScal + 30-PCA'
    llama_tabpfn_standscal['method'] = llama_tabpfn_standscal['method'] + '_default'

    llama_tabicl_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_tabicl-llama.csv"
    )
    llama_tabicl_standscal['post_processing'] = 'StandScal + 30-PCA'
    llama_tabicl_standscal['method'] = (
        llama_tabicl_standscal['method'].str.replace('_standscal', '')
    )

    llama_tabpfn_tabicl_nopca = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_NOpca_tabpfn_tabicl.csv"
    )
    llama_tabpfn_tabicl_nopca = llama_tabpfn_tabicl_nopca[
        llama_tabpfn_tabicl_nopca['method'].str.contains('llama-3.1-8b')
    ].copy()
    llama_tabpfn_tabicl_nopca['post_processing'] = 'No PCA'
    llama_tabpfn_tabicl_nopca['method'] = [
        e + '_default' if e == 'num-str_llm-llama-3.1-8b_tabpfn' else e
        for e in llama_tabpfn_tabicl_nopca['method']
    ]

    # --- Nemotron-1B × {TabPFN-2.5, TabICLv2} ---
    nemotron_tabpfn_default = results[results['method'].isin([
        'num-str_llm-llama-nemotron-embed-1b-v2_tabpfn_default'
    ])].copy()
    nemotron_tabpfn_default['post_processing'] = 'Default PCA'

    nemotron_tabicl_default = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_30pca_without_standscal_tabicl.csv"
    )
    nemotron_tabicl_default = nemotron_tabicl_default[
        nemotron_tabicl_default['method'].str.contains('nemotron')
    ].copy()
    nemotron_tabicl_default['post_processing'] = 'Default PCA'

    nemotron_tabpfn_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_standscal_pca_30.csv"
    )
    nemotron_tabpfn_standscal = nemotron_tabpfn_standscal[
        nemotron_tabpfn_standscal['method'].str.contains('nemotron')
    ].copy()
    nemotron_tabpfn_standscal['post_processing'] = 'StandScal + 30-PCA'
    nemotron_tabpfn_standscal['method'] = nemotron_tabpfn_standscal['method'] + '_default'

    nemotron_tabicl_standscal = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_30pca_with_standscal_tabicl.csv"
    )
    nemotron_tabicl_standscal = nemotron_tabicl_standscal[
        nemotron_tabicl_standscal['method'].str.contains('nemotron')
    ].copy()
    nemotron_tabicl_standscal['post_processing'] = 'StandScal + 30-PCA'

    nemotron_tabpfn_tabicl_nopca = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_NOpca_tabpfn_tabicl.csv"
    )
    nemotron_tabpfn_tabicl_nopca = nemotron_tabpfn_tabicl_nopca[
        nemotron_tabpfn_tabicl_nopca['method'].str.contains('nemotron')
    ].copy()
    nemotron_tabpfn_tabicl_nopca['post_processing'] = 'No PCA'
    nemotron_tabpfn_tabicl_nopca['method'] = [
        e + '_default' if e == 'num-str_llm-llama-nemotron-embed-1b-v2_tabpfn' else e
        for e in nemotron_tabpfn_tabicl_nopca['method']
    ]

    # --- MiniLM-L6 × {TabPFN-2.5, TabICLv2} ---
    minilml6_tabpfn_default = results[results['method'].isin([
        'num-str_llm-all-MiniLM-L6-v2_tabpfn_default'
    ])].copy()
    minilml6_tabpfn_default['post_processing'] = 'Default PCA'

    minilml6_tabicl_default = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_30pca_without_standscal_tabicl.csv"
    )
    minilml6_tabicl_default = minilml6_tabicl_default[
        minilml6_tabicl_default['method'].str.contains('MiniLM-L6')
    ].copy()
    minilml6_tabicl_default['post_processing'] = 'Default PCA'

    minilml6_tabpfn_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_standscal_pca_30.csv"
    )
    minilml6_tabpfn_standscal = minilml6_tabpfn_standscal[
        minilml6_tabpfn_standscal['method'].str.contains('MiniLM-L6')
    ].copy()
    minilml6_tabpfn_standscal['post_processing'] = 'StandScal + 30-PCA'
    minilml6_tabpfn_standscal['method'] = minilml6_tabpfn_standscal['method'] + '_default'

    minilml6_tabicl_standscal = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_30pca_with_standscal_tabicl.csv"
    )
    minilml6_tabicl_standscal['post_processing'] = 'StandScal + 30-PCA'

    minilml6_tabpfn_tabicl_nopca = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_NOpca_tabpfn_tabicl.csv"
    )
    minilml6_tabpfn_tabicl_nopca = minilml6_tabpfn_tabicl_nopca[
        minilml6_tabpfn_tabicl_nopca['method'].str.contains('MiniLM-L6')
    ].copy()
    minilml6_tabpfn_tabicl_nopca['post_processing'] = 'No PCA'
    minilml6_tabpfn_tabicl_nopca['method'] = [
        e + '_default' if e == 'num-str_llm-all-MiniLM-L6-v2_tabpfn' else e
        for e in minilml6_tabpfn_tabicl_nopca['method']
    ]

    # --- E5-base-v2 × {TabPFN-2.5, TabICLv2} ---
    e5basev2_tabpfn_default = results[results['method'].isin([
        'num-str_llm-e5-base-v2_tabpfn_default'
    ])].copy()
    e5basev2_tabpfn_default['post_processing'] = 'Default PCA'

    e5basev2_tabicl_default = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_30pca_without_standscal_tabicl.csv"
    )
    e5basev2_tabicl_default = e5basev2_tabicl_default[
        e5basev2_tabicl_default['method'].str.contains('e5-base-v2')
    ].copy()
    e5basev2_tabicl_default['post_processing'] = 'Default PCA'

    e5basev2_tabpfn_standscal = pd.read_csv(
        f"{COMPILED}/result_comparison_standscal_pca_30.csv"
    )
    e5basev2_tabpfn_standscal = e5basev2_tabpfn_standscal[
        e5basev2_tabpfn_standscal['method'].str.contains('e5-base-v2')
    ].copy()
    e5basev2_tabpfn_standscal['post_processing'] = 'StandScal + 30-PCA'
    e5basev2_tabpfn_standscal['method'] = e5basev2_tabpfn_standscal['method'] + '_default'

    e5basev2_tabicl_standscal = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_30pca_with_standscal_tabicl.csv"
    )
    e5basev2_tabicl_standscal = e5basev2_tabicl_standscal[
        e5basev2_tabicl_standscal['method'].str.contains('e5-base-v2')
    ].copy()
    e5basev2_tabicl_standscal['post_processing'] = 'StandScal + 30-PCA'

    e5basev2_tabpfn_tabicl_nopca = pd.read_csv(
        f"{COMPILED}/result_llama_nemotron_miniLM_L6_E5_base_v2_NOpca_tabpfn_tabicl.csv"
    )
    e5basev2_tabpfn_tabicl_nopca = e5basev2_tabpfn_tabicl_nopca[
        e5basev2_tabpfn_tabicl_nopca['method'].str.contains('e5-base-v2')
    ].copy()
    e5basev2_tabpfn_tabicl_nopca['post_processing'] = 'No PCA'
    e5basev2_tabpfn_tabicl_nopca['method'] = [
        e + '_default' if e == 'num-str_llm-e5-base-v2_tabpfn' else e
        for e in e5basev2_tabpfn_tabicl_nopca['method']
    ]

    # --- {OPT-6.7B, BGE-large} × {TabPFN-2.5, TabICLv2} (ported from salts) ---
    opt_bge_tabpfn_tabicl_default = pd.read_csv(
        f"{COMPILED}/result_opt_bge_tabpfn_tabicl_30pca_without_standscal.csv"
    )
    opt_bge_tabpfn_tabicl_default['method'] = [
        e + '_default'
        if e in ('num-str_llm-opt-6.7b_tabpfn', 'num-str_llm-bge-large_tabpfn')
        else e
        for e in opt_bge_tabpfn_tabicl_default['method']
    ]
    opt_bge_tabpfn_tabicl_default['post_processing'] = 'Default PCA'

    opt_bge_tabpfn_tabicl_standscal = pd.read_csv(
        f"{COMPILED}/result_opt_bge_tabpfn_tabicl_30pca_with_standscal.csv"
    )
    opt_bge_tabpfn_tabicl_standscal['method'] = [
        e + '_default'
        if e in ('num-str_llm-opt-6.7b_tabpfn', 'num-str_llm-bge-large_tabpfn')
        else e
        for e in opt_bge_tabpfn_tabicl_standscal['method']
    ]
    opt_bge_tabpfn_tabicl_standscal['post_processing'] = 'StandScal + 30-PCA'

    opt_bge_tabpfn_tabicl_nopca = pd.read_csv(
        f"{COMPILED}/result_opt_bge_tabicl_tabpfn_NOpca.csv"
    )
    opt_bge_tabpfn_tabicl_nopca['method'] = [
        e + '_default'
        if e in ('num-str_llm-opt-6.7b_tabpfn', 'num-str_llm-bge-large_tabpfn')
        else e
        for e in opt_bge_tabpfn_tabicl_nopca['method']
    ]
    opt_bge_tabpfn_tabicl_nopca['post_processing'] = 'No PCA'

    common_columns = [
        'roc_auc', 'brier_score_loss', 'f1_weighted', 'preprocess_time',
        'param_search_time', 'inference_time', 'run_time', 'data_name',
        'method', 'n_cv', 'fold_index', 'task', 'r2', 'rmse', 'post_processing',
    ]

    parts = [
        llama_qwen_ridge_xgb_extrees_default,
        nemotron_xgb_extrees_default,
        llama_qwen_nemotron_ridge_xgb_extrees_standscal,
        llama_qwen_nemotron_ridge_xgb_extrees_nopca,
        minilml6_ridge_xgb_extrees_default,
        minilml6_ridge_xgb_extrees_standscal,
        minilml6_ridge_xgb_extrees_nopca,
        e5basev2_ridge_xgb_extrees_default,
        e5basev2_ridge_xgb_extrees_standscal,
        e5basev2_ridge_xgb_extrees_nopca,
        opt_bge_ridge_xgb_extrees_default,
        opt_bge_ridge_xgb_extrees_standscal,
        opt_bge_ridge_xgb_extrees_nopca,
        qwen_tabpfn_default, qwen_tabicl_default,
        qwen_tabpfn_tabicl_standscal,
        qwen_tabpfn_nopca, qwen_tabicl_nopca,
        llama_tabpfn_default, llama_tabicl_default,
        llama_tabpfn_standscal, llama_tabicl_standscal,
        llama_tabpfn_tabicl_nopca,
        nemotron_tabpfn_default, nemotron_tabicl_default,
        nemotron_tabpfn_standscal, nemotron_tabicl_standscal,
        nemotron_tabpfn_tabicl_nopca,
        minilml6_tabpfn_default, minilml6_tabicl_default,
        minilml6_tabpfn_standscal, minilml6_tabicl_standscal,
        minilml6_tabpfn_tabicl_nopca,
        e5basev2_tabpfn_default, e5basev2_tabicl_default,
        e5basev2_tabpfn_standscal, e5basev2_tabicl_standscal,
        e5basev2_tabpfn_tabicl_nopca,
        opt_bge_tabpfn_tabicl_default,
        opt_bge_tabpfn_tabicl_standscal,
        opt_bge_tabpfn_tabicl_nopca,
    ]
    return _preprocess_results(pd.concat(
        [p[common_columns] for p in parts], ignore_index=True,
    ))


def _build_pivots(post_processing_results):
    """Compute the encoder × post-processing and (encoder, learner) × post-
    processing pivots used by the figure rendering."""
    encoder_pivot = (
        post_processing_results
        .groupby(['encoder', 'post_processing'], as_index=False)['score']
        .mean()
        .pivot(index='encoder', columns='post_processing', values='score')
    )
    learner_pivot = (
        post_processing_results
        .groupby(['encoder', 'learner', 'post_processing'], as_index=False)['score']
        .mean()
        .pivot(
            index=['encoder', 'learner'],
            columns='post_processing',
            values='score',
        )
        .reset_index()
    )
    return encoder_pivot, learner_pivot


def _tfidf_baseline(results):
    """Mean Tf-Idf score across {Ridge, XGBoost, ExtraTrees, TabPFN-2.5,
    TabICLv2}, used as the dashed reference line in every panel."""
    tfidf_main = results[results['method'].isin([
        'num-str_tabvec_ridge_default',
        'num-str_tabvec_xgb_default',
        'num-str_tabvec_extrees_default',
        'num-str_tabvec_tabpfn_default',
    ])]
    tfidf_tabicl = _preprocess_results(
        pd.read_csv(f"{COMPILED}/result_comparison_tabicl.csv")
    )
    return float(pd.concat([tfidf_main, tfidf_tabicl], ignore_index=True)['score'].mean())


# ---------------------------------------------------------------------------
# 2. Figure config (ported from salts ``_APPENDIX`` block)
# ---------------------------------------------------------------------------

COL_DEFAULT   = 'Default PCA'
COL_NO_PCA    = 'No PCA'
COL_STANDSCAL = 'StandScal + 30-PCA'
ORDER  = [COL_DEFAULT, COL_STANDSCAL, COL_NO_PCA]
COLORS = {COL_DEFAULT: '#1f77b4', COL_STANDSCAL: '#ff7f0e', COL_NO_PCA: '#2ca02c'}
SHAPES = {COL_DEFAULT: 's',       COL_STANDSCAL: 'o',       COL_NO_PCA: '^'}
TFIDF_COLOR = '#7B3FBF'

FS_TITLE     = 16
FS_YLABEL    = 15
FS_YTICK     = 9
FS_LEGEND    = 14
FS_FOOTER    = 12
FS_VAL_LABEL = 10
FS_PCT_LABEL = 10
FS_RIDGE     = 11

BG_ALPHA    = 0.25
BG_LW       = 1.0
BG_MARKER_S = 25
FG_LW       = 3.0
FG_MARKER_S = 160

RIDGE_NAME = 'Ridge'

# Architecture per encoder — drives panel placement (top vs bottom row)
# and the title/subtitle colour.
MODEL_ARCH = {
    'lm all-minilm-l6-v2':            'Encoder',
    'lm e5-base-v2':                   'Encoder',
    'lm bge-large':                    'Encoder',
    'lm llama-nemotron-embed-1b-v2':   'Encoder distilled\nfrom Decoder',
    'lm llama-3.1-8b':                 'Decoder',
    'lm qwen-3-8b':                    'Decoder',
    'lm opt-6.7b':                     'Decoder',
}
ARCH_COLORS = {
    'Encoder':                          '#D62728',
    'Decoder':                          '#E377C2',
    'Encoder distilled\nfrom Decoder':  '#8C564B',
}

DESIRED_ORDER_KEYS = ['minilm-l6', 'e5-base', 'bge', 'nemotron', 'llama-3', 'qwen', 'opt']


def _display_name(model):
    """Drop the ``LM `` prefix and shorten Nemotron for panel titles."""
    name = str(model)
    if name.startswith('LM '):
        name = name[3:]
    if 'nemotron' in name.lower():
        name = 'Nemotron-1B'
    return name


def _model_order(all_models):
    ordered = []
    for key in DESIRED_ORDER_KEYS:
        ordered.extend([m for m in all_models if key.lower() in str(m).lower()])
    unmatched = [m for m in all_models if m not in ordered]
    if unmatched:
        print(f"figure_E2: appending unmatched models at end: {unmatched}")
        ordered.extend(unmatched)
    return ordered


# ---------------------------------------------------------------------------
# 3. Panel rendering
# ---------------------------------------------------------------------------

def _draw_panel(ax, model, encoder_pivot, learner_pivot, tfidf_val, is_first_in_row):
    """Render one encoder panel: per-learner background lines, Ridge
    annotation, foreground 3-point arrow line, Tf-Idf horizontal baseline,
    arch-coloured title."""
    learner_rows = learner_pivot[learner_pivot['encoder'] == model]

    # Per-learner background lines (faint grey).
    for _, lrow in learner_rows.iterrows():
        vals = [lrow.get(c, np.nan) for c in ORDER]
        if any(pd.isna(v) for v in vals):
            continue
        ax.plot(np.arange(3), vals, color='grey',
                alpha=BG_ALPHA, lw=BG_LW, zorder=1)
        for i, (col, v) in enumerate(zip(ORDER, vals)):
            ax.scatter(i, v, s=BG_MARKER_S, color=COLORS[col],
                       marker=SHAPES[col],
                       alpha=BG_ALPHA + 0.15, edgecolor='none', zorder=1.5)

    # Ridge annotation — arrow pointing at Ridge's middle (StandScal) point.
    ridge_row = learner_rows[learner_rows['learner'] == RIDGE_NAME]
    if not ridge_row.empty:
        ridge_vals = ridge_row.iloc[0][ORDER].values
        ax.annotate(
            'Ridge',
            xy=(1, float(ridge_vals[1])),
            xytext=(-45, 0),
            textcoords='offset points',
            fontsize=FS_RIDGE, fontweight='bold',
            color='black', alpha=0.40,
            ha='right', va='center',
            arrowprops=dict(
                arrowstyle='->', color='black',
                lw=1.2, alpha=0.40,
                shrinkA=2, shrinkB=4,
            ),
            zorder=5,
        )

    # Foreground aggregated line + value labels.
    row = encoder_pivot.loc[model, ORDER].values
    ax.plot(np.arange(3), row, color='black', lw=1.0, alpha=0.4, zorder=2)
    for i, (col, val) in enumerate(zip(ORDER, row)):
        ax.scatter(i, val, s=FG_MARKER_S, color=COLORS[col],
                   marker=SHAPES[col],
                   zorder=4, edgecolor='white', linewidth=2.0)
        ax.annotate(f'{val:.4f}', (i, val), xytext=(0, 12),
                    textcoords='offset points', ha='center',
                    fontsize=FS_VAL_LABEL, fontweight='bold')

    # Arrows + percent-change labels between consecutive points.
    baseline = row[0]
    for i in range(1, 3):
        arrow = FancyArrowPatch(
            (i - 1, row[i - 1]), (i, row[i]),
            arrowstyle='->', mutation_scale=18,
            color=COLORS[ORDER[i]], lw=FG_LW, zorder=3,
        )
        ax.add_patch(arrow)
        pct = (row[i] - baseline) / baseline * 100
        ax.annotate(
            f'{pct:+.2f}%',
            ((2 * i - 1) / 2, (row[i - 1] + row[i]) / 2),
            xytext=(0, -15), textcoords='offset points',
            ha='center', fontsize=FS_PCT_LABEL,
            color=COLORS[ORDER[i]], fontweight='bold',
        )

    # Tf-Idf dashed baseline.
    ax.axhline(y=tfidf_val, color=TFIDF_COLOR, linestyle='--',
               lw=2.8, zorder=0, alpha=0.8)
    if is_first_in_row:
        ax.annotate(
            'Tf-Idf',
            xy=(0.3, tfidf_val - 0.02),
            xytext=(-8, 0),
            textcoords='offset points',
            color=TFIDF_COLOR,
            fontsize=11, fontweight='bold',
            ha='right', va='center',
            annotation_clip=False,
            zorder=10,
        )

    # Y-limit covers per-learner spread + foreground + Tf-Idf.
    learner_vals = learner_rows[ORDER].values.flatten()
    learner_vals = learner_vals[~pd.isna(learner_vals)]
    if learner_vals.size:
        y_lo = min(learner_vals.min(), row.min(), tfidf_val) - 0.02
        y_hi = max(learner_vals.max(), row.max(), tfidf_val) + 0.03
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_ylim(0.60, 0.75)

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=FS_YTICK)
    ax.set_xlim(-0.5, 2.5)

    # Arch-coloured title + italic subtitle.
    name = _display_name(model)
    parts = name.split('-')
    if len(name) > 18 and len(parts) > 2:
        title = '-'.join(parts[:2]) + '-\n' + '-'.join(parts[2:])
    else:
        title = name
    arch = MODEL_ARCH.get(str(model).lower(), '')
    arch_color = ARCH_COLORS.get(arch, 'black')
    if arch == 'Decoder':
        ax.set_facecolor('#FDF2F8')

    ax.set_title('')
    ax.text(0.5, 1.10, title, transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=FS_TITLE, fontweight='bold', color=arch_color)
    if arch:
        n_lines = arch.count('\n') + 1
        subtitle_y = 0.88 if n_lines > 1 else 0.95
        ax.text(0.5, subtitle_y, arch, transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=FS_TITLE - 4, color=arch_color, style='italic')

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def plot_post_processing_appendix():
    results = load_results()
    pp = _load_post_processing_data(results)
    encoder_pivot, learner_pivot = _build_pivots(pp)
    tfidf_val = _tfidf_baseline(results)

    encoder_order = _model_order(encoder_pivot.index.tolist())
    encoder_models = [
        m for m in encoder_order
        if MODEL_ARCH.get(str(m).lower(), '') in (
            'Encoder', 'Encoder distilled\nfrom Decoder',
        )
    ]
    decoder_models = [
        m for m in encoder_order
        if MODEL_ARCH.get(str(m).lower(), '') == 'Decoder'
    ]
    n_top, n_bot = len(encoder_models), len(decoder_models)

    fig = plt.figure(figsize=(3.2 * max(n_top, n_bot), 7))
    gs_top = fig.add_gridspec(1, n_top, left=0.06, right=0.98,
                              top=0.92, bottom=0.52, wspace=0.15)
    gs_bot = fig.add_gridspec(1, n_bot, left=0.06, right=0.78,
                              top=0.44, bottom=0.08, wspace=0.15)
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(n_top)]
    axes_bot = [fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)]

    for i, model in enumerate(encoder_models):
        _draw_panel(axes_top[i], model, encoder_pivot, learner_pivot,
                    tfidf_val, is_first_in_row=(i == 0))
    for i, model in enumerate(decoder_models):
        _draw_panel(axes_bot[i], model, encoder_pivot, learner_pivot,
                    tfidf_val, is_first_in_row=(i == 0))

    # Sync y-limits within each row, then drop redundant y-tick labels on
    # subsequent panels.
    for row_axes in (axes_top, axes_bot):
        if not row_axes:
            continue
        y_lo = min(ax.get_ylim()[0] for ax in row_axes)
        y_hi = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(y_lo, y_hi)
        for ax in row_axes[1:]:
            ax.tick_params(labelleft=False)

    if axes_top:
        axes_top[0].set_ylabel('Score (higher is better)', fontsize=FS_YLABEL, y=0.4)
    if axes_bot:
        axes_bot[0].set_ylabel('Score (higher is better)', fontsize=FS_YLABEL, y=0.4)

    legend_handles = [
        plt.Line2D([0], [0], marker=SHAPES[c], color='w',
                   markerfacecolor=COLORS[c], markeredgecolor=COLORS[c],
                   markersize=14, label=c)
        for c in ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=FS_LEGEND + 3,
        handletextpad=0.4, columnspacing=1.2,
        borderpad=0.4, handlelength=1.2,
    )
    fig.text(
        0.75, -0.02,
        'Percentages show performance difference compared to Default 30 PCA.',
        ha='right', va='bottom',
        fontsize=FS_FOOTER, color='dimgray',
    )

    save_figure(
        fig, "post_processing_comparison_per_encoder_with_learner_bg_APPENDIX",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_post_processing_appendix()
