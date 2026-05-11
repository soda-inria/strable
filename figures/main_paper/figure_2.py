"""Figure 2 — Across-learner mean score per encoder under three post-processing strategies:
Default 30-PCA (blue), Standard scaling + 30-PCA (orange), No PCA / first 30
raw embedding dimensions (green)."""

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
# 1. Data loading — duplicated with figure_E2.py per the project's policy
# ---------------------------------------------------------------------------

COMPILED = path_configs['compiled_results']


def _preprocess_results(df):
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
    """All 7 encoders × 3 post-processing variants × 5 learners. See
    ``figure_E2._load_post_processing_data`` for the same logic."""
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
    qwen_tabpfn_nopca = pd.read_csv(f"{COMPILED}/result_comparison_qwen_nopca_30.csv")
    qwen_tabpfn_nopca['post_processing'] = 'No PCA'
    qwen_tabpfn_nopca['method'] = qwen_tabpfn_nopca['method'].str.replace('-no_pca', '')
    qwen_tabicl_nopca = pd.read_csv(f"{COMPILED}/result_comparison_tabicl_qwen.csv")
    qwen_tabicl_nopca['post_processing'] = 'No PCA'
    qwen_tabicl_nopca['method'] = qwen_tabicl_nopca['method'].str.replace('_nopca', '')

    llama_tabpfn_default = results[results['method'].isin([
        'num-str_llm-llama-3.1-8b_tabpfn_default'
    ])].copy()
    llama_tabpfn_default['post_processing'] = 'Default PCA'
    llama_tabicl_default = pd.read_csv(f"{COMPILED}/result_comparison_tabicl.csv")
    llama_tabicl_default = llama_tabicl_default[
        llama_tabicl_default['method'].str.contains('llama-3.1-8b')
    ].copy()
    llama_tabicl_default['post_processing'] = 'Default PCA'
    llama_tabpfn_standscal = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
    llama_tabpfn_standscal = llama_tabpfn_standscal[
        llama_tabpfn_standscal['method'].str.contains('llama-3.1-8b')
    ].copy()
    llama_tabpfn_standscal['post_processing'] = 'StandScal + 30-PCA'
    llama_tabpfn_standscal['method'] = llama_tabpfn_standscal['method'] + '_default'
    llama_tabicl_standscal = pd.read_csv(f"{COMPILED}/result_comparison_tabicl-llama.csv")
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
    nemotron_tabpfn_standscal = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
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
    minilml6_tabpfn_standscal = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
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
    e5basev2_tabpfn_standscal = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
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


def _build_pivot(post_processing_results):
    return (
        post_processing_results
        .groupby(['encoder', 'post_processing'], as_index=False)['score']
        .mean()
        .pivot(index='encoder', columns='post_processing', values='score')
    )


def _tfidf_baseline(results):
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
# 2. Figure rendering — single row, no per-learner backgrounds
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

FG_LW       = 3.0
FG_MARKER_S = 160

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
        print(f"figure_2: appending unmatched models at end: {unmatched}")
        ordered.extend(unmatched)
    return ordered


def _draw_panel(ax, model, encoder_pivot, tfidf_val, is_first):
    row = encoder_pivot.loc[model, ORDER].values
    x_pos = np.arange(3)

    # Foreground line + scatter + per-point value labels.
    ax.plot(x_pos, row, color='black', lw=1.0, alpha=0.4, zorder=2)
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

    # Tf-Idf dashed baseline. Annotate only on the first panel to avoid
    # repeating the label across all 7.
    ax.axhline(y=tfidf_val, color=TFIDF_COLOR, linestyle='--',
               lw=2.8, zorder=0, alpha=0.8)
    if is_first:
        ax.annotate(
            'Tf-Idf',
            xy=(0.32, tfidf_val - 0.01),
            xytext=(-8, 0),
            textcoords='offset points',
            color=TFIDF_COLOR,
            fontsize=11, fontweight='bold',
            ha='right', va='center',
            annotation_clip=False,
            zorder=10,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=FS_YTICK)
    ax.set_xlim(-0.5, 2.5)

    name = _display_name(model)
    parts = name.split('-')
    title = '-'.join(parts[:2]) + '-\n' + '-'.join(parts[2:]) \
        if (len(name) > 18 and len(parts) > 2) else name
    arch = MODEL_ARCH.get(str(model).lower(), '')
    arch_color = ARCH_COLORS.get(arch, 'black')
    if arch == 'Decoder':
        ax.set_facecolor('#FDF2F8')

    ax.set_title('')
    ax.text(0.5, 1.1, title, transform=ax.transAxes,
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


def plot_post_processing_main():
    results = load_results()
    pp = _load_post_processing_data(results)
    encoder_pivot = _build_pivot(pp)
    tfidf_val = _tfidf_baseline(results)

    encoder_order = _model_order(encoder_pivot.index.tolist())
    n_models = len(encoder_order)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(3.0 * n_models, 3),
        sharey=True,
        gridspec_kw={'wspace': 0.15},
    )
    if n_models == 1:
        axes = [axes]

    for i, model in enumerate(encoder_order):
        _draw_panel(axes[i], model, encoder_pivot, tfidf_val, is_first=(i == 0))

    axes[0].set_ylabel('Score (higher is better)', fontsize=FS_YLABEL, y=0.4)

    legend_handles = [
        plt.Line2D([0], [0], marker=SHAPES[c], color='w',
                   markerfacecolor=COLORS[c], markeredgecolor=COLORS[c],
                   markersize=14, label=c)
        for c in ORDER
    ]
    LEGEND_Y = -0.05
    fig.legend(
        handles=legend_handles,
        loc='lower left',
        bbox_to_anchor=(0.2, LEGEND_Y),
        ncol=3,
        frameon=False,
        fontsize=FS_LEGEND + 3,
        handletextpad=0.4, columnspacing=1.2,
        borderpad=0.4, handlelength=1.2,
    )
    fig.text(
        0.85, LEGEND_Y + 0.07,
        'Percentages show performance difference compared to Default 30 PCA.',
        ha='right', va='bottom',
        fontsize=FS_FOOTER, color='dimgray',
    )
    plt.subplots_adjust(bottom=0.20, top=0.78)

    save_figure(fig, "post_processing_comparison_per_encoder_with_learner_bg")
    plt.close(fig)


if __name__ == "__main__":
    plot_post_processing_main()
