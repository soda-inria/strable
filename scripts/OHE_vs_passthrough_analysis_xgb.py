""" Supports the claim from table E.5 """

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from configs.path_configs import path_configs

# --- Load all 8 score DataFrames in a single Parallel call ---

base_path = f"{path_configs['results']}"

SCORE_DIRS = {
    # XGBoost (+ Ridge, ExtraTrees) runs
    'df_xgb_ohe_ss':            f"{base_path}/benchmark_llama_qwen_nemotron_30pca_with_standscal_CT30_OHE_ridge_extrees_xgboost",
    'df_xgb_ohe_no_ss':         f"{base_path}/benchmark_llama_qwen_nemotron_30pca_without_standscal_CT30_OHE_ridge_extrees_xgboost",
    'df_xgb_pass_ss':           f"{base_path}/benchmark_llama_qwen_nemotron_30pca_with_standscal_CT30_passthrough_xgboost",
    'df_xgb_pass_no_ss':        f"{base_path}/benchmark_llama_qwen_nemotron_30pca_without_standscal_CT30_passthrough_xgboost",
    # TabPFN + TabICL runs (Qwen only)
    'df_tabpfn_tabicl_pass_ss':    f"{base_path}/benchmark_qwen_standscal_30pca_tabpfn_tabicl_30_thresh",
    'df_tabpfn_tabicl_pass_no_ss': f"{base_path}/benchmark_qwen_30pca_tabpfn_tabicl_30_thresh",
    'df_tabpfn_tabicl_ohe_ss':     f"{base_path}/benchmark_qwen_standscal_30pca_tabpfn_tabicl_30_thresh_OHE",
    'df_tabpfn_tabicl_ohe_no_ss':  f"{base_path}/benchmark_qwen_30pca_tabpfn_tabicl_30_thresh_OHE",
}

# Collect (label, file) pairs across all directories first.
labelled_files = []
for label, score_dir_path in SCORE_DIRS.items():
    score_files = list(Path(score_dir_path).glob("**/score/*.csv"))
    if not score_files:
        print(f"WARNING: no files found in {score_dir_path}")
    labelled_files.extend((label, f) for f in score_files)

# Single Parallel pool reads every score CSV across all 8 directories.
read_dfs = Parallel(n_jobs=-1)(delayed(pd.read_csv)(f) for _, f in labelled_files)

dfs_by_label = {label: [] for label in SCORE_DIRS}
for (label, _), df in zip(labelled_files, read_dfs):
    dfs_by_label[label].append(df)

def _finalize(dfs):
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df['score'] = df['r2'].fillna(df['roc_auc'])
    return df

df_xgb_ohe_ss               = _finalize(dfs_by_label['df_xgb_ohe_ss'])
df_xgb_ohe_no_ss            = _finalize(dfs_by_label['df_xgb_ohe_no_ss'])
df_xgb_pass_ss              = _finalize(dfs_by_label['df_xgb_pass_ss'])
df_xgb_pass_no_ss           = _finalize(dfs_by_label['df_xgb_pass_no_ss'])
df_tabpfn_tabicl_pass_ss    = _finalize(dfs_by_label['df_tabpfn_tabicl_pass_ss'])
df_tabpfn_tabicl_pass_no_ss = _finalize(dfs_by_label['df_tabpfn_tabicl_pass_no_ss'])
df_tabpfn_tabicl_ohe_ss     = _finalize(dfs_by_label['df_tabpfn_tabicl_ohe_ss'])
df_tabpfn_tabicl_ohe_no_ss  = _finalize(dfs_by_label['df_tabpfn_tabicl_ohe_no_ss'])

print("Loaded shapes:")
print(f"  XGB OHE + SS:       {df_xgb_ohe_ss.shape}")
print(f"  XGB OHE no SS:      {df_xgb_ohe_no_ss.shape}")
print(f"  XGB Pass + SS:      {df_xgb_pass_ss.shape}")
print(f"  XGB Pass no SS:     {df_xgb_pass_no_ss.shape}")
print(f"  TabPFN/ICL OHE+SS:  {df_tabpfn_tabicl_ohe_ss.shape}")
print(f"  TabPFN/ICL OHE noSS:{df_tabpfn_tabicl_ohe_no_ss.shape}")
print(f"  TabPFN/ICL Pass+SS: {df_tabpfn_tabicl_pass_ss.shape}")
print(f"  TabPFN/ICL Pass noSS:{df_tabpfn_tabicl_pass_no_ss.shape}")

# --- Filter helpers ---

def filter_learner(df, learner_substring):
    return df[df['method'].str.contains(learner_substring, case=False, na=False)].copy()

# XGBoost: filter from Ridge+Extrees+XGBoost runs
df_xgb_ohe_ss_f = filter_learner(df_xgb_ohe_ss, 'xgb')
df_xgb_ohe_no_ss_f = filter_learner(df_xgb_ohe_no_ss, 'xgb')
df_xgb_pass_ss_f = filter_learner(df_xgb_pass_ss, 'xgb')
df_xgb_pass_no_ss_f = filter_learner(df_xgb_pass_no_ss, 'xgb')

# TabPFN and TabICL are already isolated in their runs, but filter in case
df_tabpfn_ohe_ss = filter_learner(df_tabpfn_tabicl_ohe_ss, 'tabpfn')
df_tabpfn_ohe_no_ss = filter_learner(df_tabpfn_tabicl_ohe_no_ss, 'tabpfn')
df_tabpfn_pass_ss = filter_learner(df_tabpfn_tabicl_pass_ss, 'tabpfn')
df_tabpfn_pass_no_ss = filter_learner(df_tabpfn_tabicl_pass_no_ss, 'tabpfn')

df_tabicl_ohe_ss = filter_learner(df_tabpfn_tabicl_ohe_ss, 'tabicl')
df_tabicl_ohe_no_ss = filter_learner(df_tabpfn_tabicl_ohe_no_ss, 'tabicl')
df_tabicl_pass_ss = filter_learner(df_tabpfn_tabicl_pass_ss, 'tabicl')
df_tabicl_pass_no_ss = filter_learner(df_tabpfn_tabicl_pass_no_ss, 'tabicl')

print(f"\nAfter learner filtering:")
print(f"  XGB Pass+SS: {df_xgb_pass_ss_f.shape}")
print(f"  TabPFN Pass+SS: {df_tabpfn_pass_ss.shape}")
print(f"  TabICL Pass+SS: {df_tabicl_pass_ss.shape}")

# --- Per-dataset mean scores ---

GROUP_KEYS = ['data_name', 'method']
SCORE_COL = 'score'

# --- Datasets to exclude globally ---
# device-covid19serology contains features deterministically derived from the
# target (iga_result, pan_result), causing all learners to score ~1.0 under at
# least one encoding. Excluded for all (learner, encoding) combinations to
# avoid contaminating the OHE-vs-passthrough comparison with target leakage.
EXCLUDED_DATASETS = {'device-covid19serology'}


def extract_encoder(method_name):
    name = method_name.lower()
    if 'nemotron' in name:
        return 'Nemotron-1B'
    elif 'llama' in name:
        return 'LLaMA-3.1-8B'
    elif 'qwen' in name:
        return 'Qwen-3-8B'
    else:
        return 'Other'


def mean_scores(df):
    if len(df) == 0:
        return pd.DataFrame(columns=GROUP_KEYS + [SCORE_COL])
    df = df.copy()
    df = df[~df['data_name'].isin(EXCLUDED_DATASETS)]   # <-- global filter
    df['encoder'] = df['method'].apply(extract_encoder)
    return df.groupby(GROUP_KEYS)[SCORE_COL].mean().reset_index()

# --- Delta computation helper ---

def compute_delta(df_ohe, df_pass, label_ohe='ohe', label_pass='pass'):
    s_ohe = mean_scores(df_ohe).rename(columns={SCORE_COL: f'score_{label_ohe}'})
    s_pass = mean_scores(df_pass).rename(columns={SCORE_COL: f'score_{label_pass}'})
    delta = s_ohe.merge(s_pass, on=GROUP_KEYS, how='inner')
    delta['delta'] = delta[f'score_{label_pass}'] - delta[f'score_{label_ohe}']
    delta['abs_delta'] = delta['delta'].abs()
    return delta

# --- XGBoost deltas ---
delta_xgb_ss = compute_delta(df_xgb_ohe_ss_f, df_xgb_pass_ss_f)
delta_xgb_no_ss = compute_delta(df_xgb_ohe_no_ss_f, df_xgb_pass_no_ss_f)

# --- TabPFN deltas (Qwen only) ---
delta_tabpfn_ss = compute_delta(df_tabpfn_ohe_ss, df_tabpfn_pass_ss)
delta_tabpfn_no_ss = compute_delta(df_tabpfn_ohe_no_ss, df_tabpfn_pass_no_ss)

# --- TabICL deltas (Qwen only) ---
delta_tabicl_ss = compute_delta(df_tabicl_ohe_ss, df_tabicl_pass_ss)
delta_tabicl_no_ss = compute_delta(df_tabicl_ohe_no_ss, df_tabicl_pass_no_ss)

# --- Summary reporter ---

def summarize_delta(df, label):
    print(f"\n=== {label} ===")
    if len(df) == 0:
        print("  (empty — nothing to merge)")
        return
    print(f"  N pairs: {len(df)}")
    print(f"  Mean delta (pass - ohe): {df['delta'].mean():+.4f}")
    print(f"  Median delta: {df['delta'].median():+.4f}")
    print(f"  Std of delta: {df['delta'].std():.4f}")
    print(f"  Mean absolute delta: {df['abs_delta'].mean():.4f}")
    print(f"  Max absolute delta: {df['abs_delta'].max():.4f}")
    print(f"  95th percentile |delta|: {df['abs_delta'].quantile(0.95):.4f}")
    print(f"  Fraction |delta| < 0.01: {(df['abs_delta'] < 0.01).mean():.2%}")
    print(f"  Fraction |delta| < 0.005: {(df['abs_delta'] < 0.005).mean():.2%}")

print("\n" + "="*70)
print("OHE vs PASSTHROUGH for learners that natively handle categoricals")
print("="*70)

summarize_delta(delta_xgb_ss, "XGBoost, with StandScal+PCA")
summarize_delta(delta_xgb_no_ss, "XGBoost, without StandScal+PCA")
summarize_delta(delta_tabpfn_ss, "TabPFN-2.5 (Qwen), with StandScal+PCA")
summarize_delta(delta_tabpfn_no_ss, "TabPFN-2.5 (Qwen), without StandScal+PCA")
summarize_delta(delta_tabicl_ss, "TabICLv2 (Qwen), with StandScal+PCA")
summarize_delta(delta_tabicl_no_ss, "TabICLv2 (Qwen), without StandScal+PCA")

# --- Per-encoder breakdown (XGBoost only, since it has all 3 encoders) ---

delta_xgb_ss['encoder'] = delta_xgb_ss['method'].apply(extract_encoder)
delta_xgb_no_ss['encoder'] = delta_xgb_no_ss['method'].apply(extract_encoder)

print("\n=== XGBoost per-encoder breakdown (with StandScal) ===")
print(delta_xgb_ss.groupby('encoder').agg(
    n=('delta', 'count'),
    mean_delta=('delta', 'mean'),
    mean_abs_delta=('abs_delta', 'mean'),
    max_abs_delta=('abs_delta', 'max'),
).round(4))

print("\n=== XGBoost per-encoder breakdown (without StandScal) ===")
print(delta_xgb_no_ss.groupby('encoder').agg(
    n=('delta', 'count'),
    mean_delta=('delta', 'mean'),
    mean_abs_delta=('abs_delta', 'mean'),
    max_abs_delta=('abs_delta', 'max'),
).round(4))

print("\n=== TabPFN-2.5 per-encoder breakdown (with StandScal, aligned) ===")
print(delta_tabpfn_ss.groupby('encoder').agg(
    n=('delta', 'count'),
    mean_delta=('delta', 'mean'),
    mean_abs_delta=('abs_delta', 'mean'),
    max_abs_delta=('abs_delta', 'max'),
).round(4))

print("\n=== TabPFN-2.5 per-encoder breakdown (without StandScal, aligned) ===")
print(delta_tabpfn_no_ss.groupby('encoder').agg(
    n=('delta', 'count'),
    mean_delta=('delta', 'mean'),
    mean_abs_delta=('abs_delta', 'mean'),
    max_abs_delta=('abs_delta', 'max'),
).round(4))

# --- Per-encoder breakdown (TabICLv2) ---

print("\n=== TabICLv2 per-encoder breakdown (with StandScal, aligned) ===")
print(delta_tabicl_ss.groupby('encoder').agg(
    n=('delta', 'count'),
    mean_delta=('delta', 'mean'),
    mean_abs_delta=('abs_delta', 'mean'),
    max_abs_delta=('abs_delta', 'max'),
).round(4))

print("\n=== TabICLv2 per-encoder breakdown (without StandScal, aligned) ===")
print(delta_tabicl_no_ss.groupby('encoder').agg(
    n=('delta', 'count'),
    mean_delta=('delta', 'mean'),
    mean_abs_delta=('abs_delta', 'mean'),
    max_abs_delta=('abs_delta', 'max'),
).round(4))

# --- Combined summary table across all learners ---

def summary_row(delta_df, learner, standscal):
    if len(delta_df) == 0:
        return None
    return {
        'learner': learner,
        'standscal': standscal,
        'n_pairs': len(delta_df),
        'mean_delta': delta_df['delta'].mean(),
        'mean_abs_delta': delta_df['abs_delta'].mean(),
        'max_abs_delta': delta_df['abs_delta'].max(),
        'frac_within_0.01': (delta_df['abs_delta'] < 0.01).mean(),
    }

summary_rows = [
    summary_row(delta_xgb_ss, 'XGBoost', True),
    summary_row(delta_xgb_no_ss, 'XGBoost', False),
    summary_row(delta_tabpfn_ss, 'TabPFN-2.5', True),
    summary_row(delta_tabpfn_no_ss, 'TabPFN-2.5', False),
    summary_row(delta_tabicl_ss, 'TabICLv2', True),
    summary_row(delta_tabicl_no_ss, 'TabICLv2', False),
]

summary_df = pd.DataFrame([r for r in summary_rows if r is not None])
print("\n" + "="*70)
print("COMBINED SUMMARY TABLE")
print("="*70)
print(summary_df.round(4).to_string(index=False))

print("TabPFN outliers (StandScal + PCA):")
print(delta_tabpfn_ss.sort_values('abs_delta', ascending=False).head(3)[
    ['data_name', 'method', 'score_ohe', 'score_pass', 'delta']
])

print("\nXGBoost outliers (StandScal + PCA):")
print(delta_xgb_ss.sort_values('abs_delta', ascending=False).head(5)[
    ['data_name', 'method', 'score_ohe', 'score_pass', 'delta']
])