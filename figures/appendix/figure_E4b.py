"""Figure E.4(b) — Per-dataset runtime ratio between LLaMA-3.1-8B + TabPFN-2.5 with PCA at
60-vs-30 and 120-vs-30 components, on a log-scale x-axis."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from configs.path_configs import path_configs
from figures._main import (
    dtype_map,
    encoder_map,
    learner_map,
    load_results,
    save_figure,
)


def _load_pca60(score_dir):
    files = list(Path(score_dir).glob("**/score/*.csv"))
    dfs = Parallel(n_jobs=-1)(delayed(pd.read_csv)(f) for f in files)
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = (meta[2] + "_default").replace(learner_map)
    df['PCA_dimensions'] = '60-PCA'
    return df


def _load_pca120():
    df = pd.read_csv(f"{path_configs['compiled_results']}/result_REBUTTALS_tabPFN_LLaMA8_120pca.csv")
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = (meta[2] + "_default").replace(learner_map)
    df['PCA_dimensions'] = '120-PCA'
    return df


def _build_runtime_ratios():
    results = load_results()
    df_30 = results[
        (results['learner'] == 'TabPFN-2.5')
        & (results['dtype'] == 'Num+Str')
        & (results['encoder'] == 'LM LLaMA-3.1-8B')
    ].copy()
    df_30['PCA_dimensions'] = '30-PCA'

    df_60  = _load_pca60(f"{path_configs['results']}/benchmark_llama3.1_8b_60pca_tabpfn")
    df_120 = _load_pca120()

    combined = pd.concat([df_30, df_60, df_120], axis=0)
    pivot = (
        combined
        .groupby(['data_name', 'PCA_dimensions'], as_index=False)['run_time']
        .mean()
        .pivot(index='data_name', columns='PCA_dimensions', values='run_time')
        .reset_index()
    )
    return pd.DataFrame({
        'data_name': pivot['data_name'],
        '60 vs 30':  pivot['60-PCA']  / pivot['30-PCA'],
        '120 vs 30': pivot['120-PCA'] / pivot['30-PCA'],
    })


def plot_pca_runtime_ratio():
    runtime_ratios = _build_runtime_ratios()
    melted = runtime_ratios.melt(
        id_vars=['data_name'],
        var_name='Comparison',
        value_name='Runtime Ratio',
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.set_theme(style="whitegrid")
    palette = {'120 vs 30': 'lightcoral', '60 vs 30': 'navajowhite'}
    sns.boxplot(
        data=melted, x='Runtime Ratio', y='Comparison',
        palette=palette,
        width=0.5, fliersize=4, linewidth=1.5,
        ax=ax,
    )

    # Log-scale x-axis with ×0.5 / ×1 / ×2 / ×4 / ×8 ticks.
    ax.set_xscale('log')
    major_ticks = [0.5, 1, 2, 4, 8]
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(
        [r'$\times 0.5$', r'$\times 1$', r'$\times 2$', r'$\times 4$', r'$\times 8$']
    ))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.axvline(1, color='tab:green', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.text(0.6, 0.5, "← Faster", ha='center', va='bottom',
            color='tab:green', fontweight='bold', fontsize=14,
            transform=ax.get_xaxis_transform())
    ax.text(3, 0.5, "Slower →", ha='center', va='bottom',
            color='tab:red', fontweight='bold', fontsize=14,
            transform=ax.get_xaxis_transform())

    for i, comp in enumerate(['60 vs 30', '120 vs 30']):
        median_val = runtime_ratios[comp].median()
        ax.text(4.5, i, f'Median: {median_val:.2f}x', ha='left', va='center',
                fontsize=9, fontweight='bold', backgroundcolor='white')

    ax.set_xlabel('Runtime Ratio (Log Scale)', fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(left=0.4, right=10)
    plt.tight_layout()

    save_figure(fig, "runtime_boxplot_pca30_vs_60_vs_120_delta_per_sample_size")
    plt.close(fig)


if __name__ == "__main__":
    plot_pca_runtime_ratio()
