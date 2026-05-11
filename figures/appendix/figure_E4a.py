"""Figure E.4(a) — Per-dataset score delta between LLaMA-3.1-8B + TabPFN-2.5 with PCA at
60-vs-30 and 120-vs-30 components."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
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
    """Compile per-fold scores from the 60-PCA ablation directory."""
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


def _build_score_diffs():
    """Per-dataset score deltas for 60-vs-30 and 120-vs-30 PCA dimensions."""
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
        .groupby(['data_name', 'PCA_dimensions'], as_index=False)['score']
        .mean()
        .pivot(index='data_name', columns='PCA_dimensions', values='score')
        .reset_index()
    )
    return pd.DataFrame({
        'data_name': pivot['data_name'],
        '60 vs 30':  pivot['60-PCA']  - pivot['30-PCA'],
        '120 vs 30': pivot['120-PCA'] - pivot['30-PCA'],
    })


def plot_pca_score_delta():
    score_diffs = _build_score_diffs()
    melted = score_diffs.melt(
        id_vars=['data_name'],
        var_name='Comparison',
        value_name='Score Difference',
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.set_theme(style="whitegrid")
    palette = {'120 vs 30': 'lightcoral', '60 vs 30': 'navajowhite'}
    sns.boxplot(
        data=melted, x='Score Difference', y='Comparison',
        palette=palette,
        width=0.5, fliersize=4, linewidth=1.5,
        ax=ax,
    )
    ax.axvline(0, color='dimgrey', linestyle='--', linewidth=2.5, alpha=0.8)

    ax.text(-0.12, 0.5, "← Lower PCA Better", ha='center', va='bottom',
            color='tab:green', fontweight='bold', fontsize=12,
            transform=ax.get_xaxis_transform())
    ax.text(0.15, 0.5, "Higher PCA Better →", ha='center', va='bottom',
            color='tab:red', fontweight='bold', fontsize=12,
            transform=ax.get_xaxis_transform())

    for i, comp in enumerate(['120 vs 30', '60 vs 30']):
        median_val = score_diffs[comp].median()
        ax.text(0.18, i, f'Median: {median_val:.4f}', ha='left', va='center',
                fontsize=9, fontweight='bold', backgroundcolor='white')

    ax.set_xlabel(r'$Score_{higher} - Score_{lower}$ where Score = $R^2$ & AUC',
                  fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(-0.25, 0.35)
    plt.tight_layout()

    save_figure(fig, "boxplot_pca30_vs_60_vs_120_delta_per_sample_size")
    plt.close(fig)


if __name__ == "__main__":
    plot_pca_score_delta()
