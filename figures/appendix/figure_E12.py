"""Figure E.12 — Two-panel diagnostic tying the CT=30 threshold (which routes columns
with cardinality ≥30 to LLM, otherwise to OHE/passthrough) to the
meta-feature that most destabilizes pipeline rankings"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.path_configs import path_configs
from figures._main import save_figure


# Hardcoded — sourced from the meta-feature ranking-stability analysis
# (Figure 5c). See posthoc_analysis.py for the bootstrap computation.
META_FEATURES = [
    'Stopword Density\n(prose-ness)',
    'Dict. Hit Rate\n(naturalness)',
    'Symbol Density\n(code-ness)',
    'Prop. Multiword\n(phrase-ness)',
    'Uniqueness\n(cardinality)',
    'Avg Words/Cell\n(length)',
]
META_TAUS = [0.36, 0.36, 0.33, 0.15, 0.15, 0.09]


def plot_ct30_threshold_disruptor():
    df_structure = pd.read_csv(
        f"{path_configs['base_path']}/df_structure_VSE_STRABLE_CARTE_TTB.csv"
    )
    df_structure['ct_bin'] = np.where(
        df_structure['n_unique'] < 30,
        'low (<30, OHE/passthrough)', 'high (>=30, LLM)',
    )
    d = df_structure[df_structure['col_type_heuristic'] != 'datetime'].copy()

    bin_data = {
        'native': d.loc[d['ct_bin'].str.startswith('low'),  'avg_words_per_cell'].dropna(),
        'llm':    d.loc[d['ct_bin'].str.startswith('high'), 'avg_words_per_cell'].dropna(),
    }
    mean_native = bin_data['native'].mean()
    mean_llm    = bin_data['llm'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={'width_ratios': [1, 1.3]})

    # --- Panel A: meta-feature ranking disruption ---
    ax = axes[0]
    colors = ['#cccccc'] * len(META_FEATURES)
    colors[-1] = '#d62728'
    ax.barh(META_FEATURES[::-1], META_TAUS[::-1],
            color=colors[::-1], edgecolor='black', linewidth=0.4)
    ax.set_xlabel(r"Kendall's $\tau$  (Low vs High percentile)", fontsize=12)
    ax.set_title("Ranking stability across meta-features\n(lower = bigger disruptor)",
                 fontsize=12, pad=10)
    ax.set_xlim(0, 0.42)
    ax.spines[['top', 'right']].set_visible(False)
    ax.annotate(r"lowest $\tau$ = biggest disruptor",
                xy=(0.09, 0), xytext=(0.22, -0.6),
                fontsize=10, color='#d62728', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))

    # --- Panel B: avg_words_per_cell by routing decision ---
    ax = axes[1]
    y_native = np.random.default_rng(0).normal(0, 0.06, size=len(bin_data['native']))
    y_llm    = np.random.default_rng(1).normal(1, 0.06, size=len(bin_data['llm']))

    ax.scatter(bin_data['native'], y_native, s=14, alpha=0.35,
               color='#ff7f0e', label='Treated by Learner/OHE (<30)')
    ax.scatter(bin_data['llm'], y_llm, s=14, alpha=0.35,
               color='#1f77b4', label='Treated by LLM (≥30)')

    ax.plot([mean_native, mean_native], [-0.35, 0.35],
            color='#ff7f0e', lw=3, solid_capstyle='butt')
    ax.plot([mean_llm, mean_llm], [0.65, 1.35],
            color='#1f77b4', lw=3, solid_capstyle='butt')
    ax.text(mean_native, -0.55, f"mean = {mean_native:.2f}",
            ha='center', va='top', fontsize=11,
            color='#ff7f0e', fontweight='bold')
    ax.text(mean_llm, 1.55, f"mean = {mean_llm:.2f}",
            ha='center', va='bottom', fontsize=11,
            color='#1f77b4', fontweight='bold')

    ax.annotate('', xy=(mean_llm, 0.5), xytext=(mean_native, 0.5),
                arrowprops=dict(arrowstyle='<->', color='#d62728', lw=1.5))
    ax.text((mean_native + mean_llm) / 2, 0.5,
            f"  {mean_llm / mean_native:.1f}×  ",
            ha='center', va='center', fontsize=11,
            color='#d62728', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white', edgecolor='#d62728'))

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Learner/OHE\n(<30)', 'LLM\n(≥30)'], fontsize=11)
    ax.set_xlabel('Avg words per cell', fontsize=12)
    ax.set_title("CT=30 sorts columns on the same axis\nthat most destabilizes rankings",
                 fontsize=12, pad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.8, 1.8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "CT_30_threshold_vs_string_index_kendalltau")
    plt.close(fig)


if __name__ == "__main__":
    plot_ct30_threshold_disruptor()
