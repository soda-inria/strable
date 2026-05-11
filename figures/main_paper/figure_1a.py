"""Figure 1(a) Comparison of STRABLE (108 datasets) vs OpenML."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import openml
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from configs.path_configs import path_configs
from figures._main import (
    load_dataset_summary,
    load_results,
    save_figure,
    selected_encoders,
)


# ---------------------------------------------------------------------------
# 1. Style constants
# ---------------------------------------------------------------------------

STRABLE_COLOR = '#D94F3D'
OPENML_COLOR  = '#2E86C1'
STRABLE_ALPHA = 0.50
OPENML_ALPHA  = 0.30

FS_TITLE  = 26
FS_MEDIAN = 20
FS_AXIS   = 26
FS_TICK   = 16
FS_YTICK  = 18
FS_LEGEND = 22

LW_KDE_STRABLE = 4.5
LW_KDE_OPENML  = 4.5
LW_MEDIAN      = 4.5
LW_HIST_EDGE   = 0.8
LW_SPINE       = 1.5
LW_TICK        = 1.3


# ---------------------------------------------------------------------------
# 2. Drawing helpers
# ---------------------------------------------------------------------------

def _fmt_val(v):
    if v >= 1e6: return f'{v/1e6:.1f}M'
    if v >= 1e3: return f'{v/1e3:.1f}K'
    if v >= 10:  return f'{v:.0f}'
    if v >= 1:   return f'{v:.1f}'
    return f'{v:.2f}'


def _fmt_log1p_tick(x, _pos):
    """Tick formatter that maps log1p(x) back to the original x."""
    if x < 0:
        return ''
    orig = np.expm1(x)
    if orig < 0.01: return '0'
    if orig < 1:    return f'{orig:.2f}'
    if orig < 10:   return f'{orig:.1f}'
    if orig < 1e3:  return f'{orig:.0f}'
    if orig < 1e6:  return f'{orig/1e3:.0f}K'
    return f'{orig/1e6:.0f}M'


def _safe_kde(data, x_eval, bw=0.3):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5 or np.std(data) < 1e-10:
        return None
    try:
        return gaussian_kde(data, bw_method=bw)(x_eval)
    except Exception:
        return None


def _style_ax(ax, title, square=True):
    ax.set_title(title, fontsize=FS_TITLE, fontweight='normal', pad=6)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.grid(True, alpha=0.12, which='major')

    ax.tick_params(axis='x', labelsize=FS_TICK, width=LW_TICK, length=5)
    ax.tick_params(axis='y', labelsize=FS_YTICK, width=LW_TICK, length=5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LW_SPINE)
    ax.spines['bottom'].set_linewidth(LW_SPINE)
    ax.set_box_aspect(0.75)


def _annot_inside(ax, x, y_frac, label, color):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    span = xlim[1] - xlim[0]
    frac = (x - xlim[0]) / span if span > 0 else 0.5
    frac = float(np.clip(frac, 0.08, 0.92))

    ha = 'left' if frac < 0.5 else 'right'
    offset = 6 if ha == 'left' else -6

    x_safe = xlim[0] + frac * span
    y_pos  = ylim[0] + y_frac * (ylim[1] - ylim[0])

    ax.annotate(
        f'med={label}',
        xy=(x_safe, y_pos),
        fontsize=FS_MEDIAN, fontweight='bold', color=color,
        ha=ha, va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.88,
                  ec=color, lw=0.9),
        xytext=(offset, 0),
        textcoords='offset points',
        annotation_clip=False,
    )


def _draw_panel(ax, s_raw, o_raw, title):
    """Histogram + KDE + median line for one feature, on a log1p axis."""
    s_data = pd.Series(s_raw).dropna().astype(float)
    o_data = pd.Series(o_raw).dropna().astype(float)
    s_data = s_data[np.isfinite(s_data) & (s_data >= 0)]
    o_data = o_data[np.isfinite(o_data) & (o_data >= 0)]

    s_log = np.log1p(s_data.values)
    o_log = np.log1p(o_data.values)

    combined = np.concatenate([s_log, o_log])
    lo, hi = float(combined.min()), float(combined.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.03
    lo_bin, hi_bin = max(0.0, lo - pad), hi + pad

    bin_edges = np.linspace(lo_bin, hi_bin, 22)
    bin_width = bin_edges[1] - bin_edges[0]

    ax.hist(o_log, bins=bin_edges,
            weights=np.ones(len(o_log)) / len(o_log),
            alpha=OPENML_ALPHA, color=OPENML_COLOR,
            edgecolor='white', linewidth=LW_HIST_EDGE, zorder=2)
    ax.hist(s_log, bins=bin_edges,
            weights=np.ones(len(s_log)) / len(s_log),
            alpha=STRABLE_ALPHA, color=STRABLE_COLOR,
            edgecolor='white', linewidth=LW_HIST_EDGE, zorder=3)

    x_eval = np.linspace(lo_bin, hi_bin, 500)

    kde_s = _safe_kde(s_log, x_eval)
    if kde_s is not None:
        ax.plot(x_eval, np.minimum(kde_s * bin_width, 1.0),
                color=STRABLE_COLOR, linewidth=LW_KDE_STRABLE, zorder=5)

    kde_o = _safe_kde(o_log, x_eval)
    if kde_o is not None:
        ax.plot(x_eval, np.minimum(kde_o * bin_width, 1.0),
                color=OPENML_COLOR, linewidth=LW_KDE_OPENML, linestyle='--', zorder=5)

    s_med_orig = float(np.median(s_data))
    o_med_orig = float(np.median(o_data))
    s_med_log  = float(np.log1p(s_med_orig))
    o_med_log  = float(np.log1p(o_med_orig))

    ax.axvline(s_med_log, color=STRABLE_COLOR, linewidth=LW_MEDIAN,
               alpha=0.85, zorder=6)
    ax.axvline(o_med_log, color=OPENML_COLOR, linewidth=LW_MEDIAN, linestyle='--',
               alpha=0.85, zorder=6)

    ax.set_xlim(lo_bin, hi_bin)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_log1p_tick))

    _style_ax(ax, title)

    ymax_auto = ax.get_ylim()[1]
    ax.set_ylim(0, min(1.0, max(ymax_auto, 1e-3)))

    _annot_inside(ax, s_med_log, 0.95, _fmt_val(s_med_orig), STRABLE_COLOR)
    _annot_inside(ax, o_med_log, 0.75, _fmt_val(o_med_orig), OPENML_COLOR)


# ---------------------------------------------------------------------------
# 3. Data preparation
# ---------------------------------------------------------------------------

E2E_ENCODERS = {'ContextTab', 'TabSTAR', 'CatBoost'}


def _build_data():
    """Load and prepare every series the figure needs.

    Returns
    -------
    dict[str, tuple[pandas.Series, pandas.Series]]
        Keyed by panel label; each value is ``(strable_series, openml_series)``.
    """
    np.random.seed(42)
    results = load_results()
    dataset_summary_wide = load_dataset_summary()

    df_results = results[
        (results['dtype'] == 'Num+Str')
        & (results['encoder'].isin(selected_encoders))
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()

    # ``df_results_fair`` keeps only the best learner per encoder so the
    # stopword-density panel reflects a single representative pipeline per
    # encoder (E2E learners are kept whole — they aren't paired with a
    # separate encoder).
    score = 'score'
    best_learner = (
        df_results[~df_results['encoder'].isin(E2E_ENCODERS)]
        .groupby(['encoder', 'learner'])[score]
        .mean()
        .reset_index()
        .sort_values(score, ascending=False)
        .groupby('encoder')
        .first()
        .reset_index()[['encoder', 'learner']]
    )
    df_modular_best = df_results[~df_results['encoder'].isin(E2E_ENCODERS)].merge(
        best_learner, on=['encoder', 'learner'], how='inner',
    )
    df_e2e = df_results[df_results['encoder'].isin(E2E_ENCODERS)]
    df_results_fair = pd.concat([df_modular_best, df_e2e], ignore_index=True)

    # The structure CSV is needed because ``stopword_density`` is computed
    # at the (dataset, column) level by the natural-language-test pipeline
    # and joined onto results_fair.
    df_structure = pd.read_csv(
        f"{path_configs['base_path']}/df_structure_VSE_STRABLE_CARTE_TTB.csv"
    )
    df_structure = df_structure[df_structure['col_type_heuristic'] != 'datetime']
    dataset_metrics = (
        df_structure.groupby('dataset')[['stopword_density']]
        .median()
        .reset_index()
    )
    df_results_fair = df_results_fair.merge(
        dataset_metrics, left_on='data_name', right_on='dataset', how='inner',
    )

    # OpenML side: list_datasets gives row/column/missing counts at the
    # dataset level; the cached features CSV gives per-column cardinality
    # / string length / stopword density.
    print("Fetching OpenML metadata...")
    openml_datasets = openml.datasets.list_datasets(output_format="dataframe")
    df_openml_features = pd.read_csv(f"{path_configs['openml_features']}")
    openml_dataset_stats = df_openml_features.groupby('did').agg(
        avg_cardinality=('cardinality', 'mean'),
        avg_string_length=('avg_string_length', 'mean'),
    ).reset_index()

    strable_rows     = dataset_summary_wide['num_rows']
    strable_cols     = dataset_summary_wide['num_columns']
    strable_card     = dataset_summary_wide['avg_cardinality']
    strable_strlen   = dataset_summary_wide['avg_string_length_per_cell']
    strable_stopword = df_results_fair.groupby('data_name')['stopword_density'].median()
    strable_missing  = dataset_summary_wide['prop_missing_total'].dropna()

    openml_rows     = openml_datasets['NumberOfInstances']
    openml_cols     = openml_datasets['NumberOfFeatures']
    openml_card     = openml_dataset_stats['avg_cardinality']
    openml_strlen   = openml_dataset_stats['avg_string_length']
    openml_stopword = df_openml_features.groupby('did')['stopword_density'].median()
    openml_missing  = (
        openml_datasets['NumberOfMissingValues']
        / (openml_datasets['NumberOfFeatures'] * openml_datasets['NumberOfInstances'])
    ).dropna()

    return {
        'Rows':                 (strable_rows,     openml_rows),
        'Columns':              (strable_cols,     openml_cols),
        'Cardinality':          (strable_card,     openml_card),
        'String Length':        (strable_strlen,   openml_strlen),
        'Stopword Density':     (strable_stopword, openml_stopword),
        'Prop. Missing Values': (strable_missing,  openml_missing),
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

LAYOUT_LANDSCAPE = [
    (0, 0, 'Rows'),
    (0, 1, 'Columns'),
    (0, 2, 'Cardinality'),
    (1, 0, 'String Length'),
    (1, 1, 'Stopword Density'),
    (1, 2, 'Prop. Missing Values'),
]


def plot_metadata_distribution_landscape():
    panels = _build_data()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, col, title in LAYOUT_LANDSCAPE:
        s_raw, o_raw = panels[title]
        _draw_panel(axes[row, col], s_raw, o_raw, title)

    fig.supylabel('Probability', fontsize=FS_AXIS, x=0.12)
    fig.text(0.5, 0.02, 'log(Value + 1)', ha='center', fontsize=FS_AXIS)

    legend_elements = [
        Patch(facecolor=STRABLE_COLOR, alpha=STRABLE_ALPHA,
              edgecolor='grey', lw=0.5, label='STRABLE'),
        Patch(facecolor=OPENML_COLOR,  alpha=OPENML_ALPHA,
              edgecolor='grey', lw=0.5, label='OpenML'),
    ]
    # ``frameon=False`` ported from salts to match the paper's borderless legend.
    fig.legend(
        handles=legend_elements, ncol=2, fontsize=FS_LEGEND,
        framealpha=0.95, edgecolor='lightgrey', frameon=False,
        loc='upper center', bbox_to_anchor=(0.5, 1.03),
    )
    plt.tight_layout(rect=[0.06, 0.04, 1.0, 0.93])
    plt.subplots_adjust(wspace=0.01, hspace=0.3)

    save_figure(
        fig,
        "metadata_distribution_median_mode_6_indexes_openml_comparison_landscape",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_metadata_distribution_landscape()
