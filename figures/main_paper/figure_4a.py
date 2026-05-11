"""Figure 4(a) — Pareto-frontier plot: pipeline performance (avg score across 108 datasets)
vs total run time per 1K samples.
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.path_configs import path_configs
from figures._main import (
    Y_METRIC_LABELS,
    clean_method_name,
    dtype_map,
    encoder_map,
    get_encoder_color,
    get_learner_color_simple,
    get_learner_marker,
    get_pareto_front,
    learner_colors,
    learner_map,
    learner_shapes,
    load_dataset_summary,
    load_results,
    save_figure,
    selected_encoders,
)


COMPILED = path_configs['compiled_results']
DTYPE_PREFIX = 'num-str'
Y_METRIC = 'score'
X_METRIC = 'run_time_per_1k'   # paper figure uses run_time_per_1k

E2E_MAP = {
    'CatBoost':   'CatBoost',
    'ContextTab': 'ContextTab',
    'TabSTAR':    'TabSTAR',
    'TabPFN-2.5': 'TabPFN',
    'Mambular':   'Mambular',
}

# Pipelines drawn with a hatched marker (post-processing variants the paper
# wants visually distinguished from default-PCA versions).
TARGET_HATCHES = [
    "LM LLaMA-3.1-8B (StandScal + PCA (30))",
    "LM Qwen-3-8B (No PCA (30))",
    "LM LLaMA-3.1-8B (OHE|CT=30)",
    "LM Qwen-3-8B (OHE|CT=30)",
]


# ---------------------------------------------------------------------------
# 1. Data loading — duplicates with figure_3 / figure_E2 by design
# ---------------------------------------------------------------------------

def _attach_dataset_meta(df, dataset_summary):
    df = df.merge(dataset_summary, on='data_name', how='left')
    df['run_time_per_1k']      = df['run_time']      / df['num_rows'] * 1000
    df['inference_time_per_1k'] = df['inference_time'] / df['num_rows'] * 1000
    return df


def _load_qwen_nopca_30(dataset_summary):
    df = pd.read_csv(f"{COMPILED}/result_comparison_qwen_nopca_30.csv")
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].str.replace('-no_pca', '').replace(learner_map)
    df['encoder'] = df['encoder'] + ' (No PCA (30))'
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    return _attach_dataset_meta(df, dataset_summary)


def _load_pca30_standscal_llama_tabpfn(dataset_summary):
    """LLaMA-3.1-8B + TabPFN-2.5 with standard scaling before PCA."""
    df = pd.read_csv(f"{COMPILED}/result_comparison_standscal_pca_30.csv")
    df['score']  = df['r2'].fillna(df['roc_auc'])
    df['method'] = df['method'] + '_default'
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map) + ' (StandScal + PCA (30))'
    df['learner'] = meta[2].replace(learner_map)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    df = df[df['method'] == 'num-str_llm-llama-3.1-8b_tabpfn_default']
    return _attach_dataset_meta(df, dataset_summary)


def _load_rebuttal_csv(filename, dataset_summary, learner_filter_drop=None):
    df = pd.read_csv(f"{COMPILED}/{filename}")
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0]
    df['encoder'] = meta[1].replace(encoder_map)
    df['learner'] = meta[2].replace(learner_map)
    df['method']  = df['method'].str.replace(f'{DTYPE_PREFIX}_', '', regex=False)
    df['method']  = df['method'].apply(clean_method_name)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    if learner_filter_drop is not None:
        df = df[~df['method'].isin(learner_filter_drop)]
    return _attach_dataset_meta(df, dataset_summary)


def _load_mambular_only(dataset_summary):
    df = _load_rebuttal_csv("result_REBUTTALS_mambular.csv", dataset_summary)
    return df[df['encoder'] == 'Mambular']


def _load_tabicl(dataset_summary):
    """TabICLv2 with three post-processing variants tagged on the encoder."""
    df = pd.read_csv(f"{COMPILED}/result_comparison_tabicl_all.csv")
    df['score'] = df['r2'].fillna(df['roc_auc'])
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)

    df['encoder_polished'] = ''
    rules = [
        ('num-str_llm-qwen3-8b_tabicl',           'LM Qwen-3-8B',     ' (30-PCA)'),
        ('num-str_tabvec_tabicl',                 'Tf-Idf',            ''),
        ('num-str_llm-llama-3.1-8b_tabicl',       'LM LLaMA-3.1-8B',  ' (30-PCA)'),
        ('num-str_llm-qwen3-8b_tabicl_nopca',     'LM Qwen-3-8B',     ' (No PCA (30))'),
        ('num-str_llm-llama-3.1-8b_tabicl_standscal',
                                                   'LM LLaMA-3.1-8B', ' (StandScal + PCA (30))'),
    ]
    for method_str, encoder_str, suffix in rules:
        mask = (df['method'] == method_str) & (df['encoder'] == encoder_str)
        df.loc[mask, 'encoder_polished'] = df.loc[mask, 'encoder'] + suffix
    df['encoder'] = df['encoder_polished']

    df['learner'] = df['method'].str.split('_', expand=True, n=2)[2]
    df['learner'] = df['learner'].str.replace('_nopca|_standscal', '', regex=True)
    df['learner'] = df['learner'].replace(learner_map)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    return _attach_dataset_meta(df, dataset_summary)


# ---------------------------------------------------------------------------
# 2. Drawing helpers
# ---------------------------------------------------------------------------

def _clean_encoder_label(name):
    label = name
    if label.startswith('LM '):
        label = label[3:]
    return label.replace('LLaMA-Nemotron-Embed-1B-v2', 'Nemotron-1B')


def _add_optimal_arrow(ax):
    """Draw a small green right-pointing arrow with 'Optimal' label, anchored
    in normalized axes-fractional coordinates so it survives axis rescaling."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    def fx(f):
        log_min, log_max = np.log10(xmin), np.log10(xmax)
        return 10 ** (log_min + f * (log_max - log_min))

    def fy(f):
        return ymin + f * (ymax - ymin)

    x_tail, y_tail = fx(0.28), fy(0.83)
    x_tip,  y_tip  = fx(0.04), fy(0.93)

    ax.annotate(
        '',
        xy=(x_tip, y_tip), xytext=(x_tail, y_tail),
        xycoords='data', textcoords='data',
        arrowprops=dict(
            arrowstyle=mpatches.ArrowStyle(
                'simple', tail_width=10, head_width=15, head_length=5,
            ),
            color='#2ca02c',
            mutation_scale=1.0,
        ),
        zorder=5,
    )
    ax.text(
        fx(0.17), fy(0.88), 'Optimal',
        transform=ax.transData,
        fontsize=7, fontweight='bold', color='white',
        ha='center', va='center',
        rotation=-20, rotation_mode='anchor',
        zorder=6,
    )


def _draw_panel(ax, df_agg, factor, x_metric, learner_pal, encoder_pal,
                learner_markers, unique_learners, unique_encoders, pareto_df,
                y_bottom, pareto_base):
    ax.grid(True, which='major', linestyle='--', linewidth=0.5,
            color='gray', alpha=0.3, zorder=0)

    if factor == 'encoder':
        current_palette = encoder_pal
        hue_col = 'encoder'
    else:
        current_palette = learner_pal
        hue_col = 'learner'

    mask_tuned = df_agg['learner'].str.contains('tuned')
    df_default = df_agg[~mask_tuned]
    df_tuned   = df_agg[mask_tuned]

    # Default learners — solid markers.
    sns.lineplot(
        data=df_default, x=x_metric, y=Y_METRIC,
        hue=hue_col, style='learner',
        palette=current_palette, markers=learner_markers,
        dashes=False, estimator=None, lw=0, markersize=9,
        ax=ax, legend=False,
        **{'fillstyle': 'full', 'markeredgewidth': 1.0, 'markeredgecolor': 'black'},
    )
    # Tuned learners — half-filled markers.
    sns.lineplot(
        data=df_tuned, x=x_metric, y=Y_METRIC,
        hue=hue_col, style='learner',
        palette=current_palette, markers=learner_markers,
        dashes=False, estimator=None, lw=0, markersize=9,
        ax=ax, legend=False,
        **{'fillstyle': 'left', 'markerfacecoloralt': 'white',
           'markeredgecolor': 'black', 'markeredgewidth': 1.0},
    )

    # Hatched overlay for the post-processing-variant pipelines.
    if factor == 'encoder':
        with mpl.rc_context({'hatch.color': 'white', 'hatch.linewidth': 1.2}):
            for target_enc in TARGET_HATCHES:
                subset = df_agg[df_agg['encoder'] == target_enc]
                for _, row in subset.iterrows():
                    m = learner_markers.get(row['learner'], 'o')
                    ax.scatter(
                        [row[x_metric]], [row[Y_METRIC]],
                        s=85, marker=m,
                        facecolors='none', edgecolors='white',
                        linewidths=0, hatch='////', zorder=4,
                    )

    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('')
    ax.set_ylim(bottom=y_bottom)
    ax.set_ylabel(
        f'Avg {Y_METRIC_LABELS[Y_METRIC]} ($R^2$ & AUC)' if factor == 'encoder' else '',
        fontsize=10,
    )
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_position(('axes', 0))
    ax.spines['bottom'].set_position(('axes', 0))

    # Pareto step extending from the left edge to the right edge.
    x_right_edge = ax.get_xlim()[1]
    end_point = pd.DataFrame(
        {x_metric: [x_right_edge], Y_METRIC: [pareto_df[Y_METRIC].iloc[-1]]}
    )
    pareto_extended = pd.concat([pareto_base, end_point], ignore_index=True)
    ax.step(
        pareto_extended[x_metric], pareto_extended[Y_METRIC],
        where='post', linestyle='--', color='black', linewidth=1.2, zorder=0,
    )
    _add_optimal_arrow(ax)

    # Annotated labels for visually-prominent pipelines.
    if factor == 'encoder':
        tf_rows = pareto_df[pareto_df['encoder'] == 'Tf-Idf']
        if not tf_rows.empty:
            tf_row = tf_rows.sort_values(x_metric).iloc[0]
            for x_off, y_off in [(15, 30), (35, 40)]:
                ax.annotate(
                    'Tf-Idf',
                    xy=(tf_row[x_metric], tf_row[Y_METRIC]),
                    xytext=(x_off, y_off),
                    textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    color=encoder_pal.get('Tf-Idf', 'black'),
                    ha='right', va='bottom',
                )
    else:
        for label, dy in [('TabICLv2', (2, 2)), ('TabPFN-2.5', (10, 13))]:
            rows = pareto_df[pareto_df['learner'] == label]
            if rows.empty:
                continue
            row = rows.sort_values(x_metric).iloc[0]
            va = 'bottom' if label == 'TabICLv2' else 'top'
            ax.annotate(
                label,
                xy=(row[x_metric], row[Y_METRIC]),
                xytext=dy,
                textcoords='offset points',
                fontsize=9, fontweight='bold',
                color=learner_pal.get(label, 'black'),
                ha='right', va=va,
            )

    # Title above the panel.
    title = 'Encoder' if factor == 'encoder' else 'Learner'
    ax.set_title(
        title, fontsize=12, fontweight='bold',
        loc='left',
        y=1.72,
        x=-0.2 if factor == 'encoder' else -0.06,
    )
    ax.text(
        -0.2 if factor == 'encoder' else -0.06,
        1.68,
        '(shape = learner, color = encoder)' if factor == 'encoder'
        else '(shape = learner, color = learner)',
        transform=ax.transAxes,
        fontsize=10, style='italic', color='#333333',
    )

    # Legend.
    if factor == 'encoder':
        std_encoders = sorted([e for e in unique_encoders if e not in E2E_MAP])
        e2e_encoders = sorted([e for e in unique_encoders if e in E2E_MAP])
        sorted_encoder_list = std_encoders + e2e_encoders

        enc_handles = []
        for enc in sorted_encoder_list:
            display_label = _clean_encoder_label(enc).replace(' (', '\n(', 1)
            current_color = encoder_pal[enc]
            if enc in E2E_MAP:
                learner_key = E2E_MAP[enc]
                current_color = learner_colors[learner_key]
                current_marker = learner_shapes[learner_key]
                h = mlines.Line2D(
                    [], [], color=current_color,
                    marker=current_marker, linestyle='',
                    markersize=8, label=display_label,
                )
            elif enc in TARGET_HATCHES:
                h = mpatches.Patch(
                    facecolor=current_color, edgecolor='white',
                    hatch='////', label=display_label,
                )
            else:
                h = mpatches.Patch(color=current_color, label=display_label)
            enc_handles.append(h)

        ax.legend(
            handles=enc_handles,
            loc='lower center', bbox_to_anchor=(0.48, 1.03),
            ncol=2, fontsize=8, frameon=False,
            columnspacing=0.8, labelspacing=0.3,
            handletextpad=0.5, borderaxespad=0.0,
        )
    else:
        base_order = [
            'Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 'TabSTAR',
            'ContextTab', 'CatBoost', 'Mambular', 'RealMLP', 'TabM', 'TabICLv2',
        ]
        sorted_learners = []
        for base in base_order:
            if base in unique_learners:
                sorted_learners.append(base)
            if f'{base}-tuned' in unique_learners:
                sorted_learners.append(f'{base}-tuned')
        for l in unique_learners:
            if l not in sorted_learners:
                sorted_learners.append(l)

        lrn_handles = []
        for lrn in sorted_learners:
            is_tuned = 'tuned' in lrn
            kwargs = dict(
                color=learner_pal[lrn], marker=learner_markers[lrn],
                linestyle='', markersize=8, label=lrn,
                markeredgecolor='black', markeredgewidth=1.0,
            )
            if is_tuned:
                kwargs.update(fillstyle='left', markerfacecoloralt='white')
            else:
                kwargs['fillstyle'] = 'full'
            lrn_handles.append(mlines.Line2D([], [], **kwargs))

        ax.legend(
            handles=lrn_handles,
            loc='lower center', bbox_to_anchor=(0.5, 1.05),
            ncol=2, fontsize=8, frameon=False,
            columnspacing=0.5, labelspacing=0.6,
            handletextpad=0.3, borderaxespad=0.0,
        )


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def plot_pareto_frontier():
    results = load_results()
    dataset_summary = load_dataset_summary().drop(columns=['task'])

    extra_pipelines = pd.concat([
        _load_qwen_nopca_30(dataset_summary),
        _load_pca30_standscal_llama_tabpfn(dataset_summary),
        _load_rebuttal_csv(
            "result_REBUTTALS_tabM_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv",
            dataset_summary,
        ),
        _load_rebuttal_csv(
            "result_REBUTTALS_realMLP_tfidf_minLMv6_Qwen8_LLaMA8_30pca.csv",
            dataset_summary,
            learner_filter_drop=['LM E5-base-v2 - RealMLP', 'LM Jasper-0.6B - RealMLP'],
        ),
        _load_mambular_only(dataset_summary),
        _load_tabicl(dataset_summary),
    ], axis=0, ignore_index=True)

    # Add normalized score columns the rebuttal pipelines lack.
    extra_pipelines['score_norm'] = extra_pipelines.groupby('data_name')['score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    extra_pipelines['score_norm_clip'] = extra_pipelines['score_norm'].clip(upper=1.0)
    extra_pipelines['score_norm_max1'] = extra_pipelines.groupby('data_name')['score'].transform(
        lambda x: x / x.max()
    )
    extra_pipelines['score_centred'] = extra_pipelines.groupby('data_name')['score'].transform(
        lambda x: x - x.mean()
    )

    encoders_in_scope = list(selected_encoders) + ['LM LLaMA-Nemotron-Embed-1B-v2']
    agg_cols   = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
    group_cols = ['encoder_learner', 'encoder', 'learner']

    df_agg = (
        results[
            (results['method'].str.contains(f'{DTYPE_PREFIX}_'))
            & (results['encoder'].isin(encoders_in_scope))
            & (results['method'] != 'num-str_tabpfn_tabpfn_default')
        ]
        .groupby(group_cols)[agg_cols]
        .median()
        .reset_index()
    )
    df_agg = pd.concat(
        [df_agg, extra_pipelines.groupby(group_cols)[agg_cols].median().reset_index()],
        axis=0, ignore_index=True,
    )

    # Per-encoder / per-learner colour and marker maps.
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()
    learner_pal     = {L: get_learner_color_simple(L) for L in unique_learners}
    encoder_pal     = {E: get_encoder_color(E)        for E in unique_encoders}
    learner_markers = {L: get_learner_marker(L)        for L in unique_learners}
    # E2E encoders share the colour of their learner counterpart.
    for enc_name, learner_name in E2E_MAP.items():
        if enc_name in encoder_pal:
            encoder_pal[enc_name] = learner_colors[learner_name]

    # Pareto frontier on the chosen x metric.
    pareto_df = get_pareto_front(df_agg, X_METRIC, Y_METRIC, maximize_y=True)
    y_min, y_max = df_agg[Y_METRIC].min(), df_agg[Y_METRIC].max()
    y_padding = (y_max - y_min) * 0.05
    y_bottom  = y_min - y_padding
    start_point = pd.DataFrame(
        {X_METRIC: [pareto_df[X_METRIC].iloc[0]], Y_METRIC: [y_bottom]}
    )
    pareto_base = pd.concat([start_point, pareto_df], ignore_index=True)

    print("=" * 60)
    print(f"PARETO FRONTIER: {Y_METRIC} (max) vs {X_METRIC} (min)")
    print("=" * 60)
    for _, row in pareto_df[['encoder', 'learner', X_METRIC, Y_METRIC]].sort_values(X_METRIC).iterrows():
        print(f"{str(row['encoder'])[:25]:<25} | {str(row['learner'])[:20]:<20} | "
              f"{row[X_METRIC]:<10.4f} | {row[Y_METRIC]:<8.4f}")
    print("=" * 60 + "\n")

    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(5, 4), sharey=True)
    for col_idx, factor in enumerate(['encoder', 'learner']):
        _draw_panel(
            axes[col_idx], df_agg, factor, X_METRIC,
            learner_pal, encoder_pal, learner_markers,
            unique_learners, unique_encoders,
            pareto_df, y_bottom, pareto_base,
        )

    fig.text(
        0.5, 0.10, 'Total Run Time per 1K samples (s) (Log Scale)',
        ha='center', fontsize=10,
    )
    plt.subplots_adjust(bottom=0.20, top=0.72, wspace=0.15, left=0.02, right=0.98)

    save_figure(
        fig,
        "comparative_pareto_optimality_plot_1Ksample_scale_progr_transparency_False_score_run_time_per_1k",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_pareto_frontier()
