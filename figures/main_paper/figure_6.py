"""Figure 6 — Top-10 pipelines per leading string type."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.path_configs import path_configs
from figures._main import (
    get_encoder_color,
    get_learner_hatch,
    load_results,
    save_figure,
    selected_encoders,
)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
TOP_N = 10
FONT_TITLE  = 26
FONT_TICK   = 22
FONT_LABEL  = 22
FONT_LEGEND = 20

TYPE_LABELS = {
    'categorical':     'Categorical',
    'name':            'Names',
    'free_text':       'Free Text',
    'structured_code': 'Structured Code',
    'identifier':      'Identifiers',
}

# Per-stratum x-axis range — chosen for the paper's layout so each subpanel
# zooms onto the relevant score band.
XLIM_PER_TYPE = {
    'categorical':     (0.7, 0.75),
    'name':            (0.66, 0.7),
    'free_text':       (0.85, 0.88),
    'structured_code': (0.84, 0.88),
}


def shorten_encoder_name(enc):
    """Replace the long Nemotron name with a compact label."""
    if 'Nemotron' in enc and '1B' in enc:
        return 'LM Nemotron-1B'
    return enc


def format_pipeline_label(pipeline):
    """Split ``Encoder - Learner`` into two lines.
    If encoder == learner (e2e models), show the name only once.
    """
    parts = pipeline.split(' - ')
    if len(parts) >= 2:
        enc     = shorten_encoder_name(parts[0].strip())
        learner = ' - '.join(parts[1:]).strip()
        if enc.lower() == learner.lower():
            return enc
        return f"{enc}\n{learner}"
    return shorten_encoder_name(pipeline)


def _build_dominant_type_per_dataset():
    """Per-dataset dominant column type from the column taxonomy CSV.
    Returns a DataFrame indexed by ``dataset`` with a ``dominant_type`` column.
    """
    df_structure = pd.read_csv(
        f"{path_configs['base_path']}/df_structure_VSE_STRABLE_CARTE_TTB.csv"
    )
    df_structure = df_structure[df_structure['col_type_heuristic'] != 'datetime']

    type_counts = (
        df_structure
        .groupby(['dataset', 'col_type_heuristic'])
        .size()
        .unstack(fill_value=0)
    )
    type_counts['total'] = type_counts.sum(axis=1)

    col_types = ['categorical', 'name', 'free_text', 'structured_code', 'identifier']
    for ct in col_types:
        if ct not in type_counts.columns:
            type_counts[ct] = 0
        type_counts[f'{ct}_pct'] = type_counts[ct] / type_counts['total']

    pct_cols = [f'{ct}_pct' for ct in col_types]
    type_counts['dominant_type'] = (
        type_counts[pct_cols].idxmax(axis=1).str.replace('_pct', '')
    )
    return type_counts


def plot_top10_by_leading_string_type():
    results = load_results()

    df_results = results[
        (results['dtype'] == 'Num+Str')
        & (results['encoder'].isin(selected_encoders))
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()

    type_counts = _build_dominant_type_per_dataset()
    df_results = df_results.merge(
        type_counts[['dominant_type']].reset_index(),
        left_on='data_name', right_on='dataset', how='inner',
    )

    valid_types = df_results['dominant_type'].value_counts().index.tolist()
    n_types = len(valid_types)

    score = 'score'
    mpl.rcParams['hatch.linewidth'] = 2.5

    fig, axes = plt.subplots(1, n_types, figsize=(28, 10), sharey=False)
    if n_types == 1:
        axes = [axes]

    encoders_in_figure = set()

    for i, dom_type in enumerate(valid_types):
        ax = axes[i]
        df_group = df_results[df_results['dominant_type'] == dom_type]
        n_ds = df_group['data_name'].nunique()

        if df_group.empty:
            ax.set_title(TYPE_LABELS.get(dom_type, dom_type),
                         fontsize=FONT_TITLE, fontweight='bold')
            ax.axis('off')
            continue

        pipeline_means = (
            df_group.groupby('encoder_learner')[score]
            .mean()
            .sort_values(ascending=False)
            .head(TOP_N)
        )
        top_pipelines = pipeline_means.index.tolist()
        df_top = df_group[df_group['encoder_learner'].isin(top_pipelines)]

        bar_colors, bar_hatches = [], []
        for pipeline in top_pipelines:
            parts   = pipeline.split(' - ')
            enc     = parts[0].strip() if len(parts) > 1 else pipeline
            learner = parts[-1].strip() if len(parts) > 1 else ''
            bar_colors.append(get_encoder_color(enc))
            bar_hatches.append(get_learner_hatch(learner))
            encoders_in_figure.add(enc)

        means = df_top.groupby('encoder_learner')[score].mean().reindex(top_pipelines)
        y_positions = range(TOP_N - 1, -1, -1)

        for y_pos, pipeline, color, hatch in zip(
                y_positions, top_pipelines, bar_colors, bar_hatches):
            ax.barh(
                y_pos, means[pipeline],
                color=color, edgecolor='black', linewidth=0.9, height=0.75,
            )
            if hatch:
                ax.barh(
                    y_pos, means[pipeline],
                    color='none', edgecolor='white',
                    hatch=hatch, linewidth=0, height=0.75,
                )

        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(
            [format_pipeline_label(p) for p in top_pipelines],
            fontsize=FONT_TICK, linespacing=1.2,
        )
        label = TYPE_LABELS.get(dom_type, dom_type)
        ax.set_title(f"{label} ({n_ds} datasets)",
                     fontsize=FONT_TITLE, fontweight='bold', pad=14)
        ax.set_xlabel('Avg Score ($R^2$ & AUC)', fontsize=FONT_LABEL)

        xmin, xmax = XLIM_PER_TYPE.get(dom_type, (0.55, 0.82))
        ax.set_xlim(xmin, xmax)
        # Pinning the y-limit (ported from salts) prevents bar squashing when a
        # stratum has fewer than TOP_N pipelines.
        ax.set_ylim(-0.5, TOP_N - 0.5)
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=FONT_TICK)
        sns.despine(ax=ax)

    plt.subplots_adjust(wspace=0.55)

    legend_handles = [
        mpatches.Patch(
            facecolor=get_encoder_color(enc),
            edgecolor='black', linewidth=0.8,
            label=shorten_encoder_name(enc),
        )
        for enc in sorted(encoders_in_figure)
    ]
    legend_handles.append(mpatches.Patch(
        facecolor='#cccccc', edgecolor='white',
        hatch='///', linewidth=0, label='Tuned learner',
    ))
    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),
        ncol=1,
        fontsize=FONT_LEGEND,
        frameon=True,
        title='Encoder',
        title_fontsize=FONT_LEGEND,
    )

    save_figure(
        fig,
        "encoder_learner_num+str_top10_ranking_by_leading_textfeature_type_selectedLLMs",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_top10_by_leading_string_type()
