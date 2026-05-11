"""Figure E.8 — Per-encoder grouped bar chart comparing model performance under raw labels
vs. labels that have been transformed (log, log1p, cbrt, arcsinh,
signed-log) to deal with skewness."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from configs.path_configs import path_configs
from figures._main import (
    Y_METRIC_LABELS,
    dtype_map,
    encoder_map,
    get_learner_color_simple,
    learner_map,
    load_results,
    minmax_no_clip,
    save_figure,
)


SCORE_COL = 'score_norm'

METHODS_TO_RETRIEVE = [
    'num-str_tabvec_tabpfn_default',
    'num-str_tabvec_ridge_default',
    'num-str_tabvec_extrees_default',
    'num-str_tabvec_xgb_default',
    'num-str_llm-all-MiniLM-L6-v2_extrees_default',
    'num-str_llm-all-MiniLM-L6-v2_ridge_default',
    'num-str_llm-all-MiniLM-L6-v2_tabpfn_default',
    'num-str_llm-all-MiniLM-L6-v2_xgb_default',
    'num-str_llm-llama-3.1-8b_extrees_default',
    'num-str_llm-llama-3.1-8b_tabpfn_default',
    'num-str_llm-llama-3.1-8b_ridge_default',
    'num-str_llm-llama-3.1-8b_xgb_default',
    'num-str_llm-qwen3-8b_extrees_default',
    'num-str_llm-qwen3-8b_ridge_default',
    'num-str_llm-qwen3-8b_tabpfn_default',
    'num-str_llm-qwen3-8b_xgb_default',
    'num-str_contexttab_contexttab',
]

# 60 datasets where we re-ran with raw (untransformed) labels.
DATASETS_TO_RETRIEVE = [
    "aijob_ai-ml-ds-salaries", "california-houses",
    "college-creditcard-marketing", "college-deposit-product-marketing",
    "covid-clinical-trials", "global-dams-database",
    "industry-payments-entity", "industry-payments-project",
    "foreign-gift-and-contract", "antenna-structure-registration",
    "colleges-and-universities", "electric-retail-service-territories",
    "historic-perimeters-wildfires", "electric-generating-plants",
    "hospitals", "oil-natural-gas-platform", "local-law-enforcements",
    "pol-terminal", "prison-boundaries", "power-plants",
    "transmission-towers", "schools", "discretionary-grant", "grant",
    "museums", "awarded-grants", "insurance-company-complaints",
    "first-time-nadac-rates", "managed-care-enrollment",
    "financial-management", "mlr-summary-reports",
    "national-average-drug-acquisition-cost",
    "aca-federal-upper-limits-wide", "conflict-events_wide",
    "fts-funding", "fts-requirement-and-funding", "food-prices_wide",
    "mercari", "journal-ranking_wide", "summary-of-deposit_wide",
    "sf-building-permits", "wine-dataset",
    "china-overseas-finance-inventory", "local-government-renewable-action",
    "global-power-plant", "us-school-bus-fleet",
    "total-contributions-ibrd-ida-ifc", "commitments-in-trust-funds",
    "contributions-to-financial-intermediary-funds",
    "corporate-procurement-contract-awards",
    "financial-intermediary-funds-cash-transfers",
    "disbursements-in-trust-funds",
    "financial-intermediary-funds-commitments",
    "financial-intermediary-funds-funding-decisions",
    "contract-awards-investment-project-financing",
    "ibrd-statement-loans-guarantees",
    "ifc-advisory-services-projects",
    "ifc-investment-service-projects", "miga-issued-projects",
    "recipient-executed-grants-commitments-disbursements",
    "ida-statement-credits-grants-guarantees",
]


def _load_raw_labels():
    """Load the raw-label rebuttal CSV (pipelines re-trained without the
    default skewness transform) and re-derive score / dtype / encoder /
    learner so it merges cleanly with ``results``."""
    df = pd.read_csv(
        f"{path_configs['compiled_results']}/"
        "result_skewness_tfidf_miniLMv6_llama8_ridge_xgb_tabpfn_contexttab.csv"
    )
    df['score'] = df['r2']
    meta = df['method'].str.split('_', expand=True, n=2)
    df['dtype']   = meta[0].replace(dtype_map)
    df['encoder'] = meta[1].replace(encoder_map)
    learner = meta[2]
    df['learner'] = (learner + '_default').where(
        learner.isin(['ridge', 'xgb', 'extrees', 'tabpfn']),
        learner,
    ).replace(learner_map)
    df['encoder_learner'] = df['encoder'] + ' - ' + df['learner']
    df['Label_treatment'] = 'Raw Label'
    return df


def plot_raw_vs_transformed_label():
    results = load_results()
    df_raw = _load_raw_labels()

    df_trans = results[
        results['method'].isin(METHODS_TO_RETRIEVE)
        & results['data_name'].isin(DATASETS_TO_RETRIEVE)
    ].copy()
    df_trans['Label_treatment'] = 'Transformed Label'

    combined = pd.concat([df_raw, df_trans], axis=0, ignore_index=True)
    combined[SCORE_COL] = (
        combined.groupby(['data_name', 'dtype'])['score']
                .transform(minmax_no_clip)
    )

    df_plot = (
        combined
        .groupby(['Label_treatment', 'encoder_learner'])[SCORE_COL]
        .mean()
        .reset_index()
    )
    df_plot[['encoder', 'learner']] = (
        df_plot['encoder_learner'].str.rsplit(' - ', n=1, expand=True)
    )

    learner_order = ['Ridge', 'ExtraTrees', 'TabPFN-2.5', 'XGBoost',
                     'CatBoost', 'ContextTab', 'TabSTAR']
    all_learners = [l for l in learner_order if l in df_plot['learner'].unique()]
    n_learners = len(all_learners)

    encoder_order = (
        df_plot[df_plot['Label_treatment'] == 'Transformed Label']
        .groupby('encoder')[SCORE_COL].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Per-encoder Kendall-τ between the Raw and Transformed learner rankings,
    # then mean across encoders for the figure-level τ.
    per_encoder_taus = []
    for enc in encoder_order:
        raw_rows = df_plot[(df_plot['Label_treatment'] == 'Raw Label')
                           & (df_plot['encoder'] == enc)]
        trn_rows = df_plot[(df_plot['Label_treatment'] == 'Transformed Label')
                           & (df_plot['encoder'] == enc)]
        common = sorted(set(raw_rows['learner']) & set(trn_rows['learner']))
        if len(common) < 2:
            continue
        raw_scores = [raw_rows.loc[raw_rows['learner'] == l, SCORE_COL].iat[0]
                      for l in common]
        trn_scores = [trn_rows.loc[trn_rows['learner'] == l, SCORE_COL].iat[0]
                      for l in common]
        tau, _ = kendalltau(raw_scores, trn_scores)
        per_encoder_taus.append(tau)
    avg_tau = float(np.mean(per_encoder_taus))

    n_encoders = len(encoder_order)
    highlight_encoders = {'Tf-Idf', 'LM LLaMA-3.1-8B'}

    bar_h = 0.08
    learner_gap = 0.005
    encoder_gap = n_learners * bar_h + (n_learners - 1) * learner_gap + 0.06

    positions = {}
    for ei, enc in enumerate(encoder_order):
        group_center = ei * encoder_gap
        total_h = n_learners * bar_h + (n_learners - 1) * learner_gap
        offsets = np.linspace(-total_h / 2, total_h / 2, n_learners)
        for li, learner in enumerate(all_learners):
            positions[(enc, learner)] = group_center + offsets[li]

    # Single-learner encoders (e.g. ContextTab) get anchored just above
    # the Tf-Idf block instead of stretching the y-grid.
    tfidf_ys = [y for (e, _), y in positions.items() if e == 'Tf-Idf']
    top_tfidf = max(tfidf_ys) + bar_h / 2
    for enc in encoder_order:
        learners_present = df_plot[df_plot['encoder'] == enc]['learner'].unique()
        if len(learners_present) == 1:
            positions[(enc, learners_present[0])] = top_tfidf + bar_h * 0.8

    fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=False)
    fig.subplots_adjust(wspace=1.5)

    xlims = [(0.0, 1.0), (0.0, 1.0)]
    treatments = ['Raw Label', 'Transformed Label']
    subtitles = [
        '(no transformation)',
        '(log, log1p, cbrt,\narcsinh, and signed-log)',
    ]
    MIN_STUB = 0.006

    for ax, treatment, subtitle, xlim in zip(axes, treatments, subtitles, xlims):
        df_t = df_plot[df_plot['Label_treatment'] == treatment]

        for enc in encoder_order:
            if enc in highlight_encoders:
                enc_ys = [
                    y for (e, l), y in positions.items()
                    if e == enc
                    and not df_t[(df_t['encoder'] == e) & (df_t['learner'] == l)].empty
                ]
                if not enc_ys:
                    continue
                pad = bar_h * 0.8
                ax.axhspan(min(enc_ys) - bar_h / 2 - pad,
                           max(enc_ys) + bar_h / 2 + pad,
                           color='lightgrey', alpha=0.3, zorder=0)

        for (enc, learner), y_pos in positions.items():
            row = df_t[(df_t['encoder'] == enc) & (df_t['learner'] == learner)]
            if row.empty:
                continue
            score_val = row[SCORE_COL].iat[0]
            color = get_learner_color_simple(learner)
            if treatment == 'Raw Label' and score_val <= 0:
                # Stub bar so empty/negative cells stay visible.
                ax.barh(y_pos, MIN_STUB, left=xlim[0], height=bar_h,
                        color=color, edgecolor='gray', linewidth=0.5,
                        alpha=0.35, zorder=2)
            else:
                ax.barh(y_pos, score_val, height=bar_h,
                        color=color, edgecolor='white', linewidth=0.4,
                        alpha=0.95, zorder=2)

        encoder_centers = [
            np.mean([
                y for (e, l), y in positions.items()
                if e == enc
                and not df_plot[(df_plot['encoder'] == e)
                                & (df_plot['learner'] == l)].empty
            ])
            for enc in encoder_order
        ]
        ax.set_yticks(encoder_centers)
        ax.set_yticklabels(encoder_order, fontsize=14)
        ax.set_title(f"{treatment}\n{subtitle}", fontsize=14, pad=8)
        ax.set_xlim(*xlim)
        ax.axvline(xlim[0], color='black', linewidth=0.8)
        ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=14)
        for i in range(n_encoders - 1):
            ax.axhline((i + 0.5) * encoder_gap, color='lightgray',
                       linewidth=0.6, zorder=1)
        ax.set_ylim(-encoder_gap * 0.55, (n_encoders - 0.92) * encoder_gap)

        if treatment == 'Transformed Label':
            fig.text(
                0.9, 0.28,
                f"Kendall's $\\tau$ = {avg_tau:.2f}\n(vs. raw)",
                va='center', ha='left', fontsize=11, color='#333333',
                fontstyle='italic', transform=fig.transFigure,
            )

    legend_handles = [
        mpatches.Patch(facecolor=get_learner_color_simple(l), label=l)
        for l in all_learners
    ]
    fig.legend(
        handles=legend_handles, title='Learner',
        title_fontsize=13, fontsize=12,
        loc='center left', bbox_to_anchor=(0.9, 0.5),
        frameon=True, edgecolor='lightgray', ncol=1,
    )
    fig.supxlabel(f'Avg {Y_METRIC_LABELS[SCORE_COL]}  ($R^2$)',
                  fontsize=14, y=-0.02)

    save_figure(fig, f"raw_vs_transformed_label_rankings_v2_{SCORE_COL}")
    plt.close(fig)


if __name__ == "__main__":
    plot_raw_vs_transformed_label()
