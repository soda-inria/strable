"""Shared utilities for STRABLE figure scripts."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import hashlib
import os
import time
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from configs.path_configs import path_configs


# ---------------------------------------------------------------------------
# Constants — name maps and encoder lists
# ---------------------------------------------------------------------------

dtype_map = {
    'num-str': 'Num+Str',
    'num-only': 'Num',
    'str-only': 'Str',
}

encoder_map = {
    'tabvec': 'Tf-Idf',
    'tarenc': 'TargetEncoder',
    'catboost': 'CatBoost',
    'tabstar': 'TabSTAR',
    'contexttab': 'ContextTab',
    'mambular': 'Mambular',
    'tarte': 'Tarte',
    'llm-all-MiniLM-L6-v2': 'LM All-MiniLM-L6-v2',
    'llm-all-MiniLM-L12-v2': 'LM All-MiniLM-L12-v2',
    'llm-e5-base-v2': 'LM E5-base-v2',
    'llm-e5-large-v2': 'LM E5-large-v2',
    'llm-e5-small-v2': 'LM E5-small-v2',
    'llm-fasttext': 'LM FastText',
    'llm-roberta-base': 'LM RoBERTa-base',
    'llm-roberta-large': 'LM RoBERTa-large',
    'llm-llama-3.1-8b': 'LM LLaMA-3.1-8B',
    'llm-llama-3.2-1b': 'LM LLaMA-3.2-1B',
    'llm-llama-3.2-3b': 'LM LLaMA-3.2-3B',
    'llm-qwen3-8b': 'LM Qwen-3-8B',
    'llm-qwen3-4b': 'LM Qwen-3-4B',
    'llm-qwen3-0.6b': 'LM Qwen-3-0.6B',
    'llm-opt-0.1b': 'LM OPT-0.1B',
    'llm-opt-0.3b': 'LM OPT-0.3B',
    'llm-opt-1.3b': 'LM OPT-1.3B',
    'llm-opt-2.7b': 'LM OPT-2.7B',
    'llm-opt-6.7b': 'LM OPT-6.7B',
    'llm-modernbert-base': 'LM ModernBERT-base',
    'llm-modernbert-large': 'LM ModernBERT-large',
    'llm-all-mpnet-base-v2': 'LM All-MPNet-base-v2',
    'llm-f2llm-0.6b': 'LM F2LLM-0.6B',
    'llm-f2llm-1.7b': 'LM F2LLM-1.7B',
    'llm-f2llm-4b': 'LM F2LLM-4B',
    'llm-bge-large': 'LM BGE-large',
    'llm-bge-small': 'LM BGE-small',
    'llm-bge-base': 'LM BGE-base',
    'llm-gemma-0.3b': 'LM Gemma-0.3B',
    'llm-uae-large': 'LM UAE-large',
    'llm-deberta-v3-xsmall': 'LM DeBERTa-v3-xsmall',
    'llm-deberta-v3-small': 'LM DeBERTa-v3-small',
    'llm-deberta-v3-base': 'LM DeBERTa-v3-base',
    'llm-deberta-v3-large': 'LM DeBERTa-v3-large',
    'llm-kalm-embed': 'LM KALM-embed',
    'llm-t5-small': 'LM T5-small',
    'llm-jasper-token-comp-0.6b': 'LM Jasper-0.6B',
    'llm-sentence-t5-base': 'LM Sentence-T5-base',
    'llm-sentence-t5-large': 'LM Sentence-T5-large',
    'llm-sentence-t5-xl': 'LM Sentence-T5-xl',
    'llm-sentence-t5-xxl': 'LM Sentence-T5-XXL',
    'llm-llama-nemotron-embed-1b-v2': 'LM LLaMA-Nemotron-Embed-1B-v2',
}

learner_map = {
    'ridge_default': 'Ridge',
    'xgb_default': 'XGBoost',
    'extrees_default': 'ExtraTrees',
    'catboost_default': 'CatBoost',
    'xgb_tune': 'XGBoost-tuned',
    'tabpfn_default': 'TabPFN-2.5',
    'extrees_tune': 'ExtraTrees-tuned',
    'catboost_tune': 'CatBoost-tuned',
    'contexttab': 'ContextTab',
    'tabstar': 'TabSTAR',
    'tabicl': 'TabICLv2',
    'tabm': 'TabM',
    'realmlp': 'RealMLP',
    'mambular': 'Mambular',
}

Y_METRIC_LABELS = {
    'score': 'Score',
    'score_norm': 'Normalized Score',
    'score_norm_clip': 'Clipped Normalized Score',
    'score_norm_max1': 'Max-1 Normalized Score',
    'score_centred': 'Mean-Centred Score',
    'score_clip': 'Clipped Score',
}

score_list = ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1',
              'score_centred', 'score_clip']

baseline_encoders = ['Tf-Idf', 'TargetEncoder', 'Tarte']
e2e_encoders = ['CatBoost', 'ContextTab', 'TabSTAR', 'Mambular']
selected_LLMs = [
    'LM All-MiniLM-L6-v2',
    'LM FastText',
    'LM E5-small-v2',
    'LM LLaMA-3.1-8B',
    'LM Qwen-3-8B',
    'LM Jasper-0.6B',
]
top3_LLMs = [
    'LM All-MiniLM-L6-v2',
    'LM FastText',
    'LM E5-small-v2',
]
selected_encoders = baseline_encoders + e2e_encoders + selected_LLMs
selected_encoders_top3 = baseline_encoders + e2e_encoders + top3_LLMs


# Source → category mapping (used to build the per-application-field analyses)
category_to_sources = {
    'Commerce': [
        'European-Commission', 'webrobots.io', 'mercari.com', 'Yelp Open Dataset',
    ],
    'Economy': [
        'aijobs.net', 'kaggle', 'Consumer-Financial-Protection-Bureau',
        'Federal-Deposit-Insurance-Corporation', 'data.ct.gov',
        'lendingclub.com', 'worldbankfinancesone',
    ],
    'Education': [
        'commonlit.org', 'FSA', 'Institute of Museum and Library Services', 'SCIMAGO',
    ],
    'Energy': [
        'energydata.info', 'fueleconomy.gov', 'world-resource-institute',
    ],
    'Food': [
        'BeerAdvocate.com', 'flavorsofcacao.com', 'whiskyanalysis.com',
        'Michelin', 'theramenrater.com', 'majestic.co.uk',
    ],
    'Health': [
        'ClinicalTrials.gov', 'European-Medicines-Agency', 'fda', 'HRSA',
        'Medicaid', 'osha.gov',
    ],
    'Infrastructure': ['HIFLD', 'data.sfgov.org'],
    'Social': ['OHCA'],
}


# Dataset → publication / collection year (used by the year-bin analyses).
year_to_datasets = {
    1900: ['michelin-ratings'],
    1933: ['community-banking_wide', 'summary-of-deposit_wide'],
    1965: [
        'industry-payments-entity', 'industry-payments-project',
        'first-time-nadac-rates', 'child-adult-healthcare-quality',
        'managed-care-enrollment', 'financial-management', 'mlr-summary-reports',
        'national-average-drug-acquisition-cost', 'aca-federal-upper-limits-wide',
    ],
    1970: ['osha-accidents'],
    1979: ['rasff_window', 'rasnf_notification_list'],
    1980: ['wine-dataset'],
    1982: [
        'global-dams-database', 'external-clinician-dashboard',
        'workforce-demographics-wide', 'broadband-availability',
        'health-professional-shortage-areas',
        'medically-underserved-areas-populations', 'discretionary-grant', 'grant',
        'hypertension-control-wide', 'china-overseas-finance-inventory',
        'local-government-renewable-action', 'global-power-plant',
        'us-school-bus-fleet',
    ],
    1992: [
        'conflict-events_wide', 'fts-funding', 'fts-requirement-and-funding',
        'food-prices_wide',
    ],
    1995: ['medicines', 'orphan-designations', 'paediatric-investigation-plan'],
    1996: ['beer-ratings', 'museums', 'awarded-grants'],
    1999: ['vehicles'],
    2000: ['covid-clinical-trials'],
    2002: [
        'antenna-structure-registration', 'colleges-and-universities',
        'electric-retail-service-territories', 'historic-perimeters-wildfires',
        'electric-generating-plants', 'historical-earthquake-locations',
        'historical-volcanic-locations', 'hospitals', 'oil-natural-gas-platform',
        'mobile-home-parks', 'local-law-enforcements', 'pol-terminal',
        'prison-boundaries', 'transmission-lines', 'power-plants',
        'transmission-towers', 'schools', 'ramen-ratings',
    ],
    2007: [
        'chocolate-bar-ratings', 'lending-club-loan',
        'journal-ranking_wide', 'media-ranking_wide',
    ],
    2009: ['sf-building-permits'],
    2010: [
        'california-houses', 'cohort-default-rate', 'gainful-employment',
        'foreign-gift-and-contract', 'total-contributions-ibrd-ida-ifc',
        'commitments-in-trust-funds',
        'contributions-to-financial-intermediary-funds',
        'corporate-procurement-contract-awards',
        'financial-intermediary-funds-cash-transfers',
        'disbursements-in-trust-funds',
        'financial-intermediary-funds-commitments',
        'financial-intermediary-funds-funding-decisions',
        'contract-awards-investment-project-financing',
        'ibrd-statement-loans-guarantees',
        'ifc-advisory-services-projects', 'ifc-investment-service-projects',
        'miga-issued-projects',
        'recipient-executed-grants-commitments-disbursements',
        'ida-statement-credits-grants-guarantees',
    ],
    2011: [
        'college-creditcard-marketing', 'college-deposit-product-marketing',
        'prepaid-financial-product', 'terms-cc-plans',
        'financial-product-complaint',
    ],
    2013: ['clear-corpus', 'kickstarter-projects', 'mercari', 'yelp_business'],
    2014: [
        'device-classification', 'device-covid19serology', 'cosmetic-event',
        'drug-enforcement', 'drug-drugsfda', 'device-pma', 'drug-shortages',
        'drug-ndc', 'food-enforcement', 'tobacco-problem', 'food-event',
        'animalandveterinary-event', 'tax-incentives',
        'insurance-company-complaints',
    ],
    2015: ['meta-critic_whisky'],
    2018: ['aijob_ai-ml-ds-salaries'],
}


# ---------------------------------------------------------------------------
# Score normalization helpers
# ---------------------------------------------------------------------------

def robust_minmax_clip(x):
    """Clip negatives to 0 then min-max normalize. Returns 1.0 on degenerate input."""
    x_clipped = x.clip(lower=0.0)
    if x_clipped.max() == x_clipped.min():
        return 1.0
    return (x_clipped - x_clipped.min()) / (x_clipped.max() - x_clipped.min())


def score_clip(x):
    """Clip negatives to 0 — used to drop catastrophic R² values."""
    return x.clip(lower=0.0)


def minmax_no_clip(x, max_val=None):
    """Min-max normalize without clipping negatives.
    If ``max_val`` is provided it overrides the series maximum.
    """
    if max_val is None:
        return (x - x.min()) / (x.max() - x.min())
    return (x - x.min()) / (max_val - x.min())


def mean_centred(x):
    """Subtract the per-group mean from each value."""
    return x - x.mean()


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def map_source_to_category(source):
    """Source → application-field bucket. ``Other`` if not in the table."""
    for category, src_list in category_to_sources.items():
        if source in src_list:
            return category
    return 'Other'


def map_dataset_to_year(dataset_name):
    """Dataset name → publication / collection year (or ``None``)."""
    for year, datasets in year_to_datasets.items():
        if dataset_name in datasets:
            return year
    return None


def map_year_to_macro_category(year):
    """Year → era bucket (Pre-2000 / 2000-2009 / 2010-Present)."""
    if year is None:
        return None
    if year < 2000:
        return 'Pre-2000'
    if year < 2010:
        return '2000-2009'
    return '2010-Present'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_summary():
    """Per-dataset metadata parquet (one row per dataset)."""
    return pd.read_parquet(f"{path_configs['dataset_summary_wide']}")


def load_results(min_results_per_method=303, drop_realtabpfn=True):
    """Load and prepare ``result_comparison.csv`` exactly as the original
    monolith did. Returns one row per (data_name, method, fold) plus per-row
    normalized scores and dataset metadata.

    Parameters
    ----------
    min_results_per_method : int, default 303
        Drop methods that have fewer than this many rows (i.e. that didn't
        complete on every dataset × fold combination). 303 = 108 datasets ×
        ~3 folds with some missingness allowed.
    drop_realtabpfn : bool, default True
        Drop ``RealTabPFN-2.5`` rows. The paper reports ``TabPFN-2.5`` only.
    """
    results = pd.read_csv(f"{path_configs['compiled_results']}/result_comparison.csv")
    results['score'] = results['r2'].fillna(results['roc_auc'])

    meta = results['method'].str.split('_', expand=True, n=2)
    results['dtype'] = meta[0]
    results['encoder'] = meta[1]
    results['learner'] = meta[2]

    results['dtype'] = results['dtype'].replace(dtype_map)
    results['encoder'] = results['encoder'].replace(encoder_map)
    results['learner'] = results['learner'].replace(learner_map)

    results['method_polished'] = (
        results['encoder'] + ' - ' + results['learner']
        + '\n(' + results['dtype'] + ')'
    )
    results['encoder_learner'] = results['encoder'] + ' - ' + results['learner']

    # Normalised score variants (per (data_name, dtype) group).
    grouper = results.groupby(['data_name', 'dtype'])['score']
    results['score_norm_clip'] = grouper.transform(robust_minmax_clip)
    results['score_clip']      = grouper.transform(score_clip)
    results['score_norm']      = grouper.transform(minmax_no_clip)
    results['score_norm_max1'] = grouper.transform(minmax_no_clip, max_val=1.0)
    results['score_centred']   = grouper.transform(mean_centred)

    # Join dataset metadata (drops the duplicate 'task' column).
    dataset_summary = load_dataset_summary().drop(columns=['task'])
    results = results.merge(dataset_summary, on='data_name', how='left')

    results['run_time_per_1k']      = results['run_time']      / results['num_rows'] * 1000
    results['inference_time_per_1k'] = results['inference_time'] / results['num_rows'] * 1000

    results['category']           = results['source'].apply(map_source_to_category)
    results['year']               = results['data_name'].apply(map_dataset_to_year)
    results['year_macro_category'] = results['year'].apply(map_year_to_macro_category)

    results['train_size'] = np.ceil(results['num_rows'] * (2 / 3)).astype(int)
    results['test_size']  = results['num_rows'] - results['train_size']

    if drop_realtabpfn:
        results = results[results['learner'] != 'RealTabPFN-2.5']

    if min_results_per_method is not None:
        counts = results['method'].value_counts()
        keep   = counts[counts >= min_results_per_method].index.tolist()
        results = results[results['method'].isin(keep)].reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Generic computation helpers
# ---------------------------------------------------------------------------

def clean_method_name(method_str):
    """Map a raw ``encoder_learner`` string to the human-readable form."""
    if method_str == 'catboost_catboost':
        return 'CatBoost - CatBoost'
    if '_' in method_str:
        enc_part, lrn_part = method_str.split('_', 1)
        encoder = encoder_map.get(enc_part, enc_part)
        learner = learner_map.get(lrn_part, lrn_part)
        return f"{encoder} - {learner}"
    return encoder_map.get(method_str, learner_map.get(method_str, method_str))


def get_pareto_front(df, x_col, y_col, maximize_y=True):
    """Return rows on the Pareto front of (x_col, y_col)."""
    sorted_df = df.sort_values(x_col, ascending=True)
    pareto_points = []
    if maximize_y:
        current_best_y = -float('inf')
        for _, row in sorted_df.iterrows():
            if row[y_col] > current_best_y:
                pareto_points.append(row)
                current_best_y = row[y_col]
    else:
        current_best_y = float('inf')
        for _, row in sorted_df.iterrows():
            if row[y_col] < current_best_y:
                pareto_points.append(row)
                current_best_y = row[y_col]
    return pd.DataFrame(pareto_points)


def calculate_rankings(data_frame):
    """Average rank of each algorithm across rows. Lower is better."""
    ranks_per_dataset = data_frame.rank(axis=1, ascending=False, method='min')
    return ranks_per_dataset.mean(axis=0)


def median_iqr(x):
    """``Median [Q1, Q3]`` formatted string."""
    med = x.median()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return f"{med:.0f} [{q1:.0f}, {q3:.0f}]"


def bin_feature_33_66(df, col):
    """Bin a column into Low / Med / High by 33rd and 66th percentiles."""
    bins = [0, df[col].quantile(0.33), df[col].quantile(0.66), float('inf')]
    return pd.cut(df[col], bins=bins, labels=['Low', 'Med', 'High'])


def bin_feature_median(df, col):
    """Bin a column into Low / High by the median."""
    bins = [0, df[col].quantile(0.5), float('inf')]
    return pd.cut(df[col], bins=bins, labels=['Low', 'High'])


# ---------------------------------------------------------------------------
# Palette / style
# ---------------------------------------------------------------------------

learner_colors = {
    'XGBoost':    '#D55E00',  # Vermilion
    'TabSTAR':    '#0072B2',  # Blue
    'Ridge':      '#009E73',  # Bluish Green
    'ExtraTrees': '#E69F00',  # Orange/Yellow
    'TabPFN':     '#CC79A7',  # Reddish Purple
    'ContextTab': '#56B4E9',  # Sky Blue
    'CatBoost':   '#F0E442',  # Yellow
    'RealMLP':    '#FF00FF',  # Magenta
    'TabM':       '#0E9594',  # Cyan/teal
    'TabICLv2':   '#A0522D',  # Sienna
    'Mambular':   '#7B68EE',  # Medium Slate Blue
}

learner_shapes = {
    'XGBoost':    's', 'CatBoost': 'D', 'ExtraTrees': '^',
    'Ridge':      'o', 'TabPFN':   'h', 'ContextTab': 'X',
    'TabSTAR':    'p', 'RealMLP':  '*', 'TabM':       'v',
    'TabICLv2':   'P', 'Mambular': 'H',
}

llm_base_colors = {
    'llama':    'tab:purple',
    'opt':      'tab:brown',
    'gemma':    'tab:gray',
    'fasttext': 'tab:green',
    'bge':      'tab:olive',
    'e5':       'tab:pink',
    'qwen':     'crimson',     # tab:red clashes with XGBoost
    'bert':     'navy',        # tab:blue clashes with TabSTAR
    'deberta':  'teal',        # tab:cyan clashes with ContextTab
    'roberta':  '#483D8B',     # DarkSlateBlue
    'mpnet':    '#556B2F',     # DarkOliveGreen
    'mini':     '#FF1493',     # DeepPink
    'glove':    '#4682B4',     # SteelBlue
    'jasper':   '#32CD32',     # LimeGreen
    'f2llm':    '#000000',     # Black
    'fallback': '#333333',
}


def get_hash_shade(base_color, model_name):
    """Deterministically darken/lighten ``base_color`` by a hash of ``model_name``."""
    rgb = mcolors.to_rgb(base_color)
    h = int(hashlib.sha256(model_name.encode('utf-8')).hexdigest(), 16)
    mod = ((h % 100) / 100.0) * 0.3 - 0.15
    new_rgb = [max(0, min(1, c + mod)) for c in rgb]
    return mcolors.to_hex(new_rgb)


def get_encoder_color(encoder_name):
    """Resolve an encoder display name to a palette color.
    Order matters: learners take precedence over LLM family heuristics.
    """
    name = str(encoder_name).lower()

    if 'catboost' in name:   return learner_colors['CatBoost']
    if 'contexttab' in name: return learner_colors['ContextTab']
    if 'tabstar' in name:    return learner_colors['TabSTAR']
    if 'tarte' in name:      return learner_colors.get('Tarte', '#7f7f7f')
    if 'tabpfn' in name:     return learner_colors['TabPFN']
    if 'xgb' in name:        return learner_colors['XGBoost']
    if 'mambular' in name:   return learner_colors['Mambular']

    if any(x in name for x in ['string', 'target', 'tabvec', 'onehot', 'tfidf']):
        return '#7f7f7f'

    if 'llama' in name:    return get_hash_shade(llm_base_colors['llama'], name)
    if 'qwen' in name:     return get_hash_shade(llm_base_colors['qwen'], name)
    if 'opt' in name:      return get_hash_shade(llm_base_colors['opt'], name)
    if 'gemma' in name:    return get_hash_shade(llm_base_colors['gemma'], name)
    if 'f2llm' in name:    return get_hash_shade(llm_base_colors['f2llm'], name)
    if 'roberta' in name:  return get_hash_shade(llm_base_colors['roberta'], name)
    if 'deberta' in name:  return get_hash_shade(llm_base_colors['deberta'], name)
    if 'mpnet' in name:    return get_hash_shade(llm_base_colors['mpnet'], name)
    if 'bert' in name:     return get_hash_shade(llm_base_colors['bert'], name)
    if 'mini' in name:     return get_hash_shade(llm_base_colors['mini'], name)
    if 'e5' in name:       return get_hash_shade(llm_base_colors['e5'], name)
    if 'bge' in name:      return get_hash_shade(llm_base_colors['bge'], name)
    if 'fasttext' in name: return get_hash_shade(llm_base_colors['fasttext'], name)
    if 'glove' in name:    return get_hash_shade(llm_base_colors['glove'], name)
    if 'jasper' in name:   return get_hash_shade(llm_base_colors['jasper'], name)

    return llm_base_colors['fallback']


def get_tuning_style(learner_key):
    """``fillstyle``/``markeredgewidth`` kwargs for tuned vs default markers."""
    if 'tuned' in learner_key:
        return {'fillstyle': 'none', 'markeredgewidth': 2}
    return {'fillstyle': 'full'}


def get_learner_color_simple(learner_name):
    """Substring match a learner name against ``learner_colors``."""
    for family, color in learner_colors.items():
        if family in learner_name:
            return color
    return '#333333'


def get_learner_hatch(learner_name):
    """Diagonal hatch for tuned learners, no hatch for default."""
    return '///' if 'tuned' in learner_name else ''


def get_learner_marker(learner_name):
    """Marker shape for a learner. Falls back to ``'o'`` for unknown names."""
    if learner_name in learner_shapes:
        return learner_shapes[learner_name]
    for family, marker in learner_shapes.items():
        if family in learner_name:
            return marker
    return 'o'


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def today_folder():
    """``YYYY-MM-DD`` subfolder name for today's outputs."""
    return time.strftime("%Y-%m-%d")


def pics_dir():
    """``results_pics/<today>/`` (created on first call)."""
    p = Path(path_configs['results_pics']) / today_folder()
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir():
    """``results_tables/<today>/`` (created on first call)."""
    p = Path(path_configs['results_tables']) / today_folder()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, name, ext='pdf'):
    """Save ``fig`` as ``<name>_<today>.<ext>`` under ``pics_dir()``."""
    out = pics_dir() / f"{name}_{today_folder()}.{ext}"
    fig.savefig(out, bbox_inches='tight')
    print(f"✅ Saved {out}")
    return out


def save_latex(latex_str, name):
    """Write ``latex_str`` to ``<name>_<today>.tex`` under ``tables_dir()``."""
    out = tables_dir() / f"{name}_{today_folder()}.tex"
    with open(out, 'w') as f:
        f.write(latex_str)
    print(f"✅ Saved {out}")
    return out
