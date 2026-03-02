import random
import time
from networkx import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from scipy import stats
from scipy.stats import spearmanr, kendalltau
from typing import Union, List, Dict, Set
from pandas import DataFrame, Series
from scipy.optimize import curve_fit
from scipy.stats import rankdata
pd.set_option('display.max_columns', None)
from src.utils_visualization import critical_difference_diagram
import matplotlib.ticker as ticker
import math
from scipy.optimize import fsolve
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.colors as mc
import matplotlib.colors as mcolors
import hashlib
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import RANSACRegressor
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from configs.path_configs import path_configs

results = pd.read_csv(path_configs["base_path"] + "/results/compiled_results/result_comparison.csv")

results['score'] = results['r2'].fillna(results['roc_auc'])

meta = results['method'].str.split('_', expand=True, n=2)

results['dtype'] = meta[0]
results['encoder'] = meta[1]
results['learner'] = meta[2]

########################SETUP########################

# 2. DEFINE YOUR MAPPINGS
dtype_map = {
    'num-str': 'Num+Str',
    'num-only': 'Num',
    'str-only': 'Str'
}
# check encoder list for updates
encoder_map = {
    'tabvec': 'Tf-Idf',
    'tarenc': 'TargetEncoder',
    'catboost': 'CatBoost',
    'tabpfn': 'TabPFN-2.5',
    'tabstar': 'TabSTAR',
    'contexttab': 'ContextTab',
    'tarte': 'Tarte',
    'tarte-ft': 'Tarte-FT',
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
# check learners list for updates
learner_map = {
    'ridge_default': 'Ridge', 
    'xgb_default': 'XGBoost', 
    'extrees_default': 'ExtraTrees',
    'catboost_default': 'CatBoost',
    'xgb_tune': 'XGBoost-tuned',
    'tabpfn_default': 'TabPFN-2.5',
    'extrees_tune': 'ExtraTrees-tuned',
    'catboost_tune': 'CatBoost-tuned',
    'realtabpfn_default': 'RealTabPFN-2.5',
    'contexttab': 'ContextTab',
    'tabstar': 'TabSTAR',
    'tarte_default': 'Tarte',
}

# 3. APPLY MAPPINGS
results['dtype'] = results['dtype'].replace(dtype_map)
results['encoder'] = results['encoder'].replace(encoder_map)
results['learner'] = results['learner'].replace(learner_map)

results['method_polished'] = results['encoder'] + ' - ' + results['learner'] + '\n(' + results['dtype'] + ')'

results['encoder_learner'] = results['encoder'] + ' - ' + results['learner'] 

'''
Normalised scores
'''
def robust_minmax_clip(x):
    x_clipped = x.clip(lower=0.0)
    if x_clipped.max() == x_clipped.min():
        return 1.0
    return (x_clipped - x_clipped.min()) / (x_clipped.max() - x_clipped.min())

# Apply this function to each group
results['score_norm_clip'] = results.groupby(['data_name', 'dtype'])['score'].transform(robust_minmax_clip)

def score_clip(x):
    x_clipped = x.clip(lower=0.0)
    return x_clipped

# Apply this function to each group
results['score_clip'] = results.groupby(['data_name', 'dtype'])['score'].transform(score_clip)


# create a minmax normalization function that does not clip, and has an option to define the max. if it's not defined, use the max of the series
def minmax_no_clip(x, max_val: float = None):
    """
    Min-max normalize WITHOUT clipping negatives.
    If max_val is provided it will be used as the series maximum; otherwise the series max is used.
    Returns a pandas Series with the same index as the input.
    """
    if max_val is None:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - x.min()) / (max_val - x.min())
    
results['score_norm'] = results.groupby(['data_name', 'dtype'])['score'].transform(minmax_no_clip)

results['score_norm_max1'] = results.groupby(['data_name', 'dtype'])['score'].transform(minmax_no_clip, max_val=1.0)

def mean_centred(x):
    return x - x.mean()

results['score_centred'] = results.groupby(['data_name', 'dtype'])['score'].transform(mean_centred)

'''
define score list
'''

score_list = ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred', 'score_clip']

Y_METRIC_LABELS = {
        'score': 'Score',
        'score_norm': 'Normalized Score',
        'score_norm_clip': 'Clipped Normalized Score',
        'score_norm_max1': 'Max-1 Normalized Score',
        'score_centred': 'Mean-Centred Score',
        'score_clip': 'Clipped Score'
    }

'''
merge datasets_summary for metadata
'''

dataset_summary_wide = pd.read_parquet(path_configs["base_path"] + "/dataset_summary_wide.parquet")

dataset_summary_wide.drop(columns=['task'], inplace=True)

results = results.merge(dataset_summary_wide, on='data_name', how='left')

results['run_time_per_1k'] = results['run_time'] / results['num_rows'] * 1000

results['inference_time_per_1k'] = results['inference_time'] / results['num_rows'] * 1000

'''
Add macro categories for data sources
'''
category_to_sources = {
    'Commerce': [
        'European-Commission', 
        'webrobots.io', 
        'mercari.com', 
        'Yelp Open Dataset'
    ],
    'Economy': [
        'aijobs.net', 
        'kaggle', 
        'Consumer-Financial-Protection-Bureau', 
        'Federal-Deposit-Insurance-Corporation', 
        'data.ct.gov', 
        'lendingclub.com', 
        'worldbankfinancesone'
    ],
    'Education': [
        'commonlit.org', 
        'FSA', 
        'Institute of Museum and Library Services', 
        'SCIMAGO'
    ],
    'Energy': [
        'energydata.info', 
        'fueleconomy.gov', 
        'world-resource-institute'
    ],
    'Food': [
        'BeerAdvocate.com', 
        'flavorsofcacao.com', 
        'whiskyanalysis.com', 
        'Michelin', 
        'theramenrater.com', 
        'majestic.co.uk'
    ],
    'Health': [
        'ClinicalTrials.gov', 
        'European-Medicines-Agency', 
        'fda', 
        'HRSA', 
        'Medicaid', 
        'osha.gov'
    ],
    'Infrastructure': [
        'HIFLD', 
        'data.sfgov.org'
    ],
    'Social': [
        'OHCA'
    ]
}

def map_source_to_category(source):
    for category, src_list in category_to_sources.items():
        if source in src_list:
            return category
    return 'Other'

# use the dictionary to create a new field in results called 'category'
results['category'] = results['source'].apply(map_source_to_category)

'''
Add dataset/source year or publication year
'''

year_to_datasets = {
    1900: ['michelin-ratings'],
    1933: ['community-banking_wide', 'summary-of-deposit_wide'],
    1965: [
        'industry-payments-entity', 'industry-payments-project', 'first-time-nadac-rates',
        'child-adult-healthcare-quality', 'managed-care-enrollment', 'financial-management',
        'mlr-summary-reports', 'national-average-drug-acquisition-cost', 'aca-federal-upper-limits-wide'
    ],
    1970: ['osha-accidents'],
    1979: ['rasff_window', 'rasnf_notification_list'],
    1980: ['wine-dataset'],
    1982: [
        'global-dams-database', 'external-clinician-dashboard', 'workforce-demographics-wide',
        'broadband-availability', 'health-professional-shortage-areas',
        'medically-underserved-areas-populations', 'discretionary-grant', 'grant',
        'hypertension-control-wide', 'china-overseas-finance-inventory',
        'local-government-renewable-action', 'global-power-plant', 'us-school-bus-fleet'
    ],
    1992: [
        'conflict-events_wide', 'fts-funding', 'fts-requirement-and-funding', 'food-prices_wide'
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
        'transmission-towers', 'schools', 'ramen-ratings'
    ],
    2007: [
        'chocolate-bar-ratings', 'lending-club-loan', 'journal-ranking_wide', 'media-ranking_wide'
    ],
    2009: ['sf-building-permits'],
    2010: [
        'california-houses', 'cohort-default-rate', 'gainful-employment',
        'foreign-gift-and-contract', 'total-contributions-ibrd-ida-ifc',
        'commitments-in-trust-funds', 'contributions-to-financial-intermediary-funds',
        'corporate-procurement-contract-awards', 'financial-intermediary-funds-cash-transfers',
        'disbursements-in-trust-funds', 'financial-intermediary-funds-commitments',
        'financial-intermediary-funds-funding-decisions',
        'contract-awards-investment-project-financing', 'ibrd-statement-loans-guarantees',
        'ifc-advisory-services-projects', 'ifc-investment-service-projects',
        'miga-issued-projects', 'recipient-executed-grants-commitments-disbursements',
        'ida-statement-credits-grants-guarantees'
    ],
    2011: [
        'college-creditcard-marketing', 'college-deposit-product-marketing',
        'prepaid-financial-product', 'terms-cc-plans', 'financial-product-complaint'
    ],
    2013: ['clear-corpus', 'kickstarter-projects', 'mercari', 'yelp_business'],
    2014: [
        'device-classification', 'device-covid19serology', 'cosmetic-event',
        'drug-enforcement', 'drug-drugsfda', 'device-pma', 'drug-shortages',
        'drug-ndc', 'food-enforcement', 'tobacco-problem', 'food-event',
        'animalandveterinary-event', 'tax-incentives', 'insurance-company-complaints'
    ],
    2015: ['meta-critic_whisky'],
    2018: ['aijob_ai-ml-ds-salaries']
}

def map_dataset_to_year(dataset_name):
    for year, datasets in year_to_datasets.items():
        if dataset_name in datasets:
            return year
    return None

# Apply the function to create the new column
results['year'] = results['data_name'].apply(map_dataset_to_year)

'''
macro category for year
'''

def map_year_to_macro_category(year):
    if year < 2000:
        return 'Pre-2000'
    elif year >= 2000 and year < 2010:
        return '2000-2009' 
    else:
        return '2010-Present'   

results['year_macro_category'] = results['year'].apply(map_year_to_macro_category)

'''
selected LLMs
'''
baseline_encoders = ['Tf-Idf', 'TargetEncoder']
e2e_encoders = ['CatBoost',  'TabPFN-2.5', 'ContextTab', 'TabSTAR', 'Tarte']
selected_LLMs = [
    'LM All-MiniLM-L6-v2',
    'LM FastText',
    'LM E5-small-v2',
    'LM LLaMA-3.1-8B',
    'LM Qwen-3-8B',
    'LM Jasper-0.6B'
]

top3_LLMs = [
    'LM All-MiniLM-L6-v2',
    'LM FastText',
    'LM E5-small-v2'
]

selected_encoders = baseline_encoders + e2e_encoders + selected_LLMs

selected_encoders_top3 = baseline_encoders + e2e_encoders + top3_LLMs

'''
Add train and test sizes per dataset
'''

# Use np.ceil to round up, then convert to integer
results['train_size'] = np.ceil(results['num_rows'] * (2/3)).astype(int)

# ideally, calculate test_size as the remainder to ensure they add up to num_rows exactly
results['test_size'] = results['num_rows'] - results['train_size']

'''
drop TARTE encoder
'''

# results = results[results['encoder'] != 'Tarte']

'''
drop TabPFN-2.5 encoder
'''

# temp = results[(results['dtype'] == 'Num_only') & (results['encoder'] == 'TabPFN-2.5')].copy()

# results = results[(results['encoder'] != 'TabPFN-2.5')]

# results = pd.concat([results, temp], axis=0).reset_index(drop=True)

'''
Drop RealTabPFN
'''

results = results[results['learner'] != 'RealTabPFN-2.5']

'''
keep only methods that have k results (all datasets)
'''
k = 303
count_results = results.method.value_counts()
keep_method_list = count_results[count_results >= k].index.tolist()

results = results[results.method.isin(keep_method_list)].reset_index(drop=True)


#####################FUNCTIONS##################

# Function to generate a marker of data_name, num_train, and random_state
def _generate_marker(df_score):
    df_score_ = df_score.copy()
    df_score_["marker"] = (
        df_score_["data_name"]
        + "-"
        + df_score_["n_cv"].astype(str)
        + "-"
        + df_score_["fold_index"].astype(str)
    )
    return df_score_

# Define a function to clean the method string based on your maps
def clean_method_name(method_str):
    # Handle specific edge cases where the format might not be a simple split
    # such as 'catboost_catboost' or special baselines
    if method_str == 'catboost_catboost':
        return 'CatBoost - CatBoost'
    
    # Attempt to split into encoder and learner parts
    # Assuming format is usually 'encoder_learner'
    if '_' in method_str:
        parts = method_str.split('_', 1)
        enc_part = parts[0]
        lrn_part = parts[1]
        
        # Get mapped names, default to original if not in map
        encoder = encoder_map.get(enc_part, enc_part)
        learner = learner_map.get(lrn_part, lrn_part)
        
        return f"{encoder} - {learner}"
    
    # Fallback for single-part names (like baselines)
    return encoder_map.get(method_str, learner_map.get(method_str, method_str))

def get_pareto_front(df, x_col, y_col, maximize_y=True):
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
    """
    Calculates the average rank of each algorithm across the given datasets.
    Output: Series of ranks (e.g., XGB: 1.2, Ridge: 3.4)
    """
    # Rank per dataset
    ranks_per_dataset = data_frame.rank(axis=1, ascending=False, method='min')
    # Average across datasets
    avg_ranks = ranks_per_dataset.mean(axis=0)
    return avg_ranks

def model_ref_1(x, a, b):
    """Reference 1: 1 - (a/sqrt(x)) * exp(-bx)"""
    # Note: 1.0 is the fixed asymptote here
    return 1 - (a / np.sqrt(x)) * np.exp(-b * x)

def median_iqr(x):
    """Returns string formatted as 'Median (IQR)'"""
    med = x.median()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return f"{med:.0f} [{q1:.0f}, {q3:.0f}]"

def bin_feature_33_66(df, col):
    bins = [0, df[col].quantile(0.33), df[col].quantile(0.66), float('inf')]
    labels = ['Low', 'Med', 'High']
    return pd.cut(df[col], bins=bins, labels=labels)

def bin_feature_median(df, col):
    bins = [0, df[col].quantile(0.5), float('inf')]
    labels = ['Low', 'High']
    return pd.cut(df[col], bins=bins, labels=labels)


def plot_regime(df, model_col, ax, markers):

    ax.plot([0.5, 1.0], [0.5, 1.0], '--', color='gray', alpha=0.3, zorder=1)
    
    unique_models = df[model_col].unique()
    palette = sns.color_palette("husl", len(unique_models))
    
    for i, model in enumerate(unique_models):
        subset = df[df[model_col] == model]
        ax.scatter(subset['High'], subset['Low'], 
                   marker=markers.get(model, 'o'), s=80, 
                   label=model, alpha=0.9, color=palette[i], 
                   edgecolors='white', linewidth=0.5, zorder=3)
####################### PALETTES, SHAPES & EMOJIS ####################
learner_emoji_map = {
    'Ridge': '📏',
    'ExtraTrees': '🌳',
    'XGBoost': '🌳',
    'CatBoost': '🌳',
    'TabSTAR': '🧠',
    'ContextTab': '🧠',
    'TabPFN-2.5': '🧠',
    'RealTabPFN-2.5': '🧠',
    'Tarte': '🧠'
}

encoder_emoji_map = {
    'Tf-Idf': '📊',
    'TargetEncoder': '📊',
    'CatBoost': '⚙️', 
    'ContextTab': '⚙️',
    'TabSTAR': '⚙️',
    'TabPFN-2.5': '⚙️',
    'Tarte': '⚙️',
    'LM': '🤖',
}

font_path = 'NotoColorEmoji.ttf'

# Verify file exists
if not os.path.exists(font_path):
    print(f"Warning: Font file not found at {font_path}")
    # Fallback to a standard font if missing
    emoji_prop = FontProperties() 
else:
    emoji_prop = FontProperties(fname=font_path)

# 1. Base Colors for Learners
learner_colors = {
    'XGBoost': '#D55E00',      # Vermilion
    'TabSTAR': '#0072B2',     # Blue
    'Ridge': '#009E73',   # Bluish Green
    'ExtraTrees': '#E69F00',        # Orange/Yellow
    'TabPFN': '#CC79A7',       # Reddish Purple
    'ContextTab': '#56B4E9',   # Sky Blue
    'CatBoost': '#F0E442',      # Yellow
    'Tarte': '#882255',        # Wine
    'RealTabPFN': '#CC79A7'    # Same as TabPFN
}

# 2. Shapes for Learners
learner_shapes = {
    'XGBoost': 's',       # Square
    'CatBoost': 'D',      # Diamond
    'ExtraTrees': '^',    # Triangle Up
    'Ridge': 'o',         # hexagon
    'TabPFN': 'h',        # Octagon
    'ContextTab': 'X',    # Filled Plus
    'TabSTAR': 'p',       # Pentagon
    # 'Tarte': '<'          # Triangle Down
}

llm_base_colors = {
    # --- Safe Tab10 Assignments ---
    'llama': 'tab:purple',  # Distinct from TabPFN (Reddish Purple)
    'opt': 'tab:brown',     # Distinct from ExtraTrees (Orange)
    'gemma': 'tab:gray',    # Neutral
    'fasttext': 'tab:green',# Distinct from Ridge (Teal-ish)
    'bge': 'tab:olive',     # Distinct from CatBoost (Yellow)
    'e5': 'tab:pink',       # Distinct from Llama (Purple) and TabPFN (Red-Purple)

    # --- Custom Colors to Avoid Learner Conflicts ---
    'qwen': 'crimson',      # 'tab:red' clashes with XGBoost (Vermilion)
    'bert': 'navy',         # 'tab:blue' clashes with TabSTAR (Blue)
    'deberta': 'teal',      # 'tab:cyan' clashes with ContextTab (Sky Blue)
    
    # --- Others (Distinct Hexes) ---
    'roberta': '#483D8B',   # DarkSlateBlue
    'mpnet': '#556B2F',     # DarkOliveGreen
    'mini': '#FF1493',      # DeepPink
    'glove': '#4682B4',     # SteelBlue
    'jasper': '#32CD32',    # LimeGreen
    'f2llm': '#000000',     # Black
    
    'fallback': '#333333'
}

def get_hash_shade(base_color, model_name):
    """Deterministically darkens/lightens a base color based on model name."""
    rgb = mcolors.to_rgb(base_color)
    # Use hash of name to get a consistent modifier between -0.15 and 0.15
    # We reduced the range slightly to prevent drifting into other colors
    h = int(hashlib.sha256(model_name.encode('utf-8')).hexdigest(), 16)
    mod = ((h % 100) / 100.0) * 0.3 - 0.15 
    
    new_rgb = [max(0, min(1, c + mod)) for c in rgb]
    return mcolors.to_hex(new_rgb)

# 3. Encoder Family Color Logic (Pseudocode for mapping)
def get_encoder_color(encoder_name):
    name = str(encoder_name).lower()
    
    # --- CHECK LEARNERS FIRST ---
    if 'catboost' in name: return learner_colors['CatBoost'] 
    if 'contexttab' in name: return learner_colors['ContextTab']
    if 'tabstar' in name: return learner_colors['TabSTAR']
    if 'tarte' in name: return learner_colors['Tarte']
    if 'tabpfn' in name: return learner_colors['TabPFN']
    if 'xgb' in name: return learner_colors['XGBoost']
    
    # --- BASELINES ---
    if any(x in name for x in ['string', 'target', 'tabvec', 'onehot', 'tfidf']): 
        return '#7f7f7f' # Grey

    # --- LLM FAMILIES (Expanded) ---
    # 1. LLaMA
    if 'llama' in name: return get_hash_shade(llm_base_colors['llama'], name)
    # 2. Qwen
    if 'qwen' in name: return get_hash_shade(llm_base_colors['qwen'], name)
    # 3. OPT
    if 'opt' in name: return get_hash_shade(llm_base_colors['opt'], name)
    # 4. Gemma
    if 'gemma' in name: return get_hash_shade(llm_base_colors['gemma'], name)
    # 5. F2LLM
    if 'f2llm' in name: return get_hash_shade(llm_base_colors['f2llm'], name)
    
    # --- BERT Variants (Now Split) ---
    if 'roberta' in name: return get_hash_shade(llm_base_colors['roberta'], name)
    if 'deberta' in name: return get_hash_shade(llm_base_colors['deberta'], name)
    if 'mpnet' in name: return get_hash_shade(llm_base_colors['mpnet'], name)
    if 'bert' in name: return get_hash_shade(llm_base_colors['bert'], name)
    
    # --- Embedding Variants (Now Split) ---
    if 'mini' in name: return get_hash_shade(llm_base_colors['mini'], name)
    if 'e5' in name: return get_hash_shade(llm_base_colors['e5'], name)
    if 'bge' in name: return get_hash_shade(llm_base_colors['bge'], name)
        
    # --- Simple Embeddings (Now Split) ---
    if 'fasttext' in name: return get_hash_shade(llm_base_colors['fasttext'], name)
    if 'glove' in name: return get_hash_shade(llm_base_colors['glove'], name)
    if 'jasper' in name: return get_hash_shade(llm_base_colors['jasper'], name)

    return llm_base_colors['fallback']

# 4. Tuning Logic (Matplotlib style kwargs)
def get_tuning_style(learner_key):
    if 'tuned' in learner_key:
        return {'fillstyle': 'none', 'markeredgewidth': 2} 
        # Or for "empty with dot": use a custom marker composition
    else:
        return {'fillstyle': 'full'}

# 2. Helper to get Color (Same for default/tuned)
def get_learner_color_simple(learner_name):
    for family, color in learner_colors.items():
        if family in learner_name:
            return color
    return '#333333' # Fallback

# 3. Helper to get Hatch (Texture)
def get_learner_hatch(learner_name):
    if 'tuned' in learner_name:
        return '///'  # Diagonal lines (repeat / for density)
    else:
        return ''     # No hatch (solid)

def get_learner_marker(learner_name):
    # Check exact match first
    if learner_name in learner_shapes:
        return learner_shapes[learner_name]
    # Check if it's a tuned version (e.g. "XGBoost-tuned" -> "XGBoost")
    for family, marker in learner_shapes.items():
        if family in learner_name:
            return marker
    return 'o'    

def get_learner_emoji(learner_name):
    """
    Returns the emoji + space + name for a learner.
    Handles '-tuned' suffixes automatically.
    """
    name_str = str(learner_name)
    
    # 1. Clean the name to find the 'Family' key (e.g. "XGBoost-tuned" -> "XGBoost")
    # We split by '-' and take the first part, unless it's a specific known compound
    base_name = name_str.split('-')[0]
    
    # Special handle for TabPFN versions if needed (e.g. TabPFN-2.5 -> TabPFN)
    # if 'TabPFN' in name_str:
    #     base_name = 'TabPFN'
    # if 'RealTabPFN' in name_str:
    #     base_name = 'RealTabPFN'

    # 2. Look up the emoji
    # We check if any key in the map is a substring of the learner name
    found_emoji = '❓' # Default fallback
    
    for key, emoji in learner_emoji_map.items():
        if key.lower() == base_name.lower():
            found_emoji = emoji
            break
            
    # 3. Return formatted string: "🌳 XGBoost-tuned"
    return f"{found_emoji} {name_str}"

def get_encoder_emoji(encoder_name):
    """
    Returns the emoji + space + name for an encoder.
    Logic is hierarchical: Internal > Baseline > LLM
    """
    name_str = str(encoder_name)
    
    # 1. Check for Internal/E2E (Gear ⚙️)
    # Explicit check for known internal encoders
    internal_keys = ['CatBoost', 'ContextTab', 'TabSTAR', 'TabPFN-2.5', 'Tarte']
    if any(k in name_str for k in internal_keys):
        return f"{encoder_emoji_map['CatBoost']} {name_str}"
        
    # 2. Check for Baselines (Chart 📊)
    baseline_keys = ['Tf-Idf', 'TargetEncoder']
    if any(k in name_str for k in baseline_keys):
        # We use 'Tf-Idf' as the key to fetch the Chart emoji
        return f"{encoder_emoji_map['Tf-Idf']} {name_str}"
        
    # 3. Check for LLMs (Robot 🤖)
    # Generally, everything else in your list starting with "LM" is an LLM
    if 'LM ' in name_str:
        return f"{encoder_emoji_map['LM']} {name_str}"

    # Fallback
    return f"❓ {name_str}"

##### CREATE TODAYS_FOLDER TO SAVE PLOTS #####

TODAYS_FOLDER = time.strftime("%Y-%m-%d")
# Create directory if it doesn't exist
import os
os.makedirs('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ TODAYS_FOLDER, exist_ok=True)

os.makedirs('/data/parietal/store4/soda/gblayer/salts/results_tables/'+ TODAYS_FOLDER, exist_ok=True)
    
################ BENCHMARK BUILDING PROCESS PLOTS ##################


'''
DATASETS DISTRIBUTION - HETEROGENEITY
MODE
'''
import openml
# ---------------------------------------------------------
# 1. LOAD OPENML DATA
# ---------------------------------------------------------
print("Fetching OpenML metadata...")
openml_datasets = openml.datasets.list_datasets(output_format="dataframe")

# ---------------------------------------------------------
# 2. PLOTTING
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

metrics = [
    ('num_rows', 'Rows', 'salmon', (0, 0)),
    ('num_columns', 'Columns', 'skyblue', (0, 1)),
    ('avg_cardinality', 'Cardinality', 'purple', (1, 0)),
    ('avg_string_length_per_cell', 'String Length (Avg # char)', 'green', (1, 1))
]

for col_name, title, color, (r, c) in metrics:
    ax = axes[r, c]
    
    # Extract data for this column
    data = dataset_summary_wide[col_name].dropna()
    
    # --- A. Plot STRABLE Benchmark (Your Data) ---
    sns.histplot(data, kde=True, log_scale=True, 
                 ax=ax, color=color, alpha=0.6, line_kws={'linewidth': 1.5},
                 edgecolor='black')
    
    # Lock the scale to your dataset's range
    x_min, x_max = ax.get_xlim()

    # --- CALCULATE & PLOT MODE (Red Line) ---
    # Since we are plotting on a log scale, the visual "Peak" (Mode) is best found 
    # by estimating the KDE on the log-transformed data.
    log_data = np.log10(data[data > 0]) # Handle log(0) if any
    # kde = stats.gaussian_kde(log_data)
    counts, bin_edges = np.histogram(log_data, bins='auto')
    max_idx = np.argmax(counts)

    # Create a grid to find the peak
    # x_grid = np.linspace(log_data.min(), log_data.max(), 1000)
    # pdf = kde(x_grid)
    
    # Find the peak in log space and convert back to linear space
    # mode_log = x_grid[np.argmax(pdf)]
    # mode_val = 10**mode_log
    mode_log = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
    mode_val = 10**mode_log
    
    # Add the vertical red line
    ax.axvline(x=mode_val, color='red', linestyle='-', linewidth=2, zorder=5)
    
    # --- B. Plot OpenML Distribution (Overlay) ---
    if col_name == 'num_rows':
        ax2 = ax.twinx()
        openml_rows = openml_datasets[openml_datasets["NumberOfInstances"] > 0]["NumberOfInstances"]
        sns.kdeplot(openml_rows, ax=ax2, log_scale=True, 
                    color=color, linestyle='--', linewidth=2.5, warn_singular=False,
                    zorder=10)
        ax2.set_yticks([]) 
        ax2.set_ylabel("")
        ax2.set_xlim(x_min, x_max)
        
    elif col_name == 'num_columns':
        ax2 = ax.twinx()
        openml_cols = openml_datasets[openml_datasets["NumberOfFeatures"] > 0]["NumberOfFeatures"]
        sns.kdeplot(openml_cols, ax=ax2, log_scale=True, 
                    color=color, linestyle='--', linewidth=2.5, warn_singular=False,
                    zorder=10)
        ax2.set_yticks([]) 
        ax2.set_ylabel("")
        ax2.set_xlim(x_min, x_max)
    
    # --- Mode Label ---
    if col_name=='num_rows':
        ax.text(mode_val/10, 0.9, f'{mode_val:.1f}', 
        transform=ax.get_xaxis_transform(), # Mixes data coords (x) and axes coords (y)
        color='red', fontsize=12, fontweight='bold', 
        ha='left', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    else:
        ax.text(mode_val, 0.9, f'{mode_val:.1f}', 
            transform=ax.get_xaxis_transform(), # Mixes data coords (x) and axes coords (y)
            color='red', fontsize=12, fontweight='bold', 
            ha='left', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


    # --- Formatting ---
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.15)

# ---------------------------------------------------------
# 3. LEGEND & LAYOUT
# ---------------------------------------------------------
plt.subplots_adjust(hspace=0.4, wspace=0.3, right=0.85)

fig.supxlabel("Value (Log Scale)", fontsize=16, y=0.01)
fig.supylabel("Frequency", fontsize=16)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='OpenML\nDistribution'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='STRABLE\nMode'),
    Line2D([0], [0], color='grey', linestyle='-', linewidth=2, label='STRABLE\nDistribution')
]

fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.1, 0.83), fontsize=12)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'metadata_distribution_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
DISTRIBUTION OF PROPORTION OF UNIQUE TEXT CELLS
MODE
'''
import matplotlib
# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
feature_name = 'prop_unique_text_cells'
plot_title = 'Proportion Unique Text Cells'
plot_color = 'teal' 

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))

# 1. Extract Data
# Assuming 'dataset_summary_wide' is already loaded in your environment
data = dataset_summary_wide[feature_name].dropna()

# 2. Plot Histogram + KDE (Log Scale)
sns.histplot(data, kde=True, log_scale=True, 
             ax=ax, color=plot_color, alpha=0.6, line_kws={'linewidth': 1.5},
             edgecolor='black')

# 3. Calculate Mode (Empirical from Data bins)
# Handle log(0) explicitly if data contains absolute zeros, though 'unique' implies > 0
log_data = np.log10(data[data > 0]) 
counts, bin_edges = np.histogram(log_data, bins='auto')
max_idx = np.argmax(counts)
mode_log = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
mode_val = 10**mode_log

# 4. Add Mode Line and Label
ax.axvline(x=mode_val, color='red', linestyle='-', linewidth=2, zorder=5)

# Place text slightly offset
ax.text(mode_val, 0.9, f'{mode_val:.2f}', 
        transform=ax.get_xaxis_transform(), 
        color='red', fontsize=12, fontweight='bold', 
        ha='left', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 5. Formatting
ax.set_title(plot_title, fontsize=14, fontweight='bold')
ax.set_xlabel("Proportion (Log Scale)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.grid(True, alpha=0.15)

# ---------------------------------------------------------
# LEGEND
# ---------------------------------------------------------
# Combined handle for Hist + KDE
strable_dist_handle = (
    Patch(facecolor=plot_color, alpha=0.6, edgecolor='black'),
    Line2D([0], [0], color=plot_color, linewidth=1.5)
)
mode_handle = Line2D([0], [0], color='red', linestyle='-', linewidth=2)

fig.legend(
    handles=[strable_dist_handle, mode_handle],
    labels=['STRABLE\nDistribution', 'STRABLE\nMode'],
    loc='upper right', 
    bbox_to_anchor=(0.65, 0.85),
    fontsize=10,
    handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)}
)

plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'prop_unique_text_cells_distribution_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
DATASETS DISTRIBUTION - HETEROGENEITY
MEDIAN+MODE
'''
import openml
# ---------------------------------------------------------
# 1. LOAD OPENML DATA
# ---------------------------------------------------------
# Assuming 'dataset_summary_wide' is already loaded in your environment
print("Fetching OpenML metadata...")
openml_datasets = openml.datasets.list_datasets(output_format="dataframe")

# ---------------------------------------------------------
# 2. PLOTTING
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

metrics = [
    ('num_rows', 'Rows', 'salmon', (0, 0)),
    ('num_columns', 'Columns', 'skyblue', (0, 1)),
    ('avg_cardinality', 'Cardinality', 'purple', (1, 0)),
    ('avg_string_length_per_cell', 'String Length (Avg # char)', 'green', (1, 1))
]

for col_name, title, color, (r, c) in metrics:
    ax = axes[r, c]
    
    # Extract data for this column
    data = dataset_summary_wide[col_name].dropna()
    
    # --- A. Plot STRABLE Benchmark (Your Data) ---
    sns.histplot(data, kde=True, log_scale=True, 
                 ax=ax, color=color, alpha=0.6, line_kws={'linewidth': 1.5},
                 edgecolor='black')
    
    # Lock the scale to your dataset's range
    x_min, x_max = ax.get_xlim()

    # --- B. CALCULATE & PLOT MODE (Red Line) ---
    log_data = np.log10(data[data > 0]) # Handle log(0)
    counts, bin_edges = np.histogram(log_data, bins='auto')
    max_idx = np.argmax(counts)
    
    # Estimate mode from bin center
    mode_log = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
    mode_val = 10**mode_log
    
    ax.axvline(x=mode_val, color='red', linestyle='-', linewidth=2, zorder=5)

    # --- C. CALCULATE & PLOT MEDIAN (Blue Line) ---
    median_val = data.median()
    
    # Draw vertical blue line
    ax.axvline(x=median_val, color='blue', linestyle='-', linewidth=2, zorder=5)

    # --- D. Plot OpenML Distribution (Overlay) ---
    if col_name == 'num_rows':
        ax2 = ax.twinx()
        openml_rows = openml_datasets[openml_datasets["NumberOfInstances"] > 0]["NumberOfInstances"]
        sns.kdeplot(openml_rows, ax=ax2, log_scale=True, 
                    color=color, linestyle='--', linewidth=2.5, warn_singular=False,
                    zorder=10)
        ax2.set_yticks([]) 
        ax2.set_ylabel("")
        ax2.set_xlim(x_min, x_max)
        
    elif col_name == 'num_columns':
        ax2 = ax.twinx()
        openml_cols = openml_datasets[openml_datasets["NumberOfFeatures"] > 0]["NumberOfFeatures"]
        sns.kdeplot(openml_cols, ax=ax2, log_scale=True, 
                    color=color, linestyle='--', linewidth=2.5, warn_singular=False,
                    zorder=10)
        ax2.set_yticks([]) 
        ax2.set_ylabel("")
        ax2.set_xlim(x_min, x_max)
    
    # --- LABELS ---
    transform = ax.get_xaxis_transform()
    
    # Mode Label (Top, Red)
    # Special handling for 'Rows' to avoid cutting off text on the left edge
    if col_name == 'num_rows':
        mode_x_pos = mode_val/6
        median_x_pos = median_val/4
    elif col_name == 'num_columns':
        mode_x_pos = mode_val
        median_x_pos = median_val-5
    elif col_name == 'avg_cardinality':
        mode_x_pos = mode_val
        median_x_pos = median_val-1000
    else:
        mode_x_pos = mode_val-5
        median_x_pos = median_val
            
    ax.text(mode_x_pos, 0.9, f'{mode_val:.0f}', 
            transform=transform,
            color='red', fontsize=12, fontweight='bold', 
            ha='left', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Median Label (Slightly Lower, Blue)
    # We position it at y=0.75 to minimize overlap with the Mode label
    ax.text(median_x_pos, 0.75, f'{median_val:.0f}', 
            transform=transform, 
            color='blue', fontsize=12, fontweight='bold', 
            ha='left', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # --- Formatting ---
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.15)

# ---------------------------------------------------------
# 3. LEGEND & LAYOUT
# ---------------------------------------------------------
plt.subplots_adjust(hspace=0.4, wspace=0.3, right=0.85)

fig.supxlabel("Value (Log Scale)", fontsize=16, y=0.01)
fig.supylabel("Frequency", fontsize=16)

# Custom Legend with Median Added
legend_elements = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='OpenML\nDistribution'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='STRABLE\nMode'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='STRABLE\nMedian'), # Added this
    Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='STRABLE\nDistribution')
]

fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.1, 0.83), fontsize=12)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'metadata_distribution_median_mode_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
DISTRIBUTION OF PROPORTION OF UNIQUE TEXT CELLS
MEDIAN+MODE
'''
# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
feature_name = 'prop_unique_text_cells'
plot_title = 'Proportion Unique Text Cells'
plot_color = 'teal' 

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))

# 1. Extract Data
# Assuming 'dataset_summary_wide' is already loaded in your environment
data = dataset_summary_wide[feature_name].dropna()

# 2. Plot Histogram + KDE (Log Scale)
sns.histplot(data, kde=True, log_scale=True, 
             ax=ax, color=plot_color, alpha=0.6, line_kws={'linewidth': 1.5},
             edgecolor='black')

# 3. Calculate Mode (Empirical from Data bins)
# Handle log(0) explicitly if data contains absolute zeros, though 'unique' implies > 0
log_data = np.log10(data[data > 0]) 
counts, bin_edges = np.histogram(log_data, bins='auto')
max_idx = np.argmax(counts)
mode_log = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
mode_val = 10**mode_log

# 4. Calculate Median
median_val = data.median()

# 5. Add Mode Line and Label (Red)
ax.axvline(x=mode_val, color='red', linestyle='-', linewidth=2, zorder=5)

# Place text slightly offset
ax.text(mode_val, 0.9, f'{mode_val:.2f}', 
        transform=ax.get_xaxis_transform(), 
        color='red', fontsize=12, fontweight='bold', 
        ha='left', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 6. Add Median Line and Label (Blue)
ax.axvline(x=median_val, color='blue', linestyle='-', linewidth=2, zorder=5)

# Place text lower (y=0.75) to avoid overlap with Mode label
ax.text(median_val-0.09, 0.75, f'{median_val:.2f}', 
        transform=ax.get_xaxis_transform(), 
        color='blue', fontsize=12, fontweight='bold', 
        ha='left', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 7. Formatting
ax.set_title(plot_title, fontsize=14, fontweight='bold')
ax.set_xlabel("Proportion (Log Scale)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.grid(True, alpha=0.15)

# ---------------------------------------------------------
# LEGEND
# ---------------------------------------------------------
# Combined handle for Hist + KDE
strable_dist_handle = (
    Patch(facecolor=plot_color, alpha=0.6, edgecolor='black'),
    Line2D([0], [0], color=plot_color, linewidth=1.5)
)
mode_handle = Line2D([0], [0], color='red', linestyle='-', linewidth=2)
median_handle = Line2D([0], [0], color='blue', linestyle='-', linewidth=2) # New median handle

fig.legend(
    handles=[strable_dist_handle, mode_handle, median_handle],
    labels=['STRABLE\nDistribution', 'STRABLE\nMode', 'STRABLE\nMedian'], # Added Median label
    loc='upper right', 
    bbox_to_anchor=(0.65, 0.85),
    fontsize=10,
    handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)}
)

plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'prop_unique_text_cells_distribution_median_mode_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()



'''
MACRO-SOURCE ANALYSIS: map each source to a field (education, finance, health, etc.)
repeat the transposed histogram plot and the violin plot for macro-sources
'''

# sources_list = ['aijobs.net',
#  'BeerAdvocate.com',
#  'commonlit.org',
#  'kaggle',
#  'Consumer-Financial-Protection-Bureau',
#  'ClinicalTrials.gov',
#  'energydata.info',
#  'European-Medicines-Agency',
#  'European-Commission',
#  'fda',
#  'flavorsofcacao.com',
#  'Federal-Deposit-Insurance-Corporation',
#  'FSA',
#  'fueleconomy.gov',
#  'HIFLD',
#  'HRSA',
#  'Institute of Museum and Library Services',
#  'data.ct.gov',
#  'webrobots.io',
#  'lendingclub.com',
#  'Medicaid',
#  'whiskyanalysis.com',
#  'Michelin',
#  'OHCA',
#  'mercari.com',
#  'osha.gov',
#  'SCIMAGO',
#  'data.sfgov.org',
#  'theramenrater.com',
#  'majestic.co.uk',
#  'world-resource-institute',
#  'worldbankfinancesone',
#  'Yelp Open Dataset']

# Build the same mapping but derive entries from the existing `sources_list`

category_counts = results.groupby('category')['data_name'].nunique()
category_label_map = {cat: f"{cat} ({count})" for cat, count in category_counts.items()}
results['category_with_ds_count'] = results['category'].map(category_label_map)

hist_df = results.groupby('category', as_index=False)['data_name'].nunique()

# Plotting
# Swapped figsize dimensions slightly to accommodate horizontal layout
plt.figure(figsize=(5, 4))

# Use barh instead of bar
bars = plt.barh(hist_df['category'], hist_df['data_name'], color='#5da5da', edgecolor='#333333', linewidth=1.2)

# Add the count labels to the right of each bar
for bar in bars:
    width = bar.get_width() # Get the length of the bar
    plt.text(
        width + 0.5,        # x-position: end of bar + small offset
        bar.get_y() + bar.get_height()/2, # y-position: center of bar
        int(width), 
        va='center',        # Vertically align to center
        ha='left',          # Horizontally align to left of the text point
        fontsize=16, 
        fontweight='bold'
    )

# Formatting
plt.xlabel('Number of Datasets', fontsize=16) # Swapped label
plt.ylabel('Application Field', fontsize=20)           # Swapped label
plt.xlim(0, hist_df['data_name'].max() + 1)   # Changed ylim to xlim
plt.yticks(fontsize=16)                       # Removed rotation, usually easier to read horizontally
plt.grid(axis='x', linestyle='--', alpha=0.4) # Changed grid to x-axis

# Optional: Invert y-axis if you want the first category at the top
# plt.gca().invert_yaxis()

# Adjust layout
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'histogram_macro_category_ds_count_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
DISTRIBUTION OF DATASETS PER YEAR
'''

year_counts = results.groupby('year_macro_category')['data_name'].nunique()
year_label_map = {year: f"{year} ({count})" for year, count in year_counts.items()}
results['year_with_ds_count'] = results['year_macro_category'].map(year_label_map)

hist_df = results.groupby('year_macro_category', as_index=False)['data_name'].nunique()

ordered_categories = ['Pre-2000', '2000-2009', '2010-Present']
hist_df['year_macro_category'] = pd.Categorical(
    hist_df['year_macro_category'], 
    categories=ordered_categories, 
    ordered=True
)
hist_df = hist_df.sort_values('year_macro_category')

# Plotting
plt.figure(figsize=(5, 6))
bars = plt.bar(hist_df['year_macro_category'], hist_df['data_name'], color='#ff7f0e', edgecolor='#333333', linewidth=1.2)
# Add the count labels on top of each bin
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        yval + 0.5, 
        int(yval), 
        va='bottom', 
        ha='center', 
        fontsize=16, 
        fontweight='bold'
    )

# Formatting
# plt.title('Number of Distinct Datasets by Category', fontsize=16, pad=20)
plt.ylabel('Number of Datasets', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.ylim(0, hist_df['data_name'].max() + 5)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Adjust layout to prevent label cutoff
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'histogram_year_ds_count_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
Distribution plots of R2 score for Num+Str 
'''


score = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

subset_r2 = results[
    (results['dtype'] == 'Num+Str') &
    (results['task'].isin(['regression'])) &
    (results['encoder'] != 'TabPFN-2.5') # drop TabPFN encoder
].groupby(['data_name'],as_index=False)[score].mean()

# all the negative r2 are clipped to zero
# subset_r2['r2'] = subset_r2['r2'].clip(lower=0)

# draw distribution plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=subset_r2,
    x=score,
    bins=15,
    kde=True,
    color='skyblue',
    edgecolor='black'
)
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xlabel(f"R2 {Y_METRIC_LABELS[score]}", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# for each bin of the histogram, print the upper and lower limit and the count,
# and annotate the count above each bar
for patch in ax.patches:
    bin_left = patch.get_x()
    bin_right = bin_left + patch.get_width()
    count = int(round(patch.get_height()))
    print(f"Bin lower limit: {bin_left}, Bin upper limit: {bin_right}, Count: {count}")
    if count > 0:
        ax.text(bin_left + patch.get_width() / 2, patch.get_height(),
                str(count), ha='center', va='bottom', fontsize=10)

#save picture
# format fot the pic name: plot_type + _ + metric + _ + level + date .png
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'distribution_plot_{score}_r2_regression_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Distribution plots of ROC_AUC score for Num+Str 
'''

score = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

subset_roc_auc = results[
    (results['dtype'] == 'Num+Str') &
    (~results['task'].isin(['regression'])) &
    (results['encoder'] != 'TabPFN-2.5') # drop TabPFN encoder
].groupby(['data_name'],as_index=False)[score].mean()

# draw distribution plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=subset_roc_auc,
    x=score,
    bins=15,
    kde=True,
    color='orange',
    edgecolor='black'
)
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xlabel(f"AUC {Y_METRIC_LABELS[score]}", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# for each bin of the histogram, print the upper and lower limit and the count,
# and annotate the count above each bar
for patch in ax.patches:
    bin_left = patch.get_x()
    bin_right = bin_left + patch.get_width()
    count = int(round(patch.get_height()))
    print(f"Bin lower limit: {bin_left}, Bin upper limit: {bin_right}, Count: {count}")
    if count > 0:
        ax.text(bin_left + patch.get_width() / 2, patch.get_height(),
                str(count), ha='center', va='bottom', fontsize=10)

#save picture
# format fot the pic name: plot_type + _ + metric + _ + level + date .png
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'distribution_plot_{score}_roc_auc_classification_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
LEARNER PERFORMANCE PER NUM VS NUM+STR
'''

# score = score_list[0]
# dtype = ['Num+Str', 'Num']  

# df = results.copy()
# df = df[df['dtype'].isin(dtype)]
# df = df[df['encoder'].isin(selected_encoders)]

# # by learner
# avg_performance_per_learner_dtype = df.groupby(['learner','dtype'], as_index=False)[score].mean()

# avg_performance_per_learner_dtype = avg_performance_per_learner_dtype.pivot_table(
#     index='learner',
#     columns='dtype',
#     values=score
# )

# df_melted = avg_performance_per_learner_dtype.reset_index().melt(
#     id_vars='learner', 
#     value_vars=['Num', 'Num+Str'], 
#     var_name='Data Type', 
#     value_name='Average Score'
# )

# # 3. Plotting: TODO ADD HASH BARS TO TUNED VERSION
# sns.set_theme(style="whitegrid", context="talk")
# fig, ax = plt.subplots(figsize=(5, 4))

# # Sort learners by their 'Num+Str' score for a neat ladder effect
# # We calculate the sort order from the original wide dataframe
# sort_order = avg_performance_per_learner_dtype.sort_values('Num+Str', ascending=False).index

# sns.barplot(
#     data=df_melted, 
#     y='learner', 
#     x='Average Score', 
#     hue='Data Type', 
#     order=sort_order,
#     palette={'Num': '#1f77b4', 'Num+Str': '#d62728'}, # Blue vs Red
#     edgecolor='black',
#     linewidth=1,
#     ax=ax
# )

# for i, learner_name in enumerate(sort_order):
#     hatch = get_learner_hatch(learner_name)
    
#     if hatch:
#         # Iterate over the hue containers (Num and Num+Str)
#         for container in ax.containers:
#             # The bar corresponding to this learner is at index 'i'
#             bar = container[i]
            
#             # Create an overlay patch for the White Hatch
#             # We use edgecolor='white' for the hatch lines.
#             # We use fill=False so the original color shows through.
#             rect = mpatches.Rectangle(
#                 (bar.get_x(), bar.get_y()), 
#                 bar.get_width(), 
#                 bar.get_height(), 
#                 fill=False, 
#                 hatch=hatch, 
#                 edgecolor='white', 
#                 linewidth=0, # Try to minimize border conflict
#                 alpha=0.6    # Slight transparency for the hatch
#             )
#             ax.add_patch(rect)
            
#             # Re-draw the black border on top so the white hatch doesn't overwrite it
#             border = mpatches.Rectangle(
#                 (bar.get_x(), bar.get_y()), 
#                 bar.get_width(), 
#                 bar.get_height(), 
#                 fill=False, 
#                 edgecolor='black', 
#                 linewidth=1
#             )
#             ax.add_patch(border)

# # 4. Styling
# ax.set_xlabel(f'Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)', fontsize=16)
# ax.set_xlim(0.2, 0.9)
# ax.set_ylabel('')
# # ax.set_xticklabels
# # ax.legend(title='Features', loc='lower left', bbox_to_anchor=(0.85, 0.01), fontsize=12, title_fontsize=14)

# handles, labels = ax.get_legend_handles_labels()

# # 2. Create a "Tuned" handle (Gray bar with White Hatch)
# tuned_handle = mpatches.Patch(
#     facecolor='gray', 
#     edgecolor='white', 
#     hatch='///', 
#     label='Tuned Model'
# )
# # Optional: Add a Black Border to the legend icon to match plot
# # (Matplotlib legends are tricky with multi-color edges, simpliest is just the hatch)

# # 3. Combine
# handles.append(tuned_handle)
# labels.append("Tuned")

# ax.legend(
#     handles=handles, 
#     labels=labels,
#     title='Features & Model', 
#     loc='lower left', 
#     bbox_to_anchor=(0.85, -0.02), 
#     fontsize=10, 
#     title_fontsize=16,
#     framealpha=0.9
# )

# # Optional: Add value labels to the bars
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.3f', padding=3, fontsize=16)

# # drop upper and right spines
# sns.despine(left=False, bottom=False) # Keep right spine visible

score = score_list[0]
dtype_filter = ['Num+Str', 'Num']  

df_raw = results.copy()
df_raw = df_raw[df_raw['dtype'].isin(dtype_filter)]
df_raw = df_raw[df_raw['encoder'].isin(selected_encoders)]

# drop TabPFN-2.5 encoder
df_raw = df_raw[(df_raw['method'] != 'num-str_tabpfn_tabpfn_default')]

# Calculate Sort Order
avg_perf = df_raw.groupby(['learner', 'dtype'])[score].mean().unstack()
sort_order = avg_perf.sort_values('Num+Str', ascending=False).index

# Calculate Counts for Annotation
counts = df_raw.groupby(['learner', 'dtype'])[score].count().unstack().fillna(0)

# 2. PLOTTING
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(6, 5))

sns.barplot(
    data=df_raw, 
    y='learner', 
    x=score, 
    hue='dtype', 
    order=sort_order,
    palette={'Num': '#1f77b4', 'Num+Str': '#d62728'},
    edgecolor='black',
    linewidth=1,
    errorbar=('ci', 95), 
    capsize=0.1,         
    err_kws={'linewidth': 1.5, 'color': 'black'}, 
    ax=ax
)

# 3. HATCHING LOGIC
for i, learner_name in enumerate(sort_order):
    hatch = get_learner_hatch(learner_name)
    if hatch:
        for container in ax.containers:
            if isinstance(container[i], mpatches.Rectangle): 
                bar = container[i]
                rect = mpatches.Rectangle(
                    (bar.get_x(), bar.get_y()), 
                    bar.get_width(), 
                    bar.get_height(), 
                    fill=False, 
                    hatch=hatch, 
                    edgecolor='white', 
                    linewidth=0, 
                    alpha=0.6
                )
                ax.add_patch(rect)
                border = mpatches.Rectangle(
                    (bar.get_x(), bar.get_y()), 
                    bar.get_width(), 
                    bar.get_height(), 
                    fill=False, 
                    edgecolor='black', 
                    linewidth=1
                )
                ax.add_patch(border)


# 5. STYLING & LEGEND
ax.set_xlabel(f'Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)\nwith 95% CI', fontsize=16)
ax.set_xlim(0.2, 0.95)
ax.set_ylabel('')

handles, labels = ax.get_legend_handles_labels()
tuned_handle = mpatches.Patch(facecolor='gray', edgecolor='white', hatch='///', label='Tuned Model')
handles.append(tuned_handle)
labels.append("Tuned")

ax.legend(
    handles=handles, 
    labels=labels,
    title='Features & Model', 
    loc='lower left', 
    bbox_to_anchor=(0.6, -0.02), 
    fontsize=10, 
    title_fontsize=16,
    framealpha=0.9
)

sns.despine(left=False, bottom=False)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_{score}_performance_by_learner_num+str_num_selectedLLMs_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

# by encoder
# avg_performance_per_encoder_dtype = df.groupby(['encoder','dtype'], as_index=False)[score].mean()

# avg_performance_per_encoder_dtype = avg_performance_per_encoder_dtype.pivot_table(
#     index='encoder',
#     columns='dtype',
#     values=score
# )

# avg_performance_per_encoder_dtype.dropna(axis=0, inplace=True)

'''
PIPELINE PERFORMANCE PER NUM VS NUM+STR
'''

score = score_list[0]
dtype = ['Num+Str', 'Num']  

df = results.copy()
df = df[df['dtype'].isin(dtype)]
df = df[df['encoder'].isin(selected_encoders)]

# drop TabPFN-2.5 encoder
df = df[(df['method'] != 'num-str_tabpfn_tabpfn_default')]



## by encoder-learner
avg_performance_per_encoder_learner_dtype = df.groupby(['encoder_learner','learner','dtype'], as_index=False)[score].mean()

avg_performance_per_encoder_learner_dtype = avg_performance_per_encoder_learner_dtype.pivot_table(
    index=['encoder_learner','learner'],
    columns='dtype',
    values=score
)

avg_performance_per_encoder_learner_dtype['Num'] = avg_performance_per_encoder_learner_dtype['Num'].fillna(
    avg_performance_per_encoder_learner_dtype.groupby(level='learner')['Num'].transform('mean')
)

avg_performance_per_encoder_learner_dtype.reset_index(inplace=True)

avg_performance_per_encoder_learner_dtype.drop(columns=['learner'], inplace=True)

df_plot = avg_performance_per_encoder_learner_dtype.copy()

# Recover Encoder and Learner from the string "Encoder - Learner"
# We assume the format is consistently "Encoder - Learner"
df_plot[['encoder', 'learner']] = df_plot['encoder_learner'].str.split(' - ', expand=True)

fig, ax = plt.subplots(figsize=(8, 5)) 

# 1. Background Zones
lims = [df_plot[['Num', 'Num+Str']].min().min() * 0.98, df_plot[['Num', 'Num+Str']].max().max() * 1.02]
ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0) # Diagonal

# Red Zone (Better on Num Only)
ax.fill_between(lims, [0,0], lims, color='#d62728', alpha=0.05, zorder=0)
ax.text(lims[1]*0.95, lims[0]*1.05, "Better on\nNUMERIC ONLY", color='darkred', ha='right', va='bottom', fontweight='bold')

# Green Zone (Better with Strings)
ax.fill_between(lims, lims, [2,2], color='#2ca02c', alpha=0.05, zorder=0)
ax.text(lims[0]*1.60, lims[1]*0.95, "Better with\nSTRINGS", color='darkgreen', ha='left', va='top', fontweight='bold')

# 2. Plotting Loop
for _, row in df_plot.iterrows():
    # Style Logic
    color = get_encoder_color(row['encoder'])
    marker = get_learner_marker(row['learner'])
    is_tuned = 'tuned' in row['learner']
    
    style_kwargs = {
        'marker': marker,
        'color': color,
        'markersize': 12,
        'linestyle': '',
        'label': '_nolegend_' # Prevent automatic legend creation
    }
    
    if is_tuned:
        style_kwargs.update({
            'fillstyle': 'left',
            'markerfacecoloralt': 'white',
            'markeredgecolor': 'black',
            'markeredgewidth': 1.0
        })
    else:
        style_kwargs.update({
            'fillstyle': 'full',
            'markeredgecolor': 'black',
            'markeredgewidth': 0.5
        })
        
    ax.plot(row['Num'], row['Num+Str'], **style_kwargs)

# 3. Aesthetics
ax.set_xlim(0.32, 0.75)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel(f'{Y_METRIC_LABELS[score]} on Numeric Only', fontsize=14)
ax.set_ylabel(f'{Y_METRIC_LABELS[score]} on Numeric + String', fontsize=14)

# --- 4. LEGEND GENERATION (Side-by-Side) ---

# A. Legend for Learners (Shapes) - LEFT COLUMN
unique_learners = sorted(df_plot['learner'].unique())
learner_handles = []

for lrn in unique_learners:
    m = get_learner_marker(lrn)
    is_tuned = 'tuned' in lrn
    
    # Create line handle to match scatter style
    if is_tuned:
        h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                   fillstyle='left', markerfacecoloralt='white', 
                   markeredgewidth=1.0, markersize=10)
    else:
        h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                   fillstyle='full', markeredgewidth=0.5, markersize=10)
    learner_handles.append(h)

# Place immediately to the right of the plot axis (x=1.02)
leg_learners = plt.legend(
    handles=learner_handles, 
    title=r"$\bf{Learner}$" + "\n" + r"$\it{(shape)}$",
    loc='upper left', 
    bbox_to_anchor=(0.98, 1.0), 
    frameon=False,
    ncol=1, 
    fontsize=12,
    title_fontsize=14,
    labelspacing=0.8,
    handletextpad=0.4
)
ax.add_artist(leg_learners) # Manually add to prevent overwrite

# B. Legend for Encoders (Colors) - RIGHT COLUMN
unique_encoders = sorted(df_plot['encoder'].unique())
encoder_handles = []

for enc in unique_encoders:
    c = get_encoder_color(enc)
    h = mpatches.Patch(facecolor=c, edgecolor='black', label=enc, linewidth=0.5)
    encoder_handles.append(h)

# Place further to the right (x=1.35) so it sits next to the first legend
plt.legend(
    handles=encoder_handles, 
    title=r"$\bf{Encoder}$" + "\n" + r"$\it{(color)}$",
    loc='upper left', 
    bbox_to_anchor=(1.49, 1.0), 
    ncol=1,
    frameon=False,
    fontsize=12,
    title_fontsize=14,
    labelspacing=0.8,
    handletextpad=0.4
)

# 5. Save and Show
sns.despine()
plt.subplots_adjust(right=0.6)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_{score}_performance_by_encoder-learner_num+str_num_selectedLLMs_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
AVG PERFORMANCE PER ENCODER
'''

dtype = 'Num+Str'

# plot_data = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders))].copy()
plot_data = results[(results['dtype'] == dtype)].copy()

# drop TabPFN-2.5 encoder
plot_data = plot_data[(plot_data['method'] != 'num-str_tabpfn_tabpfn_default')]

plot_data

# 1. Prepare Data (using your provided snippet)
encoder_performance = plot_data.groupby(['encoder'], as_index=False)[score].mean().sort_values(by=[score], ascending=False)

# 2. Create Plot
plt.figure(figsize=(5, 15)) # Width, Height

# Generate palette list matching the sorted order of encoders
palette_list = [get_encoder_color(enc) for enc in encoder_performance['encoder']]

ax = sns.barplot(
    data=encoder_performance,
    y='encoder',
    x=score,
    palette=palette_list,
    edgecolor='black',
    linewidth=0.5
)

# 3. Add Value Labels
# fmt='%.3f' matches the precision in your screenshot
# ax.bar_label(ax.containers[0], fmt='%.3f', padding=5, fontsize=16)

# 4. Styling
ax.set_xlabel(f'Average {Y_METRIC_LABELS[score]} ($R^2$ & AUC)', fontsize=18, ha='right', x=1.0)
ax.set_ylabel('') # Remove y-label as the names are self-explanatory
ax.set_xlim(0.6, 0.75) # Add room for labels
sns.despine()

today_date = time.strftime("%Y-%m-%d")
PIC_NAME = f'barplot_encoder_performance_{today_date}.pdf'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
AVG PERFORMANCE PER LEARNER
'''

plt.rcParams['hatch.linewidth'] = 2.0  
plt.rcParams['hatch.color'] = 'white'

dtype = 'Num+Str'

# --- 2. Prepare Data ---
plot_data = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders))].copy()
learner_performance = plot_data.groupby(['learner'], as_index=False)[score].mean().sort_values(by=[score], ascending=False)

# --- 3. Create Plot ---
plt.figure(figsize=(5, 4))

palette_list = [get_learner_color_simple(learner) for learner in learner_performance['learner']]

ax = sns.barplot(
    data=learner_performance,
    y='learner',
    x=score,
    palette=palette_list,
    edgecolor='black',
    linewidth=0.5
)

# --- 4. Apply White Hatching DIRECTLY to Bars ---
# We iterate through the patch objects (the bars themselves)
for i, bar in enumerate(ax.patches):
    # Get the learner name corresponding to this bar
    # (The order of patches matches the order of the dataframe rows because we didn't use 'hue')
    learner_name = learner_performance.iloc[i]['learner']
    
    if 'tuned' in learner_name:
        # We need to overlay a hatch. 
        # Modifying the existing bar's hatch often leads to the hatch taking the facecolor.
        # The safest way is to add a NEW patch exactly on top that is JUST the white hatch.
        
        hatch_patch = mpatches.Rectangle(
            (bar.get_x(), bar.get_y()),
            bar.get_width(),
            bar.get_height(),
            fill=False, 
            edgecolor='white', 
            hatch='///', 
            linewidth=0, 
            alpha=0.6,
            zorder=2 # Ensure it is on top of the colored bar
        )
        ax.add_patch(hatch_patch)

# --- 5. Labels & Styling ---
ax.bar_label(ax.containers[0], fmt='%.3f', padding=5, fontsize=16)

ax.set_xlabel(f'Average {Y_METRIC_LABELS[score]} ($R^2$ & AUC)', fontsize=18)
ax.set_ylabel('') 
ax.set_xlim(0.6, 0.76)

# --- 6. Custom Legend ---
tuned_handle = mpatches.Patch(
    facecolor='gray', 
    edgecolor='white', 
    hatch='///', 
    label='Tuned Model'
)

plt.legend(
    handles=[tuned_handle], 
    loc='lower right', 
    bbox_to_anchor=(1.0, 0.0),
    frameon=False, 
    fontsize=16,
    handletextpad=0.5
)

sns.despine()
today_date = time.strftime("%Y-%m-%d")
PIC_NAME = f'barplot_learner_performance_{today_date}.pdf'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Average performance when using combined (Num+Str) features.
Encoders, e2e models and selected LLMs
'''

dtype = 'Num+Str'
select_llm = 'selected' #'all_llm', 'top3', 'selected'
score = score_list[5]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

for task in ['all_task', 'classification', 'regression']:
    if (score == 'score') & (task == 'all_task'):
        set_xlim_min = 0.56
        set_xlim_max = 0.78
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score_norm') & (task == 'all_task'):
        set_xlim_min = 0.6
        set_xlim_max = 1.0
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score') & (task == 'regression'):
        set_xlim_min = 0.45
        set_xlim_max = 0.72
        bbox_to_anchor=(1.53, 0.0)
    elif (score == 'score_norm') & (task == 'regression'):
        set_xlim_min = 0.75
        set_xlim_max = 1.0
        bbox_to_anchor=(1.33, 0.0)
    elif (score == 'score') & (task == 'classification'):
        set_xlim_min = 0.75
        set_xlim_max = 0.9
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score_norm') & (task == 'classification'):
        set_xlim_min = 0.6
        set_xlim_max = 0.95
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score_clip') & (task == 'all_task'):
        set_xlim_min = 0.55
        set_xlim_max = 0.75
        bbox_to_anchor=(1.53, 0.0)
    elif (score == 'score_clip') & (task == 'regression'):
        set_xlim_min = 0.5
        set_xlim_max = 0.7
        bbox_to_anchor=(1.53, 0.0)
    elif (score == 'score_clip') & (task == 'classification'):
        set_xlim_min = 0.6
        set_xlim_max = 0.9
        bbox_to_anchor=(1.53, 0.0)


    if select_llm == 'all_llm':
        df = results.copy()
    elif select_llm == 'top3':
        df = results[results['encoder'].isin(selected_encoders_top3)].copy()
    else:
        df = results[results['encoder'].isin(selected_encoders)].copy()

    
    if task == 'all_task':
        set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)"
        pass
    elif task == 'regression':
        df = df[df['task'] == 'regression'].copy()
        set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($R^2$)"
    else: 
        set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($AUC$)"
        df = df[df['task'] != 'regression'].copy()

    plot_data = df[(df['dtype'] == dtype)].copy()

    plot_data = plot_data[(plot_data['method'] != 'num-str_tabpfn_tabpfn_default')]

    unique_learners = sorted(df['learner'].unique())

    color_map = {l: get_learner_color_simple(l) for l in unique_learners}
    hatch_map = {l: get_learner_hatch(l) for l in unique_learners}

    learner_sort_order = [
        'TabPFN-2.5', 'XGBoost-tuned', 'XGBoost',  
        'ExtraTrees-tuned', 'ExtraTrees', 'Ridge', 'CatBoost', 'CatBoost-tuned', 
        'ContextTab', 'TabSTAR', 'Tarte'
    ]
    sort_map = {name: i for i, name in enumerate(learner_sort_order)}
    ordered_learners = unique_learners

    #order by max value of tabular learner
    encoder_order = (
        plot_data.groupby(['encoder','learner'], as_index=False)[score].mean()
        .sort_values(by=['encoder',score], ascending=[True,False])
        .groupby('encoder').head(1)[['encoder',score]]
        .sort_values(by=score, ascending=False)
        .set_index(['encoder']).index
    )


    #separate e2e models from the rest
    e2e_keywords = ['CatBoost', 'ContextTab', 'TabSTAR', 'TabPFN', 'Tarte', 'E2E']

    def is_e2e(name):
        return any(k in str(name) for k in e2e_keywords)

    # 3. Filter into two groups, preserving original order
    e2e_models = [enc for enc in encoder_order if is_e2e(enc)]
    other_models = [enc for enc in encoder_order if not is_e2e(enc)]

    # 4. Concatenate: E2E first (Top), then Others
    new_encoder_order = pd.Index(e2e_models + other_models, name='encoder')

    print("--- New Order (E2E on Top) ---")
    print(new_encoder_order)

    # encoder_order = new_encoder_order

    plt.rcParams['font.sans-serif'] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams['font.family'] = "sans-serif" 
    sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
    sns.set_context("paper")

    fixed_bar_height = 3.5   # Standard height
    inter_group_gap = 6    
    intra_group_sep = 0.01

    # shrink_learners = ['ContextTab', 'TabSTAR', 'TabPFN-2.5']

    fig, ax = plt.subplots(figsize=(3, 6))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    current_y = 0
    yticks_locs = []
    prev_separator = 2.0

    # ============================================
    # 3. DYNAMIC ITERATIVE PLOTTING
    # ============================================
    for group_idx, encoder in enumerate(encoder_order):
        
        enc_df = plot_data[plot_data['encoder'] == encoder]

        present_learners = [l for l in ordered_learners if not enc_df[enc_df['learner'] == l].empty]
        
        print(f"Encoder: {encoder} | Present Learners: {present_learners}")
        
        present_learners.sort(key=lambda x: sort_map.get(x, 999))
        
        # 1. Get dictionary of {learner: mean_score} for ranking
        # learner_scores = enc_df.groupby('learner')[score].mean().to_dict()
        
        # # 2. Group learners into "Families"
        # #    (e.g., 'XGBoost' and 'XGBoost-tuned' -> 'XGBoost')
        # families = {}
        # for l in learner_scores.keys():
        #     base_name = l.replace('-tuned', '')
        #     if base_name not in families:
        #         families[base_name] = []
        #     families[base_name].append(l)
            
        # # 3. Score families by their BEST member
        # #    This ensures the family floats to the top if one version is very good
        # family_max_scores = {
        #     fam: max([learner_scores[m] for m in members]) 
        #     for fam, members in families.items()
        # }
        
        # # 4. Sort families Descending (Best score at top)
        # sorted_families = sorted(families.keys(), key=lambda f: family_max_scores[f], reverse=True)
        
        # # 5. Flatten into final list with Tuned > Default logic
        # present_learners = []
        # for fam in sorted_families:
        #     members = families[fam]
        #     # Sort inside family: 'tuned' returns True (1), default False (0).
        #     # reverse=True puts Tuned first in the list (Top of the visual group).
        #     members.sort(key=lambda x: 'tuned' in x, reverse=True)
        #     present_learners.extend(members)
            
        num_learners = len(present_learners)
        group_top = current_y
        
        for i, learner in enumerate(present_learners):
            learner_data = enc_df[enc_df['learner'] == learner][score]
            mean_score = learner_data.mean()
            
            bar_y = group_top - (i * (fixed_bar_height + intra_group_sep))
            
            visual_height = fixed_bar_height

            c = color_map[learner]
            h = hatch_map[learner]

            # Bars at zorder 3
            ax.barh(bar_y, mean_score, 
                    height=visual_height, 
                    color=c, 
                    edgecolor='white', 
                    hatch=h,
                    linewidth=0.5, 
                    zorder=3
                    )
        
        top_bar_y = group_top
        bottom_bar_y = group_top - ((num_learners - 1) * (fixed_bar_height + intra_group_sep))
        group_center_y = (top_bar_y + bottom_bar_y) / 2
        yticks_locs.append(group_center_y)

        # Define the separator line (midpoint of the gap)
        separator_y = bottom_bar_y - (fixed_bar_height / 2) - (inter_group_gap / 2)
        
        # --- NEW: ALTERNATING BANDS ---
        # We draw a band from the previous separator down to the current separator
        # This wraps the entire block of bars + half the gap above and below
        if group_idx % 2 == 1:
            ax.axhspan(separator_y, prev_separator, color='#e0e0e0', zorder=0, lw=0)

        # Update trackers
        prev_separator = separator_y
        current_y = bottom_bar_y - inter_group_gap - fixed_bar_height

        # is_last_four = group_idx >= (len(encoder_order) - 5)
        # current_gap = inter_group_gap * 1.5 if is_last_four else inter_group_gap

        # separator_y = bottom_bar_y - fixed_bar_height/2 - current_gap/2
        
        # if encoder != encoder_order[-1]:
        #     ax.axhline(separator_y, color='gray', alpha=0.15, linewidth=0.5, zorder=1)

        # current_y = bottom_bar_y - current_gap - fixed_bar_height

    e2e_ordered_names = [
        'ContextTab', 
        'TabSTAR', 
        'CatBoost', 
        'CatBoost-tuned'
    ]
    modular_ordered_names = [
        'TabPFN-2.5', 
        'XGBoost', 
        'XGBoost-tuned', 
        'ExtraTrees', 
        'ExtraTrees-tuned', 
        'Ridge'
    ]
    e2e_list = [l for l in e2e_ordered_names if l in unique_learners]
    modular_list = [l for l in modular_ordered_names if l in unique_learners]

    ax.set_yticks(yticks_locs)
    # in encoder_order, rename TabPFN-2.5 to "E2E TabPFN-2.5"
    encoder_order_renamed = [name if name != 'TabPFN-2.5' else 'E2E TabPFN-2.5' for name in encoder_order]
    # final_labels = [get_encoder_emoji(name) for name in encoder_order_renamed]
    ax.set_yticklabels(encoder_order_renamed, fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel(set_x_label, fontsize=18, ha='right', x=1.0)
    # ax.set_xlabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=18, ha='right', x=1.0)
    ax.set_ylabel("Encoder (Num+Str)", fontsize=18, labelpad=-2, y=0.45)
    # ax.set_ylabel("Encoder (Num+Str)", fontsize=18, labelpad=-2, y=0.45)
    ax.set_xlim(set_xlim_min, set_xlim_max)
    # ax.set_ylim(current_y, 5.0)
    ax.set_ylim(prev_separator + (inter_group_gap/4), 5.0) 
    ax.xaxis.grid(True, linestyle='-', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    def create_patch(l):
        return mpatches.Patch(
            facecolor=color_map[l], edgecolor='white', hatch=hatch_map[l], label=l, linewidth=0.5
        )

    handles_e2e = [create_patch(l) for l in e2e_list]
    handles_mod = [create_patch(l) for l in modular_list]

    header_e2e = mpatches.Patch(visible=False, label=r"$\bf{E2E\ Models}$")
    header_mod = mpatches.Patch(visible=False, label=r"$\bf{Modular\ Learners}$")
    header_base = mpatches.Patch(visible=False, label=r"$\bf{Baselines}$")

    # Combine final list
    final_handles = [header_e2e] + handles_e2e + [header_mod] + handles_mod

    legend = ax.legend(
        handles=final_handles,
        fontsize=7.5,
        loc='lower right',
        bbox_to_anchor=bbox_to_anchor, 
        # bbox_to_anchor=(1.2, 0.0), 
        frameon=True,
        framealpha=1, 
        facecolor='white',     # Background color of the box
        edgecolor='black',
        fancybox=False,        # Set to False for sharp corners, True for rounded
        borderpad=0.6,
        handletextpad=0.4,
        labelspacing=0.4
    )

    # Align text to the left so headers look correct
    for text in legend.get_texts():
        text.set_ha('left')

    sns.despine(left=False, bottom=False) # Keep right spine visible

    ax.tick_params(axis='y', zorder=10)

    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'avg_performance_{dtype}_{score}_{select_llm}_{task}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()



'''
ALL LLMS
Average performance when using combined (Num+Str) features.
Encoders, e2e models and all LLMs
'''

dtype = 'Num+Str'
select_llm = 'all_llm' #'all_llm', 'top3', 'selected'
score = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']
task = 'all_task' # 'all_task', 'classification', 'regression'

if (score == 'score') & (task == 'all_task'):
    set_xlim_min = 0.55
    set_xlim_max = 0.75
elif (score == 'score_norm') & (task == 'all_task'):
    set_xlim_min = 0.6
    set_xlim_max = 1.0
elif (score == 'score') & (task == 'regression'):
    set_xlim_min = 0.45
    set_xlim_max = 0.72
elif (score == 'score_norm') & (task == 'regression'):
    set_xlim_min = 0.55
    set_xlim_max = 0.95
elif (score == 'score') & (task == 'classification'):
    set_xlim_min = 0.75
    set_xlim_max = 0.86
elif (score == 'score_norm') & (task == 'classification'):
    set_xlim_min = 0.4
    set_xlim_max = 0.95


if select_llm == 'all_llm':
    df = results.copy()
elif select_llm == 'top3':
    df = results[results['encoder'].isin(selected_encoders_top3)].copy()
else:
    df = results[results['encoder'].isin(selected_encoders)].copy()

  
if task == 'all_task':
    set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)"
    pass
elif task == 'regression':
    df = df[df['task'] == 'regression'].copy()
    set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($R^2$)"
else: 
     set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($AUC$)"
     df = df[df['task'] != 'regression'].copy()

plot_data = df[(df['dtype'] == dtype)].copy()
unique_learners = sorted(df['learner'].unique())

color_map = {l: get_learner_color_simple(l) for l in unique_learners}
hatch_map = {l: get_learner_hatch(l) for l in unique_learners}

learner_sort_order = [
    'TabPFN-2.5', 'XGBoost-tuned', 'XGBoost',  
    'ExtraTrees-tuned', 'ExtraTrees', 'Ridge', 'CatBoost', 'CatBoost-tuned', 
    'ContextTab', 'TabSTAR', 'Tarte'
]
sort_map = {name: i for i, name in enumerate(learner_sort_order)}
ordered_learners = unique_learners

encoder_order = (
    plot_data.groupby(['encoder','learner'], as_index=False)[score].mean()
    .sort_values(by=['encoder',score], ascending=[True,False])
    .groupby('encoder').head(1)[['encoder',score]]
    .sort_values(by=score, ascending=False)
    .set_index(['encoder']).index
)

#separate e2e models from the rest
e2e_keywords = ['CatBoost', 'ContextTab', 'TabSTAR', 'TabPFN', 'Tarte', 'E2E']

def is_e2e(name):
    return any(k in str(name) for k in e2e_keywords)

# 3. Filter into two groups, preserving original order
e2e_models = [enc for enc in encoder_order if is_e2e(enc)]
other_models = [enc for enc in encoder_order if not is_e2e(enc)]

# 4. Concatenate: E2E first (Top), then Others
new_encoder_order = pd.Index(e2e_models + other_models, name='encoder')

print("--- New Order (E2E on Top) ---")
print(new_encoder_order)

# encoder_order = new_encoder_order

plt.rcParams['font.sans-serif'] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams['font.family'] = "sans-serif" 
sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
sns.set_context("paper")

fixed_bar_height = 6   # Standard height
inter_group_gap = 4    
intra_group_sep = 0.01

# shrink_learners = ['ContextTab', 'TabSTAR', 'TabPFN-2.5']

fig, ax = plt.subplots(figsize=(15, 40))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

current_y = 0
yticks_locs = []
prev_separator = 2.0

for group_idx, encoder in enumerate(encoder_order):
    
    enc_df = plot_data[plot_data['encoder'] == encoder]

    present_learners = [l for l in ordered_learners if not enc_df[enc_df['learner'] == l].empty]
    
    print(f"Encoder: {encoder} | Present Learners: {present_learners}")
    
    present_learners.sort(key=lambda x: sort_map.get(x, 999))
        
    num_learners = len(present_learners)
    group_top = current_y
    
    for i, learner in enumerate(present_learners):
        learner_data = enc_df[enc_df['learner'] == learner][score]
        mean_score = learner_data.mean()
        
        bar_y = group_top - (i * (fixed_bar_height + intra_group_sep))
        
        visual_height = fixed_bar_height

        c = color_map[learner]
        h = hatch_map[learner]

        # Bars at zorder 3
        ax.barh(bar_y, mean_score, 
                height=visual_height, 
                color=c, 
                edgecolor='white', 
                hatch=h,
                linewidth=0.5, 
                zorder=3
                )
    
    top_bar_y = group_top
    bottom_bar_y = group_top - ((num_learners - 1) * (fixed_bar_height + intra_group_sep))
    group_center_y = (top_bar_y + bottom_bar_y) / 2
    yticks_locs.append(group_center_y)

    separator_y = bottom_bar_y - (fixed_bar_height / 2) - (inter_group_gap / 2)
    
    if group_idx % 2 == 1:
        ax.axhspan(separator_y, prev_separator, color='#e0e0e0', zorder=0, lw=0)

    prev_separator = separator_y
    current_y = bottom_bar_y - inter_group_gap - fixed_bar_height

e2e_ordered_names = [
    'ContextTab', 
    'TabSTAR', 
    'CatBoost', 
    'CatBoost-tuned'
]
modular_ordered_names = [
    'TabPFN-2.5', 
    'XGBoost', 
    'XGBoost-tuned', 
    'ExtraTrees', 
    'ExtraTrees-tuned', 
    'Ridge'
]
e2e_list = [l for l in e2e_ordered_names if l in unique_learners]
modular_list = [l for l in modular_ordered_names if l in unique_learners]

ax.set_yticks(yticks_locs)
# in encoder_order, rename TabPFN-2.5 to "E2E TabPFN-2.5"
encoder_order_renamed = [name if name != 'TabPFN-2.5' else 'E2E TabPFN-2.5' for name in encoder_order]
# final_labels = [get_encoder_emoji(name) for name in encoder_order_renamed]
ax.set_yticklabels(encoder_order_renamed, fontsize=40)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='x', labelsize=40)
ax.set_xlabel(set_x_label, fontsize=40, ha='right', x=0.5)
# ax.set_xlabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=18, ha='right', x=1.0)
ax.set_ylabel("Encoder (Num+Str)", fontsize=40, labelpad=-5, y=0.45)
# ax.set_ylabel("Encoder (Num+Str)", fontsize=18, labelpad=-2, y=0.45)
ax.set_xlim(set_xlim_min, set_xlim_max)
# ax.set_ylim(current_y, 5.0)
ax.set_ylim(prev_separator + (inter_group_gap/4), 5.0) 
ax.xaxis.grid(True, linestyle='-', alpha=0.2, zorder=0)
ax.set_axisbelow(True)

def create_patch(l):
    return mpatches.Patch(
        facecolor=color_map[l], edgecolor='white', hatch=hatch_map[l], label=l, linewidth=0.5
    )

handles_e2e = [create_patch(l) for l in e2e_list]
handles_mod = [create_patch(l) for l in modular_list]

header_e2e = mpatches.Patch(visible=False, label=r"$\bf{E2E\ Models}$")
header_mod = mpatches.Patch(visible=False, label=r"$\bf{Modular\ Learners}$")
header_base = mpatches.Patch(visible=False, label=r"$\bf{Baselines}$")

# Combine final list
final_handles = [header_e2e] + handles_e2e + [header_mod] + handles_mod

legend = ax.legend(
    handles=final_handles,
    fontsize=25,
    loc='lower right',
    bbox_to_anchor=(0.95, 0.0), 
    # bbox_to_anchor=(1.2, 0.0), 
    frameon=True,
    framealpha=1, 
    facecolor='white',     # Background color of the box
    edgecolor='black',
    fancybox=False,        # Set to False for sharp corners, True for rounded
    borderpad=0.6,
    handletextpad=0.4,
    labelspacing=0.4
)

# Align text to the left so headers look correct
for text in legend.get_texts():
    text.set_ha('left')

sns.despine(left=False, bottom=False) # Keep right spine visible

ax.tick_params(axis='y', zorder=10)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_performance_{dtype}_{score}_all_llm_{task}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()



'''
Critical Difference Diagram - selected LLMs, baselines and E2E models
'''

dtype = 'num-str'
score = score_list[0]  
level = 'data_name'  # 'fold_index' or 'data_name'
for task in ['all_task', 'classification', 'regression']:

    # task = 'regression'

    if task == 'classification':
        df_score = results[results['encoder'].isin(selected_encoders)].copy()
        df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)
        df_score = df_score[df_score['task'] != 'regression'].copy()
    elif task == 'regression':
        df_score = results[results['encoder'].isin(selected_encoders)].copy()
        df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)
        df_score = df_score[df_score['task'] == 'regression'].copy()
    else:
        df_score = results[results['encoder'].isin(selected_encoders)].copy()
        df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)

    # drop TabPFN-2.5 encoder
    df_score = df_score[(df_score['method'] != 'num-str_tabpfn_tabpfn_default')]

    df_score['method'] = df_score['method'].str.replace(f'{dtype}_', '', regex=False)


    # Apply the cleaning logic
    df_score['method'] = df_score['method'].apply(clean_method_name)


    # Ranks and test results
    if level == 'fold_index':
        df_score_final = _generate_marker(df_score)

        avg_rank = (
            df_score_final.groupby(["marker"], group_keys=True)  # marker
            [score].rank(pct=False, ascending=False)
            .groupby(df_score_final.method)
            .mean()
        )
        avg_rank = -1 * avg_rank

        test_results_CF = sp.posthoc_conover_friedman(
            df_score_final,
            melted=True,
            block_col="marker",
            block_id_col="marker",
            group_col="method",
            y_col=score,
        )
    else:
        df_agg = df_score.groupby(["data_name", "method"], as_index=False)[score].mean()
        df_agg["rank"] = df_agg.groupby("data_name")[score].rank(ascending=False)
        avg_rank = df_agg.groupby(['method'])['rank'].mean()

        avg_rank = -1 * avg_rank

        df_agg = df_score.groupby(["data_name", "method"], as_index=False)[score].mean()

        # 2. Pivot to check for missing values (Incomplete Blocks)
        df_pivot = df_agg.pivot(index="data_name", columns="method", values=score)

        # 3. Check if any method is missing for any dataset
        if df_pivot.isnull().values.any():
            print(f"Warning: Dropping {df_pivot.isnull().any(axis=1).sum()} datasets with missing method scores.")
            # Friedman test REQUIRES complete blocks. We must drop datasets where ANY method failed.
            df_pivot = df_pivot.dropna(axis=0)

        df_clean = df_pivot.reset_index().melt(id_vars="data_name", var_name="method", value_name=score)

        # 5. Run the Post-hoc Test
        test_results_CF = sp.posthoc_conover_friedman(
            df_clean,
            melted=True,
            block_col="data_name",
            block_id_col="data_name",
            group_col="method",
            y_col=score
        )

    # test_results_CF = sp.posthoc_conover_friedman(
    #     df_score_final,
    #     melted=False,
    #     block_col="data_name",
    #     block_id_col="data_name",
    #     group_col="method",
    #     y_col=score,
    # )

    # pivot_df_wilson_holm = df_score_final.pivot_table(
    #     index='marker', 
    #     columns='method', 
    #     values=score, 
    #     aggfunc='mean'
    # )

    # assert not pivot_df_wilson_holm.isna().any().any()

    # pvals = sp.posthoc_wilcoxon(
    #     pivot_df_wilson_holm.values.T,
    #     p_adjust="holm"
    # )

    # test_results_wilson_holm = pvals.copy()
    # test_results_wilson_holm.index = pivot_df_wilson_holm.columns
    # test_results_wilson_holm.columns = pivot_df_wilson_holm.columns

    # test_results = test_results_wilson_holm.copy()
    test_results = test_results_CF.copy()

    test_results = test_results.replace(0, 1e-100)  # Required for visualization


    # Lines
    # line_style = {model: "-" for model in models}
    models = df_score.method.unique()
    line_style = {model: "-" for model in models}
    for model in models:
        if "TargetEncoder" in model:
            line_style[model] = "--"
        if "LLM" in model:
            line_style[model] = "-."

    print(f"Total models: {len(models)}")
    palette_by_learner = {}
    palette_by_encoder = {}

    for model in models:
        # Assuming format is "Encoder - Learner" based on your previous split code
        parts = model.split(' - ')
        encoder_part = parts[0]
        learner_part = parts[-1]
        
        # 1. Learner Palette
        palette_by_learner[model] = get_learner_color_simple(learner_part)
        
        # 2. Encoder Palette
        palette_by_encoder[model] = get_encoder_color(encoder_part)

    name_map = {}
    for model, rank_val in avg_rank.items():
        # rank_val is negative, so we take abs()
        name_map[model] = f"{model} ({abs(rank_val):.1f})"

    # Rename Index/Columns in Data
    avg_rank_plot = avg_rank.rename(index=name_map)
    test_results_plot = test_results.rename(index=name_map, columns=name_map)

    # Update Palettes & Styles to match new names
    palette_by_learner_plot = {name_map[k]: v for k, v in palette_by_learner.items() if k in name_map}
    palette_by_encoder_plot = {name_map[k]: v for k, v in palette_by_encoder.items() if k in name_map}
    line_style_plot = {name_map[k]: v for k, v in line_style.items() if k in name_map}

    # ==========================================
    # 4. PLOT VERSION 1: COLORED BY LEARNER
    # ==========================================
    sns.set_theme(style="white", font_scale=1)

    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 5))

    critical_difference_diagram(
        ranks=avg_rank_plot,
        sig_matrix=test_results_plot,
        label_fmt_left="{label}",
        label_fmt_right=" {label}",
        label_props={"fontsize": 10},
        crossbar_props={"color": "black", "linewidth": 1},
        marker_props={"marker": ""},
        elbow_props={"linewidth": 1.5},
        text_h_margin=1.2,
        
        color_palette=palette_by_learner_plot,  # <--- APPLIED HERE
        line_style=line_style_plot,
        
        bold_control=True,
        v_space=4,
        ax=ax1,
    )
    n_models = len(models)

    # # 1. Round up to the nearest multiple of 5 to get the maximum tick
    # max_tick = math.ceil(n_models / 5) * 5

    # # 2. Generate the descending range with a step of 5
    # #    range(start, stop, step) -> stop is exclusive, so we go down to 0 to include 5
    # tick_range = list(range(max_tick, 0, -5))

    # # 3. Add None at the beginning as requested
    # xticklabels = [None] + tick_range
    # # Adjust ticks based on your rank range (e.g. if max rank is 40)
    # ax1.set_xticklabels(xticklabels, fontsize=15)

    # Create ticks every 5 steps, but ensuring 1 and Max are included
    major_ticks = list(range(0, n_models-4, 5)) 
    if n_models not in major_ticks:
        major_ticks.append(n_models) 

    # Filter 0 (ranks start at 1) and sort
    major_ticks = sorted([t for t in major_ticks if t > 0])

    # Convert to negative space (since CD diagram uses negative ranks)
    plot_ticks = [-t for t in major_ticks]

    # Apply ticks
    ax1.set_xticks(plot_ticks)
    ax1.set_xticklabels(major_ticks, fontsize=12)

    # Force limits to show full range (Left=Max Rank, Right=Rank 1)
    # In negative space: Left = -N, Right = 0
    ax1.set_xlim(-(n_models - 4), 0)


    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'critical_difference_diagram_custom_Jun_selectedLLMs_friedman_colorbylearner_{level}_{score}_{task}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()



    # ==========================================
    # 5. PLOT VERSION 2: COLORED BY ENCODER
    # ==========================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 5))

    critical_difference_diagram(
        ranks=avg_rank_plot,
        sig_matrix=test_results_plot,
        label_fmt_left="{label}",
        label_fmt_right=" {label}",
        label_props={"fontsize": 10},
        crossbar_props={"color": "black", "linewidth": 1},
        marker_props={"marker": ""},
        elbow_props={"linewidth": 1.5},
        text_h_margin=1.2,
        
        color_palette=palette_by_encoder_plot,  # <--- APPLIED HERE
        line_style=line_style_plot,
        
        bold_control=True,
        v_space=4,
        ax=ax2,
    )
    n_models = len(models)

    # 1. Round up to the nearest multiple of 5 to get the maximum tick
    # max_tick = math.ceil(n_models / 5) * 5

    # # 2. Generate the descending range with a step of 5
    # #    range(start, stop, step) -> stop is exclusive, so we go down to 0 to include 5
    # tick_range = list(range(max_tick, 0, -5))

    # # 3. Add None at the beginning as requested
    # xticklabels = [None] + tick_range
    # ax2.set_xticklabels(xticklabels, fontsize=15)

    major_ticks = list(range(0, n_models-4, 5)) 
    if n_models not in major_ticks:
        major_ticks.append(n_models) 

    # Filter 0 (ranks start at 1) and sort
    major_ticks = sorted([t for t in major_ticks if t > 0])

    # Convert to negative space (since CD diagram uses negative ranks)
    plot_ticks = [-t for t in major_ticks]

    # Apply ticks
    ax2.set_xticks(plot_ticks)
    ax2.set_xticklabels(major_ticks, fontsize=12)

    # Force limits to show full range (Left=Max Rank, Right=Rank 1)
    # In negative space: Left = -N, Right = 0
    ax2.set_xlim(-(n_models - 4), 0)


    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'critical_difference_diagram_custom_Jun_selectedLLMs_friedman_colorbyencoder_{level}_{score}_{task}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()

'''
COMPARATIVE PARETO PLOTS - split into 2 
using big LLMs list, baselines and E2E models
legend should be included
'''

dtype = 'num-str'
progressive_transparency = False

for metric in score_list:

    # metric = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']
    
    Y_METRIC = metric

    # 1. Update Data Aggregation for the current metric
    if 'encoder_learner' in results.columns:
        agg_cols = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
        group_cols = ['encoder_learner', 'encoder', 'learner']
        df_agg = results[(results['method'].str.contains(f'{dtype}_')) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(group_cols)[agg_cols].median().reset_index()
        
    # drop TabPFN-2.5 encoder

    HIGHER_SCORE_IS_BETTER = True

    # We need explicit dictionaries for every unique value in the dataframe
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()

    # A. Marker Palette (Always based on Learner)
    # Maps "XGBoost" -> 's', "XGBoost-tuned" -> 's'
    learner_markers_dict = {L: get_learner_marker(L) for L in unique_learners}

    # B. Color Palettes
    # Palette for Right Plot (Hue = Learner)
    learner_palette_dict = {L: get_learner_color_simple(L) for L in unique_learners}

    # Palette for Left Plot (Hue = Encoder)
    encoder_palette_dict = {E: get_encoder_color(E) for E in unique_encoders}

    # 2. Re-initialize Plotting Parameters
    sns.set_style("white")
    ROW_METRICS = ['inference_time_per_1k', 'run_time_per_1k']
    COL_FACTORS = ['encoder', 'learner']
    COL_TITLES = ['Encoder', 'Learner'] 
    ROW_LABELS = ['Inference Time per 1K samples (s)', 'Total Run Time per 1K samples (s)']

    # --- PALETTE PREPARATION ---
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()

    learner_pal = {L: get_learner_color_simple(L) for L in unique_learners}
    encoder_pal = {E: get_encoder_color(E) for E in unique_encoders}
    learner_markers = {L: get_learner_marker(L) for L in unique_learners}

    # 3. INDENTED PLOTTING LOOP
    for row_idx, x_metric in enumerate(ROW_METRICS):
        fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
        pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, True)

        # save pareto_df to latex for later analysis
        today_date = time.strftime("%Y-%m-%d")
        filename = f"pareto_frontier_{Y_METRIC}_vs_{x_metric}_{today_date}.tex"
        save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{TODAYS_FOLDER}/{filename}'
        with open(save_path, 'w') as f:
            f.write(pareto_df.to_latex(index=False))

        # =======================================================
        # NEW CODE: PRINT PARETO FRONTIER DETAILS
        # =======================================================
        print(f"\n" + "="*60)
        print(f"PARETO FRONTIER: {Y_METRIC} (Max) vs {x_metric} (Min)")
        print("="*60)
        
        # Select and rename columns for cleaner output
        display_cols = ['encoder', 'learner', x_metric, Y_METRIC]
        
        # Sort by the X metric (Time) to show the progression from Fast->Slow
        frontier_view = pareto_df[display_cols].sort_values(by=x_metric)
        
        # Print formatted table
        print(f"{'Encoder':<25} | {'Learner':<20} | {'Time (s)':<10} | {'Score':<8}")
        print("-" * 75)
        
        for _, row in frontier_view.iterrows():
            e = str(row['encoder'])[:25] # Truncate long names for display
            l = str(row['learner'])[:20]
            t = row[x_metric]
            s = row[Y_METRIC]
            print(f"{e:<25} | {l:<20} | {t:<10.4f} | {s:<8.4f}")
        print("="*60 + "\n")
        # =======================================================

        y_bottom = df_agg[Y_METRIC].min()  # Or hardcode e.g., 0.0
        # Right X: Use a value larger than your max X to ensure it hits the edge
        x_right_edge = df_agg[x_metric].max() * 5.0 

        # 3. Create extension points
        # Point A: (First X, Bottom Y) -> Creates vertical line from axis up to first point
        start_point = pd.DataFrame({
            x_metric: [pareto_df[x_metric].iloc[0]], 
            Y_METRIC: [y_bottom]
        })
        
        # Point B: (Right Edge X, Last Y) -> Creates horizontal line to the right
        end_point = pd.DataFrame({
            x_metric: [x_right_edge], 
            Y_METRIC: [pareto_df[Y_METRIC].iloc[-1]]
        })

        pareto_extended = pd.concat([start_point, pareto_df, end_point], ignore_index=True)

        for col_idx, factor in enumerate(COL_FACTORS):
            ax = axes[col_idx]

            # --- NEW CODE: CONNECT ENCODER DOTS ---
            # Only apply this to the 'encoder' plot (left side)
            # if factor == 'encoder':
            #     for enc in unique_encoders:
            #         # 1. Get all points for this specific encoder
            #         enc_subset = df_agg[df_agg['encoder'] == enc]
                    
            #         # 2. Sort by the X-axis metric so the line is drawn sequentially
            #         enc_subset = enc_subset.sort_values(by=x_metric)
                    
            #         # 3. Draw line if there are multiple points
            #         if len(enc_subset) > 1:
            #             ax.plot(
            #                 enc_subset[x_metric], 
            #                 enc_subset[Y_METRIC], 
            #                 color=encoder_pal[enc], 
            #                 linewidth=1.0,       # Thin line
            #                 alpha=0.4,           # Semi-transparent to not distract
            #                 zorder=1             # Important: Draw BEHIND the markers
            #             )
            # --------------------------------------
            
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3, zorder=0)

            # --- SETUP CONTEXT ---
            if factor == 'encoder':
                current_palette = encoder_pal
                hue_col = 'encoder'
                style_col = 'learner'
            else:
                current_palette = learner_pal
                hue_col = 'learner'
                style_col = 'learner'

            if progressive_transparency == False:
                # --- SPLIT DATA ---
                mask_tuned = df_agg['learner'].str.contains('tuned')
                df_default = df_agg[~mask_tuned]
                df_tuned = df_agg[mask_tuned]

                # --- PLOT 1: DEFAULTS (Full Fill) ---
                sns.lineplot(
                    data=df_default, x=x_metric, y=Y_METRIC,
                    hue=hue_col, style=style_col,
                    palette=current_palette, markers=learner_markers,
                    dashes=False, estimator=None, lw=0, markersize=9,
                    ax=ax, legend=False, # Disable auto-legend
                    **{'fillstyle': 'full', 'markeredgewidth': 1.0, 'markeredgecolor': 'white', 'markeredgecolor': 'black'}
                )
                
                # --- PLOT 2: TUNED (Half Fill with Black Contour) ---
                sns.lineplot(
                    data=df_tuned, x=x_metric, y=Y_METRIC,
                    hue=hue_col, style=style_col,
                    palette=current_palette, markers=learner_markers,
                    dashes=False, estimator=None, lw=0, markersize=9,
                    ax=ax, legend=False, # Disable auto-legend
                    # Key fix: Black edge to show the contour of the half-filled shape
                    **{'fillstyle': 'left', 'markerfacecoloralt': 'white', 
                    'markeredgecolor': 'black', 'markeredgewidth': 1.0} 
                )
                
                # --- PARETO LINE ---
                # ax.step(
                #     pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
                #     linestyle='--', color='black', linewidth=1.2, zorder=0
                # )
                ax.step(
                    pareto_extended[x_metric], 
                    pareto_extended[Y_METRIC], 
                    where='post',
                    linestyle='--', 
                    color='black', 
                    linewidth=1.2, 
                    zorder=0
                )
                
                # --- AXIS FORMATTING ---
                ax.set_box_aspect(1) 
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                # ax.set_title(COL_TITLES[col_idx], fontsize=12, fontweight='bold', pad=105) 
                # Main Title
                if factor == 'encoder':
                    ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                                loc='left', y=1.68, x=-0.25) 
                    subtitle_text = "(shape = learner, color = encoder)"
                    ax.text(0.2, 1.72,  subtitle_text, transform=ax.transAxes, 
                            fontsize=10, style='italic', color='#333333')
                else:
                    ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                                loc='left', y=1.68, x=-0.15) 
                    subtitle_text = "(shape = learner, color = learner)" 
                    ax.text(0.25, 1.72,  subtitle_text, transform=ax.transAxes, 
                            fontsize=10, style='italic', color='#333333')

                ax.set_xlabel('')
                ax.set_ylim(bottom=y_bottom)

                if col_idx == 0:
                    ax.set_ylabel(f'Avg {Y_METRIC_LABELS[Y_METRIC]} ($R^2$ & AUC)', fontsize=10)
                else:
                    ax.set_ylabel('')
                
                # sns.despine(ax=ax, trim=False, offset=0)
                # WITH THIS MANUAL SPINE CONFIGURATION:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Force Left and Bottom spines to stay at the "axes" 0-coordinate
                # This ensures they meet perfectly at the corner
                ax.spines['left'].set_position(('axes', 0))
                ax.spines['bottom'].set_position(('axes', 0))
                
                # Ensure they extend fully
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                # 1. Map Encoder Names to Learner Keys
                e2e_map = {
                    'CatBoost': 'CatBoost',
                    'ContextTab': 'ContextTab',
                    'TabSTAR': 'TabSTAR',
                    'TabPFN-2.5': 'TabPFN',
                    # 'Tarte': 'Tarte'
                }

                # 2. Create a 'markers' dictionary for the Encoders
                # Start with default 'D' (Diamond) for everyone
                encoder_markers = {enc: 'D' for enc in unique_encoders}

                # Overwrite the E2E models with their Learner shapes
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_markers:
                        encoder_markers[enc_name] = learner_shapes[learner_name]

                # IMPORTANT: Ensure your encoder_pal (colors) also matches the learner colors for these 3
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_pal:
                        encoder_pal[enc_name] = learner_colors[learner_name]

                # --- MANUAL LEGEND GENERATION ---
                if factor == 'encoder':
                    std_encoders = [e for e in unique_encoders if e not in e2e_map]
                    e2e_encoders = [e for e in unique_encoders if e in e2e_map]
                    
                    # 2. Sort both lists independently
                    std_encoders.sort()
                    e2e_encoders.sort()
                    
                    # 3. Combine: Standard first, E2E last
                    sorted_encoder_list = std_encoders + e2e_encoders

                    enc_handles = []
                    for enc in sorted_encoder_list:
                        current_color = encoder_pal[enc]
                        # current_marker = 'D'  # Default Diamond
                        if enc in e2e_map:
                            learner_key = e2e_map[enc]
                            current_color = learner_colors[learner_key] # Force learner color
                            current_marker = learner_shapes[learner_key] # Force shape color
                            h = mlines.Line2D([], [], color=current_color, marker=current_marker, 
                                        linestyle='', markersize=8, label=enc)
                        else:
                            h = mpatches.Patch(color=current_color, label=enc)
                        enc_handles.append(h)
                                        
                    ax.legend(
                        handles=enc_handles,
                        loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.9  
                    )
                
                else:
                    # RIGHT LEGEND: Shape/Color by Learner, Grouped (Base, Tuned)
                    # 1. Define specific sorting order
                    base_order = ['Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 
                                'TabSTAR', 'ContextTab', 'CatBoost', 
                                # 'Tarte'
                                ]
                    
                    sorted_learners = []
                    # Add families in order: Base then Tuned
                    for base in base_order:
                        if base in unique_learners: 
                            sorted_learners.append(base)
                        if f"{base}-tuned" in unique_learners:
                            sorted_learners.append(f"{base}-tuned")
                    
                    # Catch leftovers
                    for l in unique_learners:
                        if l not in sorted_learners:
                            sorted_learners.append(l)

                    # 2. Build Handles
                    lrn_handles = []
                    for lrn in sorted_learners:
                        is_tuned = 'tuned' in lrn
                        marker = learner_markers[lrn]
                        color = learner_pal[lrn]
                        
                        if is_tuned:
                            # Half-filled with black edge
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='left', markerfacecoloralt='white',
                                            markeredgecolor='black', markeredgewidth=1.0)
                        else:
                            # Full filled
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='full', markeredgecolor='black',markeredgewidth=1.0)
                        lrn_handles.append(h)

                    ax.legend(
                        handles=lrn_handles,
                        loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.4
                    )
            else:
                # --- PREPARE LAYERS (Progressive Transparency) ---
                base_priority = ['Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 
                                'TabSTAR', 'ContextTab', 'CatBoost'
                                # , 'Tarte'
                                ]
                
                # KEY CHANGE 1: STRICT SORTING
                # Primary Sort: Base Family order (CatBoost bottom, Tarte top)
                # Secondary Sort: 'tuned' status (Default=0 (Bottom), Tuned=1 (Top))
                z_learners = sorted(list(unique_learners), 
                                    key=lambda x: (
                                        base_priority.index(x.split('-')[0]) if x.split('-')[0] in base_priority else 999,
                                        1 if 'tuned' in x else 0 
                                    ))
                
                total_layers = len(z_learners)

                # 2. Iterate and Plot Layer by Layer
                for i, lrn_name in enumerate(z_learners):
                    
                    # A. Calculate Progressive Alpha (Opacity)
                    # Bottom Layer (i=0) -> 1.0 (Opaque)
                    # Top Layer (i=Max)  -> 0.4 (Transparent)
                    if total_layers > 1:
                        curr_alpha = 1.0 - (0.6 * (i / (total_layers - 1)))
                    else:
                        curr_alpha = 1.0

                    subset = df_agg[df_agg['learner'] == lrn_name]
                    if subset.empty:
                        continue

                    is_tuned = 'tuned' in lrn_name
                    
                    if is_tuned:
                        # KEY CHANGE 2: Transparent Empty Half
                        # Changed markerfacecoloralt from 'white' to 'none'
                        style_kwargs = {
                            'fillstyle': 'left', 
                            'markerfacecoloralt': 'none', # See-through!
                            'markeredgecolor': 'black', 
                            'markeredgewidth': 1.0
                        }
                    else:
                        # Default: Full-filled
                        style_kwargs = {
                            'fillstyle': 'full', 
                            'markeredgecolor': 'black', 
                            'markeredgewidth': 1.0
                        }

                    sns.lineplot(
                        data=subset, x=x_metric, y=Y_METRIC,
                        hue=hue_col, style=style_col,
                        palette=current_palette, markers=learner_markers,
                        dashes=False, estimator=None, lw=0, markersize=9,
                        ax=ax, legend=False,
                        alpha=curr_alpha, # Apply calculated transparency
                        **style_kwargs
                    )
                
                # --- PARETO LINE ---
                ax.step(
                    pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
                    linestyle='--', color='black', linewidth=1.2, zorder=0
                )
                
                # --- AXIS FORMATTING ---
                ax.set_box_aspect(1) 
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                             loc='left', y=1.68) 
            
                # 2. Subtitle (Push slightly below title but above legend, e.g., y=1.35)
                subtitle_text = "(shape = learner, color = encoder)" if factor == 'encoder' else "(shape = learner, color = learner)" 
                ax.text(0.4, 1.72,  subtitle_text, transform=ax.transAxes, 
                        fontsize=10, style='italic', color='#333333')

                ax.set_xlabel('')
                if col_idx == 0:
                    ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=10)
                else:
                    ax.set_ylabel('')

                # sns.despine(ax=ax, trim=False, offset=0) 
                # WITH THIS MANUAL SPINE CONFIGURATION:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Force Left and Bottom spines to stay at the "axes" 0-coordinate
                # This ensures they meet perfectly at the corner
                ax.spines['left'].set_position(('axes', 0))
                ax.spines['bottom'].set_position(('axes', 0))
                
                # Ensure they extend fully
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                e2e_map = {
                    'CatBoost': 'CatBoost',
                    'ContextTab': 'ContextTab',
                    'TabSTAR': 'TabSTAR',
                    'TabPFN-2.5': 'TabPFN',
                    # 'Tarte': 'Tarte'
                }

                # 2. Create a 'markers' dictionary for the Encoders
                # Start with default 'D' (Diamond) for everyone
                encoder_markers = {enc: 'D' for enc in unique_encoders}

                # Overwrite the E2E models with their Learner shapes
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_markers:
                        encoder_markers[enc_name] = learner_shapes[learner_name]

                # IMPORTANT: Ensure your encoder_pal (colors) also matches the learner colors for these 3
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_pal:
                        encoder_pal[enc_name] = learner_colors[learner_name]

                # --- LEGEND GENERATION ---
                if factor == 'encoder':
                    std_encoders = [e for e in unique_encoders if e not in e2e_map]
                    e2e_encoders = [e for e in unique_encoders if e in e2e_map]
                    
                    # 2. Sort both lists independently
                    std_encoders.sort()
                    e2e_encoders.sort()
                    
                    # 3. Combine: Standard first, E2E last
                    sorted_encoder_list = std_encoders + e2e_encoders

                    enc_handles = []
                    for enc in sorted_encoder_list:
                        current_color = encoder_pal[enc]
                        # current_marker = 'D'  # Default Diamond
                        if enc in e2e_map:
                            learner_key = e2e_map[enc]
                            current_color = learner_colors[learner_key] # Force learner color
                            current_marker = learner_shapes[learner_key] # Force shape color
                            h = mlines.Line2D([], [], color=current_color, marker=current_marker, 
                                        linestyle='', markersize=8, label=enc)
                        else:
                            h = mpatches.Patch(color=current_color, label=enc)
                        enc_handles.append(h)
                    
                    ax.legend(
                        handles=enc_handles,
                        loc='lower center', bbox_to_anchor=(0.5, 1.05),
                        ncol=2, fontsize=8, frameon=False, 
                        title=None, columnspacing=0.8
                    )
                else:
                    # Legend Logic (Same as before)
                    sorted_learners_leg = []
                    for base in base_priority:
                        if base in unique_learners: sorted_learners_leg.append(base)
                        if f"{base}-tuned" in unique_learners: sorted_learners_leg.append(f"{base}-tuned")
                    for l in unique_learners:
                        if l not in sorted_learners_leg: sorted_learners_leg.append(l)

                    lrn_handles = []
                    for lrn in sorted_learners_leg:
                        is_tuned = 'tuned' in lrn
                        marker = learner_markers[lrn]
                        color = learner_pal[lrn]
                        # Keep legend opaque (alpha=1) for clarity
                        if is_tuned:
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='left', markerfacecoloralt='white', # Keep white for legend visibility
                                            markeredgecolor='black', markeredgewidth=1.0)
                        else:
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='full', markeredgecolor='black', markeredgewidth=1.0)
                        lrn_handles.append(h)

                    ax.legend(
                        handles=lrn_handles,
                        loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.4
                    )

        fig.text(0.5, 0.12, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=10)
        plt.subplots_adjust(bottom=0.13, top=0.75, wspace=0.50, left=0.05, right=0.99)


        today_date = time.strftime("%Y-%m-%d")
        format = 'pdf'
        PIC_NAME = f'comparative_pareto_optimality_plot_1Ksample_scale_progr_transparency_{progressive_transparency}_{Y_METRIC}_{x_metric}_{today_date}.{format}'
        plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
        plt.show()


'''
NUM ONLY
COMPARATIVE PARETO PLOTS - split into 2 
using big LLMs list, baselines and E2E models
legend should be included
'''

dtype = 'num-only'
progressive_transparency = False

for metric in score_list:
    
    Y_METRIC = metric

    # 1. Update Data Aggregation for the current metric
    if 'encoder_learner' in results.columns:
        agg_cols = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
        group_cols = ['encoder_learner', 'encoder', 'learner']
        df_agg = results[(results['method'].str.contains(f'{dtype}_')) & (results['encoder'].isin(selected_encoders))].groupby(group_cols)[agg_cols].median().reset_index()
        
    HIGHER_SCORE_IS_BETTER = True

    # We need explicit dictionaries for every unique value in the dataframe
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()

    # A. Marker Palette (Always based on Learner)
    # Maps "XGBoost" -> 's', "XGBoost-tuned" -> 's'
    learner_markers_dict = {L: get_learner_marker(L) for L in unique_learners}

    # B. Color Palettes
    # Palette for Right Plot (Hue = Learner)
    learner_palette_dict = {L: get_learner_color_simple(L) for L in unique_learners}

    # Palette for Left Plot (Hue = Encoder)
    encoder_palette_dict = {E: get_encoder_color(E) for E in unique_encoders}

    # 2. Re-initialize Plotting Parameters
    sns.set_style("white")
    ROW_METRICS = ['inference_time_per_1k', 'run_time_per_1k']
    COL_FACTORS = ['encoder', 'learner']
    COL_TITLES = ['Encoder', 'Learner'] 
    ROW_LABELS = ['Inference Time per 1K samples (s)', 'Total Run Time per 1K samples (s)']

    # --- PALETTE PREPARATION ---
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()

    learner_pal = {L: get_learner_color_simple(L) for L in unique_learners}
    encoder_pal = {E: get_encoder_color(E) for E in unique_encoders}
    learner_markers = {L: get_learner_marker(L) for L in unique_learners}

    # 3. INDENTED PLOTTING LOOP
    for row_idx, x_metric in enumerate(ROW_METRICS):
        fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
        pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, True)

        # =======================================================
        # NEW CODE: PRINT PARETO FRONTIER DETAILS
        # =======================================================
        print(f"\n" + "="*60)
        print(f"PARETO FRONTIER: {Y_METRIC} (Max) vs {x_metric} (Min)")
        print("="*60)
        
        # Select and rename columns for cleaner output
        display_cols = ['encoder', 'learner', x_metric, Y_METRIC]
        
        # Sort by the X metric (Time) to show the progression from Fast->Slow
        frontier_view = pareto_df[display_cols].sort_values(by=x_metric)
        
        # Print formatted table
        print(f"{'Encoder':<25} | {'Learner':<20} | {'Time (s)':<10} | {'Score':<8}")
        print("-" * 75)
        
        for _, row in frontier_view.iterrows():
            e = str(row['encoder'])[:25] # Truncate long names for display
            l = str(row['learner'])[:20]
            t = row[x_metric]
            s = row[Y_METRIC]
            print(f"{e:<25} | {l:<20} | {t:<10.4f} | {s:<8.4f}")
        print("="*60 + "\n")
        # =======================================================

        y_bottom = df_agg[Y_METRIC].min()  # Or hardcode e.g., 0.0
        # Right X: Use a value larger than your max X to ensure it hits the edge
        x_right_edge = df_agg[x_metric].max() * 5.0 

        # 3. Create extension points
        # Point A: (First X, Bottom Y) -> Creates vertical line from axis up to first point
        start_point = pd.DataFrame({
            x_metric: [pareto_df[x_metric].iloc[0]], 
            Y_METRIC: [y_bottom]
        })
        
        # Point B: (Right Edge X, Last Y) -> Creates horizontal line to the right
        end_point = pd.DataFrame({
            x_metric: [x_right_edge], 
            Y_METRIC: [pareto_df[Y_METRIC].iloc[-1]]
        })

        pareto_extended = pd.concat([start_point, pareto_df, end_point], ignore_index=True)

        for col_idx, factor in enumerate(COL_FACTORS):
            ax = axes[col_idx]

            # --- NEW CODE: CONNECT ENCODER DOTS ---
            # Only apply this to the 'encoder' plot (left side)
            # if factor == 'encoder':
            #     for enc in unique_encoders:
            #         # 1. Get all points for this specific encoder
            #         enc_subset = df_agg[df_agg['encoder'] == enc]
                    
            #         # 2. Sort by the X-axis metric so the line is drawn sequentially
            #         enc_subset = enc_subset.sort_values(by=x_metric)
                    
            #         # 3. Draw line if there are multiple points
            #         if len(enc_subset) > 1:
            #             ax.plot(
            #                 enc_subset[x_metric], 
            #                 enc_subset[Y_METRIC], 
            #                 color=encoder_pal[enc], 
            #                 linewidth=1.0,       # Thin line
            #                 alpha=0.4,           # Semi-transparent to not distract
            #                 zorder=1             # Important: Draw BEHIND the markers
            #             )
            # --------------------------------------
            
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3, zorder=0)

            # --- SETUP CONTEXT ---
            if factor == 'encoder':
                current_palette = encoder_pal
                hue_col = 'encoder'
                style_col = 'learner'
            else:
                current_palette = learner_pal
                hue_col = 'learner'
                style_col = 'learner'

            if progressive_transparency == False:
                # --- SPLIT DATA ---
                mask_tuned = df_agg['learner'].str.contains('tuned')
                df_default = df_agg[~mask_tuned]
                df_tuned = df_agg[mask_tuned]

                # --- PLOT 1: DEFAULTS (Full Fill) ---
                sns.lineplot(
                    data=df_default, x=x_metric, y=Y_METRIC,
                    hue=hue_col, style=style_col,
                    palette=current_palette, markers=learner_markers,
                    dashes=False, estimator=None, lw=0, markersize=9,
                    ax=ax, legend=False, # Disable auto-legend
                    **{'fillstyle': 'full', 'markeredgewidth': 1.0, 'markeredgecolor': 'white', 'markeredgecolor': 'black'}
                )
                
                # --- PLOT 2: TUNED (Half Fill with Black Contour) ---
                sns.lineplot(
                    data=df_tuned, x=x_metric, y=Y_METRIC,
                    hue=hue_col, style=style_col,
                    palette=current_palette, markers=learner_markers,
                    dashes=False, estimator=None, lw=0, markersize=9,
                    ax=ax, legend=False, # Disable auto-legend
                    # Key fix: Black edge to show the contour of the half-filled shape
                    **{'fillstyle': 'left', 'markerfacecoloralt': 'white', 
                    'markeredgecolor': 'black', 'markeredgewidth': 1.0} 
                )
                
                # --- PARETO LINE ---
                # ax.step(
                #     pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
                #     linestyle='--', color='black', linewidth=1.2, zorder=0
                # )
                ax.step(
                    pareto_extended[x_metric], 
                    pareto_extended[Y_METRIC], 
                    where='post',
                    linestyle='--', 
                    color='black', 
                    linewidth=1.2, 
                    zorder=0
                )
                
                # --- AXIS FORMATTING ---
                ax.set_box_aspect(1) 
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                # ax.set_title(COL_TITLES[col_idx], fontsize=12, fontweight='bold', pad=105) 
                # Main Title
                ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                             loc='left', y=1.68) 
            
                # 2. Subtitle (Push slightly below title but above legend, e.g., y=1.35)
                subtitle_text = "(shape = learner, color = encoder)" if factor == 'encoder' else "(shape = learner, color = learner)" 
                ax.text(0.4, 1.72,  subtitle_text, transform=ax.transAxes, 
                        fontsize=10, style='italic', color='#333333')
                    
                ax.set_xlabel('')
                ax.set_ylim(bottom=y_bottom)

                if col_idx == 0:
                    ax.set_ylabel(f'Avg {Y_METRIC_LABELS[Y_METRIC]} ($R^2$ & AUC)', fontsize=10)
                else:
                    ax.set_ylabel('')
                
                # sns.despine(ax=ax, trim=False, offset=0)
                # WITH THIS MANUAL SPINE CONFIGURATION:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Force Left and Bottom spines to stay at the "axes" 0-coordinate
                # This ensures they meet perfectly at the corner
                ax.spines['left'].set_position(('axes', 0))
                ax.spines['bottom'].set_position(('axes', 0))
                
                # Ensure they extend fully
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                # 1. Map Encoder Names to Learner Keys
                e2e_map = {
                    'CatBoost': 'CatBoost',
                    'ContextTab': 'ContextTab',
                    'TabSTAR': 'TabSTAR',
                    'TabPFN-2.5': 'TabPFN',
                    'Tarte': 'Tarte'
                }

                # 2. Create a 'markers' dictionary for the Encoders
                # Start with default 'D' (Diamond) for everyone
                encoder_markers = {enc: 'D' for enc in unique_encoders}

                # Overwrite the E2E models with their Learner shapes
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_markers:
                        encoder_markers[enc_name] = learner_shapes[learner_name]

                # IMPORTANT: Ensure your encoder_pal (colors) also matches the learner colors for these 3
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_pal:
                        encoder_pal[enc_name] = learner_colors[learner_name]

                # --- MANUAL LEGEND GENERATION ---
                if factor == 'encoder':
                    std_encoders = [e for e in unique_encoders if e not in e2e_map]
                    e2e_encoders = [e for e in unique_encoders if e in e2e_map]
                    
                    # 2. Sort both lists independently
                    std_encoders.sort()
                    e2e_encoders.sort()
                    
                    # 3. Combine: Standard first, E2E last
                    sorted_encoder_list = std_encoders + e2e_encoders

                    enc_handles = []
                    for enc in sorted_encoder_list:
                        current_color = encoder_pal[enc]
                        # current_marker = 'D'  # Default Diamond
                        if enc in e2e_map:
                            learner_key = e2e_map[enc]
                            current_color = learner_colors[learner_key] # Force learner color
                            current_marker = learner_shapes[learner_key] # Force shape color
                            h = mlines.Line2D([], [], color=current_color, marker=current_marker, 
                                        linestyle='', markersize=8, label=enc)
                        else:
                            h = mpatches.Patch(color=current_color, label=enc)
                        enc_handles.append(h)
                                        
                    ax.legend(
                        handles=enc_handles,
                        loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.9  
                    )
                
                else:
                    # RIGHT LEGEND: Shape/Color by Learner, Grouped (Base, Tuned)
                    # 1. Define specific sorting order
                    base_order = ['Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 
                                'TabSTAR', 'ContextTab', 'CatBoost', 'Tarte']
                    
                    sorted_learners = []
                    # Add families in order: Base then Tuned
                    for base in base_order:
                        if base in unique_learners: 
                            sorted_learners.append(base)
                        if f"{base}-tuned" in unique_learners:
                            sorted_learners.append(f"{base}-tuned")
                    
                    # Catch leftovers
                    for l in unique_learners:
                        if l not in sorted_learners:
                            sorted_learners.append(l)

                    # 2. Build Handles
                    lrn_handles = []
                    for lrn in sorted_learners:
                        is_tuned = 'tuned' in lrn
                        marker = learner_markers[lrn]
                        color = learner_pal[lrn]
                        
                        if is_tuned:
                            # Half-filled with black edge
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='left', markerfacecoloralt='white',
                                            markeredgecolor='black', markeredgewidth=1.0)
                        else:
                            # Full filled
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='full', markeredgecolor='black',markeredgewidth=1.0)
                        lrn_handles.append(h)

                    ax.legend(
                        handles=lrn_handles,
                        loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.4
                    )
            else:
                # --- PREPARE LAYERS (Progressive Transparency) ---
                base_priority = ['Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 
                                'TabSTAR', 'ContextTab', 'CatBoost', 'Tarte']
                
                # KEY CHANGE 1: STRICT SORTING
                # Primary Sort: Base Family order (CatBoost bottom, Tarte top)
                # Secondary Sort: 'tuned' status (Default=0 (Bottom), Tuned=1 (Top))
                z_learners = sorted(list(unique_learners), 
                                    key=lambda x: (
                                        base_priority.index(x.split('-')[0]) if x.split('-')[0] in base_priority else 999,
                                        1 if 'tuned' in x else 0 
                                    ))
                
                total_layers = len(z_learners)

                # 2. Iterate and Plot Layer by Layer
                for i, lrn_name in enumerate(z_learners):
                    
                    # A. Calculate Progressive Alpha (Opacity)
                    # Bottom Layer (i=0) -> 1.0 (Opaque)
                    # Top Layer (i=Max)  -> 0.4 (Transparent)
                    if total_layers > 1:
                        curr_alpha = 1.0 - (0.6 * (i / (total_layers - 1)))
                    else:
                        curr_alpha = 1.0

                    subset = df_agg[df_agg['learner'] == lrn_name]
                    if subset.empty:
                        continue

                    is_tuned = 'tuned' in lrn_name
                    
                    if is_tuned:
                        # KEY CHANGE 2: Transparent Empty Half
                        # Changed markerfacecoloralt from 'white' to 'none'
                        style_kwargs = {
                            'fillstyle': 'left', 
                            'markerfacecoloralt': 'none', # See-through!
                            'markeredgecolor': 'black', 
                            'markeredgewidth': 1.0
                        }
                    else:
                        # Default: Full-filled
                        style_kwargs = {
                            'fillstyle': 'full', 
                            'markeredgecolor': 'black', 
                            'markeredgewidth': 1.0
                        }

                    sns.lineplot(
                        data=subset, x=x_metric, y=Y_METRIC,
                        hue=hue_col, style=style_col,
                        palette=current_palette, markers=learner_markers,
                        dashes=False, estimator=None, lw=0, markersize=9,
                        ax=ax, legend=False,
                        alpha=curr_alpha, # Apply calculated transparency
                        **style_kwargs
                    )
                
                # --- PARETO LINE ---
                ax.step(
                    pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
                    linestyle='--', color='black', linewidth=1.2, zorder=0
                )
                
                # --- AXIS FORMATTING ---
                ax.set_box_aspect(1) 
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                             loc='left', y=1.68) 
            
                # 2. Subtitle (Push slightly below title but above legend, e.g., y=1.35)
                subtitle_text = "(shape = learner, color = encoder)" if factor == 'encoder' else "(shape = learner, color = learner)" 
                ax.text(0.4, 1.72,  subtitle_text, transform=ax.transAxes, 
                        fontsize=10, style='italic', color='#333333')

                ax.set_xlabel('')
                if col_idx == 0:
                    ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=10)
                else:
                    ax.set_ylabel('')

                # sns.despine(ax=ax, trim=False, offset=0) 
                # WITH THIS MANUAL SPINE CONFIGURATION:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Force Left and Bottom spines to stay at the "axes" 0-coordinate
                # This ensures they meet perfectly at the corner
                ax.spines['left'].set_position(('axes', 0))
                ax.spines['bottom'].set_position(('axes', 0))
                
                # Ensure they extend fully
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                e2e_map = {
                    'CatBoost': 'CatBoost',
                    'ContextTab': 'ContextTab',
                    'TabSTAR': 'TabSTAR',
                    'TabPFN-2.5': 'TabPFN',
                    'Tarte': 'Tarte'
                }

                # 2. Create a 'markers' dictionary for the Encoders
                # Start with default 'D' (Diamond) for everyone
                encoder_markers = {enc: 'D' for enc in unique_encoders}

                # Overwrite the E2E models with their Learner shapes
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_markers:
                        encoder_markers[enc_name] = learner_shapes[learner_name]

                # IMPORTANT: Ensure your encoder_pal (colors) also matches the learner colors for these 3
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_pal:
                        encoder_pal[enc_name] = learner_colors[learner_name]

                # --- LEGEND GENERATION ---
                if factor == 'encoder':
                    std_encoders = [e for e in unique_encoders if e not in e2e_map]
                    e2e_encoders = [e for e in unique_encoders if e in e2e_map]
                    
                    # 2. Sort both lists independently
                    std_encoders.sort()
                    e2e_encoders.sort()
                    
                    # 3. Combine: Standard first, E2E last
                    sorted_encoder_list = std_encoders + e2e_encoders

                    enc_handles = []
                    for enc in sorted_encoder_list:
                        current_color = encoder_pal[enc]
                        # current_marker = 'D'  # Default Diamond
                        if enc in e2e_map:
                            learner_key = e2e_map[enc]
                            current_color = learner_colors[learner_key] # Force learner color
                            current_marker = learner_shapes[learner_key] # Force shape color
                            h = mlines.Line2D([], [], color=current_color, marker=current_marker, 
                                        linestyle='', markersize=8, label=enc)
                        else:
                            h = mpatches.Patch(color=current_color, label=enc)
                        enc_handles.append(h)
                    
                    ax.legend(
                        handles=enc_handles,
                        loc='lower center', bbox_to_anchor=(0.5, 1.05),
                        ncol=2, fontsize=8, frameon=False, 
                        title=None, columnspacing=0.8
                    )
                else:
                    # Legend Logic (Same as before)
                    sorted_learners_leg = []
                    for base in base_priority:
                        if base in unique_learners: sorted_learners_leg.append(base)
                        if f"{base}-tuned" in unique_learners: sorted_learners_leg.append(f"{base}-tuned")
                    for l in unique_learners:
                        if l not in sorted_learners_leg: sorted_learners_leg.append(l)

                    lrn_handles = []
                    for lrn in sorted_learners_leg:
                        is_tuned = 'tuned' in lrn
                        marker = learner_markers[lrn]
                        color = learner_pal[lrn]
                        # Keep legend opaque (alpha=1) for clarity
                        if is_tuned:
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='left', markerfacecoloralt='white', # Keep white for legend visibility
                                            markeredgecolor='black', markeredgewidth=1.0)
                        else:
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='full', markeredgecolor='black', markeredgewidth=1.0)
                        lrn_handles.append(h)

                    ax.legend(
                        handles=lrn_handles,
                        loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.4
                    )

        fig.text(0.5, 0.12, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=10)
        plt.subplots_adjust(bottom=0.13, top=0.75, wspace=0.50, left=0.05, right=0.99)


        today_date = time.strftime("%Y-%m-%d")
        format = 'pdf'
        PIC_NAME = f'num_only_comparative_pareto_optimality_plot_1Ksample_scale_progr_transparency_{progressive_transparency}_{Y_METRIC}_{x_metric}_{today_date}.{format}'
        plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
        plt.show()


'''
PARETO PLOTS
- One pareto plot for TabPFN2.5 with only encoders
- One pareto plot by encoders
- One pareto plot by learners
'''

score = score_list[0]  

# --- 2. DATA PREPARATION ---
df_analysis = results.copy()
df_analysis = df_analysis[(df_analysis['dtype'].isin(['Num+Str'])) & (df_analysis['encoder'].isin(selected_encoders))].copy()
df_analysis['string_diversity_Bin'] = bin_feature_33_66(df_analysis, 'string_diversity')


def plot_pareto_frontier_custom(df, x_col, y_col, group_col, title, mode='encoder', ax=None):
    """
    Plots Pareto front + Scatter points.
    mode: 'encoder', 'learner', or 'interaction'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Aggregation
    if mode == 'interaction':
         df_agg = df.groupby(['encoder', 'learner'], as_index=False)[[x_col, y_col]].median()
    else:
         df_agg = df.groupby([group_col], as_index=False)[[x_col, y_col]].median()

    # 2. Pareto Calculation (Minimize Time, Maximize Score)
    # pareto_df = get_pareto_front(df_agg, x_col, y_col, max_x=False, max_y=True)
    pareto_df = get_pareto_front(df_agg, x_col, y_col, maximize_y=True)
    
    # Extension points
    y_bottom = df_agg[y_col].min() * 0.99
    x_right_edge = df_agg[x_col].max() * 5.0
    
    start_point = pd.DataFrame({x_col: [pareto_df[x_col].iloc[0]], y_col: [y_bottom]})
    end_point = pd.DataFrame({x_col: [x_right_edge], y_col: [pareto_df[y_col].iloc[-1]]})
    pareto_extended = pd.concat([start_point, pareto_df, end_point], ignore_index=True)

    # 3. Plot Pareto Line
    ax.step(
        pareto_extended[x_col], pareto_extended[y_col], 
        where='post', linestyle='--', color='black', linewidth=1.2, zorder=0
    )

    # 4. Plot Points
    for _, row in df_agg.iterrows():
        # Determine Attributes
        if mode == 'interaction':
            enc_name = row['encoder']
            lrn_name = row['learner']
            label_name = f"{enc_name} + {lrn_name}"
        elif mode == 'encoder':
            enc_name = row[group_col]
            lrn_name = 'Average' # Use default shape
            label_name = enc_name
        elif mode == 'learner':
            enc_name = 'Average' # Use default color
            lrn_name = row[group_col]
            label_name = lrn_name

        # Colors & Shapes
        if mode == 'learner':
             color = get_learner_color_simple(lrn_name)
        else:
             color = get_encoder_color(enc_name)
             
        marker = get_learner_marker(lrn_name)
        is_tuned = 'tuned' in lrn_name

        style_kwargs = {
            'color': color, 'marker': marker, 'markersize': 11,
            'linestyle': '', 'alpha': 0.9, 'zorder': 3
        }
        
        if is_tuned:
            style_kwargs.update({'fillstyle': 'left', 'markerfacecoloralt': 'white', 
                                 'markeredgecolor': 'black', 'markeredgewidth': 1.0})
        else:
            style_kwargs.update({'fillstyle': 'full', 'markeredgecolor': 'black', 
                                 'markeredgewidth': 0.5})

        ax.plot(row[x_col], row[y_col], label=label_name, **style_kwargs)

    # 5. Styling
    ax.set_xscale('log')
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Run Time per 1K Samples (s)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    sns.despine(ax=ax)

# --- EXECUTION ---
x_metric = 'run_time_per_1k'
y_metric = score_list[0]

'''
One pareto plot for TabPFN2.5 with only encoders
'''
plt.figure(figsize=(6, 5))
subset_tabpfn = df_analysis[df_analysis['learner'] == 'TabPFN-2.5'].copy()

plot_pareto_frontier_custom(
    subset_tabpfn, x_metric, y_metric, 
    group_col='encoder', 
    title='Fixed Learner: TabPFN-2.5', 
    mode='encoder', ax=plt.gca()
)

# Custom Legend for Encoders
handles, labels = plt.gca().get_legend_handles_labels()
# Filter duplicates if any
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Encoders")
# increase size of both axis labels
plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'pareto_front_by_encoder_for_tabpfn_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
One pareto plot by encoders
'''
plt.figure(figsize=(6, 5))
plot_pareto_frontier_custom(
    df_analysis, x_metric, y_metric, 
    group_col='encoder', 
    title='Aggregated by Encoder', 
    mode='encoder', ax=plt.gca()
)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Encoders")
plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'pareto_front_by_encoder_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
One pareto plot by learners
'''
plt.figure(figsize=(6, 5))

plot_pareto_frontier_custom(
    df_analysis, x_metric, y_metric, 
    group_col='learner', 
    title='Aggregated by Learner', 
    mode='learner', ax=plt.gca()
)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Learners")
plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'pareto_front_by_learner_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
VSE vs STRABLE comparisonXGBoost with only encoders and only for classification tasks
'''

import matplotlib.patheffects as pe

subset_xgboost = results.copy()
balanced_problems = ['kickstarter-projects', 'animalandveterinary-event', 'device-pma', 'drug-drugsfda']
subset_xgboost = subset_xgboost[(subset_xgboost['learner'] == 'XGBoost') & (subset_xgboost['task'] == 'b-classification') & (subset_xgboost['data_name'].isin(balanced_problems))]

# 1. Define the specific encoders to plot
selected_encoders = [
    'Tf-Idf',
    'LM BGE-large',
    'LM Qwen-3-8B',
    'LM LLaMA-3.1-8B'
]

# 2. Filter by Encoder AND Data Type
# We explicitly keep only 'Num+Str' and 'Str' to satisfy the requirements and avoid the ValueError
df_plot = subset_xgboost[
    (subset_xgboost['encoder'].isin(selected_encoders)) & 
    (subset_xgboost['dtype'].isin(['Num+Str', 'Str']))
].copy()

# 3. Compute Ranks
# Rank 1 = Best Score
df_plot['rank'] = df_plot.groupby(['data_name', 'dtype'])['score'].rank(ascending=False)

# 4. Binning num_rows
bins = [0, 1000, 2000, 4000, 8000, 20000, 50000, np.inf]
labels = [1000, 2000, 4000, 8000, 20000, 50000, 100000]

df_plot['size_bin'] = pd.cut(df_plot['num_rows'], bins=bins, labels=labels)

# Ensure size_bin is numeric for interpolation
df_plot['size_bin'] = df_plot['size_bin'].astype(float)

# 5. Define Style & Palette
palette = {
    'Tf-Idf': '#C85250',           # Red
    'TargetEncoder': '#7F7F7F',    # Gray
    'LM BGE-large': '#5D85C3',     # Blue
    'LM Qwen-3-8B': '#9B59B6',     # Purple
    'LM LLaMA-3.1-8B': '#E74C3C'   # Red/Pink
}

# Setup the figure
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.set_context("talk")

# 1. Plotting
ax = sns.lineplot(
    data=df_plot,
    x='size_bin',
    y='rank',
    hue='encoder',
    style='dtype',
    palette=palette,
    markers=True,
    dashes={'Num+Str': (1, 0), 'Str': (2, 2)},
    linewidth=2.5,
    err_style='band',
    ci=95,
    legend=False 
)

# 2. Add Top Legend
legend_handles = [
    Line2D([0], [0], color='black', lw=2.5, linestyle='-', label='Original table (Num+Str)'),
    Line2D([0], [0], color='black', lw=2.5, linestyle='--', label='Using only text entries (Str)')
]

ax.legend(
    handles=legend_handles, 
    loc='lower center', 
    bbox_to_anchor=(0.5, 1.02), 
    ncol=2, 
    frameon=True,
    handletextpad=0.5,
    columnspacing=2
)

# 3. Add Inline Labels (ROBUST VERSION)
label_positions = {
    'Tf-Idf': 10000,
    'LM BGE-large': 40000,
    'LM Qwen-3-8B': 60000,
    'LM LLaMA-3.1-8B': 80000 
}

for encoder in selected_encoders:
    # Get the target X position you defined
    target_x = label_positions.get(encoder, 4000)
    
    # Filter for the specific line we want to label (Solid line / Num+Str)
    subset = df_plot[(df_plot['encoder'] == encoder) & (df_plot['dtype'] == 'Num+Str')]
    
    # Group by size_bin to get the actual (x, y) points that make up the line
    # We drop NaNs to ensure we have a valid path to interpolate on
    line_data = subset.groupby('size_bin')['rank'].mean().reset_index().dropna()
    line_data = line_data.sort_values('size_bin')
    
    if len(line_data) >= 2:
        # valid_xs are the bin positions where data actually exists (e.g., 2000, 4000, 8000...)
        valid_xs = line_data['size_bin'].values
        valid_ys = line_data['rank'].values
        
        # np.interp calculates the Y at target_x based on the existing points
        # It handles cases where target_x is between any two bins, or even if it needs to extrapolate slightly
        y_target = np.interp(target_x, valid_xs, valid_ys)
        
        # Place the text
        txt = ax.text(
            x=target_x, 
            y=y_target, 
            s=encoder, 
            color=palette[encoder], 
            fontweight='bold',
            ha='center', 
            va='bottom'
        )
        
        # Add white outline
        txt.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])
    else:
        print(f"Not enough data points to interpolate label for {encoder}")

# 4. Formatting
plt.gca().invert_yaxis()
plt.xlabel("Dataset Size (Num Rows)")
plt.ylabel("Mean Rank (Lower is Better)")
# Only limit the view, don't limit the data processing
plt.xlim(2000, 100000) 

sns.despine()
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'sample_size_performance_num-str_str-only_xgboost_binary_classification_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

# plot_pareto_frontier_custom(
#     subset_xgboost, x_metric, y_metric, 
#     group_col='encoder', 
#     title='Fixed Learner: XGBoost, binary classification', 
#     mode='encoder', ax=plt.gca()
# )

# # Custom Legend for Encoders
# handles, labels = plt.gca().get_legend_handles_labels()
# # Filter duplicates if any
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Encoders")
# # increase size of both axis labels
# plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
# plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'pareto_front_by_encoder_for_xgboost_binary_classification_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

'''
BARPLOT: KENDALLTAU of NUM+STR BY HIGH/LOW BIN of CARDINALITY, STRING LENGTH and STRING DIVERSITY
'''

# --- 1. CONFIGURATION ---
score = score_list[0]
percentile = 33 

# --- 2. DATA PREPARATION ---
df_analysis = results.copy()
df_analysis = df_analysis[(df_analysis['dtype'].isin(['Num+Str'])) & 
                          (df_analysis['encoder'].isin(selected_encoders))].copy()

# Apply binning
df_analysis['Card_Bin'] = bin_feature_33_66(df_analysis, 'avg_cardinality')
df_analysis['Str_Bin'] = bin_feature_33_66(df_analysis, 'avg_string_length_per_cell')
df_analysis['n_col_Bin'] = bin_feature_33_66(df_analysis, 'num_columns')
df_analysis['n_row_Bin'] = bin_feature_33_66(df_analysis, 'num_rows')
df_analysis['string_diversity_Bin'] = bin_feature_33_66(df_analysis, 'string_diversity')

heterogeneity_dimensions = ['Card_Bin', 'Str_Bin', 'n_col_Bin', 'n_row_Bin', 'string_diversity_Bin']
heterogeneity_feature_map = {
    'Card_Bin': 'Cardinality',
    'Str_Bin': 'String Length',
    'n_col_Bin': 'Num Columns',
    'n_row_Bin': 'Num Rows',
    'string_diversity_Bin': 'String Diversity'
}

# --- 3. CALCULATE METRICS ---
metrics_data = []

for dim in heterogeneity_dimensions:
    # Group by learner and bin to get average scores per pipeline first
    df_pipeline = df_analysis.groupby(['encoder_learner', dim], as_index=False)[score].mean()
    df_pivot = df_pipeline.pivot(index='encoder_learner', columns=dim, values=score)
    
    # A. Kendall Tau (Stability)
    tau, _ = kendalltau(df_pivot['Low'], df_pivot['High'])
    
    # B. Average Performance (High vs Low)
    # We Average the performance across all learners for the specific bin
    # (This gives the general difficulty/impact of that bin)
    avg_high = df_pivot['High'].mean()
    avg_low = df_pivot['Low'].mean()
    
    metrics_data.append({
        'Feature': dim,
        'FeatureName': heterogeneity_feature_map.get(dim, dim),
        'KendallTau': tau,
        'High': avg_high,
        'Low': avg_low
    })

df_metrics = pd.DataFrame(metrics_data)

# Sort by Kendall Tau (to align both plots by stability)
df_metrics = df_metrics.sort_values('KendallTau', ascending=True)

# Reshape for the Right Plot (Long format for hue plotting)
df_long = df_metrics.melt(id_vars=['FeatureName', 'KendallTau'], 
                          value_vars=['High', 'Low'], 
                          var_name='Bin', value_name='AvgScore')

# --- 4. PLOTTING ---
sns.set_theme(style="whitegrid", context="paper") # Reduced font size context
fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True)

# --- LEFT PLOT: STABILITY ---
# We use a neutral or distinct palette for the features
palette_left = sns.color_palette("tab10", n_colors=len(df_metrics))

sns.barplot(
    data=df_metrics,
    x='KendallTau',
    y='FeatureName',
    palette=palette_left,
    edgecolor='black',
    linewidth=0.8,
    ax=axes[0]
)

axes[0].set_xlabel('Stability of Model Ranks\n(Kendall $\\tau$ High vs Low)', fontsize=16, labelpad=11, ha='right', x=0.9)
axes[0].set_ylabel('', )
axes[0].set_xlim(0.45, 0.65) # Adjust x-limits to fit your specific data range
axes[0].tick_params(axis='y', labelsize=16)  # <--- Change 14 to your desired font size
axes[0].tick_params(axis='x', labelsize=14)  # Keep X axis size separate if needed
axes[0].grid(True, axis='x', alpha=0.5)

# --- RIGHT PLOT: IMPACT (High vs Low) ---
# Custom colors: Light Red (High) and Light Green (Low)
custom_palette = {'High': '#ff9999', 'Low': '#99ff99'} # Light red and light green

sns.barplot(
    data=df_long,
    x='AvgScore',
    y='FeatureName',
    hue='Bin',
    palette=custom_palette,
    edgecolor='black',
    linewidth=0.8,
    ax=axes[1]
)

axes[1].set_xlabel(f'Impact on Avg {Y_METRIC_LABELS[score]}\n($R^2$ & AUC)', fontsize=16, labelpad=11, ha='center', x=0.7, multialignment='center')
axes[1].set_ylabel('', fontsize=18)
# axes[1].tick_params(axis='y', labelsize=14)  # <--- Change 14 to your desired font size
axes[1].tick_params(axis='x', labelsize=14)  # Keep X axis size separate if needed
axes[1].grid(True, axis='x', alpha=0.5)
axes[1].set_xlim(0.6, 0.8) 

# Legend settings
axes[1].legend(title=None, bbox_to_anchor=(0.59, 0.3), loc='upper left', borderaxespad=0, frameon=False, fontsize=16)

sns.despine(left=True, bottom=False)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'kendalltau_heterogeneity_axis_barplot_{percentile}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
BARPLOT: KENDALLTAU of NUM+STR BY HIGH/LOW BIN of CARDINALITY, STRING LENGTH and STRING DIVERSITY
drop n_col and n_row
'''

# --- 1. CONFIGURATION ---
score = score_list[0]
percentile = 33 

# --- 2. DATA PREPARATION ---
# We need the full results (both Num and Num+Str) to compare rankings
df_analysis = results.copy()

df_analysis = df_analysis[(df_analysis['dtype'].isin(['Num+Str'])) & (df_analysis['encoder'].isin(selected_encoders))].copy()

df_analysis['Card_Bin'] = bin_feature_33_66(df_analysis, 'avg_cardinality')
df_analysis['Str_Bin'] = bin_feature_33_66(df_analysis, 'avg_string_length_per_cell')
df_analysis['n_col_Bin'] = bin_feature_33_66(df_analysis, 'num_columns')
df_analysis['n_row_Bin'] = bin_feature_33_66(df_analysis, 'num_rows')
df_analysis['string_diversity_Bin'] = bin_feature_33_66(df_analysis, 'string_diversity')


# compute avg performance per pipeline and heterogeneity
def compute_kendalltau_between_bins(df, feature_bin_col, score_col):
    df_pipeline_heterogeneity_axis = df.groupby(['encoder_learner', feature_bin_col], as_index=False)[score_col].mean()
    df_pipeline_heterogeneity_axis_pivot = df_pipeline_heterogeneity_axis.pivot(index='encoder_learner', columns=feature_bin_col, values=score_col)
    #select only low and high
    df_pipeline_heterogeneity_axis_pivot = df_pipeline_heterogeneity_axis_pivot[['Low', 'High']]
    #compute kendalltau
    corr, _ = kendalltau(df_pipeline_heterogeneity_axis_pivot['Low'], df_pipeline_heterogeneity_axis_pivot['High'])
    print(f"Kendall Tau between Low and High {feature_bin_col} bins: {corr:.4f}")
    return corr

heterogeneity_dimensions = ['Card_Bin', 'Str_Bin', 'n_col_Bin', 'n_row_Bin', 'string_diversity_Bin']

df_kendalltau_heterogeneity = pd.DataFrame()
for dim in heterogeneity_dimensions:
    cor = compute_kendalltau_between_bins(df_analysis, dim, score)
    df_kendalltau_heterogeneity = pd.concat([df_kendalltau_heterogeneity, pd.DataFrame([{'Feature': dim, 'KendallTau': cor}])])

heterogeneity_feature_map = {
    'Card_Bin': 'Cardinality',
    'Str_Bin': 'String Length',
    'n_col_Bin': 'Num Columns',
    'n_row_Bin': 'Num Rows',
    'string_diversity_Bin': 'String Diversity'
}

# --- 4. HORIZONTAL BARPLOT FOR HETEROGENEITY ---
df_plot = df_kendalltau_heterogeneity.copy()

# Map feature codes to readable names
df_plot['FeatureName'] = df_plot['Feature'].map(heterogeneity_feature_map).fillna(df_plot['Feature'])

# Sort for a clean horizontal bar chart (largest at top)
df_plot = df_plot[~df_plot['Feature'].isin(['n_col_Bin', 'n_row_Bin'])]
df_plot = df_plot.sort_values('KendallTau', ascending=True)


sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(4, 3))

# Use a tab* palette (tab10) sized to the number of bars
palette = sns.color_palette("tab10", n_colors=len(df_plot))

sns.barplot(
    data=df_plot,
    x='KendallTau',
    y='FeatureName',
    palette=palette,
    edgecolor='black',
    linewidth=0.8,
    ax=ax
)

# Styling
ax.set_xlabel(f'Kendall $\\tau$(High vs Low)', fontsize=16, labelpad=12, x=0.40)
ax.set_ylabel('')
ax.set_xlim(0.45, 0.65)  # full Kendall tau range
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Annotate bar values
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=4, fontsize=12)

sns.despine(left=False, bottom=False)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'kendalltau_heterogeneity_axis_barplot_drop_ncol_nrow_{percentile}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()



'''
string diversity pareto frontier
'''

# ==========================================
# FIGURE 4: High String Diversity (Pipelines)
# ==========================================

for bin in ['Low', 'High']:

    plt.figure(figsize=(6, 5)) # Slightly wider to accommodate double legend


    # 1. Filter Data for High String Diversity
    high_div_df = df_analysis[df_analysis['string_diversity_Bin'] == bin].copy()

    # 2. Plot Pareto Frontier & Points
    # We use the previously defined custom plotting function
    plot_pareto_frontier_custom(
        high_div_df, 
        x_metric, 
        y_metric, 
        group_col=None, 
        title=f'{bin} String Diversity', 
        mode='interaction', 
        ax=plt.gca()
    )

    ax = plt.gca()

    # --- LEGEND 1: LEARNERS (SHAPES) ---
    learner_handles = []
    sorted_learners = sorted(high_div_df['learner'].unique())

    for lrn in sorted_learners:
        m = get_learner_marker(lrn)
        is_tuned = 'tuned' in lrn
        
        # Create handle matching the plot style
        if is_tuned:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='left', markerfacecoloralt='white', markeredgewidth=1, markersize=10)
        else:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='full', markersize=10)
        learner_handles.append(h)

    # Add First Legend (Learners) - Anchor Top Right
    legend1 = plt.legend(
        handles=learner_handles, 
        title="Learners (Shape)", 
        loc='upper left', 
        ncol=2,
        bbox_to_anchor=(1, 1.15),
        columnspacing=0.1, 
        frameon=False
    )
    ax.add_artist(legend1) # Critical: Add it manually so the second legend doesn't overwrite it


    # --- LEGEND 2: ENCODERS (COLORS) ---
    encoder_handles = []
    sorted_encoders = sorted(high_div_df['encoder'].unique())

    for enc in sorted_encoders:
        c = get_encoder_color(enc)
        # Use a square patch for colors
        h = Patch(facecolor=c, edgecolor='black', label=enc)
        encoder_handles.append(h)

    # Add Second Legend (Encoders) - Anchor Bottom Right
    plt.legend(
        handles=encoder_handles, 
        title="Encoders (Color)", 
        loc='lower left', 
        bbox_to_anchor=(1, -0.16),
        columnspacing=0.1, 
        frameon=False,
        ncol=2 # Keep it vertical list
    )
    # plt.ylim(0.645, 0.84)
    plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
    plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'{bin}_string_diversity_pareto_frontier_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()


'''
high string diversity drop TabPFN as encoders
'''

plt.figure(figsize=(6, 5)) # Slightly wider to accommodate double legend

# 1. Filter Data for High String Diversity
high_div_df = df_analysis[(df_analysis['string_diversity_Bin'] == 'High') & (df_analysis['encoder'] != 'TabPFN-2.5')].copy()

# 2. Plot Pareto Frontier & Points
# We use the previously defined custom plotting function
plot_pareto_frontier_custom(
    high_div_df, 
    x_metric, 
    y_metric, 
    group_col=None, 
    title='High String Diversity - after dropping TabPFN-2.5 as encoder', 
    mode='interaction', 
    ax=plt.gca()
)

ax = plt.gca()

# --- LEGEND 1: LEARNERS (SHAPES) ---
learner_handles = []
sorted_learners = sorted(high_div_df['learner'].unique())

for lrn in sorted_learners:
    m = get_learner_marker(lrn)
    is_tuned = 'tuned' in lrn
    
    # Create handle matching the plot style
    if is_tuned:
        h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                   fillstyle='left', markerfacecoloralt='white', markeredgewidth=1, markersize=10)
    else:
        h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                   fillstyle='full', markersize=10)
    learner_handles.append(h)

# Add First Legend (Learners) - Anchor Top Right
legend1 = plt.legend(
    handles=learner_handles, 
    title="Learners (Shape)", 
    loc='upper left', 
    ncol=2,
    bbox_to_anchor=(1, 1.15),
    columnspacing=0.1, 
    frameon=False
)
ax.add_artist(legend1) # Critical: Add it manually so the second legend doesn't overwrite it


# --- LEGEND 2: ENCODERS (COLORS) ---
encoder_handles = []
sorted_encoders = sorted(high_div_df['encoder'].unique())

for enc in sorted_encoders:
    c = get_encoder_color(enc)
    # Use a square patch for colors
    h = Patch(facecolor=c, edgecolor='black', label=enc)
    encoder_handles.append(h)

# Add Second Legend (Encoders) - Anchor Bottom Right
plt.legend(
    handles=encoder_handles, 
    title="Encoders (Color)", 
    loc='lower left', 
    bbox_to_anchor=(1, -0.16),
    columnspacing=0.1, 
    frameon=False,
    ncol=2 # Keep it vertical list
)
plt.ylim(0.695, 0.84)
plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'high_string_diversity_pareto_front_no_tabpfn_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()



'''
string length pareto frontier
'''

for bin in ['Low', 'High']:

    plt.figure(figsize=(6, 5)) # Slightly wider to accommodate double legend

    # 1. Filter Data for High String Length
    high_div_df = df_analysis[df_analysis['Str_Bin'] == bin].copy()

    # 2. Plot Pareto Frontier & Points
    # We use the previously defined custom plotting function
    plot_pareto_frontier_custom(
        high_div_df, 
        x_metric, 
        y_metric, 
        group_col=None, 
        title=f'{bin} String Length', 
        mode='interaction', 
        ax=plt.gca()
    )

    ax = plt.gca()

    # --- LEGEND 1: LEARNERS (SHAPES) ---
    learner_handles = []
    sorted_learners = sorted(high_div_df['learner'].unique())

    for lrn in sorted_learners:
        m = get_learner_marker(lrn)
        is_tuned = 'tuned' in lrn
        
        # Create handle matching the plot style
        if is_tuned:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='left', markerfacecoloralt='white', markeredgewidth=1, markersize=10)
        else:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='full', markersize=10)
        learner_handles.append(h)

    # Add First Legend (Learners) - Anchor Top Right
    legend1 = plt.legend(
        handles=learner_handles, 
        title="Learners (Shape)", 
        loc='upper left', 
        ncol=2,
        bbox_to_anchor=(1, 1.15),
        columnspacing=0.1, 
        frameon=False
    )
    ax.add_artist(legend1) # Critical: Add it manually so the second legend doesn't overwrite it


    # --- LEGEND 2: ENCODERS (COLORS) ---
    encoder_handles = []
    sorted_encoders = sorted(high_div_df['encoder'].unique())

    for enc in sorted_encoders:
        c = get_encoder_color(enc)
        # Use a square patch for colors
        h = Patch(facecolor=c, edgecolor='black', label=enc)
        encoder_handles.append(h)

    # Add Second Legend (Encoders) - Anchor Bottom Right
    plt.legend(
        handles=encoder_handles, 
        title="Encoders (Color)", 
        loc='lower left', 
        bbox_to_anchor=(1, -0.16),
        columnspacing=0.1, 
        frameon=False,
        ncol=2 # Keep it vertical list
    )
    # plt.ylim(0.645, 0.84)
    plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
    plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'{bin}_string_length_pareto_frontier_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()


'''
num rows pareto frontier
'''

for bin in ['Low', 'High']:

    plt.figure(figsize=(6, 5)) # Slightly wider to accommodate double legend

    # 1. Filter Data for High String Length
    high_div_df = df_analysis[df_analysis['n_row_Bin'] == bin].copy()

    # 2. Plot Pareto Frontier & Points
    # We use the previously defined custom plotting function
    plot_pareto_frontier_custom(
        high_div_df, 
        x_metric, 
        y_metric, 
        group_col=None, 
        title=f'{bin} Num Rows', 
        mode='interaction', 
        ax=plt.gca()
    )

    ax = plt.gca()

    # --- LEGEND 1: LEARNERS (SHAPES) ---
    learner_handles = []
    sorted_learners = sorted(high_div_df['learner'].unique())

    for lrn in sorted_learners:
        m = get_learner_marker(lrn)
        is_tuned = 'tuned' in lrn
        
        # Create handle matching the plot style
        if is_tuned:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='left', markerfacecoloralt='white', markeredgewidth=1, markersize=10)
        else:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='full', markersize=10)
        learner_handles.append(h)

    # Add First Legend (Learners) - Anchor Top Right
    legend1 = plt.legend(
        handles=learner_handles, 
        title="Learners (Shape)", 
        loc='upper left', 
        ncol=2,
        bbox_to_anchor=(1, 1.15),
        columnspacing=0.1, 
        frameon=False
    )
    ax.add_artist(legend1) # Critical: Add it manually so the second legend doesn't overwrite it


    # --- LEGEND 2: ENCODERS (COLORS) ---
    encoder_handles = []
    sorted_encoders = sorted(high_div_df['encoder'].unique())

    for enc in sorted_encoders:
        c = get_encoder_color(enc)
        # Use a square patch for colors
        h = Patch(facecolor=c, edgecolor='black', label=enc)
        encoder_handles.append(h)

    # Add Second Legend (Encoders) - Anchor Bottom Right
    plt.legend(
        handles=encoder_handles, 
        title="Encoders (Color)", 
        loc='lower left', 
        bbox_to_anchor=(1, -0.16),
        columnspacing=0.1, 
        frameon=False,
        ncol=2 # Keep it vertical list
    )
    # plt.ylim(0.645, 0.84)
    plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
    plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'{bin}_num_rows_pareto_frontier_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()



'''
cardinality pareto frontier
'''

for bin in ['Low', 'High']:

    plt.figure(figsize=(6, 5)) # Slightly wider to accommodate double legend

    # 1. Filter Data for High String Length
    high_div_df = df_analysis[df_analysis['Card_Bin'] == bin].copy()

    # 2. Plot Pareto Frontier & Points
    # We use the previously defined custom plotting function
    plot_pareto_frontier_custom(
        high_div_df, 
        x_metric, 
        y_metric, 
        group_col=None, 
        title=f'{bin} Cardinality', 
        mode='interaction', 
        ax=plt.gca()
    )

    ax = plt.gca()

    # --- LEGEND 1: LEARNERS (SHAPES) ---
    learner_handles = []
    sorted_learners = sorted(high_div_df['learner'].unique())

    for lrn in sorted_learners:
        m = get_learner_marker(lrn)
        is_tuned = 'tuned' in lrn
        
        # Create handle matching the plot style
        if is_tuned:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='left', markerfacecoloralt='white', markeredgewidth=1, markersize=10)
        else:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='full', markersize=10)
        learner_handles.append(h)

    # Add First Legend (Learners) - Anchor Top Right
    legend1 = plt.legend(
        handles=learner_handles, 
        title="Learners (Shape)", 
        loc='upper left', 
        ncol=2,
        bbox_to_anchor=(1, 1.15),
        columnspacing=0.1, 
        frameon=False
    )
    ax.add_artist(legend1) # Critical: Add it manually so the second legend doesn't overwrite it


    # --- LEGEND 2: ENCODERS (COLORS) ---
    encoder_handles = []
    sorted_encoders = sorted(high_div_df['encoder'].unique())

    for enc in sorted_encoders:
        c = get_encoder_color(enc)
        # Use a square patch for colors
        h = Patch(facecolor=c, edgecolor='black', label=enc)
        encoder_handles.append(h)

    # Add Second Legend (Encoders) - Anchor Bottom Right
    plt.legend(
        handles=encoder_handles, 
        title="Encoders (Color)", 
        loc='lower left', 
        bbox_to_anchor=(1, -0.16),
        columnspacing=0.1, 
        frameon=False,
        ncol=2 # Keep it vertical list
    )
    # plt.ylim(0.645, 0.84)
    plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
    plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'{bin}_cardinality_pareto_frontier_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()


'''
n col pareto frontier
'''

for bin in ['Low', 'High']:

    plt.figure(figsize=(6, 5)) # Slightly wider to accommodate double legend


    # 1. Filter Data for High String Length
    high_div_df = df_analysis[df_analysis['n_col_Bin'] == bin].copy()

    # 2. Plot Pareto Frontier & Points
    # We use the previously defined custom plotting function
    plot_pareto_frontier_custom(
        high_div_df, 
        x_metric, 
        y_metric, 
        group_col=None, 
        title=f'{bin} Num Columns', 
        mode='interaction', 
        ax=plt.gca()
    )

    ax = plt.gca()

    # --- LEGEND 1: LEARNERS (SHAPES) ---
    learner_handles = []
    sorted_learners = sorted(high_div_df['learner'].unique())

    for lrn in sorted_learners:
        m = get_learner_marker(lrn)
        is_tuned = 'tuned' in lrn
        
        # Create handle matching the plot style
        if is_tuned:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='left', markerfacecoloralt='white', markeredgewidth=1, markersize=10)
        else:
            h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                    fillstyle='full', markersize=10)
        learner_handles.append(h)

    # Add First Legend (Learners) - Anchor Top Right
    legend1 = plt.legend(
        handles=learner_handles, 
        title="Learners (Shape)", 
        loc='upper left', 
        ncol=2,
        bbox_to_anchor=(1, 1.15),
        columnspacing=0.1, 
        frameon=False
    )
    ax.add_artist(legend1) # Critical: Add it manually so the second legend doesn't overwrite it


    # --- LEGEND 2: ENCODERS (COLORS) ---
    encoder_handles = []
    sorted_encoders = sorted(high_div_df['encoder'].unique())

    for enc in sorted_encoders:
        c = get_encoder_color(enc)
        # Use a square patch for colors
        h = Patch(facecolor=c, edgecolor='black', label=enc)
        encoder_handles.append(h)

    # Add Second Legend (Encoders) - Anchor Bottom Right
    plt.legend(
        handles=encoder_handles, 
        title="Encoders (Color)", 
        loc='lower left', 
        bbox_to_anchor=(1, -0.16),
        columnspacing=0.1, 
        frameon=False,
        ncol=2 # Keep it vertical list
    )
    # plt.ylim(0.645, 0.84)
    plt.xlabel('Run Time per 1K Samples (s)', fontsize=16)
    plt.ylabel(f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)", fontsize=16)
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'{bin}_n_col_pareto_frontier_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()


'''
PERFORMANCE BY CARDINALITY AND STRING LENGTH REGIMES
'''


# --- 1. SETTINGS & STYLES ---
sns.set_theme(style="whitegrid", context="paper")

# --- 4. PLOTTING FUNCTION ---
def plot_regime_custom(df, ax, title, feature_type, set_xlim_min=0.45, set_xlim_max=1.0, set_ylim_min=0.45, set_ylim_max=1.0):
    # 1. Diagonal & Backgrounds
    lims = [0.0, 1.0]
    ax.plot(lims, lims, ls='--', c='grey', alpha=0.5, zorder=0)
    
    # Regions
    # Below Diagonal (High Bin > Low Bin) -> Red
    ax.fill_between(lims, [0,0], lims, color='red', alpha=0.05, transform=ax.transData)
    # Above Diagonal (Low Bin > High Bin) -> Green
    ax.fill_between(lims, lims, [1.1, 1.1], color='green', alpha=0.05, transform=ax.transData)
    
    # ---------------------------------------------------------
    # NEW: RANSAC REGRESSOR LOGIC (CLIPPED)
    # ---------------------------------------------------------
    if len(df) > 5: # Only fit if we have enough points
        # Prepare Data
        X = df[['High-bin']].values
        y = df['Low-bin'].values
        
        # Fit RANSAC
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        
        # 1. Generate dense prediction line across X range
        # (Use 500 points to ensure smooth clipping)
        line_X = np.linspace(set_xlim_min, set_xlim_max, 500).reshape(-1, 1)
        line_y = ransac.predict(line_X)
        
        # 2. Filter: Keep only points where Y is between 0.7 and 1.0
        mask = (line_y >= 0.4) & (line_y <= 1.0)
        
        line_X_clipped = line_X[mask]
        line_y_clipped = line_y[mask]
        
        # 3. Plot clipped line
        if len(line_X_clipped) > 1:
            ax.plot(line_X_clipped, line_y_clipped, color='cornflowerblue', linewidth=3, 
                    alpha=0.8, zorder=2, label='_nolegend_')
    # ---------------------------------------------------------

    # 2. Plot Points Loop
    unique_pipelines = df['pipeline'].unique()
    
    for pipe in unique_pipelines:
        subset = df[df['pipeline'] == pipe]
        if len(subset) == 0: continue
        
        # Parse info
        enc = subset['encoder'].iloc[0]
        lrn = subset['learner'].iloc[0]
        
        color = get_encoder_color(enc)
        marker = get_learner_marker(lrn)
        
        # Tuning Logic
        is_tuned = 'tuned' in lrn
        
        style_kwargs = {
            'color': color,
            'marker': marker,
            'markersize': 10,
            'linestyle': '',
            'alpha': 0.9,
            'zorder': 3 
        }
        
        if is_tuned:
            style_kwargs.update({
                'fillstyle': 'left',
                'markerfacecoloralt': 'white',
                'markeredgecolor': 'black',
                'markeredgewidth': 1.0
            })
        else:
            style_kwargs.update({
                'fillstyle': 'full',
                'markeredgecolor': 'white',
                'markeredgewidth': 0.5
            })
            
        # Plot
        ax.plot(subset['High-bin'], subset['Low-bin'], label=pipe, **style_kwargs)
        ax.set_xlim(set_xlim_min, set_xlim_max)
        ax.set_ylim(set_ylim_min, set_ylim_max)

    # 3. Styling & Annotations
    ax.set_title(title+' '+title_tag, fontsize=18, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 4. Descriptive Text Annotations (Same as before...)
    ax.text(0.05, 0.98, 
            f"The algorithm is better on datasets\nwith LOW {feature_type}\nfeatures", 
            transform=ax.transAxes, fontsize=16, color='darkgreen', 
            va='top', ha='left', weight='bold')
    ax.text(0.15, 0.82, "↑", transform=ax.transAxes, fontsize=20, color=f'darkgreen', weight='bold', rotation=45)
    
    # Axis Labels
    if feature_type == 'cardinality':
        ax.set_xlabel('Score on high cardinality datasets', fontsize=20, labelpad=10)
        ax.set_ylabel('Score on low cardinality datasets', fontsize=20, labelpad=10)
        if upper_33:
            ax.text(0.95, 0.05, 
                    f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
                    transform=ax.transAxes, fontsize=16, color='darkred', 
                    va='bottom', ha='right', weight='bold')
            ax.text(0.8, 0.25, "↓", transform=ax.transAxes, fontsize=24, color='darkred', weight='bold', rotation=45)
        # else:
        #      ax.text(0.95, 0.05, 
        #             f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
        #             transform=ax.transAxes, fontsize=11, color='darkred', 
        #             va='bottom', ha='right', weight='bold')
        #      ax.text(0.85, 0.2, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)
    else:
        ax.set_xlabel('Score on high string length datasets', fontsize=20, labelpad=10)
        ax.set_ylabel('Score on low string length datasets', fontsize=20, labelpad=10)
        if upper_33:
            ax.text(0.95, 0.05, 
                    f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
                    transform=ax.transAxes, fontsize=16, color='darkred', 
                    va='bottom', ha='right', weight='bold')
            ax.text(0.8, 0.25, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)
        # else:
        #     ax.text(0.95, 0.05, 
        #             f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
        #             transform=ax.transAxes, fontsize=11, color='darkred', 
        #             va='bottom', ha='right', weight='bold')
        #     ax.text(0.85, 0.2, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)



def apply_annotations(ax, df, position_dict):
    for target, offset in position_dict.items():
        # 1. Find the point
        row = df[df['pipeline'] == target]
        if len(row) == 0: continue
            
        x = row['High-bin'].values[0]
        y = row['Low-bin'].values[0]
        
        # 2. Determine alignment based on offset direction
        # If pushing Right (x>0), align text Left. If pushing Left (x<0), align text Right.
        ha = 'left' if offset[0] >= 0 else 'right'
        
        # If pushing Up (y>0), align Bottom. If pushing Down (y<0), align Top.
        va = 'bottom' if offset[1] >= 0 else 'top'

        # 3. Draw Annotation
        ax.annotate(
            target, 
            xy=(x, y), 
            xytext=offset,
            textcoords='offset points',
            fontsize=12, 
            fontweight='bold', 
            color='black',
            ha=ha, 
            va=va
        )


score = score_list[0] # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

# task = 'all_task'

for task in ['regression', 'classification', 'all_task']:

    if task == 'regression':
        title_tag = '($R^2$)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] == 'regression')].copy()
    elif task == 'classification':
        title_tag = '(AUC)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] != 'regression')].copy()
    else:
        title_tag = '(Avg $R^2$ & AUC)'
        df_plot = results[results['dtype'] == 'Num+Str'].copy()

    upper_33 = True # keep 33 percentile (drop median)

    if upper_33:
        percentile = 33
        df_plot['Card_Bin'] = bin_feature_33_66(df_plot, 'avg_cardinality')
        df_plot['Str_Bin'] = bin_feature_33_66(df_plot, 'avg_string_length_per_cell')
    # else:
    #     percentile = 50
    #     df_plot['Card_Bin'] = bin_feature_median(df_plot, 'avg_cardinality')
    #     df_plot['Str_Bin'] = bin_feature_median(df_plot, 'avg_string_length_per_cell')


    df_plot = df_plot[df_plot['encoder'].isin(selected_encoders)].copy()

    df_plot = df_plot[(df_plot['method'] != 'num-str_tabpfn_tabpfn_default')]

    # show all rows
    pd.set_option('display.max_rows', None)

    # Aggregate scores
    card_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'Card_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    str_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'Str_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    # Merge Low / High into columns
    card_wide = card_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='Card_Bin',
        values=score
    ).reset_index()

    str_wide = str_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='Str_Bin',
        values=score
    ).reset_index()

    rename_map = {'High': 'High-bin', 'Low': 'Low-bin'}
    card_wide = card_wide.rename(columns=rename_map)
    str_wide = str_wide.rename(columns=rename_map)

    # Combine names to create unique pipeline identifiers
    card_wide['pipeline'] = card_wide['encoder'] + " + " + card_wide['learner']
    str_wide['pipeline'] = str_wide['encoder'] + " + " + str_wide['learner']

    # --- 5. MAIN EXECUTION ---
    fig = plt.figure(figsize=(16, 8)) # Increased size for better readability

    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # --- APPLY TO PLOTS ---

    # Run Plots
    # Assumes 'High-bin' and 'Low-bin' are column names in card_wide/str_wide
    if upper_33:
        plot_regime_custom(card_wide, ax0, "Cardinality", "cardinality", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)
        plot_regime_custom(str_wide, ax1, "String Length", "string length", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)

        cardinality_positions = {
        "TabPFN-2.5 + TabPFN-2.5":       (10, 10),   # Push Right & Up
        "ContextTab + ContextTab":       (-3, -5), # Push Left & Down (avoid top edge)
        "Tf-Idf + TabPFN-2.5":     (-5, 10)  # Push Left & Down
        }

        # --- Configuration for Plot 2 (String Length) ---
        string_positions = {
            "TabPFN-2.5 + TabPFN-2.5":       (-1, 10),   # Push Right & Up
            "ContextTab + ContextTab":       (-20, -1),   # Push Left (avoid right spine)
            "Tf-Idf + TabPFN-2.5":     (-20, 2),    # Push Left harder (avoid edge)
            # "LM Qwen-3-8B + TabPFN-2.5":     (-10, -10),
            # "LM LLaMA-3.1-8B + TabPFN-2.5":   (-15, -1)
        }

        # --- APPLY SEPARATELY ---
        # apply_annotations(ax0, card_wide, cardinality_positions)
        # apply_annotations(ax1, str_wide, string_positions)

    # --- 6. LEGEND GENERATION ---
    # We build a custom legend to handle the "Tuned" look nicely
    handles, labels = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()

    # Merge
    temp_dict = dict(zip(labels + l1, handles + h1))
    sorted_labels = sorted(temp_dict.keys())
    sorted_handles = [temp_dict[k] for k in sorted_labels]

    # Place Legend
    fig.legend(
        sorted_handles, 
        sorted_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=5, 
        fontsize=12,
        frameon=False,
        columnspacing=0.05,
        handletextpad=0.1
    )


    # Layout adjustments
    plt.subplots_adjust(bottom=0.5, top=1.15, left=0.08, right=0.98)

    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'performance_per_cardinality_string_length_1by2_percentile_{percentile}_{task}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()

'''
REPEAT SAME FIGURE FOR NUM_ROWS, NUM_COLS
'''

score = score_list[0] # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

# --- 4. PLOTTING FUNCTION ---
def plot_regime_custom(df, ax, title, feature_type, set_xlim_min=0.45, set_xlim_max=1.0, set_ylim_min=0.45, set_ylim_max=1.0):
    # 1. Diagonal & Backgrounds
    lims = [0.0, 1.0]
    ax.plot(lims, lims, ls='--', c='grey', alpha=0.5, zorder=0)
    
    # Regions
    # Below Diagonal (High Bin > Low Bin) -> Red
    ax.fill_between(lims, [0,0], lims, color='red', alpha=0.05, transform=ax.transData)
    # Above Diagonal (Low Bin > High Bin) -> Green
    ax.fill_between(lims, lims, [1.1, 1.1], color='green', alpha=0.05, transform=ax.transData)
    
    # ---------------------------------------------------------
    # NEW: RANSAC REGRESSOR LOGIC (CLIPPED)
    # ---------------------------------------------------------
    if len(df) > 5: # Only fit if we have enough points
        # Prepare Data
        X = df[['High-bin']].values
        y = df['Low-bin'].values
        
        # Fit RANSAC
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        
        # 1. Generate dense prediction line across X range
        # (Use 500 points to ensure smooth clipping)
        line_X = np.linspace(set_xlim_min, set_xlim_max, 500).reshape(-1, 1)
        line_y = ransac.predict(line_X)
        
        # 2. Filter: Keep only points where Y is between 0.4 and 1.0
        mask = (line_y >= 0.4) & (line_y <= 1.0)
        
        line_X_clipped = line_X[mask]
        line_y_clipped = line_y[mask]
        
        # 3. Plot clipped line
        if len(line_X_clipped) > 1:
            ax.plot(line_X_clipped, line_y_clipped, color='cornflowerblue', linewidth=3, 
                    alpha=0.8, zorder=2, label='_nolegend_')
    # ---------------------------------------------------------

    # 2. Plot Points Loop
    unique_pipelines = df['pipeline'].unique()
    
    for pipe in unique_pipelines:
        subset = df[df['pipeline'] == pipe]
        if len(subset) == 0: continue
        
        # Parse info
        enc = subset['encoder'].iloc[0]
        lrn = subset['learner'].iloc[0]
        
        color = get_encoder_color(enc)
        marker = get_learner_marker(lrn)
        
        # Tuning Logic
        is_tuned = 'tuned' in lrn
        
        style_kwargs = {
            'color': color,
            'marker': marker,
            'markersize': 10,
            'linestyle': '',
            'alpha': 0.9,
            'zorder': 3 
        }
        
        if is_tuned:
            style_kwargs.update({
                'fillstyle': 'left',
                'markerfacecoloralt': 'white',
                'markeredgecolor': 'black',
                'markeredgewidth': 1.0
            })
        else:
            style_kwargs.update({
                'fillstyle': 'full',
                'markeredgecolor': 'white',
                'markeredgewidth': 0.5
            })
            
        # Plot
        ax.plot(subset['High-bin'], subset['Low-bin'], label=pipe, **style_kwargs)
        ax.set_xlim(set_xlim_min, set_xlim_max)
        ax.set_ylim(set_ylim_min, set_ylim_max)

    # 3. Styling & Annotations
    ax.set_title(title+' '+title_tag, fontsize=18, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 4. Descriptive Text Annotations (Same as before...)
    ax.text(0.05, 0.98, 
            f"The algorithm is better on datasets\nwith LOW {feature_type}\n", 
            transform=ax.transAxes, fontsize=16, color='darkgreen', 
            va='top', ha='left', weight='bold')
    ax.text(0.15, 0.82, "↑", transform=ax.transAxes, fontsize=20, color=f'darkgreen', weight='bold', rotation=45)
    
    # Axis Labels
    if feature_type == 'row count':
        ax.set_xlabel('Score on high row count datasets', fontsize=16, labelpad=10)
        ax.set_ylabel('Score on low row count datasets', fontsize=16, labelpad=10)
        if upper_33:
            ax.text(0.9, 0.05, 
                    f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\n", 
                    transform=ax.transAxes, fontsize=16, color='darkred', 
                    va='bottom', ha='right', weight='bold')
            ax.text(0.8, 0.25, "↓", transform=ax.transAxes, fontsize=24, color='darkred', weight='bold', rotation=45)
    else:
        ax.set_xlabel('Score on high column count datasets', fontsize=16, labelpad=10)
        ax.set_ylabel('Score on low column count datasets', fontsize=16, labelpad=10)
        if upper_33:
            ax.text(0.95, 0.1, 
                    f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\n", 
                    transform=ax.transAxes, fontsize=16, color='darkred', 
                    va='bottom', ha='right', weight='bold')
            ax.text(0.85, 0.3, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)


def apply_annotations(ax, df, position_dict):
    for target, offset in position_dict.items():
        # 1. Find the point
        row = df[df['pipeline'] == target]
        if len(row) == 0: continue
            
        x = row['High-bin'].values[0]
        y = row['Low-bin'].values[0]
        
        # 2. Determine alignment based on offset direction
        # If pushing Right (x>0), align text Left. If pushing Left (x<0), align text Right.
        ha = 'left' if offset[0] >= 0 else 'right'
        
        # If pushing Up (y>0), align Bottom. If pushing Down (y<0), align Top.
        va = 'bottom' if offset[1] >= 0 else 'top'

        # 3. Draw Annotation
        ax.annotate(
            target, 
            xy=(x, y), 
            xytext=offset,
            textcoords='offset points',
            fontsize=11, 
            fontweight='bold', 
            color='black',
            ha=ha, 
            va=va
        )
        
# task = 'classification'

for task in ['regression', 'classification', 'all_task']:

    if task == 'regression':
        title_tag = '($R^2$)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] == 'regression')].copy()
    elif task == 'classification':
        title_tag = '(AUC)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] != 'regression')].copy()
    else:
        title_tag = '(Avg $R^2$ & AUC)'
        df_plot = results[results['dtype'] == 'Num+Str'].copy()
    upper_33 = True # keep 33 percentile (drop median)

    if upper_33:
        percentile = 33
        df_plot['n_col_Bin'] = bin_feature_33_66(df_plot, 'num_columns')
        df_plot['n_row_Bin'] = bin_feature_33_66(df_plot, 'num_rows')


    df_plot = df_plot[df_plot['encoder'].isin(selected_encoders)].copy()

    df_plot = df_plot[(df_plot['method'] != 'num-str_tabpfn_tabpfn_default')]

    # Aggregate scores
    row_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'n_row_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    col_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'n_col_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    # Merge Low / High into columns
    row_wide = row_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='n_row_Bin',
        values=score
    ).reset_index()

    col_wide = col_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='n_col_Bin',
        values=score
    ).reset_index()

    rename_map = {'High': 'High-bin', 'Low': 'Low-bin'}
    row_wide = row_wide.rename(columns=rename_map)
    col_wide = col_wide.rename(columns=rename_map)

    # Combine names to create unique pipeline identifiers
    row_wide['pipeline'] = row_wide['encoder'] + " + " + row_wide['learner']
    col_wide['pipeline'] = col_wide['encoder'] + " + " + col_wide['learner']

    # --- 1. SETTINGS & STYLES ---
    sns.set_theme(style="whitegrid", context="paper")

    # --- 5. MAIN EXECUTION ---
    fig = plt.figure(figsize=(16, 8)) # Increased size for better readability

    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # --- APPLY TO PLOTS ---

    # Run Plots
    # Assumes 'High-bin' and 'Low-bin' are column names in card_wide/str_wide
    if upper_33:

        plot_regime_custom(row_wide, ax0, "Row Count", "row count", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)
        plot_regime_custom(col_wide, ax1, "Column Count", "column count", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)

        cardinality_positions = {
        "TabPFN-2.5 + TabPFN-2.5":       (10, 10),   # Push Right & Up
        "ContextTab + ContextTab":       (-3, -5), # Push Left & Down (avoid top edge)
        "Tf-Idf + TabPFN-2.5":     (-5, 10)  # Push Left & Down
        }

        # --- Configuration for Plot 2 (String Length) ---
        string_positions = {
            "TabPFN-2.5 + TabPFN-2.5":       (1, -15),   # Push Right & Up
            "ContextTab + ContextTab":       (-20, 0),   # Push Left (avoid right spine)
            "Tf-Idf + TabPFN-2.5":     (-20, 2),    # Push Left harder (avoid edge)
            # "LM Qwen-3-8B + TabPFN-2.5":     (-10, -10),
            # "LM LLaMA-3.1-8B + TabPFN-2.5":   (-15, -1)
        }

        # --- APPLY SEPARATELY ---
        # apply_annotations(ax0, card_wide, cardinality_positions)
        # apply_annotations(ax1, str_wide, string_positions)

    # --- 6. LEGEND GENERATION ---
    # We build a custom legend to handle the "Tuned" look nicely
    handles, labels = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()

    # Merge
    temp_dict = dict(zip(labels + l1, handles + h1))
    sorted_labels = sorted(temp_dict.keys())
    sorted_handles = [temp_dict[k] for k in sorted_labels]

    # Place Legend
    fig.legend(
        sorted_handles, 
        sorted_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=5, 
        fontsize=12,
        frameon=False,
        columnspacing=0.05,
        handletextpad=0.1
    )


    # Layout adjustments
    plt.subplots_adjust(bottom=0.5, top=1.15, left=0.08, right=0.98)
    
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'performance_per_row_col_count_1by2_percentile_{percentile}_{task}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()

'''
PERFORMANCE BY N_GRAM (STRING DIVERSITY)
'''

score = score_list[0] # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

def plot_regime_custom(df, ax, title, feature_type, set_xlim_min=0.45, set_xlim_max=1.0, set_ylim_min=0.45, set_ylim_max=1.0):
    # 1. Diagonal & Backgrounds
    lims = [0.0, 1.0]
    ax.plot(lims, lims, ls='--', c='grey', alpha=0.5, zorder=0)
    
    # Regions
    # Below Diagonal (High Bin > Low Bin) -> Red
    ax.fill_between(lims, [0,0], lims, color='red', alpha=0.05, transform=ax.transData)
    # Above Diagonal (Low Bin > High Bin) -> Green
    ax.fill_between(lims, lims, [1.1, 1.1], color='green', alpha=0.05, transform=ax.transData)
    
    # ---------------------------------------------------------
    # NEW: RANSAC REGRESSOR LOGIC (CLIPPED)
    # ---------------------------------------------------------
    if len(df) > 5: # Only fit if we have enough points
        # Prepare Data
        X = df[['High-bin']].values
        y = df['Low-bin'].values
        
        # Fit RANSAC
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        
        # 1. Generate dense prediction line across X range
        # (Use 500 points to ensure smooth clipping)
        line_X = np.linspace(set_xlim_min, set_xlim_max, 500).reshape(-1, 1)
        line_y = ransac.predict(line_X)
        
        # 2. Filter: Keep only points where Y is between 0.4 and 0.9
        mask = (line_y >= 0.4) & (line_y <= 0.9)
        
        line_X_clipped = line_X[mask]
        line_y_clipped = line_y[mask]
        
        # 3. Plot clipped line
        if len(line_X_clipped) > 1:
            ax.plot(line_X_clipped, line_y_clipped, color='cornflowerblue', linewidth=3, 
                    alpha=0.8, zorder=2, label='_nolegend_')
    # ---------------------------------------------------------

    # 2. Plot Points Loop
    unique_pipelines = df['pipeline'].unique()
    
    for pipe in unique_pipelines:
        subset = df[df['pipeline'] == pipe]
        if len(subset) == 0: continue
        
        # Parse info
        enc = subset['encoder'].iloc[0]
        lrn = subset['learner'].iloc[0]
        
        color = get_encoder_color(enc)
        marker = get_learner_marker(lrn)
        
        # Tuning Logic
        is_tuned = 'tuned' in lrn
        
        style_kwargs = {
            'color': color,
            'marker': marker,
            'markersize': 10,
            'linestyle': '',
            'alpha': 0.9,
            'zorder': 3 
        }
        
        if is_tuned:
            style_kwargs.update({
                'fillstyle': 'left',
                'markerfacecoloralt': 'white',
                'markeredgecolor': 'black',
                'markeredgewidth': 1.0
            })
        else:
            style_kwargs.update({
                'fillstyle': 'full',
                'markeredgecolor': 'white',
                'markeredgewidth': 0.5
            })
            
        # Plot
        ax.plot(subset['High-bin'], subset['Low-bin'], label=pipe, **style_kwargs)
        ax.set_xlim(set_xlim_min, set_xlim_max)
        ax.set_ylim(set_ylim_min, set_ylim_max)

    # 3. Styling & Annotations
    ax.set_title(title+' '+title_tag, fontsize=18, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 4. Descriptive Text Annotations (Same as before...)
    ax.text(0.05, 0.98, 
            f"The algorithm is better on datasets\nwith LOW {feature_type}\n  ", 
            transform=ax.transAxes, fontsize=10, color='darkgreen', 
            va='top', ha='left', weight='bold')
    ax.text(0.15, 0.72, "↑", transform=ax.transAxes, fontsize=20, color=f'darkgreen', weight='bold', rotation=45)
    
    # Axis Labels
    ax.set_xlabel('Score on high string diversity datasets', fontsize=16, labelpad=10)
    ax.set_ylabel('Score on low string diversity datasets', fontsize=16, labelpad=10)
    ax.text(0.95, 0.05, 
            f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\n", 
            transform=ax.transAxes, fontsize=10, color='darkred', 
            va='bottom', ha='right', weight='bold')
    ax.text(0.4, 0.2, "↓", transform=ax.transAxes, fontsize=24, color='darkred', weight='bold', rotation=45)

def apply_annotations(ax, df, position_dict):
    for target, offset in position_dict.items():
        # 1. Find the point
        row = df[df['pipeline'] == target]
        if len(row) == 0: continue
            
        x = row['High-bin'].values[0]
        y = row['Low-bin'].values[0]
        
        # 2. Determine alignment based on offset direction
        # If pushing Right (x>0), align text Left. If pushing Left (x<0), align text Right.
        ha = 'left' if offset[0] >= 0 else 'right'
        
        # If pushing Up (y>0), align Bottom. If pushing Down (y<0), align Top.
        va = 'bottom' if offset[1] >= 0 else 'top'

        # 3. Draw Annotation
        ax.annotate(
            target, 
            xy=(x, y), 
            xytext=offset,
            textcoords='offset points',
            fontsize=12, 
            fontweight='bold', 
            color='black',
            ha=ha, 
            va=va
        )

for task in ['regression', 'classification', 'all_task']:

    if task == 'regression':
        title_tag = '($R^2$)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] == 'regression')].copy()
    elif task == 'classification':
        title_tag = '(AUC)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] != 'regression')].copy()
    else:
        title_tag = '(Avg $R^2$ & AUC)'
        df_plot = results[results['dtype'] == 'Num+Str'].copy()


    upper_33 = True # keep 33 percentile (drop median)

    if upper_33:
        percentile = 33
        df_plot['string_diversity_Bin'] = bin_feature_33_66(df_plot, 'string_diversity')


    df_plot = df_plot[df_plot['encoder'].isin(selected_encoders)].copy()

    df_plot = df_plot[(df_plot['method'] != 'num-str_tabpfn_tabpfn_default')]

    # Aggregate scores
    string_diversity_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'string_diversity_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    # Merge Low / High into columns
    string_diversity_wide = string_diversity_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='string_diversity_Bin',
        values=score
    ).reset_index()

    rename_map = {'High': 'High-bin', 'Low': 'Low-bin'}
    string_diversity_wide = string_diversity_wide.rename(columns=rename_map)

    # Combine names to create unique pipeline identifiers
    string_diversity_wide['pipeline'] = string_diversity_wide['encoder'] + " + " + string_diversity_wide['learner']

    # --- 1. SETTINGS & STYLES ---
    sns.set_theme(style="whitegrid", context="paper")

    # --- 5. MAIN EXECUTION ---
    fig = plt.figure(figsize=(5, 4)) 

    gs = GridSpec(1, 1)
    gs.update(left=0.15, right=0.99, top=0.99, bottom=0.05)
    ax0 = fig.add_subplot(gs[0])

    # --- 2. PLOTTING ---
    if upper_33:
        # Run the plot on ax0
        plot_regime_custom(string_diversity_wide, ax0, "String Diversity", "string diversity", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)

        # Note: I commented out the annotations since you had them commented
        # apply_annotations(ax0, card_wide, cardinality_positions)

    # --- 3. LEGEND GENERATION ---
    # Fix: Only get handles from ax0 (ax1 does not exist in this configuration)
    handles, labels = ax0.get_legend_handles_labels()

    # Sort logic
    temp_dict = dict(zip(labels, handles))
    sorted_labels = sorted(temp_dict.keys())
    sorted_handles = [temp_dict[k] for k in sorted_labels]

    # Place Legend
    # Changed to 'center left' at (1.02, 0.5) to anchor it right next to the plot
    fig.legend(
        sorted_handles, 
        sorted_labels,
        loc='center left',      # Anchor the left center of the legend...
        bbox_to_anchor=(1.02, 0.5), # ...to the right edge of the figure
        ncol=2, 
        fontsize=10,            # Reduced slightly to fit better
        frameon=False,
        columnspacing=0.05,
        handletextpad=0.1
    )

        
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'performance_per_n_gram_1by2_percentile_{percentile}_{task}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()


'''
AVERAGE PERFORMANCE OF ENCODER (NUM+STR) VS KENDALLTAU CORRELATION(KENDALLTAU CORRELATION(NUM+STR VS NUM))
'''
dtype = 'Num+Str'
score = 'score'
encoder_performance_byavg = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(['encoder'], as_index=False)[score].mean()

encoder_performance_bymedian = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(['encoder'], as_index=False)[score].median()
encoder_performance_bymedian.rename(columns={score: 'score_median'}, inplace=True)

encoder_performance_bymax = (
    results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(['encoder','learner'], as_index=False)[score].mean()
    .sort_values(by=['encoder',score], ascending=[True,False])
    .groupby('encoder').head(1)[['encoder','learner', score]]
    .sort_values(by=score, ascending=False)
    .reset_index(drop=True)
)

merge_df = pd.merge(encoder_performance_byavg, encoder_performance_bymax, on='encoder', suffixes=('_avg', '_max'))
merge_df = pd.merge(merge_df, encoder_performance_bymedian, on='encoder')
merge_df.rename(columns={'learner':'learner_max'}, inplace=True)

df_pivot = results[(results['dtype'].isin(['Num', 'Num+Str'])) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].pivot_table(
    index=['encoder', 'learner'],
    columns='dtype',
    values=score,
    aggfunc='mean'  # Average if there are multiple runs/folds
)

def calculate_tau(group):
    # Drop learners that don't have BOTH scores (avoids errors)
    valid_data = group[['Num', 'Num+Str']]
            
    tau, _ = kendalltau(valid_data['Num'], valid_data['Num+Str'])
    return tau

# show all rows
pd.set_option('display.max_rows', None)

df_pivot['Num'] = df_pivot['Num'].fillna(
    df_pivot.groupby(level='learner')['Num'].transform('mean')
)

# compute difference between Num+Str and Num
# df_pivot['difference'] = df_pivot['Num+Str'] - df_pivot['Num']

# merge_df['avg_difference'] = df_pivot.groupby('encoder', as_index=False)['difference'].mean()

df_pivot.dropna(subset=['Num+Str'], inplace=True)

merge_df['kendall_tau'] = df_pivot.groupby('encoder', as_index=False).apply(calculate_tau)

merge_df.sort_values(by='score_max', ascending=False)

merge_df.sort_values(by=['kendall_tau','score_max'], ascending=[True,False])

# drop E2E
merge_df.dropna(axis=0, inplace = True)
merge_df = merge_df[merge_df['encoder'] != 'CatBoost']
merge_df.reset_index(drop=True, inplace=True)

#plot BY MAX
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(4, 3))

# Scatter Plot
sns.scatterplot(
    data=merge_df, 
    x='score_max', 
    y='kendall_tau', 
    s=200,          # Size of markers
    color='#1f77b4', # Standard blue
    edgecolor='black', 
    alpha=0.8,
    ax=ax
)

#check the table
# merge_df[['encoder', 'score_max', 'kendall_tau']]

# 3. Add Labels for each point (Encoder names)
ax.text(0.699576+0.002, 0.400000 + 0.02, 'Tarte', fontsize=9, weight='bold',color='#333333')
ax.text(0.694746+0.002, 0.0 + 0.02, 'LM Jasper-0.6B', fontsize=9, weight='bold',color='#333333')
ax.text(0.718, -0.1, 'LM LLaMA-3.1-8B', fontsize=9, weight='bold',color='#333333')
ax.text(0.715, -0.25, 'LM Qwen-3-8B', fontsize=9, weight='bold',color='#333333')
ax.text(0.695, 0.866667 + 0.02, 'TargetEncoder', fontsize=9, weight='bold',color='#333333')
ax.text(0.744753+0.002, 0.866667 + 0.02, 'Tf-Idf', fontsize=9, weight='bold',color='#333333')
ax.text(0.722780-0.006, 0.81, 'LM All-MiniLM-L6-v2', fontsize=9, weight='bold',color='#333333')
ax.text(0.703, 0.695, 'LM E5-small-v2', fontsize=9, weight='bold',color='#333333')
ax.text(0.726188+0.002, 0.695, 'LM FastText', fontsize=9, weight='bold',color='#333333')


# 4. Styling
ax.set_xlabel('Maximum Score Achieved by Encoder', labelpad=10, fontsize=12)
ax.set_ylabel('Kendall $\\tau$(Num+Str,Num)', labelpad=10, fontsize=12)

# Optional: Add reference lines
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
sns.despine()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'kendalltau_per_max_score_encoder_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


#plot BY MEDIAN
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(4, 3))

# Scatter Plot
sns.scatterplot(
    data=merge_df, 
    x='score_median', 
    y='kendall_tau', 
    s=200,          # Size of markers
    color='#1f77b4', # Standard blue
    edgecolor='black', 
    alpha=0.8,
    ax=ax
)

#check the table
# merge_df[['encoder', 'score_median', 'kendall_tau']].sort_values(by=['kendall_tau'], ascending=False)


# 3. Add Labels for each point (Encoder names)
ax.text(0.63, 0.9, 'TargetEncoder', fontsize=9, weight='bold',color='#333333')
ax.text(0.75, 0.9, 'Tf-Idf', fontsize=9, weight='bold',color='#333333')
ax.text(0.63, 0.45, 'Tarte', fontsize=9, weight='bold',color='#333333')
ax.text(0.65, 0.70, 'LM FastText', fontsize=9, weight='bold',color='#333333')
ax.text(0.7, 0.75, 'LM All-MiniLM-L6-v2', fontsize=9, weight='bold',color='#333333')
ax.text(0.73, 0.70, 'LM E5-small-v2', fontsize=9, weight='bold',color='#333333')
ax.text(0.696320+0.002, 0.0 + 0.15, 'LM Jasper-0.6B', fontsize=9, weight='bold',color='#333333')
ax.text(0.718, -0.1, 'LM LLaMA-3.1-8B', fontsize=9, weight='bold',color='#333333')
ax.text(0.718, -0.25, 'LM Qwen-3-8B', fontsize=9, weight='bold',color='#333333')


# 4. Styling
ax.set_xlabel('Median Score Achieved by Encoder', labelpad=10, fontsize=12)
ax.set_ylabel('Kendall $\\tau$(Num+Str,Num)', labelpad=10, fontsize=12)

# Optional: Add reference lines
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(-0.3,1)
ax.set_xlim(0.6,0.85)
sns.despine()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'kendalltau_per_median_score_encoder_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
ranking change of learners for different encoders between numeric and num+str
'''

# 1. Prepare Data: Calculate Ranks for each (Encoder, Learner) pair
# We utilize the pivot table 'df_pivot' from your previous code
# Rows: (Encoder, Learner), Cols: [Num, Num+Str]

# Reset index to work with a flat dataframe
df_ranks = df_pivot.reset_index()

# Define ranking function (Higher score = Lower Rank #1)
def get_learner_ranks(df, score_col):
    df[f'rank_{score_col}'] = df[score_col].rank(ascending=False)
    return df

# Apply ranking within each encoder group for the 'Num+Str' score
df_ranks = df_ranks.groupby('encoder', group_keys=False).apply(lambda x: get_learner_ranks(x, 'Num+Str'))

# For 'Num' column, the ranking is theoretically identical across all encoder groups 
# (since the encoder doesn't affect the Num-only score). 
# We calculate the global 'Num' ranking once to ensure consistency.
numeric_scores = df_ranks[['learner', 'Num']].drop_duplicates('learner').set_index('learner')['Num']
numeric_ranks = numeric_scores.rank(ascending=False)
df_ranks['rank_Num'] = df_ranks['learner'].map(numeric_ranks)

# 2. Select Representative Encoders to Plot
# We contrast "Stable" (High Tau) vs "Shifty" (Low Tau) encoders

# drop TF-IDF for clarity AND USE PARETO OPTIMALITY LLM ORDER
encoders_to_plot = results[(results['dtype'].isin(['Num+Str'])) & (results['encoder'].isin(['TargetEncoder', 'Tf-Idf','LM FastText', 'LM E5-small-v2', 'LM All-MiniLM-L6-v2', 'LM Jasper-0.6B', 'LM Qwen-3-8B', 'LM LLaMA-3.1-8B', 'Tarte']))].groupby(['encoder'], as_index=False)['run_time_per_1k'].mean().sort_values(by='run_time_per_1k', ascending=True)['encoder'].to_list()
# (results['learner']!='ExtraTrees-tuned') & 
fig, axes = plt.subplots(1, 9, figsize=(24, 10), sharey=True)

# Define X-axis points
x_points = [0, 1]
x_labels = ['Numeric\nOnly', 'Numeric\n+ String']

# Iterate over the selected encoders
for idx, enc in enumerate(encoders_to_plot):
    ax = axes[idx]
    
    # Get data for this specific encoder
    data = df_ranks[df_ranks['encoder'] == enc].copy()
    
    # Plot lines for each learner
    for _, row in data.iterrows():
        learner = row['learner']
        # if learner == 'ExtraTrees-tuned':
        #     continue  # Skip ExtraTrees-tuned
        rank_start = row['rank_Num']
        rank_end = row['rank_Num+Str']
        
        # Style using your existing maps
        color = get_learner_color_simple(learner)
        marker = get_learner_marker(learner)
        
        # Plot Line connecting Start Rank -> End Rank
        ax.plot(x_points, [rank_start, rank_end], 
                color=color, marker=marker, markersize=10, 
                linewidth=2.5, alpha=0.8)
        
        # Add text labels to the left of the first plot for learner names
        if idx == 0: 
            ax.text(-0.15, rank_start+0.25, learner, ha='right', va='center', 
                    fontsize=15, color=color, fontweight='bold')
            
    # Styling the Subplot
    ax.set_xticks(x_points)
    ax.set_xticklabels(x_labels, fontsize=16, fontweight='bold')
    ax.set_title(f"{enc}", fontsize=16, fontweight='bold', pad=15)
    
    # Invert Y axis so Rank 1 is at the top
    if idx == 0:
        ax.invert_yaxis()
        ax.set_ylabel("Learner Rank (1 = Best)", fontsize=20, labelpad=45)
        # Set integer ticks only
        max_rank = int(data['rank_Num'].max())
        ax.set_yticks(range(1, max_rank + 1))
    
    # Add vertical grid lines only
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.grid(axis='y', linestyle='-', alpha=0.1) # Faint horizontal guide
    
    # Add Tau value to title/annotation
    # (Assuming merge_df exists from your previous code)
    if 'merge_df' in locals():
        try:
            tau = merge_df[merge_df['encoder'] == enc]['kendall_tau'].values[0]
            ax.text(0.5, 0.98, f"Stability $\\tau$ = {tau:.2f}", 

                    transform=ax.transAxes, ha='center', color='dimgray', fontsize=14, style='italic')
        except:
            pass

    sns.despine(left=True, bottom=True)


plt.tight_layout(w_pad=1.0)

# Save & Show
today_date = time.strftime("%Y-%m-%d")
PIC_NAME = f'learner_rank_bump_chart_{today_date}.pdf'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
SAMPLING DIAGRAM
Integrate Marine's comments
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_schema():
    # Increased height to fit the new step comfortably
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # --- Styles ---
    # Blue: Data Pools
    box_style = dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1565C0", lw=2)
    # Green: Sampling
    sample_style = dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#2E7D32", lw=2)
    # Purple: Evaluation
    eval_style = dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec="#7B1FA2", lw=2)
    # Orange: Ranking (Processing)
    proc_style = dict(boxstyle="ellipse,pad=0.3", fc="#FFF3E0", ec="#EF6C00", lw=2)
    # Red: Final Metric
    tau_style = dict(boxstyle="darrow,pad=0.3", fc="#FFEBEE", ec="#C62828", lw=2)
    
    # Added shrinkA=0, shrinkB=0 to ensure the arrow doesn't stop short of the coordinates
    arrow_props = dict(arrowstyle="->", color="#555555", lw=2, mutation_scale=15, shrinkA=0, shrinkB=0)

    # --- Y-Coordinates for Layers (Centers) ---
    y_main = 9.8
    y_part = 8.5
    y_sub = 6.8
    y_eval = 4.8
    y_rank = 2.8
    y_tau = 1.0

    # --- Half-Heights (Approximate distance from center to edge) ---
    h_main = 0.45
    h_part = 0.45
    h_sub = 0.45
    h_eval = 0.55  # Taller because of 3 lines
    h_rank = 0.45
    h_tau = 0.5

    # --- 1. Main Pool ---
    ax.text(5, y_main+0.5, "Total Datasets Pool\n($M$ Datasets)", ha="center", va="center", size=12, bbox=box_style)

    # --- 2. Partitions (Disjoint) ---
    # Arrows: Start at Bottom of Main, End at Top of Partitions
    ax.annotate("", xy=(3, y_part + h_part), xytext=(5, y_main - h_main+0.5), arrowprops=arrow_props, zorder=0)
    ax.annotate("", xy=(7, y_part + h_part), xytext=(5, y_main - h_main+0.5), arrowprops=arrow_props, zorder=0)
    
    ax.text(3, y_part, "Partition A\n($M/2$ datasets)", ha="center", va="center", size=11, bbox=box_style)
    ax.text(7, y_part, "Partition B\n($M/2$ datasets)", ha="center", va="center", size=11, bbox=box_style)
    
    ax.text(5, 9.3, "Random Split (Disjoint)", ha="center", va="center", size=9, color="#555555", 
            bbox=dict(fc="white", ec="none"), zorder=0)

    # --- LOOP BOX ---
    rect = patches.FancyBboxPatch((0.5, 1.8), 9, 5.8, boxstyle="round,pad=0.2", 
                                  linewidth=2, edgecolor='gray', facecolor='none', linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(0.5, 7.0, "Bootstrap \n $K$ times", size=10, color="gray", weight="bold")

    # --- 3. Sampling (Green) ---
    # Start at Bottom of Partitions, End at Top of Subsamples
    ax.annotate("", xy=(3, y_sub + h_sub), xytext=(3, y_part - h_part), arrowprops=arrow_props)
    ax.annotate("", xy=(7, y_sub + h_sub), xytext=(7, y_part - h_part), arrowprops=arrow_props)
    
    ax.text(3, y_sub, "Subsample $S_1$\n(Size $N$)", ha="center", va="center", size=11, bbox=sample_style)
    ax.text(7, y_sub, "Subsample $S_2$\n(Size $N$)", ha="center", va="center", size=11, bbox=sample_style)

    # --- 4. Evaluation (Purple) ---
    # Start at Bottom of Subsamples, End at Top of Eval
    ax.annotate("", xy=(3, y_eval + h_eval), xytext=(3, y_sub - h_sub), arrowprops=arrow_props)
    ax.annotate("", xy=(7, y_eval + h_eval), xytext=(7, y_sub - h_sub), arrowprops=arrow_props)
    
    eval_text = "Evaluate Avg Score\n(R2/AUC) per Model\nacross $N$ datasets"
    ax.text(3, y_eval, eval_text, ha="center", va="center", size=10, bbox=eval_style)
    ax.text(7, y_eval, eval_text, ha="center", va="center", size=10, bbox=eval_style)

    # --- 5. Rankings (Orange) ---
    # Start at Bottom of Eval, End at Top of Rankings
    ax.annotate("", xy=(3, y_rank + h_rank), xytext=(3, y_eval - h_eval), arrowprops=arrow_props)
    ax.annotate("", xy=(7, y_rank + h_rank), xytext=(7, y_eval - h_eval), arrowprops=arrow_props)
    
    ax.text(3, y_rank, "Compute Rankings\n$R_1$", ha="center", va="center", size=11, bbox=proc_style)
    ax.text(7, y_rank, "Compute Rankings\n$R_2$", ha="center", va="center", size=11, bbox=proc_style)

    # --- 6. Kendall Tau (Red) ---
    # Start at Bottom of Rankings, End at Top of Kendall
    ax.annotate("", xy=(5, y_tau + h_tau), xytext=(3, y_rank - h_rank), arrowprops=arrow_props, zorder=1)
    ax.annotate("", xy=(5, y_tau + h_tau), xytext=(7, y_rank - h_rank), arrowprops=arrow_props, zorder=1)
    
    ax.text(5, y_tau, "Kendall $\\tau(R_1, R_2)$", ha="center", va="center", size=12, bbox=tau_style)

    plt.tight_layout()
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'sampling_diagram_two_benchmarks_convergence_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()

draw_schema()



'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
the benchmark to converge to the same ranking?
BOOTSTRAPPED CURVE FITTING (EXTRAPOLATION)
'''

results_copy = results[(results['dtype']=='Num+Str') & (results['method'] != 'num-str_tabpfn_tabpfn_default')].copy()

pivot_df = results_copy.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# PAY ATTENTION TO DROP NANS IN THE CORRECT DIMENSION
pivot_df = pivot_df.dropna(axis=1)  

# pivot_df = pivot_df.dropna(axis=0)  

df = pivot_df.copy()

# take only those index that contain Num+Str in any column name
df = df[[col for col in df.columns if 'Num+Str' in col]]

# CLEANING: Drop rows with NaN if any algorithm failed on a dataset
# (Rankings break if one algorithm has a NaN and others don't)
# if df.isna().sum().sum() > 0:
#     print(f"Warning: Dropping {df.isna().any(axis=1).sum()} datasets containing NaN values.")
#     df = df.dropna()

n_datasets = len(df)
print(f"Total valid datasets for analysis: {n_datasets}")

# split pivot_df into 2 subsamples for kendall-tau (of equal size)
mid_point = n_datasets // 2
# sample randomly for df_subsample1, compute size, and sample the same size from rest for df_subsample2
indices = df.index.tolist()

# Parameters
# We start at N=10 and go up to the full dataset count in steps of 5
sample_sizes = range(10, mid_point + 1, 1) 
n_iterations = 2000 
stability_scores = []

# Step B: Bootstrapping Loop
for n in sample_sizes:
    print(f"Running simulations for N={n} datasets...")
    for i in range(n_iterations):
        print(f"  Iteration {i+1}/{n_iterations}...")
        random.shuffle(indices)
        df_subsample1 = df.loc[indices[:mid_point], :]
        df_subsample2 = df.loc[indices[mid_point:mid_point*2], :]

        # print("sizes of subsamples:", df_subsample1.shape, df_subsample2.shape)

        # check that sampled datasets in each subsample are different
        assert len(set(df_subsample1.index).intersection(set(df_subsample2.index))) == 0, "Subsamples overlap!"

        # 1. Subsample N datasets (randomly select N rows)
        subset1 = df_subsample1.sample(n=n, replace=False)
        subset2 = df_subsample2.sample(n=n, replace=False)
        
        # 2. Compute rankings on this subset
        subset_rankings_1 = calculate_rankings(subset1)
        subset_rankings_2 = calculate_rankings(subset2)
        
        # 3. Compare subset ranking vs. true ranking
        # We use kendall-tau correlation to see if the ordering is preserved
        corr, _ = kendalltau(subset_rankings_1, subset_rankings_2)
        
        stability_scores.append({
            'N_datasets': n,
            'Kendalltau_Correlation': corr
        })

df_stability = pd.DataFrame(stability_scores)

# Setup
n_bootstraps = 2000
target_y = 0.95
# Calculate the disagreement percentage based on the Kendall Tau formula: (1 - tau) / 2
disagreement_pct = ((1 - target_y) / 2) * 100
max_plot_x = 3000
x_range_smooth = np.linspace(25, max_plot_x, 300)

# Storage for results
results_kendalltau_extrap = {
    'ref1': {'popt': [], 'curves': [], 'preds': []}
}

print(f"Starting {n_bootstraps} bootstrap iterations...")

for k in range(n_bootstraps):
    boot_sample = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
    df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
    X, Y = df_agg['N_datasets'], df_agg['Kendalltau_Correlation']
    
    # --- FIT MODEL REF 1 ---
    try:
        # Fixed asymptote at 1.0
        p_r1, _ = curve_fit(model_ref_1, X, Y, p0=[0.5, 0.05], bounds=([0, 0], [10, np.inf]))
        
        # 1. STORE THE PARAMETERS HERE
        results_kendalltau_extrap['ref1']['popt'].append(p_r1)
        
        # Finding N requires numerical solver since N is in sqrt and exp
        
        func = lambda n: model_ref_1(n, *p_r1) - target_y
        req_N = fsolve(func, x0=50)[0]
        if 0 < req_N < 10000:
            results_kendalltau_extrap['ref1']['preds'].append(req_N)
            results_kendalltau_extrap['ref1']['curves'].append(model_ref_1(x_range_smooth, *p_r1))
    except: pass

# Convert the list of arrays into a 2D NumPy array (rows=bootstraps, cols=[a, b])
all_popt = np.array(results_kendalltau_extrap['ref1']['popt'])

# compute the line before 25
x_range_dotted = np.linspace(3, 25, 50)

# 2. Re-generate curves for this specific range using your saved bootstrap parameters
#    (We use all_popt which you created earlier: all_popt = np.array(...))
curves_dotted_list = []
for p in all_popt:
    curves_dotted_list.append(model_ref_1(x_range_dotted, *p))

# 3. Calculate the median curve (consistent with your green line logic)
med_line_dotted = np.median(np.array(curves_dotted_list), axis=0)

# Calculate the optimal (mean) parameters
mean_params = np.mean(all_popt, axis=0)
print(f"Optimal parameter a: {mean_params[0]:.4f}")
print(f"Optimal parameter b: {mean_params[1]:.4f}")

# Calculate 95% Confidence Intervals
ci_lower = np.percentile(all_popt, 2.5, axis=0)
ci_upper = np.percentile(all_popt, 97.5, axis=0)

print(f"95% CI for a: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
print(f"95% CI for b: [{ci_lower[1]:.4f}, {ci_upper[1]:.4f}]")

tau_for_strable_benchmark = 1 - (mean_params[0]/np.sqrt(n_datasets)) * np.exp(-mean_params[1]*n_datasets)
disagreement_pct_strable = ((1 - tau_for_strable_benchmark) / 2) * 100

# --- VISUALIZATION ---
plt.rcParams.update({'font.size': 8}) # Standard academic base size
fig, ax = plt.subplots(figsize=(5, 4))

# 1. Plot Observed Data
df_real_agg = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['median', 'sem'])
ax.errorbar(df_real_agg['N_datasets'], df_real_agg['median'], yerr=df_real_agg['sem'], 
             fmt='o', color='blue', markersize=3, elinewidth=0.8, 
             label='Observed (Median ± SE)', zorder=10)

# 2. Extract and Plot Bootstrap CI Band
if len(results_kendalltau_extrap['ref1']['curves']) > 0:
    curves = np.array(results_kendalltau_extrap['ref1']['curves'])
    med_line = np.median(curves, axis=0)
    # Calculate 95% Confidence Interval
    low_ci = np.percentile(curves, 2.5, axis=0)
    high_ci = np.percentile(curves, 97.5, axis=0)
    
    # Render the Model and the Confidence Interval
    # ax.fill_between(x_range_smooth, low_ci, high_ci, color='green', alpha=0.4, 
    #             edgecolor='green', linewidth=0.5, label='95% Bootstrap CI')
    ax.plot(x_range_dotted, med_line_dotted, color='green', 
            linewidth=1.5, linestyle=':') # Dotted style
    
    ax.plot(x_range_smooth, med_line, color='green', linewidth=1.5, 
            label=r'$1 - \frac{a}{\sqrt{N}} * e^{-bN}$')
    
    x_range_smooth_oracle = np.linspace(3, max_plot_x, 300)
    oracle_curve = 1 - (mean_params[0] / (2 * np.sqrt(x_range_smooth_oracle))) * np.exp(-mean_params[1] * x_range_smooth_oracle)
    
    ax.plot(x_range_smooth_oracle, oracle_curve, color="#A900D3", linestyle='--', linewidth=2, 
            label=r'Oracle Correlation: $1 - \frac{a}{2\sqrt{N}} e^{-bN}$')

# 3. Reference Lines & STRABLE Metrics
ax.axvline(x=n_datasets, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=tau_for_strable_benchmark, color='black', linestyle=':', linewidth=1)

# Shortened annotations for 6x3 readability
ax.text(n_datasets - 35, 0.83, f"STRABLE\nsize: {n_datasets}", 
        color='red', fontweight='bold', fontsize=10)

# Calculate and display disagreement succinctly
annot_text = (f"$\\tau={tau_for_strable_benchmark:.1f}$\n"
              f"Disagreement:{disagreement_pct_strable:.1f}%")
ax.annotate(annot_text, xy=(n_datasets, tau_for_strable_benchmark), 
            xytext=(111, 0.865),
            # arrowprops=dict(arrowstyle="->", color='gray'),
            fontsize=9
            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )

ax.text(10, 0.925, "asymptotic agreement to oracle\n(theoretical correction)", 
        color='#A900D3', 
        fontsize=9, 
        fontweight='bold', 
        rotation=10, 
        ha='left', va='bottom')

ax.text(45, 0.88, "asymptotic agreement of\ntwo independent benchmarks", 
        color='green', 
        fontsize=9, 
        fontweight='bold', 
        rotation=10, 
        ha='left', va='bottom')

# --- FINAL POLISH ---
ax.set_xlabel('Number of Datasets (N)', fontsize=12)
ax.set_ylabel('Kendall $\\tau$ correlation\nbetween two benchmarks', fontsize=12)
# add a second y axis on the right
# ax2 = ax.twinx()
# ax2.set_ylabel('Kendall $\\tau$ correlation to the oracle rank', fontsize=12)
# ax2.set_ylim(0.69, 1.0)
ax.legend(loc='lower right', fontsize=11, frameon=True)
ax.grid(True, alpha=0.2)
ax.set_ylim(0.69, 1.0)
ax.set_xlim(-5, 170)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_greenfunction_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
LEAVE-ONE-CATEGORY-OUT V4:
'''

# 1. Prepare the Pivot Table
# Ensure we have one score per model per dataset

results_copy = results[(results['dtype']=='Num+Str') & (results['method'] != 'num-str_tabpfn_tabpfn_default')].copy()

pivot_source = results_copy.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# select only algos that are Num+Str
pivot_source = pivot_source[[col for col in pivot_source.columns if 'Num+Str' in col]].copy()

# Drop any methods/datasets that have NaN values
pivot_source = pivot_source.dropna(axis=1) #drop columns with NaN

# pivot_source = pivot_source.dropna(axis=0) #drop columns with NaN

valid_datasets = pivot_source.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['category_with_ds_count'].first()

# 2. LOSO Calculation Loop
sources = results['category_with_ds_count'].unique()
loso_correlations = []

for src in sources:
    # Split datasets into current source and all other sources
    datasets_in_source = dataset_to_source[dataset_to_source == src].index
    
    # Filter pivot table
    df_src = pivot_source.loc[datasets_in_source]
    
    # Vector 1: Rank of each model on Source_i (averaged across datasets in that source)
    # Higher score is better -> ascending=False
    ranks_src = df_src.mean().rank(ascending=False)

    print(f"{len(ranks_src)} models ranked on category {src}.")
    
    # Vector 2: Average rank of each model on all other categories
    # We rank per dataset first, then average those ranks
    ranks_others_per_ds = pivot_source.rank(axis=1, ascending=False)
    avg_ranks_others = ranks_others_per_ds.mean()

    print(f"{len(avg_ranks_others)} models ranked on other categories excluding {src}.")

    # check that both vectors have the same models
    print(f"Category: {src}")
    assert set(ranks_src.index) == set(avg_ranks_others.index), "Model mismatch between category and others"
    # Calculate Kendall Tau between the two ranking vectors
    corr, _ = kendalltau(ranks_src, avg_ranks_others)
    
    loso_correlations.append({
        'Category': src,
        'Correlation': corr,
        'N_datasets': len(datasets_in_source)
    })

df_loso = pd.DataFrame(loso_correlations).sort_values('Correlation', ascending=False)
#drop rows with NaN correlation
df_loso = df_loso.dropna(subset=['Correlation'])

# Visualization - Transposed
plt.figure(figsize=(6, 5)) # Increased height to accommodate Y-axis labels
colors = plt.cm.viridis(np.linspace(0, 1, len(df_loso)))

# Use barh for horizontal bars
bars = plt.barh(df_loso['Category'], df_loso['Correlation'], color=colors, alpha=0.8)

# Add value labels for each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.01,                # Position slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2, # Center vertically in the bar
        f'{width:.3f}',             # The Kendall Tau value
        va='center', 
        fontsize=16, 
        fontweight='bold'
    )

# Formatting
# plt.title('Leave-One-Category-Out: $\\tau$(Category_i, Benchmark)', fontsize=16, pad=20)
plt.xlabel('Kendall $\\tau$ Rank Correlation', fontsize=16)
plt.ylabel('Excluded Category ($Category_i$)', fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0, 1.0) # Extended limit to make room for text labels
plt.grid(axis='x', alpha=0.3)
plt.legend(loc='lower right')

# plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_v4_check_transposed_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
cosine similarity heatmap
'''
from datasets_metadata_recap import wide_datasets as data_list_wide
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- 1. DATA PREPARATION FUNCTION ---
def prepare_tabvec(X_raw):
    """Generates embeddings using skrub's StringEncoder."""
    from skrub import StringEncoder, TableVectorizer, SquashingScaler
    
    # TableVectorizer to clean and handle types
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_cleaned = cleaner.fit_transform(X_raw)

    # Encode using StringEncoder for high cardinality (text)
    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        high_cardinality=text_encoder,
        numeric=num_transformer
    )

    return encoder.fit_transform(X_cleaned)

# --- 2. CONFIG & PATHS ---
df_name = 'ramen-ratings' #'ramen-ratings', 'hospitals'
base_path = '/data/parietal/store4/soda/gblayer/salts/data/llm_embeding'
# Adjust raw data path to where your original CSV/Parquet lives
processed_data_path = f'/data/parietal/store4/soda/gblayer/salts/data/data_processed/{df_name}/data.parquet' 
processed_json_path = f'/data/parietal/store4/soda/gblayer/salts/data/data_processed/{df_name}/config.json'

#read the json
import json
with open(processed_json_path, 'r') as f:
    processed_json = json.load(f)

print(f"Processing dataset: {df_name}...")

# Dictionary to store processed numpy arrays for the heatmap
embeddings_dict = {}

# --- 3. LOAD LLM EMBEDDINGS (Pre-computed) ---
llm_models = {
    'MiniLM-L6-v2': f'{base_path}/llm-all-MiniLM-L6-v2/llm-all-MiniLM-L6-v2|{df_name}.parquet',
    'LLaMA-3.1-8B': f'{base_path}/llm-llama-3.1-8b/llm-llama-3.1-8b|{df_name}.parquet',
    'Qwen3-8B': f'{base_path}/llm-qwen3-8b/llm-qwen3-8b|{df_name}.parquet'
}

for name, path in llm_models.items():
    try:
        df = pd.read_parquet(path)
        emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
        emb_cols.sort(key=lambda x: int(x[1:]))
        embeddings_dict[name] = df[emb_cols].values
        print(f"Loaded {name} shape: {embeddings_dict[name].shape}")
    except Exception as e:
        print(f"Could not load {name}: {e}")

# --- 4. GENERATE STRINGENCODER EMBEDDINGS (Live) ---
try:
    print("Generating StringEncoder (TabVec) embeddings...")
    # Loading the raw features (not the embeddings) to pass into your function
    X_raw = pd.read_parquet(processed_data_path) 

    #drop target column if exists
    X_raw = X_raw.drop(columns=[processed_json['target_name']], errors='ignore')

    # Use your function to get the feature matrix
    xt_encoded = prepare_tabvec(X_raw)
    
    # If it's a sparse matrix (skrub sometimes returns sparse), convert to dense
    if hasattr(xt_encoded, "toarray"):
        print("Converting sparse matrix to dense array for Tf-Idf...")
        embeddings_dict['Tf-Idf'] = xt_encoded.toarray()
    else:
        print("Using dense output for Tf-Idf...")
        embeddings_dict['Tf-Idf'] = xt_encoded.values if hasattr(xt_encoded, 'values') else xt_encoded
    
    print(f"Generated Tf-Idf shape: {embeddings_dict['Tf-Idf'].shape}")
except Exception as e:
    print(f"Failed to generate Tf-Idf: {e}")

# --- 5. PLOTTING HEATMAPS (2x2 Reorganization) ---
# Define the order to ensure they appear in the right quadrants
plot_order = ['LLaMA-3.1-8B', 'Qwen3-8B', 'Tf-Idf', 'MiniLM-L6-v2']
# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes_flat = axes.flatten()

for i, name in enumerate(plot_order):
    if name not in embeddings_dict:
        continue
        
    embs = embeddings_dict[name]
    ax = axes_flat[i]
    
    # Take first 50 samples
    sample_size = min(50, embs.shape[0])
    embs_heat = embs[:sample_size]
    
    # Compute similarity
    sim_matrix = cosine_similarity(embs_heat)
    
    # Calculate Mean Similarity (ignoring diagonal)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    avg_sim = sim_matrix[mask].mean()
    
    # Plotting
    # cbar_ax ensures the colorbar doesn't "steal" space from the plot area
    sns.heatmap(
        sim_matrix, 
        ax=ax, 
        cmap='viridis', 
        vmin=0, 
        vmax=1, 
        square=True, 
        cbar=True,
        cbar_kws={"shrink": 0.8}
    )
    
    # Increase font sizes here
    ax.set_title(f"{name}\nAvg CosSim: {avg_sim:.3f}", fontsize=12, fontweight='bold', pad=15)
    
    # Ensure axes are clean but uniform
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('on') # 'on' helps keep the square frame visible for alignment


today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_cosine_sim_matrix_{df_name}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
cosine similarity heatmap after pca
The Tf-Idf generated by skrub.StringEncoder is the exception; 
it includes an internal SVD (PCA-like) step by default, 
so it is already low-dimensional.
'''

from datasets_metadata_recap import wide_datasets as data_list_wide
import re
import time
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. DATA PREPARATION FUNCTION ---
def prepare_tabvec(X_raw):
    """Generates embeddings using skrub's StringEncoder."""
    from skrub import StringEncoder, TableVectorizer, SquashingScaler
    
    # TableVectorizer to clean and handle types
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_cleaned = cleaner.fit_transform(X_raw)

    # Encode using StringEncoder for high cardinality (text)
    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        high_cardinality=text_encoder,
        numeric=num_transformer
    )

    return encoder.fit_transform(X_cleaned)

# --- 2. CONFIG & PATHS ---
df_name = 'hospitals'
base_path = '/data/parietal/store4/soda/gblayer/salts/data/llm_embeding'
# Adjust raw data path to where your original CSV/Parquet lives
processed_data_path = f'/data/parietal/store4/soda/gblayer/salts/data/data_processed/{df_name}/data.parquet' 
processed_json_path = f'/data/parietal/store4/soda/gblayer/salts/data/data_processed/{df_name}/config.json'

#read the json
import json
with open(processed_json_path, 'r') as f:
    processed_json = json.load(f)

print(f"Processing dataset: {df_name}...")

# Dictionary to store processed numpy arrays for the heatmap
embeddings_dict = {}

# --- 3. LOAD LLM EMBEDDINGS (Pre-computed) ---
llm_models = {
    'MiniLM-L6-v2': f'{base_path}/llm-all-MiniLM-L6-v2/llm-all-MiniLM-L6-v2|{df_name}.parquet',
    'LLaMA-3.1-8B': f'{base_path}/llm-llama-3.1-8b/llm-llama-3.1-8b|{df_name}.parquet',
    'Qwen3-8B': f'{base_path}/llm-qwen3-8b/llm-qwen3-8b|{df_name}.parquet'
}

for name, path in llm_models.items():
    try:
        df = pd.read_parquet(path)
        emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
        emb_cols.sort(key=lambda x: int(x[1:]))
        embeddings_dict[name] = df[emb_cols].values
        print(f"Loaded {name} shape: {embeddings_dict[name].shape}")
    except Exception as e:
        print(f"Could not load {name}: {e}")

# --- 4. GENERATE STRINGENCODER EMBEDDINGS (Live) ---
try:
    print("Generating StringEncoder (TabVec) embeddings...")
    # Loading the raw features (not the embeddings) to pass into your function
    X_raw = pd.read_parquet(processed_data_path) 

    #drop target column if exists
    X_raw = X_raw.drop(columns=[processed_json['target_name']], errors='ignore')

    # Use your function to get the feature matrix
    xt_encoded = prepare_tabvec(X_raw)
    
    # If it's a sparse matrix (skrub sometimes returns sparse), convert to dense
    if hasattr(xt_encoded, "toarray"):
        embeddings_dict['Tf-Idf'] = xt_encoded.toarray()
    else:
        embeddings_dict['Tf-Idf'] = xt_encoded.values if hasattr(xt_encoded, 'values') else xt_encoded
    
    print(f"Generated Tf-Idf shape: {embeddings_dict['Tf-Idf'].shape}")
except Exception as e:
    print(f"Failed to generate Tf-Idf: {e}")

from sklearn.decomposition import PCA # <--- ADD IMPORT

# --- 5. PLOTTING HEATMAPS (With PCA applied) ---
plot_order = ['LLaMA-3.1-8B', 'Qwen3-8B', 'Tf-Idf', 'MiniLM-L6-v2']
fig, axes = plt.subplots(2, 2, figsize=(6, 6)) # Increased size slightly
axes_flat = axes.flatten()

for i, name in enumerate(plot_order):
    if name not in embeddings_dict:
        continue
        
    embs = embeddings_dict[name]
    ax = axes_flat[i]
    
    # --- STEP A: APPLY PCA (Simulation of prepare_llm) ---
    # We only apply PCA to LLMs. Tf-Idf (StringEncoder) is already dimension 30.
    if name != 'Tf-Idf':
        # We fit PCA on the available unique embeddings to simulate the reduction
        pca = PCA(n_components=30, random_state=1234)
        embs_reduced = pca.fit_transform(embs)
    else:
        # Tf-Idf is already reduced by skrub
        embs_reduced = embs

    # --- STEP B: SELECT SAMPLES FOR HEATMAP ---
    # Take first 50 samples from the REDUCED embeddings
    sample_size = min(50, embs_reduced.shape[0])
    embs_heat = embs_reduced[:sample_size]
    
    # Compute similarity
    sim_matrix = cosine_similarity(embs_heat)
    
    # Calculate Mean Similarity (ignoring diagonal)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    avg_sim = sim_matrix[mask].mean()
    
    # Plotting
    sns.heatmap(
        sim_matrix, 
        ax=ax, 
        cmap='viridis', 
        vmin=0, 
        vmax=1, 
        square=True, 
        cbar=True,
        cbar_kws={"shrink": 0.8}
    )
    
    # Update title to reflect PCA status
    status = " (30-PCA)" if name != 'Tf-Idf' else "+ SVD (Internal)"
    ax.set_title(f"{name}{status}\nAvg CosSim: {avg_sim:.3f}", fontsize=11, fontweight='bold', pad=10)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('on')

plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_cosine_sim_matrix_after_30_pca_{df_name}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Y(height) = singular value avg across datasets for each model --> 1 per LLM model
X(point) = singular value number (1 to 30)
'''
from datasets_metadata_recap import wide_datasets as data_list_wide
from scipy.linalg import svd
import re

def prepare_tabvec(X_raw):
    """Generates embeddings using skrub's StringEncoder."""
    from skrub import StringEncoder, TableVectorizer, SquashingScaler
    
    # TableVectorizer to clean and handle types
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_cleaned = cleaner.fit_transform(X_raw)

    # Encode using StringEncoder for high cardinality (text)
    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        high_cardinality=text_encoder,
        numeric=num_transformer
    )

    return encoder.fit_transform(X_cleaned)

# --- 1. CONFIGURATION ---
# List of models as requested
selected_models_raw = [
    'llm-all-MiniLM-L6-v2',
    'llm-fasttext',
    'llm-e5-small-v2',
    'llm-llama-3.1-8b',
    'llm-qwen3-8b',
    'llm-jasper-token-comp-0.6b'
]

# Assuming data_list_wide is defined elsewhere in your script
num_singular_values = 30

# results_svd = {model: [] for model in selected_models_raw}

# # --- 2. COMPUTATION LOOP ---
# print("Computing singular values across datasets...")
# for df_name in data_list_wide:
#     for model_name in selected_models_raw:
#         print(f"Processing model: {model_name} on dataset: {df_name}...")
#         try:
#             # Path handling (assuming standard naming convention from previous steps)
#             # Adjust path formatting based on your specific directory structure
#             file_path = f'/data/parietal/store4/soda/gblayer/salts/data/llm_embeding/{model_name}/{model_name}|{df_name}.parquet'
            
#             df = pd.read_parquet(file_path)
            
#             # Extract embedding columns (X0, X1, X2...)
#             emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
#             emb_cols.sort(key=lambda x: int(x[1:]))
            
#             # Normalize or center embeddings for more stable spectral analysis
#             embs = df[emb_cols].values
#             embs = embs - np.mean(embs, axis=0) # Centering
            
#             # Compute SVD - we only need the singular values (S)
#             # full_matrices=False is faster and sufficient for S
#             _, S, _ = svd(embs, full_matrices=False)
            
#             # Take only the first 30 singular values
#             # If the embedding dimension is < 30, pad with zeros
#             s_top = np.zeros(num_singular_values)
#             s_top[:min(len(S), num_singular_values)] = S[:min(len(S), num_singular_values)]
            
#             # Store results_svd for averaging
#             results_svd[model_name].append(s_top)
            
#         except Exception as e:
#             # Silently skip missing datasets for specific models if necessary
#             continue

# svd_records = []

# for model_name, datasets_sv in results_svd.items():
#     for dataset_idx, sv_array in enumerate(datasets_sv):
#         for rank, value in enumerate(sv_array):
#             svd_records.append({
#                 'model': model_name,
#                 'dataset_id': dataset_idx, # Or use df_name if you tracked it
#                 'rank': rank + 1,          # 1 to 30
#                 'singular_value': value
#             })

# # Create DataFrame
# svd_df = pd.DataFrame(svd_records)

# # Save to Parquet
# save_path = '/data/parietal/store4/soda/gblayer/salts/results/singular_values_compiled.parquet'
# svd_df.to_parquet(save_path, index=False)
# print(f"Singular values saved successfully to {save_path}")

# upload svd_df
# svd_df_existing = pd.read_parquet('/data/parietal/store4/soda/gblayer/salts/results/singular_values_compiled.parquet')

# print("Computing singular values for StringEncoder (Tf-Idf)...")
# tfidf_svd_results = []

# for dataset_idx, df_name in enumerate(data_list_wide):
#     try:
#         print(f"Processing StringEncoder on dataset: {df_name}...") 
#         # Load Raw Data
#         processed_data_path = f'/data/parietal/store4/soda/gblayer/salts/data/data_processed/{df_name}/data.parquet' 
#         processed_json_path = f'/data/parietal/store4/soda/gblayer/salts/data/data_processed/{df_name}/config.json'

#         #read the json
#         import json
#         with open(processed_json_path, 'r') as f:
#             processed_json = json.load(f)

#         df_raw = pd.read_parquet(processed_data_path)
        
#         # Drop the target column if it exists (usually 'y' or similar, modify as needed)
#         if processed_json['target_name'] in df_raw.columns:
#             df_raw = df_raw.drop(columns=[processed_json['target_name']])
            
#         # Generate Embeddings
#         # Note: prepare_tabvec returns a numpy array or sparse matrix
#         embs_tfidf = prepare_tabvec(df_raw)
        
#         # Ensure dense array for SVD
#         if hasattr(embs_tfidf, "toarray"):
#             embs_tfidf = embs_tfidf.toarray()
            
#         # Centering
#         embs_tfidf = embs_tfidf - np.mean(embs_tfidf, axis=0)
        
#         # Compute SVD
#         _, S, _ = svd(embs_tfidf, full_matrices=False)
        
#         # Take top 30
#         s_top = np.zeros(num_singular_values)
#         s_top[:min(len(S), num_singular_values)] = S[:min(len(S), num_singular_values)]
        
#         # Append records
#         for rank, value in enumerate(s_top):
#             tfidf_svd_results.append({
#                 'model': 'StringEncoder',
#                 'dataset_id': dataset_idx,
#                 'rank': rank + 1,
#                 'singular_value': value
#             })
#     except Exception as e:
#         print(f"Skipping dataset {df_name} due to error: {e}")
#         continue
        
# # Combine DataFrames
# svd_df_tfidf = pd.DataFrame(tfidf_svd_results)
# svd_df_final = pd.concat([svd_df_existing, svd_df_tfidf], ignore_index=True)

# save the final combined dataframe
# save_path = '/data/parietal/store4/soda/gblayer/salts/results/singular_values_compiled_with_tfidf.parquet'
# svd_df_final.to_parquet(save_path, index=False)
# print(f"Final singular values (including StringEncoder) saved to {save_path}")

svd_df_final = pd.read_parquet('/data/parietal/store4/soda/gblayer/salts/results/singular_values_compiled_with_tfidf.parquet')

# --- 5. PREPARE DATA FOR PLOTTING ---
name_mapping = {
    'llm-all-MiniLM-L6-v2': 'LM All-MiniLM-L6-v2',
    'llm-fasttext': 'LM FastText',
    'llm-e5-small-v2': 'LM E5-small-v2',
    'llm-llama-3.1-8b': 'LM LLaMA-3.1-8B',
    'llm-qwen3-8b': 'LM Qwen-3-8B',
    'llm-jasper-token-comp-0.6b': 'LM Jasper-0.6B',
    'StringEncoder': 'Tf-Idf'  # Map StringEncoder to Tf-Idf
}

svd_df_final['display_name'] = svd_df_final['model'].map(name_mapping)

avg_svd_per_model_rank = svd_df_final.groupby(['display_name','rank'], as_index=False)['singular_value'].mean()
avg_svd_per_model_rank_pivot = avg_svd_per_model_rank.pivot_table(
    index='rank', 
    columns='display_name', 
    values='singular_value'
)

avg_svd_per_model_rank_pivot = avg_svd_per_model_rank_pivot / avg_svd_per_model_rank_pivot.max()

# plt.tight_layout()
plt.figure(figsize=(5, 4))

# Iterate through columns (models) of your pivoted DataFrame
for model_name in avg_svd_per_model_rank_pivot.columns:
    
    # Extract Series: Index is 'rank', Values are 'singular_value'
    series = avg_svd_per_model_rank_pivot[model_name]
    
    # Define specific style for Tf-Idf to distinguish it from LLMs
    if model_name == 'Tf-Idf':
        line_color = 'black'
        linestyle = '--'
        marker = 'x'
        alpha = 1.0
    else:
        line_color = get_encoder_color(model_name) # Assuming this function exists
        linestyle = '-'
        marker = 'o'
        alpha = 0.8
    
    # Plot
    plt.plot(series.index, 
                series.values, 
                marker=marker, 
                label=model_name, 
                color=line_color,
                linestyle=linestyle,
                linewidth=2, 
                markersize=5,
                alpha=alpha)

# --- 2. FORMATTING ---
# Adding log scale for Y is highly recommended because Llama's magnitude 
# hides the detail of other models (MiniLM, E5, Jasper)
# plt.yscale('log') 

# plt.title('Average Singular Values across Datasets per LLM', fontsize=16, fontweight='bold')
plt.xlabel('Singular Value Number (Rank)', fontsize=14)
plt.ylabel(f'Normalized Singular Value\n($SV_i / SV_{{max}}$)', fontsize=14)

# Dynamic xticks based on the actual max rank in your dataframe
max_rank = int(avg_svd_per_model_rank_pivot.index.max())
plt.xticks(range(0, max_rank + 2, 2), fontsize=10)

plt.legend(fontsize=8, bbox_to_anchor=(0.55, 1), loc='upper left')
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'singular_values_1to30_avg_across_datasets_per_model_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
conditioning number
X: log (largest singular value / smallest singular value) --> 1 per LLM model
Y: TabPFN performance with that LLM model
'''

from scipy.stats import pearsonr

name_mapping = {
    'llm-all-MiniLM-L6-v2': 'LM All-MiniLM-L6-v2',
    'llm-fasttext': 'LM FastText',
    'llm-e5-small-v2': 'LM E5-small-v2',
    'llm-llama-3.1-8b': 'LM LLaMA-3.1-8B',
    'llm-qwen3-8b': 'LM Qwen-3-8B',
    'llm-jasper-token-comp-0.6b': 'LM Jasper-0.6B',
    'StringEncoder': 'Tf-Idf'  # Map StringEncoder to Tf-Idf
}

selected_models = list(name_mapping.values())

save_path = '/data/parietal/store4/soda/gblayer/salts/results/singular_values_compiled_with_tfidf.parquet'
svd_df = pd.read_parquet(save_path)

# name_mapping = {
#     'llm-all-MiniLM-L6-v2': 'LM All-MiniLM-L6-v2',
#     'llm-fasttext': 'LM FastText',
#     'llm-e5-small-v2': 'LM E5-small-v2',
#     'llm-llama-3.1-8b': 'LM LLaMA-3.1-8B',
#     'llm-qwen3-8b': 'LM Qwen-3-8B',
#     'llm-jasper-token-comp-0.6b': 'LM Jasper-0.6B'
# }

# Filter performance results
perf_df = results[
    (results['dtype'] == 'Num+Str') & 
    (results['encoder'].isin(selected_models)) & 
    (results['learner'] == 'TabPFN-2.5')
]

# Calculate mean performance per encoder across all datasets
model_performance = perf_df.groupby('encoder')['score'].mean()

# --- 2. CALCULATE LOG CONDITION NUMBERS ---
def get_log_cond(group):
    s_max = group[group['rank'] == 1]['singular_value'].values[0]
    s_min = group[group['rank'] == 30]['singular_value'].values[0]
    return np.log(s_max / s_min) if s_min > 0 else np.nan

# Calculate per model, per dataset, then average across datasets
cond_per_dataset = svd_df.groupby(['display_name', 'dataset_id']).apply(get_log_cond)
model_cond_numbers = cond_per_dataset.groupby('display_name').mean().to_dict()

# --- 3. PLOTTING ---
plt.figure(figsize=(5, 4))

# Align the data for plotting
plot_x = []
plot_y = []
labels = []

for model in selected_models:
    if model in model_cond_numbers and model in model_performance:
        plot_x.append(model_cond_numbers[model])
        plot_y.append(model_performance[model])
        labels.append(model)

# Create Scatter Plot
plt.scatter(plot_x, plot_y, s=150, c='royalblue', alpha=0.7, edgecolors='black')

# Annotate each point
for i, txt in enumerate(labels):
    if txt == 'LM All-MiniLM-L6-v2':
        plt.annotate(txt, (plot_x[i]-0.03, plot_y[i]-0.02), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')
    elif txt == 'LM E5-small-v2':
        plt.annotate(txt, (plot_x[i]-0.04, plot_y[i]), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')
    elif txt == 'Tf-Idf':
        plt.annotate(txt, (plot_x[i], plot_y[i]-0.01), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')
    elif txt == 'LM FastText':
        plt.annotate(txt, (plot_x[i], plot_y[i]-0.015), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')
    elif txt == 'LM LLaMA-3.1-8B':
        plt.annotate(txt, (plot_x[i]-0.5, plot_y[i]), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')
    elif txt == 'LM Qwen-3-8B':
        plt.annotate(txt, (plot_x[i]-0.3, plot_y[i]), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')
    else:
        plt.annotate(txt, (plot_x[i], plot_y[i]), xytext=(8, 8), 
                 textcoords='offset points', fontsize=11, fontweight='bold')


# plt.title('TabPFN Performance vs. Embedding Conditioning', fontsize=16, fontweight='bold')
plt.xlabel(f'Conditioning Number = $\\log(\\frac{{SV_1}}{{SV_{{30}}}})$', fontsize=14)
plt.ylabel('Average TabPFN-2.5 Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Optional: Add a trendline to see the correlation
if len(plot_x) > 1:
    r, _ = pearsonr(plot_x, plot_y)
    z = np.polyfit(plot_x, plot_y, 1)
    p = np.poly1d(z)
    plt.plot(plot_x, p(plot_x), "r--", alpha=0.5, linewidth=2, 
             label=f'Linear Regression Fit\n($Pearson={r:.2f}$)')
# --- ADDED: Display the legend ---
plt.legend(loc=(0.35,0.50), fontsize=12)

plt.tight_layout()



today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'conditioning_number_tabpfn_selected_llms_1to30_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
PCA ABLATION
main message: if we take a higher number of PC - say 60 - on of the embeddings of Llama-3.1-8b
and we train TabPFN on those 60 PCs, performance should not improve (or should improve very little, to be able to state that there are diminishing returns)
because the extra PCs are mostly noise
and the signal is already captured in the first 30 PCs (coming from a big LLM model)

- we need to compare PCA 30 vs PCA 60 for Llama-3.1-8b 
- vs TabPFN performance
'''

import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from glob import glob

pca_ablation_path = '/data/parietal/store4/soda/gblayer/salts/results/pca-ablation'

# compile results
score_dir = Path(pca_ablation_path)
score_files = list(score_dir.glob("**/score/*.csv"))

# Extract and concat results for PCA 60
df_score_ = Parallel(n_jobs=-1)(delayed(pd.read_csv)(args) for args in score_files)
df_score_pca_60 = pd.concat(df_score_, axis=0)
df_score_pca_60.reset_index(drop=True, inplace=True)

#preprocess pca_60
df_score_pca_60['score'] = df_score_pca_60['r2'].fillna(df_score_pca_60['roc_auc'])
meta_pca_60 = df_score_pca_60['method'].str.split('_', expand=True, n=2)
df_score_pca_60['dtype'] = meta_pca_60[0]
df_score_pca_60['encoder'] = meta_pca_60[1]
df_score_pca_60['learner'] = meta_pca_60[2]
df_score_pca_60['dtype'] = df_score_pca_60['dtype'].replace(dtype_map)
df_score_pca_60['encoder'] = df_score_pca_60['encoder'].replace(encoder_map)
df_score_pca_60['learner'] = df_score_pca_60['learner'].replace(learner_map)
df_score_pca_60['method_polished'] = df_score_pca_60['encoder'] + ' - ' + df_score_pca_60['learner'] + '\n(' + df_score_pca_60['dtype'] + ')'
df_score_pca_60['encoder_learner'] = df_score_pca_60['encoder'] + ' - ' + df_score_pca_60['learner'] 

# filter results for LLaMA-3.1-8b and TabPFN-2.5 for num-str
df_score_pca_30 = results[(results['dtype'] == 'Num+Str') & (results['encoder'] == 'LM LLaMA-3.1-8B') & (results['learner'] == 'TabPFN-2.5')]

score = score_list[0]

avg_score_llama_tabpfn_pca30 = df_score_pca_30.groupby(['data_name'], as_index=False)[score].mean()

avg_score_llama_tabpfn_pca60 = df_score_pca_60.groupby(['data_name'], as_index=False)[score].mean()

# merge both on data_name
merged_scores = pd.merge(avg_score_llama_tabpfn_pca30, avg_score_llama_tabpfn_pca60, on='data_name', suffixes=('_pca30', '_pca60'))

# add num_rows per data_name
num_rows_per_data = results[['data_name', 'num_rows']].drop_duplicates()
merged_scores = pd.merge(merged_scores, num_rows_per_data, on='data_name', how='left')

pearson_corr = merged_scores['score_pca30'].corr(merged_scores['score_pca60'])

# --- 3. Plotting ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(5, 4))

# Scatterplot
scatter = sns.scatterplot(
    data=merged_scores,
    x='score_pca30',
    y='score_pca60',
    alpha=0.7,
    edgecolor='w',
    s=100
)

# Reference Line (Diagonal)
line_ref, = plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Equal Performance')

# --- 4. Custom Legend ---
corr_handle = mlines.Line2D([], [], color='none', label=f'Pearson Corr: {pearson_corr:.3f}')
plt.legend(handles=[line_ref], fontsize=11, loc=(0.02, 0.63), framealpha=0.9)

# Labels
plt.xlabel(f'LLaMA-3.1-8B+TabPFN-2.5\nPCA 30 Score ($R^2$ & AUC)', fontsize=12)
plt.ylabel(f'LLaMA-3.1-8B+TabPFN-2.5\nPCA 60 Score ($R^2$ & AUC)', fontsize=12)

# Axis Limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# --- 5. Annotations ---

# Upper Left Triangle Annotation (PCA 60 is better)
# Pointing towards the top-left corner (0.05, 0.95)
# To shorten the arrow while keeping the angle, we move the text position (xytext)
# closer to the arrow tip (xy).
plt.annotate(
    "PCA with 60 PC\nhas a higher score",
    xy=(0.05, 0.95),             # Points to the actual angle/corner
    xytext=(0.25, 0.80),         # Text sits closer to the arrow tip
    arrowprops=dict(arrowstyle="->", color='tab:red', lw=1.5),
    fontsize=10, color='tab:red', ha='center', fontweight='bold'
)

# Lower Right Triangle Annotation (PCA 30 is better)
plt.annotate(
    "PCA with 30 PC\nhas a higher score",
    xy=(0.95, 0.05),             # Points to the actual angle/corner
    xytext=(0.8, 0.25),         # Text sits above it
    arrowprops=dict(arrowstyle="->", color='tab:green', lw=1.5),
    fontsize=10, color='tab:green', ha='center', fontweight='bold'
)

plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'llama3_8b-tabpfn-2.5_pca30_vs_60_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Add analysis per sample size
we want to show that there is a return for datasets of higher sample size
i.e., for datasets with more samples, having PCA 60 helps more than PCA 30
'''

merged_scores['delta_score'] = merged_scores['score_pca60'] - merged_scores['score_pca30']

# Plotting
plt.figure(figsize=(6, 5))
sns.set_theme(style="whitegrid")

# Main Scatter plot
# Log scale for X usually reveals the trend better for sample size
g = sns.scatterplot(
    data=merged_scores,
    x='num_rows',
    y='delta_score',
    s=80,
    alpha=0.7,
    edgecolor='k',
    color='royalblue'
)

# Add a reference line at 0 (No difference)
plt.axhline(0, color='black', linestyle='--', linewidth=1, label='No Difference')

# Add a trend line (Lowess) to show if bigger datasets benefit more
sns.regplot(
    data=merged_scores,
    x='num_rows',
    y='delta_score',
    scatter=False,
    lowess=True, # Non-linear smooth fit
    color='tab:red',
    line_kws={'linestyle': '-', 'linewidth': 3, 'label': 'Trend (Lowess)'}
)

# Formatting
plt.xscale('log')
plt.xlabel('Dataset Sample Size (Log Scale)', fontsize=12)
plt.ylabel('Performance Gain from PCA 60\n($Score_{60} - Score_{30}$)', fontsize=12)
# plt.title('Does PCA-60 help on larger datasets?', fontsize=13, fontweight='bold', pad=15)

# Annotations areas
ylim = max(abs(merged_scores['delta_score'].min()), abs(merged_scores['delta_score'].max())) * 1.1
plt.ylim(-ylim, ylim)

plt.axhspan(0, ylim, color='tab:red', alpha=0.05)
plt.text(merged_scores['num_rows'].min(), ylim*0.8, "  PCA 60 is better", color='tab:red', fontweight='bold')

plt.axhspan(-ylim, 0, color='tab:green', alpha=0.05)
plt.text(merged_scores['num_rows'].min(), -ylim*0.8, "  PCA 30 is better", color='tab:green', fontweight='bold')

plt.legend(loc='upper right')
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'pca30_vs_60_delta_per_sample_size_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
HORIZONTAL BARPLOT OF PCA 30 VS 60 PERFORMANCE DIFFERENCE
we want to show that the performance difference between PCA 60 and PCA 30
is small for most datasets
'''

merged_scores['score_diff'] = merged_scores['score_pca60'] - merged_scores['score_pca30']

# 2. Setup Plot
plt.figure(figsize=(4, 2))
sns.set_theme(style="whitegrid")

# 3. Create Horizontal Boxplot
# We use a neutral or slight 'warm' color to indicate the comparison
ax = sns.boxplot(
    x=merged_scores['score_diff'], 
    color='navajowhite', 
    width=0.4, 
    fliersize=4, 
    linewidth=1.5
)

# 4. Add the Grey Line at Zero (The Supervisor's Request)
plt.axvline(0, color='dimgrey', linestyle='--', linewidth=2.5, alpha=0.8)

# 5. Add Annotations to Interpret the Plot
# Get plot bounds for text placement
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim() # Boxplot usually implies y=0 is center

# Label "PCA 30 is Better" (Left side)
plt.text(0.15, 0.15, "PCA 60 is Better", 
         ha='center', va='bottom', color='tab:red', fontweight='bold', fontsize=10)

# Label "PCA 60 is Better" (Right side)
plt.text(-0.16, 0.15, "PCA 30 is Better", 
         ha='center', va='bottom', color='tab:green', fontweight='bold', fontsize=10)

# Add specific summary stats as text
median_val = merged_scores['score_diff'].median()
plt.text(0.15, -0.15, f' Median: {median_val:.4f}', 
         ha='center', va='bottom', color='black', fontweight='bold', fontsize=9, backgroundcolor='white')

# 6. Titles and Labels
# plt.title('Performance Delta Distribution: LLaMA-3.1-8B (60 PCs) vs (30 PCs)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('$Score_{60} - Score_{30}$ where Score=$(R^2&AUC)$', fontsize=12)
plt.yticks([]) # Hide y-axis ticks as it's a single variable
plt.xlim(-0.25, 0.25)
plt.ylim(-0.25,0.25)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'boxplot_pca30_vs_60_delta_per_sample_size_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
RUNTIME ANALYSIS PCA-30 VS PCA-60
'''

pca_60_avg_runtime_per_dataset = df_score_pca_60.groupby(['data_name'], as_index=False)['run_time'].mean()

pca_30_avg_runtime_per_dataset = df_score_pca_30.groupby(['data_name'], as_index=False)['run_time'].mean()

#merge both on data_name
merged_runtimes = pd.merge(pca_30_avg_runtime_per_dataset, pca_60_avg_runtime_per_dataset, on='data_name', suffixes=('_pca30', '_pca60'))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import time

# =============================================================================
# RUNTIME ANALYSIS: PCA 60 vs PCA 30 RATIO
# =============================================================================

# Calculate the Ratio: Time(60) / Time(30)
# Ratio > 1 means PCA 60 is slower (expected for quadratic complexity)
merged_runtimes['time_ratio'] = merged_runtimes['run_time_pca60'] / merged_runtimes['run_time_pca30']

# 2. Setup Plot
plt.figure(figsize=(4, 2))
sns.set_theme(style="whitegrid")

# 3. Create Horizontal Boxplot
# Using log scale internally first to handle outliers, but we explicitly set scale below
ax = sns.boxplot(
    x=merged_runtimes['time_ratio'], 
    color='thistle', # Light purple/pink to distinguish from accuracy plots
    width=0.4, 
    fliersize=4, 
    linewidth=1.5
)

# 4. Set Log Scale
ax.set_xscale('log')

# 5. Custom Ticks with LaTeX Formatting (The Supervisor's Request)
# We define specific points of interest: x1, x2, x4 (quadratic jump), x8, x10
major_ticks = [1, 2, 4, 8]
ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
ax.xaxis.set_major_formatter(ticker.FixedFormatter([r'$\times 1$', r'$\times 2$', r'$\times 4$', r'$\times 8$']))

# Add minor ticks for visual context between the major ones
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

ax.xaxis.set_minor_formatter(ticker.NullFormatter())

# 6. Add Reference Line at x1 (Neutral)
plt.axvline(1, color='tab:green', linestyle='--', linewidth=2.5, alpha=0.8)

# 7. Annotations
# Get bounds
x_min, x_max = plt.xlim()

# Label "Slower" (Right side)
plt.text(3.5, -0.1, "→ Slower\n(PCA 60 takes\nmore time)", 
         ha='left', va='bottom', color='tab:red', fontweight='bold', fontsize=10)

# Label "Same Speed" (At x1)
plt.text(1.2, -0.15, "Same\nSpeed", 
         ha='center', va='bottom', color='tab:green', fontweight='bold', fontsize=9)

# Add Median Annotation
median_val = merged_runtimes['time_ratio'].median()
plt.text(median_val+0.3, 0.1, f' Median: {median_val:.1f}x', 
         ha='center', va='bottom', color='black', fontweight='bold', fontsize=9, 
         backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# 8. Titles and Labels
# plt.title('Runtime Overhead: LLaMA-3.1-8B (60 PCs) vs (30 PCs)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Runtime Ratio (Log Scale)', fontsize=12)
plt.yticks([]) # Hide y-axis ticks

# Adjust x-limits to ensure x1 is visible and the right side isn't cut off
# We ensure at least 0.8 to see the 'faster' side if any exist, and max + padding
plt.xlim(left=0.8, right=max(merged_runtimes['time_ratio'].max(), 5)) 
plt.ylim(-0.25,0.25)
plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'runtime_boxplot_pca30_vs_60_delta_per_sample_size_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')

plt.show()


'''
RUNTIME ANALYSIS ENTIRE STRABLE STUDY
'''

results['run_time'].sum()#72780014.6906


'''
CUMULATIVE VARIANCE EXPLAINED: PCA 30 VS 60
we want to show that the cumulative variance explained by 
PCA 60 is not that much higher than PCA 30
'''

from datasets_metadata_recap import wide_datasets as data_list_wide
from scipy.linalg import svd
import re

# Configuration
target_model = 'llm-llama-3.1-8b'
display_name = 'LLaMA-3.1-8B'
max_rank = 60

cumulative_variance_records = []

print(f"Computing Cumulative Variance for {display_name}...")

for df_name in data_list_wide:
    print(f"Processing dataset: {df_name}...")
    try:
        # Construct path (Adjust based on your exact folder structure if needed)
        file_path = f'/data/parietal/store4/soda/gblayer/salts/data/llm_embeding/{target_model}/{target_model}|{df_name}.parquet'
        
        # Read Data
        df = pd.read_parquet(file_path)
        
        # Extract Embeddings
        emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
        # Ensure we have enough columns
        if len(emb_cols) == 0: continue
        
        embs = df[emb_cols].values
        
        # Center the data (crucial for PCA/Variance analysis)
        embs = embs - np.mean(embs, axis=0)
        
        # Compute SVD (returns all singular values)
        _, S, _ = svd(embs, full_matrices=False)
        
        # Calculate Explained Variance Ratio
        eigenvalues = S ** 2
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        
        # Calculate Cumulative Variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Pad with 1.0 if we have fewer than max_rank dimensions
        if len(cumulative_variance) < max_rank:
            pad_width = max_rank - len(cumulative_variance)
            cumulative_variance = np.pad(cumulative_variance, (0, pad_width), constant_values=1.0)
        else:
            cumulative_variance = cumulative_variance[:max_rank]
            
        cumulative_variance_records.append(cumulative_variance)

    except Exception as e:
        print(f"Skipping {df_name}: {e}")
        continue

# Average across datasets
cumulative_variance_matrix = np.array(cumulative_variance_records)
avg_cumulative_variance = np.mean(cumulative_variance_matrix, axis=0)

# --- PLOTTING PART 1 ---
plt.figure(figsize=(6, 4))
sns.set_theme(style="whitegrid")

ranks = np.arange(1, max_rank + 1)
plt.plot(ranks, avg_cumulative_variance, color='#d62728', lw=3, label=display_name)

# Annotations for PCA 30 vs 60
var_30 = avg_cumulative_variance[29] # Index 29 is rank 30
var_60 = avg_cumulative_variance[59] # Index 59 is rank 60
gain = var_60 - var_30

# Vertical Line at 30
plt.axvline(x=30, color='black', linestyle='--', alpha=0.6, label='Standard PCA (30)')

# Highlight the Gain
plt.fill_between(ranks[29:60], avg_cumulative_variance[29:60], var_30, color='gray', alpha=0.2)
plt.annotate(f'+{gain*100:.1f}% Variance\n(Dim 31-60)', 
             xy=(45, (var_30 + var_60)/2), 
             ha='center', va='center', fontsize=9, fontweight='bold', color='#333')

# Point at 30
plt.scatter([30], [var_30], color='black', zorder=5)
plt.text(32, var_30 - 0.05, f'{var_30*100:.1f}% Explained', fontsize=10, fontweight='bold')

plt.title(f'Cumulative Explained Variance: {display_name}', fontsize=12, fontweight='bold')
plt.xlabel('Number of Principal Components', fontsize=11)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=11)
plt.ylim(0, 1.05)
plt.xlim(0, 60)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'cumulative_variance_llama3_8b_pca60_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Dimension of the embeddings before PCA vs TabPFN-2.5 performance.
x = TabPFN-2.5 performance (after 30th PCA)
y = original embedding dimension (before PCA)
vertical barplot - does it make sense? 
I would have to run TabPFN on the original embedding dimension (without PCA)
'''

selected_LLMs = ['LM All-MiniLM-L6-v2',
 'LM FastText',
 'LM E5-small-v2',
 'LM LLaMA-3.1-8B',
 'LM Qwen-3-8B',
 'LM Jasper-0.6B']

selected_models_raw = [
    'llm-all-MiniLM-L6-v2',
    'llm-fasttext',
    'llm-e5-small-v2',
    'llm-llama-3.1-8b',
    'llm-qwen3-8b',
    'llm-jasper-token-comp-0.6b'
]

name_mapping = {
    'llm-all-MiniLM-L6-v2': 'LM All-MiniLM-L6-v2',
    'llm-fasttext': 'LM FastText',
    'llm-e5-small-v2': 'LM E5-small-v2',
    'llm-llama-3.1-8b': 'LM LLaMA-3.1-8B',
    'llm-qwen3-8b': 'LM Qwen-3-8B',
    'llm-jasper-token-comp-0.6b': 'LM Jasper-0.6B',
    'StringEncoder': 'Tf-Idf'  # Map StringEncoder to Tf-Idf
}

# for each model, compute avg tabpfn performance across datasets
score = score_list[0]
avg_performance_per_selected_llm_tabpfn = results[(results['learner'] == 'TabPFN-2.5') & (results['dtype'] == 'Num+Str') & (results['encoder'].isin(selected_LLMs))].groupby(['encoder'], as_index=False)[score].mean()

# for each model, compute the original embedding dimension (before PCA)
from glob import glob
import re
embedding_path = '/data/parietal/store4/soda/gblayer/salts/data/llm_embeding'
embedding_dims = {}
for model in selected_models_raw:
    model_path = f'{embedding_path}/{model}'
    # List all files for this model
    files = glob(f'{model_path}/*.parquet')
    if len(files) == 0:
        print(f"No files found for model: {model}")
        continue
    # Read the first file to get the embedding dimension (assuming all files for the same model have the same dimension)
    df = pd.read_parquet(files[0])
    emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
    embedding_dims[name_mapping[model]] = len(emb_cols)

# add column with embedding dimension to the avg_performance_per_selected_llm_tabpfn dataframe
avg_performance_per_selected_llm_tabpfn['embedding_dim'] = avg_performance_per_selected_llm_tabpfn['encoder'].map(embedding_dims)

# create a new column with the embedding dimension within the embedding name in brackets
avg_performance_per_selected_llm_tabpfn['encoder_with_dim'] = avg_performance_per_selected_llm_tabpfn.apply(lambda row: f"{row['encoder']} ({row['embedding_dim']})", axis=1)

df = avg_performance_per_selected_llm_tabpfn.copy()
custom_palette = {name: get_encoder_color(name) for name in df['encoder']}

# ---------------------------------------------------------
# 3. PLOT 2: HORIZONTAL BARPLOT
# ---------------------------------------------------------
df_sorted = df.sort_values('score', ascending=False)

plt.figure(figsize=(3, 4))

# Switched x and y for horizontal orientation
ax = sns.barplot(
    data=df_sorted,
    y='encoder_with_dim',    # Categories on Y-axis
    x='score',      # Values on X-axis
    hue='encoder', 
    palette=custom_palette,
    edgecolor='black',
    dodge=False
)
# drop legend
ax.legend_.remove()

plt.xlabel('TabPFN-2.5 Score', fontsize=12)
plt.ylabel('')
plt.xlim(0.6, 0.75) # Zoom to show differences

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'performance_tabpfn_llms_dimensions_before_pca_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

########################PREPPING REBUTTALS EXPERIMENTS########################

'''
From VSE
Fig 1: more sophisticated embedding improve performance across varying training sizes
Y = Mean Rank (lower is better)
X = Number of training samples
continuous line: num+str
dashed line: str
XGBoost for b-classification: Tf-Idf, LM FastText, LM BGE-large, LM Sentence-T5-XXL
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from adjustText import adjust_text
import numpy as np
import plotly.express as px

# 1. Filter the Data
balanced_problems = ['kickstarter-projects', 'animalandveterinary-event', 'device-pma', 'drug-drugsfda']

df = results[(results['task'] == 'b-classification') & 
             (results['learner'] == 'XGBoost') & 
             (results['dtype'].isin(['Num+Str', 'Str'])) & 
             (results['encoder'].isin(['Tf-Idf', 'LM FastText', 'LM BGE-large', 'LM Sentence-T5-XXL'])) &
             (results['data_name'].isin(balanced_problems))
             ].copy()


group_cols = ['data_name', 'learner', 'encoder', "dtype", "train_size"]

melted_results = df.groupby(group_cols)['score'].mean().reset_index()

px.strip(
    melted_results,
    y = "data_name",
    x = "score",
    color="encoder",
    hover_data=melted_results.columns,
    height=1500
)

to_plot = melted_results.copy()
to_plot["rank"] = to_plot.groupby(["data_name", "train_size"])["score"].rank(ascending=False)

## full training size
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects

sns.set(style="whitegrid", font_scale=2)

# Define a FacetGrid by 'dtype'
g = sns.FacetGrid(to_plot, col="dtype", height=8, aspect=1.3)

# Create lineplot on the FacetGrid
g.map_dataframe(sns.lineplot, x="train_size", y="rank", hue="encoder", linewidth=4)
g.set_titles(col_template="{col_name}")

# --- FIX: Shrink the x-axis to (1000, 5000) ---
# g.set(xlim=(1000, 5000)) 
# -------------------------------------------

# Prepare to annotate the lines with adjustText
for ax, feature in zip(g.axes.flatten(), to_plot['dtype'].unique()):
    lines = ax.lines
    texts = []
    # Note: Ensure to_plot['encoder'].unique() aligns with the actual line order
    # It is safer to use the hue_order or get labels directly if needed
    for i, (model, line) in enumerate(zip(to_plot['encoder'].unique(), lines)):
        # Check if line has enough data points before indexing
        if len(line.get_xdata()) > 2:
            index = -2 - (i % 2)
            x_last = line.get_xdata()[index] + 0.1
            y_last = line.get_ydata()[index] - 0.02

            text = ax.text(x_last, y_last, model, color=line.get_color())
            text.set_path_effects([PathEffects.withStroke(linewidth=7, foreground='white')])
            texts.append(text)
    
    # Adjust text per axis to ensure they don't overlap locally
    adjust_text(texts, ax=ax)

# Set labels and title
g.set_axis_labels("Number of training samples", "Rank (lower is better)")

plt.show()

## VSE max training size

import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects

sns.set(style="whitegrid", font_scale=2)

# --- Define Manual Positions ---
# Adjust these (x, y) coordinates based on your visual preference.
# The keys must match your 'dtype' values and 'encoder' names exactly.
manual_positions = {
    'Num+Str': {
        'Tf-Idf': (3000, 4.5),         # Example: (x=4000, y=Rank 4.5)
        'LM FastText': (4000, 7.5),
        'LM BGE-large': (2500, 3.1),
        'LM Sentence-T5-XXL': (3500, 2.2)
    },
    'Str': {
        'Tf-Idf': (4000, 5.5),
        'LM FastText': (2500, 7.0),
        'LM BGE-large': (3500, 4.5),
        'LM Sentence-T5-XXL': (2500, 1.1)
    }
}

# --- Create Plot ---
g = sns.FacetGrid(to_plot, col="dtype", height=8, aspect=1.3)
g.map_dataframe(sns.lineplot, x="train_size", y="rank", hue="encoder", linewidth=4)
g.set_titles(col_template="{col_name}")
g.set(xlim=(1000, 5000))

# --- Manual Annotation Loop ---
# We loop through the axes and use the title (which is the dtype name) 
# to look up the correct positions.
for ax in g.axes.flatten():
    # 1. Identify which plot this is (Num+Str or Str)
    # The title is usually set to the col_name value (e.g., "Num+Str")
    current_dtype = ax.get_title() 
    
    # 2. Get the color mapping from the lines already plotted
    # We create a dictionary {model_name: color} to ensure text matches line color
    line_colors = {}
    for line in ax.get_lines():
        # Skip confidence intervals (which start with '_')
        if not line.get_label().startswith("_"):
            line_colors[line.get_label()] = line.get_color()

    # 3. Add Text using Manual Coordinates
    if current_dtype in manual_positions:
        coords_map = manual_positions[current_dtype]
        
        for model, (x_pos, y_pos) in coords_map.items():
            # Only label if we found a matching line color (safety check)
            if model in line_colors:
                color = line_colors[model]
                
                text = ax.text(x_pos, y_pos, model, 
                               color=color, 
                               fontweight='bold', 
                               fontsize=24,
                               ha='left', va='center')
                
                # Add white outline for readability
                text.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='white')])

# Set labels
g.set_axis_labels("Number of training samples", "Rank (lower is better)")

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'xgboost_balanced_binary_classif_4_encoders_num-str_vs_str_{today_date}.{format}'
g.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')


'''
Take away: Results match VSE: for Ridge and XGBoost, balanced binary classification, small training size (1000-5000): more sophisticated embedding improve performance across varying training sizes. 
problem is we have only 4 balanced binary classification problems and only 1 dataset whose training size is below 5000 (animalandveterinary-event) - so not sure how much this is robust
also: our sample sizes are bigger!
'''

'''
From VSE
Fig 3: The number of unique ngrams per row predicts the gain better than the length of the text entries.
Y = Gain Percentage (%) 
X = Mean entry length (in characters) / Number of Unique N-grams for 1000 rows
XGBoost (Str) for balanced b-classification and LM Sentence-T5-XXL and Tf-Idf instead of OpenAI embeddings over MinHashEncoder
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from adjustText import adjust_text
import numpy as np
import plotly.express as px

# 1. Filter the Data
balanced_problems = ['kickstarter-projects', 'animalandveterinary-event', 'device-pma', 'drug-drugsfda']

df = results[#(results['task'] == 'regression') &
             (results['learner'].isin(['XGBoost'])) & 
             (results['dtype'].isin(['Str'])) & 
             (results['encoder'].isin(['Tf-Idf', 'LM Sentence-T5-XXL'])) #&
            #  (results['data_name'].isin(balanced_problems))
             ].copy()

df_diversity_encoder_perf = df.groupby(['string_diversity', 'encoder'])['score'].mean().reset_index()


df_diversity_encoder_perf_pivot = df_diversity_encoder_perf.pivot(index='string_diversity', columns='encoder', values='score').reset_index()

# compute % gain
df_diversity_encoder_perf_pivot['gain_percentage'] = (df_diversity_encoder_perf_pivot['LM Sentence-T5-XXL'] - df_diversity_encoder_perf_pivot['Tf-Idf']) / df_diversity_encoder_perf_pivot['Tf-Idf'] * 100

# plot
plt.figure(figsize=(6, 4))
sns.set_theme(style="whitegrid")

# REPLACE scatterplot WITH regplot
ax = sns.regplot(
    data=df_diversity_encoder_perf_pivot,
    x='string_diversity',
    y='gain_percentage',
    
    # Customizing the points (scatter_kws)
    scatter_kws={'s': 100, 'color': 'royalblue', 'edgecolor': 'k', 'alpha': 0.7},
    ci=None,
    # Customizing the line (line_kws) - make it black and thick like the paper
    line_kws={'color': 'black', 'linewidth': 2},
    
    # Optional: If the x-axis in the paper is log-scale, add logx=True
    # logx=True 
)

plt.xlabel('Number of Unique N-grams for 1000 Rows', fontsize=12)
plt.ylabel('Gain Percentage (%)', fontsize=12)

# Add annotations for each point (Same as before)
# for i, row in df_diversity_encoder_perf_pivot.iterrows():
#     # Adding a small offset to Y so text doesn't overlap the dot
#     text_y = row['gain_percentage'] + 1 
    
#     text = f"{row['string_diversity']}\n({row['gain_percentage']:.1f}%)"
#     ax.text(row['string_diversity'], text_y, text, 
#             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black',
#             path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

# If the paper uses a log scale for the X-axis (common for n-grams), uncomment this:
# ax.set_xscale('log')
plt.ylim(-50,20)
# plt.ylim(-2,5)

plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
# PIC_NAME = f'xgboost_balanced_binary_classif_ngram_vs_gain_percentage_tfidf_t5xxl_{today_date}.{format}'
# PIC_NAME = f'xgboost_binary_classif_ngram_vs_gain_percentage_tfidf_t5xxl_{today_date}.{format}'
PIC_NAME = f'xgboost_ngram_vs_gain_percentage_tfidf_t5xxl_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
we have some weird characteristic of the datasets or something in the preprocessing
for which even if we have the correct ngram computation - we still do not get a % 
improvement with LLM.
'''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from adjustText import adjust_text
import numpy as np
import plotly.express as px

# 1. Filter the Data
balanced_problems = ['kickstarter-projects', 'animalandveterinary-event', 'device-pma', 'drug-drugsfda']

df = results[ #(results['task'] == 'b-classification') &
             (results['learner'].isin(['XGBoost'])) & 
             (results['dtype'].isin(['Str'])) & 
             (results['encoder'].isin(['Tf-Idf', 'LM Sentence-T5-XXL'])) #&
             #(results['data_name'].isin(balanced_problems))
             ].copy()

df_diversity_encoder_perf = df.groupby(['avg_string_length_per_cell', 'encoder'])['score'].mean().reset_index()


df_diversity_encoder_perf_pivot = df_diversity_encoder_perf.pivot(index='avg_string_length_per_cell', columns='encoder', values='score').reset_index()

# compute % gain
df_diversity_encoder_perf_pivot['gain_percentage'] = (df_diversity_encoder_perf_pivot['LM Sentence-T5-XXL'] - df_diversity_encoder_perf_pivot['Tf-Idf']) / df_diversity_encoder_perf_pivot['Tf-Idf'] * 100

# plot
plt.figure(figsize=(6, 4))
sns.set_theme(style="whitegrid")

# REPLACE scatterplot WITH regplot
ax = sns.regplot(
    data=df_diversity_encoder_perf_pivot,
    x='avg_string_length_per_cell',
    y='gain_percentage',
    
    # Customizing the points (scatter_kws)
    scatter_kws={'s': 100, 'color': 'royalblue', 'edgecolor': 'k', 'alpha': 0.7},
    ci=None,
    # Customizing the line (line_kws) - make it black and thick like the paper
    line_kws={'color': 'black', 'linewidth': 2},
    
    # Optional: If the x-axis in the paper is log-scale, add logx=True
    # logx=True 
)

plt.xlabel('Mean entry length (in characters)', fontsize=12)
plt.ylabel('Gain Percentage (%)', fontsize=12)

# Add annotations for each point (Same as before)
# for i, row in df_diversity_encoder_perf_pivot.iterrows():
#     # Adding a small offset to Y so text doesn't overlap the dot
#     text_y = row['gain_percentage'] + 1 
    
#     text = f"{row['string_diversity']}\n({row['gain_percentage']:.1f}%)"
#     ax.text(row['string_diversity'], text_y, text, 
#             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black',
#             path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

# If the paper uses a log scale for the X-axis (common for n-grams), uncomment this:
# ax.set_xscale('log')
plt.ylim(-50,20)

plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
# PIC_NAME = f'xgboost_balanced_binary_classif_ngram_vs_gain_percentage_tfidf_t5xxl_{today_date}.{format}'
PIC_NAME = f'xgboost_string_length_vs_gain_percentage_tfidf_t5xxl_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
from VSE
Fig 4: Being better on classical embedding tasks translates
to being better on tabular analytics. But this is only the
case in the diverse entries regime, where the number of unique
ngrams in the column is large enough. The gain is computed
for each column by replacing the MinHash encoding by a
language model embedding (+ PCA), and averaged accross
columns 
'''

df = results[
             (results['learner'].isin(['XGBoost'])) & 
             (results['dtype'].isin(['Num+Str'])) & 
             (~results['encoder'].isin(['ContextTab', 'TabSTAR', 'TabPFN-2.5', 'CatBoost', 'TargetEncoder']))
             ].copy()

meta = df['method'].str.split('_', expand=True, n=2)

df['encoder_raw'] = meta[1]


hf_map = {
    # --- Encoders / Baselines (Not on HF) ---
    'tabvec': '-',
    'tarenc': '-',
    'catboost': '-',
    'tabpfn': '-',
    'tabstar': '-',
    'contexttab': '-',
    'tarte': '-',
    'llm-fasttext': '-',
    
    # --- Sentence Transformers / BERT family ---
    'llm-all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'llm-all-MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2',
    'llm-all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
    'llm-e5-base-v2': 'intfloat/e5-base-v2',
    'llm-e5-large-v2': 'intfloat/e5-large-v2',
    'llm-e5-small-v2': 'intfloat/e5-small-v2',
    'llm-roberta-base': 'FacebookAI/roberta-base',
    'llm-roberta-large': 'FacebookAI/roberta-large',
    'llm-modernbert-base': 'answerdotai/ModernBERT-base',
    'llm-modernbert-large': 'answerdotai/ModernBERT-large',
    'llm-deberta-v3-xsmall': 'microsoft/deberta-v3-xsmall',
    'llm-deberta-v3-small': 'microsoft/deberta-v3-small',
    'llm-deberta-v3-base': 'microsoft/deberta-v3-base',
    'llm-deberta-v3-large': 'microsoft/deberta-v3-large',
    
    # --- BGE Family ---
    'llm-bge-large': 'BAAI/bge-large-en-v1.5',
    'llm-bge-small': 'BAAI/bge-small-en-v1.5',
    'llm-bge-base': 'BAAI/bge-base-en-v1.5',
    
    # --- LLaMA Family ---
    'llm-llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
    'llm-llama-3.2-1b': 'meta-llama/Llama-3.2-1B',
    'llm-llama-3.2-3b': 'meta-llama/Llama-3.2-3B',
    
    # --- Qwen Family ---
    'llm-qwen3-8b': 'Qwen/Qwen3-Embedding-8B',
    'llm-qwen3-4b': 'Qwen/Qwen3-Embedding-4B',
    'llm-qwen3-0.6b': 'Qwen/Qwen3-Embedding-0.6B',
    
    # --- OPT Family (Note: 0.1b -> 125m, 0.3b -> 350m) ---
    'llm-opt-0.1b': 'facebook/opt-125m',
    'llm-opt-0.3b': 'facebook/opt-350m',
    'llm-opt-1.3b': 'facebook/opt-1.3b',
    'llm-opt-2.7b': 'facebook/opt-2.7b',
    'llm-opt-6.7b': 'facebook/opt-6.7b',
    
    # --- F2LLM Family ---
    'llm-f2llm-0.6b': 'codefuse-ai/F2LLM-0.6B',
    'llm-f2llm-1.7b': 'codefuse-ai/F2LLM-1.7B',
    'llm-f2llm-4b': 'codefuse-ai/F2LLM-4B',
    
    # --- T5 Family ---
    'llm-t5-small': 'google-t5/t5-small',
    'llm-sentence-t5-base': 'sentence-transformers/sentence-t5-base',
    'llm-sentence-t5-large': 'sentence-transformers/sentence-t5-large',
    'llm-sentence-t5-xl': 'sentence-transformers/sentence-t5-xl',
    'llm-sentence-t5-xxl': 'sentence-transformers/sentence-t5-xxl',
    
    # --- Others ---
    'llm-gemma-0.3b': 'google/gemma-3-270m',
    'llm-uae-large': 'WhereIsAI/UAE-Large-V1',
    'llm-kalm-embed': 'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5',
    'llm-jasper-token-comp-0.6b': 'infgrad/Jasper-Token-Compression-600M', # Check specific version if needed
}


# Create 'Hugging Face' column using the raw 'method' column
df['Hugging Face'] = df['encoder_raw'].map(hf_map).fillna('-')

mteb_scores = {
        'sentence-transformers/all-MiniLM-L6-v2': 56.03,
        'intfloat/e5-base-v2': 61.67, # e5-base-v2
        'intfloat/e5-large-v2': 62.79,
        'intfloat/e5-small-v2': 61.32,
        'BAAI/bge-large-en-v1.5': 65.89,
        'BAAI/bge-base-en-v1.5': 65.14,
        'BAAI/bge-small-en-v1.5': 64.30,
        'Qwen/Qwen3-Embedding-8B': 75.23, # High performer
        'Qwen/Qwen3-Embedding-4B': 74.61,
        'Qwen/Qwen3-Embedding-0.6B': 70.47,
        'codefuse-ai/F2LLM-0.6B': 70.03,
        'codefuse-ai/F2LLM-1.7B': 72.01,
        'codefuse-ai/F2LLM-4B': 73.67,
        'WhereIsAI/UAE-Large-V1': 66.4,
        'infgrad/Jasper-Token-Compression-600M': 74.75,
        'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5': 71.29,
        'sentence-transformers/sentence-t5-base':60.3,
        'sentence-transformers/sentence-t5-large':77.67,
        'sentence-transformers/sentence-t5-xl':76.58,
        'sentence-transformers/sentence-t5-xxl':66.13,
    }


df['MTEB (En) Score'] = df['Hugging Face'].map(mteb_scores).fillna(np.nan)

#round the MTEB score to integer
df['MTEB (En) Score'] = df['MTEB (En) Score'].round(2)

# create column n_gram_thresh < 3K and n_gram_thresh >= 3K based on the string_diversity column
df['n_gram_thresh'] = df['string_diversity'].apply(lambda x: '< 3K' if x < 3000 else '>= 3K')

baseline_name = 'tabvec'

df['encoder_raw'] = df['encoder_raw'].astype(str)

df_pivot = df.pivot_table(
    index=['data_name', 'n_gram_thresh'], 
    columns='encoder_raw', 
    values='score', 
    aggfunc='mean'
)

# Find the baseline column (flexible search)
baseline_col = [c for c in df_pivot.columns if baseline_name in c.lower() or baseline_name in c.lower()]
if not baseline_col:
    raise ValueError(f"Could not find baseline '{baseline_name}' in columns: {df_pivot.columns.tolist()}")
baseline_col = baseline_col[0]
print(f"Using baseline column: {baseline_col}")

# Subtract baseline from all columns to get GAIN
df_gain = df_pivot.sub(df_pivot[baseline_col], axis=0)

# Melt back to long format
df_plot = df_gain.reset_index().melt(
    id_vars=['data_name', 'n_gram_thresh'], 
    var_name='encoder_raw', 
    value_name='gain'
)

# 3. RE-ADD THE MTEB SCORES (Crucial Step)
# ----------------------------
# The MTEB column was lost during pivot, so we map it back onto df_plot
df_plot['Hugging Face'] = df_plot['encoder_raw'].map(hf_map).fillna('-')
df_plot['MTEB (En) Score'] = df_plot['Hugging Face'].map(mteb_scores)

# Filter: Remove rows with no MTEB score (baselines) and drop the baseline itself (gain=0)
df_plot = df_plot.dropna(subset=['MTEB (En) Score'])
df_plot = df_plot[df_plot['gain'] != 0]


# --- 4. AGGREGATE FOR PLOTTING ---
# We want one point per Model per Regime (Mean Gain)
# df_agg = df_plot.groupby(['encoder_raw', 'n_gram_thresh', 'MTEB (En) Score'])['gain'].agg(['mean']).reset_index()
# df_agg.rename(columns={'mean': 'Mean Gain'}, inplace=True)

df_all = df_agg.groupby(['encoder_raw', 'MTEB (En) Score'])['Mean Gain'].mean().reset_index()
df_all['n_gram_thresh'] = 'All'

# 3. COMBINE
df_final = pd.concat([df_agg, df_all], ignore_index=True)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress

# 1. Setup Colors
color_map = {
    '< 3K': '#d62728',   # Red
    '>= 3K': '#1f77b4',  # Blue
    'All': '#2ca02c'     # Green
}

# 2. Base Plot (Points Only)
fig = px.scatter(
    df_final,
    x="MTEB (En) Score",
    y="Mean Gain",
    color="n_gram_thresh",
    hover_data=["encoder_raw"],
    color_discrete_map=color_map,
    labels={
        "Mean Gain": "Gain over Tf-Idf (R² & AUC)", 
        "MTEB (En) Score": "MTEB Retrieval Score",
        "n_gram_thresh": "Regime"
    },
    template="simple_white",
    title="Gain for Embedding models over Tf-Idf"
)

# 3. Manually Add Trendlines
# We loop through each group, calculate the line, and add it as a trace.
for regime in df_final['n_gram_thresh'].unique():
    # Get data for this group
    group_data = df_final[df_final['n_gram_thresh'] == regime]
    
    # Skip if not enough points
    if len(group_data) < 2:
        continue
        
    # Calculate Linear Regression (y = mx + b)
    slope, intercept, r_value, p_value, std_err = linregress(
        group_data["MTEB (En) Score"], 
        group_data["Mean Gain"]
    )
    
    # Create line points (min x to max x)
    x_range = np.linspace(group_data["MTEB (En) Score"].min(), group_data["MTEB (En) Score"].max(), 100)
    y_range = slope * x_range + intercept
    
    # Add the line trace
    fig.add_trace(
        go.Scatter(
            x=x_range, 
            y=y_range, 
            mode='lines',
            line=dict(color=color_map.get(regime, 'black'), width=3),
            name=f"{regime} Trend",
            showlegend=False, # Hide from legend to keep it clean
            hoverinfo='skip'
        )
    )

# 4. Apply Final Styling
fig.update_traces(
    marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    margin=dict(l=50, r=50, b=100, t=100, pad=4),
    paper_bgcolor="white",
    font=dict(family="Arial, monospace", size=23, color="black"),
    legend=dict(
        orientation="h",
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(255, 255, 255, 0.9)"
    )
)

# Zero Line
fig.add_shape(
    type="line",
    x0=df_final["MTEB (En) Score"].min() - 2,
    x1=df_final["MTEB (En) Score"].max() + 2,
    y0=0, y1=0,
    line=dict(color="black", width=2, dash="dash"),
    layer="below"
)

# 5. Save Logic (Robust)
today_date = time.strftime("%Y-%m-%d")
TODAYS_FOLDER = today_date
save_dir = f'/data/parietal/store4/soda/gblayer/salts/results_pics/{TODAYS_FOLDER}/'
os.makedirs(save_dir, exist_ok=True)

pdf_path = os.path.join(save_dir, f'xgboost_gain_percentage_over_tfidf_vs_MTEB_by_ngram_together_{today_date}.pdf')
html_path = os.path.join(save_dir, f'xgboost_gain_percentage_over_tfidf_vs_MTEB_by_ngram_together_{today_date}.html')

# 2. Try Saving
try:
    print(f"Attempting to save PDF to {pdf_path}...")
    fig.write_image(pdf_path, format='pdf', width=1000, height=600)
    print("✅ Success!")
except Exception as e:
    print(f"❌ PDF Export Failed. Error: {e}")
    print(f"⚠️ Saving as HTML instead to {html_path}")
    fig.write_html(html_path)
    print("✅ HTML Saved. Open this file in your browser and 'Print to PDF'.")

fig.show()

'''
--- DIAGNOSIS ---
Original Row Counts: n_gram_thresh
>= 3K    13803
< 3K       129
Name: count, dtype: int64

Baseline (tabvec) counts per regime:
n_gram_thresh
>= 3K    321
< 3K       3
Name: count, dtype: int64

Rows in df_plot (valid gains) per regime:
n_gram_thresh
>= 3K    2140
< 3K       20
Name: count, dtype: int64
-----------------

Datasets in < 3K: 1
Datasets in >= 3K: 107

Checking values for model: llm-all-MiniLM-L6-v2
   n_gram_thresh  Mean Gain
0           < 3K  -0.216649
1          >= 3K  -0.020280
40           All  -0.022099
'''

import pandas as pd
import plotly.express as px

# 1. PREPARE AGGREGATED DATA (Per Regime)
# This assumes you have already run the pivot/melt to get 'df_plot'
# df_plot columns: [dataset, n_gram_thresh, encoder_raw, gain, MTEB (En) Score]

# Calculate mean gain for each regime (<3K, >=3K)
df_agg = df_plot.groupby(['encoder_raw', 'n_gram_thresh', 'MTEB (En) Score'])['gain'].agg(['mean']).reset_index()
df_agg.rename(columns={'mean': 'Mean Gain'}, inplace=True)

# 2. CALCULATE "ALL" (Macro-Average / Regime-Weighted)
# CRITICAL CHANGE: We aggregate from 'df_agg' (the summaries), NOT 'df_plot' (the raw data).
# This gives equal weight to the <3K line and the >=3K line.
df_all = df_agg.groupby(['encoder_raw', 'MTEB (En) Score'])['Mean Gain'].mean().reset_index()
df_all['n_gram_thresh'] = 'All'

# 3. COMBINE
df_final = pd.concat([df_agg, df_all], ignore_index=True)

# 4. PLOT
color_map = {
    '< 3K': '#d62728',   # Red
    '>= 3K': '#1f77b4',  # Blue
    'All': '#2ca02c'     # Green
}

category_order = ["< 3K", ">= 3K", "All"]

fig = px.scatter(
    df_final,
    x="MTEB (En) Score",
    y="Mean Gain",
    color="n_gram_thresh",
    
    # 3 Separate Panels
    facet_col="n_gram_thresh", 
    facet_col_wrap=3, 
    category_orders={"n_gram_thresh": category_order},
    
    trendline="ols",
    hover_data=["encoder_raw"],
    color_discrete_map=color_map,
    labels={"Mean Gain": "Gain over Tf-Idf (R² / AUC)", "MTEB (En) Score": "MTEB Score"},
    title="Gain over Tf-Idf by Regime (Macro-Average)"
)

# 5. STYLING
fig.update_traces(line=dict(width=5), selector=dict(mode="lines"))
fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode="markers"))

fig.update_layout(
    autosize=False,
    width=1200, height=500,
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial, monospace", size=18, color="black"),
    showlegend=False,
    margin=dict(l=50, r=50, b=80, t=80)
)

# Add Zero Line
fig.add_shape(
    type="line",
    x0=df_final["MTEB (En) Score"].min() - 2,
    x1=df_final["MTEB (En) Score"].max() + 2,
    y0=0, y1=0,
    line=dict(color="black", width=2, dash="dash"),
    row="all", col="all"
)

# Clean Titles
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

today_date = time.strftime("%Y-%m-%d")
TODAYS_FOLDER = today_date
save_dir = f'/data/parietal/store4/soda/gblayer/salts/results_pics/{TODAYS_FOLDER}/'
os.makedirs(save_dir, exist_ok=True)

pdf_path = os.path.join(save_dir, f'xgboost_gain_percentage_over_tfidf_vs_MTEB_by_ngram_split_{today_date}.pdf')
html_path = os.path.join(save_dir, f'xgboost_gain_percentage_over_tfidf_vs_MTEB_by_ngram_split_{today_date}.html')

# 2. Try Saving
try:
    print(f"Attempting to save PDF to {pdf_path}...")
    fig.write_image(pdf_path, format='pdf', width=1000, height=600)
    print("✅ Success!")
except Exception as e:
    print(f"❌ PDF Export Failed. Error: {e}")
    print(f"⚠️ Saving as HTML instead to {html_path}")
    fig.write_html(html_path)
    print("✅ HTML Saved. Open this file in your browser and 'Print to PDF'.")

fig.show()

'''
HORIZONTAL BARPLOT
Models run for TabPFN-2.5
all encoders with TabPFN-2.5 including big models modifications:
Qwen-3-8B: 30 first embeddings, 32 first embeddings
Llam-3.1-8B: 60 PCA, L2 normalization before 30-PCA
'''

dtype = 'Num+Str'
df = results[(results['learner'] == 'TabPFN-2.5') & (results['dtype'] == dtype) & (results['encoder']!='TabPFN-2.5')] 

# add '30-PCA' to the end of the encoder name if encoder name starts with LM
df['encoder_polished'] = df['encoder'].apply(lambda x: x + ' (30-PCA)' if x.startswith('LM') else x)

# add big model modifications

#### Qwen-3-8B: 30 first embeddings, 32 first embeddings

## 30 first embeddings

save_path_30 = "/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison_qwen_nopca_30.csv"
qwen_nopca_30 = pd.read_csv(save_path_30)

qwen_nopca_30['score'] = qwen_nopca_30['r2'].fillna(qwen_nopca_30['roc_auc'])

meta_nopca_30 = qwen_nopca_30['method'].str.split('_', expand=True, n=2)

qwen_nopca_30['dtype'] = meta_nopca_30[0]
qwen_nopca_30['encoder'] = meta_nopca_30[1]
qwen_nopca_30['learner'] = meta_nopca_30[2]

qwen_nopca_30['dtype'] = qwen_nopca_30['dtype'].replace(dtype_map)

qwen_nopca_30['encoder'] = qwen_nopca_30['encoder'].replace(encoder_map)

# drop '-no_pca' from the learner name
qwen_nopca_30['learner'] = qwen_nopca_30['learner'].str.replace('-no_pca', '')
qwen_nopca_30['learner'] = qwen_nopca_30['learner'].replace(learner_map)

qwen_nopca_30['encoder_polished'] = qwen_nopca_30['encoder'] + ' (' + 'No PCA (30)' + ')'


#preprocess

## 32 first embeddings

save_path_32 = "/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison_qwen_nopca_32.csv"
qwen_nopca_32 = pd.read_csv(save_path_32)

qwen_nopca_32['score'] = qwen_nopca_32['r2'].fillna(qwen_nopca_32['roc_auc'])

meta_nopca_32 = qwen_nopca_32['method'].str.split('_', expand=True, n=2)
qwen_nopca_32['dtype'] = meta_nopca_32[0]
qwen_nopca_32['encoder'] = meta_nopca_32[1]
qwen_nopca_32['learner'] = meta_nopca_32[2]

qwen_nopca_32['dtype'] = qwen_nopca_32['dtype'].replace(dtype_map)

qwen_nopca_32['encoder'] = qwen_nopca_32['encoder'].replace(encoder_map)

# drop '-no_pca' from the learner name
qwen_nopca_32['learner'] = qwen_nopca_32['learner'].str.replace('-no_pca', '')
qwen_nopca_32['learner'] = qwen_nopca_32['learner'].replace(learner_map)

qwen_nopca_32['encoder_polished'] = qwen_nopca_32['encoder'] + ' (' + 'No PCA (32)' + ')'



#### Llama-3.1-8B: 60 PCA, StandScal before 30-PCA


## 60 PCA - NEED TO RERUN

# import pandas as pd
# from pathlib import Path
# from joblib import Parallel, delayed
# from glob import glob
# pca_ablation_path = '/data/parietal/store4/soda/gblayer/salts/results/pca-ablation'
# score_dir = Path(pca_ablation_path)
# score_files = list(score_dir.glob("**/score/*.csv"))
# df_score_ = Parallel(n_jobs=-1)(delayed(pd.read_csv)(args) for args in score_files)
# df_score_pca_60 = pd.concat(df_score_, axis=0)
# df_score_pca_60.reset_index(drop=True, inplace=True)
# df_score_pca_60['score'] = df_score_pca_60['r2'].fillna(df_score_pca_60['roc_auc'])
# meta_pca_60 = df_score_pca_60['method'].str.split('_', expand=True, n=2)
# df_score_pca_60['dtype'] = meta_pca_60[0]
# df_score_pca_60['encoder'] = meta_pca_60[1]
# df_score_pca_60['learner'] = meta_pca_60[2]
# df_score_pca_60['dtype'] = df_score_pca_60['dtype'].replace(dtype_map)

# # drop -60 from encoder name
# df_score_pca_60['encoder'] = df_score_pca_60['encoder'].str.replace('-60', '')

# df_score_pca_60['encoder'] = df_score_pca_60['encoder'].replace(encoder_map)

# df_score_pca_60['learner'] = df_score_pca_60['learner'].replace(learner_map)

# df_score_pca_60['encoder_polished'] = df_score_pca_60['encoder'].apply(lambda x: x + ' (60-PCA)' if x.startswith('LM') else x)


## StandScal before 30-PCA

save_path_30 = "/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison_llama_standscal_pca_30.csv"
llama_pca_30_standscal = pd.read_csv(save_path_30)

llama_pca_30_standscal['score'] = llama_pca_30_standscal['r2'].fillna(llama_pca_30_standscal['roc_auc'])

meta_pca_30 = llama_pca_30_standscal['method'].str.split('_', expand=True, n=2)

llama_pca_30_standscal['dtype'] = meta_pca_30[0]
llama_pca_30_standscal['encoder'] = meta_pca_30[1]
llama_pca_30_standscal['learner'] = meta_pca_30[2]

llama_pca_30_standscal['dtype'] = llama_pca_30_standscal['dtype'].replace(dtype_map)

llama_pca_30_standscal['encoder'] = llama_pca_30_standscal['encoder'].replace(encoder_map)
llama_pca_30_standscal['encoder_polished'] = llama_pca_30_standscal['encoder'] + ' (' + 'StandScal + PCA (30)' + ')'

llama_pca_30_standscal['learner'] = llama_pca_30_standscal['learner'] + "_default"
llama_pca_30_standscal['learner'] = llama_pca_30_standscal['learner'].replace(learner_map)

### add Qwen-3-8B with 5 first embeddings

# Save
save_path = "/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison_qwen_nopca_5.csv"
qwen_nopca_5 = pd.read_csv(save_path)

qwen_nopca_5['score'] = qwen_nopca_5['r2'].fillna(qwen_nopca_5['roc_auc'])

meta_nopca_5 = qwen_nopca_5['method'].str.split('_', expand=True, n=2)
qwen_nopca_5['dtype'] = meta_nopca_5[0]
qwen_nopca_5['encoder'] = meta_nopca_5[1]
qwen_nopca_5['learner'] = meta_nopca_5[2]

qwen_nopca_5['dtype'] = qwen_nopca_5['dtype'].replace(dtype_map)

qwen_nopca_5['encoder'] = qwen_nopca_5['encoder'].replace(encoder_map)

# drop '-no_pca' from the learner name
qwen_nopca_5['learner'] = qwen_nopca_5['learner'].str.replace('-no_pca', '')
qwen_nopca_5['learner'] = qwen_nopca_5['learner'].replace(learner_map)

qwen_nopca_5['encoder_polished'] = qwen_nopca_5['encoder'] + ' (' + 'No PCA (5)' + ')'


#### add Qwen-3-4B with 30 first embeddings

save_path = "/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison_qwen3_4b_nopca_30.csv"
qwen3_4b_nopca_30 = pd.read_csv(save_path)

qwen3_4b_nopca_30['score'] = qwen3_4b_nopca_30['r2'].fillna(qwen3_4b_nopca_30['roc_auc'])

meta_nopca_30 = qwen3_4b_nopca_30['method'].str.split('_', expand=True, n=2)
qwen3_4b_nopca_30['dtype'] = meta_nopca_30[0]
qwen3_4b_nopca_30['encoder'] = meta_nopca_30[1]
qwen3_4b_nopca_30['learner'] = meta_nopca_30[2]

qwen3_4b_nopca_30['dtype'] = qwen3_4b_nopca_30['dtype'].replace(dtype_map)
qwen3_4b_nopca_30['encoder'] = qwen3_4b_nopca_30['encoder'].replace(encoder_map)

# drop '-no_pca' from the learner name
qwen3_4b_nopca_30['learner'] = qwen3_4b_nopca_30['learner'].str.replace('-no_pca', '')
qwen3_4b_nopca_30['learner'] = qwen3_4b_nopca_30['learner'].replace(learner_map)

qwen3_4b_nopca_30['encoder_polished'] = qwen3_4b_nopca_30['encoder'] + ' (' + 'No PCA (30)' + ')'

# select common columns
common_cols_tabpfn = ['r2', 'rmse', 'preprocess_time', 'param_search_time', 'inference_time',
       'run_time', 'data_name', 'method', 'n_cv', 'fold_index', 'task',
       'roc_auc', 'brier_score_loss','f1_weighted', 'score', 'dtype', 'encoder',
       'learner', 'encoder_polished']


# concat -NEED TO RERUN THE 60-PCA FOR LLAMA-3.1-8B, OTHERWISE THE CONCATENATION WILL BE WRONG
# encoders_tabpfn = pd.concat([df[common_cols_tabpfn], qwen_nopca_30[common_cols_tabpfn], 
# qwen_nopca_32[common_cols_tabpfn], df_score_pca_60[common_cols_tabpfn], 
# llama_pca_30_standscal[common_cols_tabpfn], qwen_nopca_5[common_cols_tabpfn], 
# qwen3_4b_nopca_30[common_cols_tabpfn]], axis=0, ignore_index=True)

encoders_tabpfn = pd.concat([df[common_cols_tabpfn], qwen_nopca_30[common_cols_tabpfn], 
qwen_nopca_32[common_cols_tabpfn], llama_pca_30_standscal[common_cols_tabpfn], qwen_nopca_5[common_cols_tabpfn], 
qwen3_4b_nopca_30[common_cols_tabpfn]], axis=0, ignore_index=True)


encoders_tabpfn['encoder_polished'].value_counts()

encoders_tabpfn['dtype'].value_counts()

task = 'all_task' #['all_task', 'classification', 'regression']

if task == 'regression':
    df_encoders_tabpfn = encoders_tabpfn[encoders_tabpfn['task'] == 'regression']
elif task == 'classification':
    df_encoders_tabpfn = encoders_tabpfn[encoders_tabpfn['task'] != 'regression']
else:
    df_encoders_tabpfn = encoders_tabpfn.copy()

score = 'score'

# check encoders

df_encoders_tabpfn['encoder_polished'].value_counts()

prep_barplot = df_encoders_tabpfn.groupby(['encoder_polished'], as_index=False)[score].mean()

df_sorted = prep_barplot.copy() #prep_barplot.set_index('encoder_polished').loc[encoder_order_tabpfn].reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP: Get the Matplotlib/Seaborn Colorblind Palette (10 distinct colors)
cb_palette = sns.color_palette("colorblind", 10).as_hex()

# 2. MAPPING: Assign ONE color per Family
# We group the encoders into 9 distinct families (leaving 1 color as fallback)
family_colors = {
    # Large Language Models
    'llama':        cb_palette[0], # Blue
    'qwen':         cb_palette[1], # Orange
    'e5':           cb_palette[2], # Green
    'minilm':       cb_palette[3], # Red
    'jasper':       cb_palette[8], # Yellow/Olive
    
    # Baselines / Single Encoders
    'targetencoder': cb_palette[4], # Purple
    'tarte':         cb_palette[5], # Brown
    'tf-idf':        cb_palette[6], # Pink
    'fasttext':      cb_palette[7], # Gray
    
    # Fallback
    'fallback':      cb_palette[9]  # Cyan
}

def get_encoder_color(encoder_name):
    name = str(encoder_name).lower()
    
    # Check families in priority order
    if 'llama' in name:         return family_colors['llama']
    if 'qwen' in name:          return family_colors['qwen']
    if 'e5' in name:            return family_colors['e5']
    if 'minilm' in name:        return family_colors['minilm']
    if 'jasper' in name:        return family_colors['jasper']
    
    if 'targetencoder' in name: return family_colors['targetencoder']
    if 'tarte' in name:         return family_colors['tarte']
    if 'tf-idf' in name or 'tfidf' in name: return family_colors['tf-idf']
    if 'fasttext' in name:      return family_colors['fasttext']

    return family_colors['fallback']

# 3. PLOTTING
row_colors = [get_encoder_color(name) for name in df_sorted['encoder_polished']]

plt.figure(figsize=(5, 6))
ax = sns.barplot(
    data=df_sorted,
    y='encoder_polished',
    x='score',
    palette=row_colors,  # Colors are now identical for siblings
    edgecolor='black',   # Adds a border to make bars distinct even if colors match
    dodge=False
)

plt.xlabel('TabPFN-2.5 Average Score (R2 & AUC)', fontsize=12)
plt.ylabel('')
plt.xlim(0.55, 0.78) 


today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_performance_tabpfn_{dtype}_{score}_with_modifications_{task}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
Examples where LLM+TabPFN/XGBoost do much better than Tf-Idf+TabPFN/XGBoost:
'''

from configs.path_configs import path_configs
from datasets_metadata_recap import wide_datasets
from skrub import TableVectorizer
import scipy

def calculate_complexity(series):
    """
    Returns (Uniqueness Ratio, Normalized Entropy)
    Uniqueness Ratio: unique_count / total_count (1.0 = All unique IDs, 0.0 = Constants)
    Normalized Entropy: 0.0 (Order) to 1.0 (Chaos/Max Information)
    """
    s = series.astype(str).dropna()
    n = len(s)
    if n == 0: return 0.0, 0.0
    
    # Get counts of each unique value
    counts = s.value_counts()
    n_unique = len(counts)
    
    # 1. Uniqueness Ratio
    uniqueness = n_unique / n
    
    # 2. Normalized Shannon Entropy
    # entropy = -sum(p * log(p)). Normalized by log(n_unique) so it's 0-1.
    if n_unique <= 1:
        norm_entropy = 0.0
    else:
        probs = counts / n
        entropy = scipy.stats.entropy(probs)
        # Normalize by max possible entropy (log of unique count)
        norm_entropy = entropy / np.log(n_unique)
        
    return uniqueness, norm_entropy

print("--- Processing STRABLE Datasets ---")
strable_path = path_configs["path_data_processed"]
results_complexity = []
for data_name in wide_datasets:
    try:
        path = f"{strable_path}/{data_name}/data.parquet"
        df = pd.read_parquet(path)
        
        # STRABLE LOGIC: Use Skrub with YOUR threshold (0)
        # All string columns are kept as text.
        cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
        df_processed = cleaner.fit_transform(df)
        
        # Identify text columns
        text_cols = [c for c in df_processed.select_dtypes(['object', 'string', 'category']).columns]
        
        for col in text_cols:
            uniq, ent = calculate_complexity(df[col])
            results_complexity.append({'benchmark': 'STRABLE', 'data_name': data_name, 'col': col, 'uniqueness': uniq, 'entropy': ent})
            
    except Exception as e:
        print(f"Skipping STRABLE {data_name}: {e}")

df_results_complexity = pd.DataFrame(results_complexity)

# for each dataset compute avg performance per pipeline
score_for_data_pipeline = df_encoders_tabpfn.groupby(['data_name','encoder_polished'], as_index=False)['score'].mean()

# for each dataset identify encoder_polished with max score
best_pipeline_per_dataset = score_for_data_pipeline.loc[score_for_data_pipeline.groupby('data_name')['score'].idxmax()]

# for each encoder_polished count how many datasets we have
num_dfs_per_pipeline_and_avg_score_per_pipeline = best_pipeline_per_dataset.groupby(['encoder_polished'], as_index=False).agg({'data_name':'nunique','score':'mean'})
best_pipeline_per_df_with_df_char = pd.merge(best_pipeline_per_dataset, results[['data_name', 'avg_string_length_per_cell', 'avg_cardinality', 'string_diversity']].drop_duplicates(), how='left', on='data_name')

avg_entropy_per_data = df_results_complexity.groupby(['data_name'], as_index=False)['entropy'].mean()

best_pipeline_per_df_with_df_char = pd.merge(best_pipeline_per_df_with_df_char, avg_entropy_per_data, on='data_name', how='left')

# dataset characteristics for best pipelines
avg_card_divers_stringlength = best_pipeline_per_df_with_df_char.groupby(['encoder_polished'], as_index=False).agg({'data_name':'nunique', 'avg_string_length_per_cell':'mean', 'avg_cardinality':'mean', 'string_diversity':'mean','entropy':'mean'}).sort_values(by='data_name', ascending=False).reset_index(drop=True)

median_card_divers_stringlength = best_pipeline_per_df_with_df_char.groupby(
    ['encoder_polished'], as_index=False
).agg({
    'avg_string_length_per_cell': median_iqr, 
    'avg_cardinality': median_iqr, 
    'string_diversity': median_iqr,
    'entropy': median_iqr
}).reset_index(drop=True)

#merge
encoder_polished_df_characteristics = pd.merge(avg_card_divers_stringlength, median_card_divers_stringlength, on='encoder_polished', suffixes=('_mean', '_median_iqr'))
encoder_polished_df_characteristics.rename(columns={'data_name':'num_of_datasets_for_which_pipeline_is_best'}, inplace=True)

encoder_polished_df_characteristics = pd.merge(encoder_polished_df_characteristics, num_dfs_per_pipeline_and_avg_score_per_pipeline[['encoder_polished', 'score']], on='encoder_polished', how='left')

# dtype = 'Num+Str'
# df = results[(results['learner'].isin(['TabPFN-2.5'])) & (results['dtype'] == dtype) & (results['encoder']!='TabPFN-2.5')] 
best_pipeline_per_df_with_df_char[['avg_string_length_per_cell', 'avg_cardinality', 'string_diversity','entropy']].drop_duplicates().describe()

# example of datasets where LMs are better: prepaid-financial-product, vehicles
best_pipeline_per_df_with_df_char.sort_values(by='score', ascending=False)[['data_name', 'encoder_polished', 'score', 'avg_string_length_per_cell', 'avg_cardinality', 'string_diversity','entropy']]

# create an indicator: is LM or not
best_pipeline_per_df_with_df_char['is_LM'] = best_pipeline_per_df_with_df_char['encoder_polished'].apply(lambda x: 'LM' in x)

best_pipeline_per_df_with_df_char.groupby(['is_LM']).agg({'avg_string_length_per_cell':'mean', 'avg_cardinality':'mean', 'string_diversity':'mean','entropy':'mean'})

best_pipeline_per_df_with_df_char.groupby(['encoder_polished']).agg({'avg_string_length_per_cell':'mean', 'avg_cardinality':'mean', 'string_diversity':'mean','entropy':'mean'}).sort_values(by='entropy', ascending=False)

'''
LLMs are winning pipelines for datasets with higher Normalized entropy.
'''

# 1. Define the numerical columns we will use for the percentile logic
metrics = [
    'avg_string_length_per_cell', 
    'avg_cardinality', 
    'string_diversity',
    'entropy'
]

# 2. Get the TRUE global distribution (one row per dataset)
# We drop duplicates so datasets with more pipelines don't skew the percentiles
global_datasets = best_pipeline_per_df_with_df_char[['data_name'] + metrics].drop_duplicates()

# Calculate the 33rd and 67th percentiles for the global distribution
percentiles = {}
for col in metrics:
    p33 = global_datasets[col].quantile(0.33)
    p67 = global_datasets[col].quantile(0.67)
    percentiles[col] = (p33, p67)

# 4. Helper function to categorize where the encoder's average sits globally
def categorize(val, p33, p67):
    if pd.isna(val):
        return "N/A"
    elif val <= p33:
        return "Lower 33% (0-33)"
    elif val >= p67:
        return "Upper 33% (67-100)"
    else:
        return "Middle 33% (34-66)"

# 5. Apply the categorization
for col in metrics:
    p33, p67 = percentiles[col]
    
    # Clean up the output column name (remove the '_mean' suffix for readability)
    base_name = col + '_mean'
    
    # Create the insight category column
    encoder_polished_df_characteristics[f'{base_name}_insight'] = encoder_polished_df_characteristics[base_name].apply(lambda x: categorize(x, p33, p67))

'''
datasets for which Tf-Idf - TabPFN-2.5 is the winning algorithm (has the best avg performance on num+str)
across the 3 folds, have middle size string diversity.
datasets for which LM Qwen-3-8B - TabPFN-2.5 (with 30-PCA) have an avg string diversity of 40K, and are in the Upper 33% (67-100)
datasets for which LM Qwen-3-4B - TabPFN-2.5 (with 30-PCA) have an avg string diversity of 39K, and are in the Upper 33% (67-100)
datasets for which LM Llama-3.1-8B - TabPFN-2.5 (with 30-PCA) have an avg string diversity of 40K, and are in the Upper 33% (67-100)
'''

encoder_polished_df_characteristics.sort_values(by=['entropy_mean_insight','num_of_datasets_for_which_pipeline_is_best','score'], ascending=[False,False,False])[['entropy_mean_insight','num_of_datasets_for_which_pipeline_is_best','encoder_polished','entropy_mean','score']]

avg_entropy_vs_score_best_model = encoder_polished_df_characteristics.sort_values(by='entropy_mean', ascending=False)[['encoder_polished','score','entropy_mean','entropy_mean_insight']].reset_index(drop=True)

# scatterplot of avg_entropy_vs_score_best_model
plt.figure(figsize=(5, 4))

# 1. Add the vertical spans for the 3 entropy categories
# To make them stretch seamlessly across the whole plot area:
plt.axvspan(0.50, 0.643, color='blue', alpha=0.15, label='Lower 33%')
plt.axvspan(0.643, 0.767, color='gray', alpha=0.15, label='Middle 33%')
plt.axvspan(0.767, 0.85, color='red', alpha=0.15, label='Upper 33%')

# 2. Your original scatter plot code
scatter = plt.scatter(
    avg_entropy_vs_score_best_model['entropy_mean'],
    avg_entropy_vs_score_best_model['score'],
    # Note: cmap='viridis' only works if you pass a 'c=' argument, 
    # e.g., c=avg_entropy_vs_score_best_model['some_column']
    s=100,
    edgecolor='black',
    zorder=3 # Ensures dots are drawn on top of the grid and spans
)

# 3. Labels and formatting
plt.xlabel('Average Normalized Entropy of\nText Columns (0-1)', fontsize=12)
plt.ylabel('Average Score of Best Pipeline\n(R2 & AUC)', fontsize=12)
plt.grid(True, zorder=0)

# 4. Add a legend to explain the background colors
# You can adjust the location ('best', 'lower right', etc.) depending on where your dots are
plt.legend(loc='best', fontsize=9)
plt.xlim(0.50, 0.85)  # Adjust x-axis limits to focus on the range of entropy values

plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'entropy_vs_score_best_pipeline_{task}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
interesting thing: when we order by entropy, we have a clear split by use of PCA vs use of no PCA.
Low Entropy (Imbalanced): Don't use PCA. You need the full resolution of the embeddings to ensure the rare minority strings aren't accidentally deleted as "low variance noise."
High Entropy (Diverse/Uniform): Always use PCA. You need to compress the massive, noisy embedding space into a few dense features so the tabular model can actually learn from it.
'''

'''
1. Tf-Idf dominates the Lower 33% of Entropy
2. Look closely at the Lower 33% category. The LLM pipelines that win here are predominantly Qwen models without PCA.
3. Middle 33% is completely monopolized by LM LLaMA-3.1-8B (StandScal + PCA (30))
'''

'''
Entropy vs string length and string diversity from best_pipeline_per_df_with_df_char
with matplotlib with trend line
'''
#strin length
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
scatter = plt.scatter(
    best_pipeline_per_df_with_df_char['avg_string_length_per_cell'],
    best_pipeline_per_df_with_df_char['entropy'],
    cmap='viridis',
    s=100,
    edgecolor='black'
)
z = np.polyfit(best_pipeline_per_df_with_df_char['avg_string_length_per_cell'], best_pipeline_per_df_with_df_char['entropy'], 1)
p = np.poly1d(z)
plt.plot(best_pipeline_per_df_with_df_char['avg_string_length_per_cell'], p(best_pipeline_per_df_with_df_char['avg_string_length_per_cell']), color='red', linestyle='--')
plt.xlabel('Average String Length per Cell', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.grid(True)
correlation = best_pipeline_per_df_with_df_char['avg_string_length_per_cell'].corr(best_pipeline_per_df_with_df_char['entropy'])
plt.title(f'Correlation: {correlation:.2f}', fontsize=12)
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'string_length_vs_entropy_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

#string diversity
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
scatter = plt.scatter(
    best_pipeline_per_df_with_df_char['string_diversity'],
    best_pipeline_per_df_with_df_char['entropy'],
    cmap='viridis',
    s=100,
    edgecolor='black'
)
z = np.polyfit(best_pipeline_per_df_with_df_char['string_diversity'], best_pipeline_per_df_with_df_char['entropy'], 1)
p = np.poly1d(z)
plt.plot(best_pipeline_per_df_with_df_char['string_diversity'], p(best_pipeline_per_df_with_df_char['string_diversity']), color='red', linestyle='--')
plt.xlabel('String Diversity', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.grid(True)
correlation = best_pipeline_per_df_with_df_char['string_diversity'].corr(best_pipeline_per_df_with_df_char['entropy'])
plt.title(f'Correlation: {correlation:.2f}', fontsize=12)
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'string_diversity_vs_entropy_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()





# ==========================================
# TABLE 1.2: Task Distribution across categories
# ==========================================

df_datasets = results.drop_duplicates(subset=['data_name']).copy()

print(f"Analysis performed on {len(df_datasets)} unique datasets across {df_datasets['category'].nunique()} sources.")

table1 = pd.crosstab(df_datasets['category'], df_datasets['task'])

# Add Total column
table1['Total'] = table1.sum(axis=1)

# save as latex
table1_final = table1.copy()

# Add a 'Total' row at the bottom
table1_final.loc['Total'] = table1_final.sum()

# Reset index so 'category' becomes a column for the LaTeX table
table1_latex = table1_final.reset_index()
table1_latex = table1_latex.rename(columns={'category': 'Category'})

df_to_export = table1_latex.copy()

# Shorten names for the ICML double-column layout
df_to_export = df_to_export.rename(columns={
    'b-classification': 'b-class',
    'm-classification': 'm-class',
    'regression': 'reg'
})

# 2. Use the Styler with LaTeX-specific bolding
# This creates a \textbf{...} wrapper that LaTeX understands
styler = df_to_export.style.hide(axis="index")

# Bold the "Total" column
styler.format(subset=['Total'], formatter="\\textbf{{{}}}")

# Bold the "Total" row (the last row)
styler.format(subset=pd.IndexSlice[df_to_export.index[-1], :], formatter="\\textbf{{{}}}")

# 3. Export to LaTeX using the modern Styler method
today_date = time.strftime("%Y-%m-%d")
filename = f"category_task_contingency_table_{today_date}.tex"
save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{TODAYS_FOLDER}/{filename}'

# This method is compatible with newer Pandas versions and handles booktabs correctly
latex_output = styler.to_latex(
    hrules=True,
    caption="Distribution of curated datasets across categories and task types.",
    label="tab:dataset_distribution_table1.5",
    column_format="l" + "c" * (len(df_to_export.columns) - 1),
    position="t",
    position_float="centering"
)

# 4. Inject Compactness Commands
# This pulls the columns together as seen in your preferred screenshot
compact_latex = latex_output.replace(
    r"\begin{tabular}", 
    r"\setlength{\tabcolsep}{3pt} \small" + "\n" + r"\begin{tabular}"
)

# 5. Save the file
with open(save_path, 'w') as f:
    f.write(compact_latex)

# ==========================================
# 3. TABLE 2: Aggregated features of tabular datasets across sources
# ==========================================


# Select features to analyze from your columns
features_to_analyze = ['num_rows','num_columns', 'num_text_columns', 'avg_string_length_per_cell', 'avg_cardinality', 'avg_tfidf_cosine_similarity', 'prop_missing_text_cells', 'prop_unique_text_cells']

# drop num_rows_y and rename num_rows_x to num_rows
# df_datasets = df_datasets.rename(columns={'num_rows_x': 'num_rows'}).drop(columns=['num_rows_y'])

# Group by Source and apply the custom formatter
table2 = df_datasets.groupby('category', as_index=False)[features_to_analyze].agg(median_iqr)


print("\n--- Table 2: Aggregated features (Median + IQR) ---")
table2


feature_names_map = {
    'category':'Category',
    'num_rows': 'Number of Rows',
    'num_columns': 'Number of Columns',
    'num_text_columns': 'Number of String Columns',
    'avg_string_length_per_cell': 'Avg. String Length',
    'avg_cardinality': 'Cardinality',
    'avg_tfidf_cosine_similarity': 'Semantic Similarity',
    'prop_missing_text_cells': 'Prop. Missing Values in String Columns',
    'prop_unique_text_cells': 'Prop. Unique Values in String Columns'
}

table2.rename(columns=feature_names_map, inplace=True)


# split into 2 tables: first 3 columns in one table and the rest in another table
# first 3 columns
table2_part1 = table2[['Number of Rows', 'Number of Columns', 'Number of String Columns']]
table2_part1.index = table2['Category']


table2_part1 = table2[['Category', 'Number of Rows', 'Number of Columns', 'Number of String Columns']].copy()

# 2. Initialize the Styler
# We hide the index to avoid the extra column of numbers
styler = table2_part1.style.hide(axis="index")

# 3. Format the data
# Bold the 'Category' column using LaTeX syntax
styler.format(subset=['Category'], formatter="\\textbf{{{}}}")

# Format the numbers to 2 decimal places (as seen in your example output)
# This applies to all columns except Category
num_cols = ['Number of Rows', 'Number of Columns', 'Number of String Columns']
styler.format(subset=num_cols, precision=2)

# 4. Setup file path
today_date = time.strftime("%Y-%m-%d")
filename = f"dataset_features_part1_{today_date}.tex"
save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{TODAYS_FOLDER}/{filename}'

# 5. Export to LaTeX
# We use 'l' for Category and 'c' for the data columns
col_format = "lccc" 

latex_output = styler.to_latex(
    hrules=True,
    caption="Summary statistics (Part 1) of curated datasets by field.",
    label="tab:dataset_features_part1_2026-01-18",
    column_format=col_format,
    position="t",
    position_float="centering"
)

# 6. Inject the compactness and font size commands
# This matches your requested formatting exactly
compact_latex = latex_output.replace(
    r"\begin{tabular}", 
    r"\setlength{\tabcolsep}{4pt} \footnotesize" + "\n" + r"\begin{tabular}"
)

# 7. Save the file
with open(save_path, 'w') as f:
    f.write(compact_latex)

print(f"Table 1 saved exactly as requested to: {save_path}")


# rest of the columns
table2_part2 = table2.drop(columns=['Category','Semantic Similarity', 'Prop. Missing Values in String Columns', 'Prop. Unique Values in String Columns'])
table2_part2.index = table2['Category']


header_map = {
    'Number of Rows': 'Number of Rows', 
    'Number of Columns': 'Number of Columns', 
    'Number of String Columns': 'Number of String Columns',
    'Avg. String Length': '\\makecell{Avg. String\\\\Length}',
    'Cardinality': 'Cardinality',
}

# 'Semantic Similarity': '\\makecell{Semantic\\\\Similarity}',
# 'Prop. Missing Values in String Columns': '\\makecell{Prop. Missing Values\\\\in String Columns}',
# 'Prop. Unique Values in String Columns': '\\makecell{Prop. Unique Values\\\\in String Columns}'


# 2. Prepare the data
# Assuming table2_part2 is your second dataframe
df_export = table2_part2.copy()
df_export.columns = [header_map.get(col, col) for col in df_export.columns]
df_export = df_export.reset_index().rename(columns={'index': 'Category'})

# 3. Initialize Styler
styler = df_export.style.hide(axis="index")

# 4. Apply LaTeX-specific formatting
# Bold the 'Category' column
styler.format(subset=['Category'], formatter="\\textbf{{{}}}")

# Disable automatic escaping so our \makecell and \textbf commands work
styler.format(escape=None)

# 5. Define file path
today_date = time.strftime("%Y-%m-%d")
filename = f"dataset_features_part2_{today_date}.tex"
save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{TODAYS_FOLDER}/{filename}'

# 6. Export to LaTeX
# We use 'c' for data columns; makecell handles the width automatically
num_data_cols = len(df_export.columns) - 1
col_format = "l" + "c" * num_data_cols

latex_output = styler.to_latex(
    hrules=True,
    caption="Summary statistics of curated datasets by category.",
    label="tab:dataset_features_part2",
    column_format=col_format,
    position="t",
    position_float="centering"
)

# 7. Inject compactness tweaks manually
# \footnotesize and \tabcolsep reduce the footprint to fit a single column
compact_latex = latex_output.replace(
    r"\begin{tabular}", 
    r"\setlength{\tabcolsep}{3pt} \footnotesize" + "\n" + r"\begin{tabular}"
)

# 8. Save
with open(save_path, 'w') as f:
    f.write(compact_latex)

print(f"Compact Table 2 with multiline headers saved to: {save_path}")


'''
GLM
'''
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import time
import os

# --------------------------------------------------------------
# 1. HELPER: Custom LaTeX Table Generator
# --------------------------------------------------------------
def save_custom_latex(model, algo_name, save_path):
    """
    Manually writes a LaTeX table ensuring:
    - Clean row names (removed "Upper 33%" from rows)
    - Detailed Caption explaining the High/Low comparison
    - Bold p-values < 0.05 with stars (Intercept excluded)
    """
    summary = model.summary()
    results_as_html = summary.tables[1].as_html()
    df_results = pd.read_html(results_as_html, header=0, index_col=0)[0]
    
    # Rename Index: Cleaner names, explanation moved to caption
    name_map = {
        'Intercept': 'Baseline Score (Low Bins)',
        'C(Card_Bin)[T.High]': 'High Cardinality',
        'C(Str_Bin)[T.High]': 'High String Length',
        'C(n_col_Bin)[T.High]': 'High Num Columns',
        'C(n_row_Bin)[T.High]': 'High Num Rows',
        'C(string_diversity_Bin)[T.High]': 'High String Diversity'
    }
    
    df_results.index = [name_map.get(idx, idx) for idx in df_results.index]
    
    # Start LaTeX String
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\begin{tabular}{lcccccc}\n\\toprule\n"
    latex_str += " & \\textbf{coef} & \\textbf{std err} & \\textbf{z} & \\textbf{P$> |$z$|$} & \\textbf{[0.025} & \\textbf{0.975]} \\\\\n\\midrule\n"
    
    for idx, row in df_results.iterrows():
        # Format metrics
        coef = f"{row['coef']:.4f}"
        stderr = f"{row['std err']:.4f}"
        z_val = f"{row['z']:.3f}"
        ci_lower = f"{row['[0.025']:.3f}"
        ci_upper = f"{row['0.975]']:.3f}"
        
        # P-value formatting
        p_val_raw = row['P>|z|']
        
        # Never bold the Intercept/Baseline
        if "Baseline" in idx:
            p_val_str = f"{p_val_raw:.3f}" 
        elif p_val_raw < 0.05:
            p_val_str = f"\\textbf{{{p_val_raw:.3f}*}}" 
        else:
            p_val_str = f"{p_val_raw:.3f}"
            
        row_name_clean = idx.replace("_", "\\_").replace("%", "\\%")
        
        latex_str += f"{row_name_clean} & {coef} & {stderr} & {z_val} & {p_val_str} & {ci_lower} & {ci_upper} \\\\\n"
        
    latex_str += "\\bottomrule\n\\end{tabular}\n"
    
    # --- UPDATED CAPTION ---
    algo_clean = algo_name.replace("_", "\\_").replace("+", " + ")
    latex_str += f"\\caption{{GLM Analysis for: {algo_clean} (High=Upper 33 percentile vs Low=Lower 33 percentile)}}\n"
    latex_str += "\\label{tab:glm_" + algo_name.replace(" ", "_").replace("+", "") + "}\n"
    latex_str += "\\end{table}"
    
    with open(save_path, 'w') as f:
        f.write(latex_str)

# --------------------------------------------------------------
# 2. Prepare Data & Strict Filtering
# --------------------------------------------------------------
# Filter for Num+Str and target configs
df_analysis = results[
    (results['dtype'] == 'Num+Str') & 
    (results['encoder'].isin(selected_encoders))
].copy()

# 1. Apply Binning
cols_to_bin = {
    'Card_Bin': 'avg_cardinality',
    'Str_Bin': 'avg_string_length_per_cell',
    'n_col_Bin': 'num_columns',
    'n_row_Bin': 'num_rows',
    'string_diversity_Bin': 'string_diversity'
}

for bin_col, feat_col in cols_to_bin.items():
    df_analysis[bin_col] = bin_feature_33_66(df_analysis, feat_col)

# 2. STRICT FILTERING: High vs Low ONLY
print(f"Original Row Count: {len(df_analysis)}")

for bin_col in cols_to_bin.keys():
    df_analysis[bin_col] = df_analysis[bin_col].astype(str)
    df_analysis = df_analysis[df_analysis[bin_col].isin(['Low', 'High'])]
    df_analysis[bin_col] = pd.Categorical(
        df_analysis[bin_col], 
        categories=['Low', 'High'], 
        ordered=True
    )

print(f"Filtered Row Count (Intersection): {len(df_analysis)}")

# --------------------------------------------------------------
# 3. Loop: Multivariate GLM per Encoder-Learner
# --------------------------------------------------------------
print("-" * 110)
print(f"{'Algorithm':<40} | {'Feature':<35} | {'Coeff':<8} | {'P-val':<8}")
print("-" * 110)

target_configs = df_analysis['encoder_learner'].drop_duplicates().to_list()

for algo in target_configs:
    df_algo = df_analysis[df_analysis['encoder_learner'] == algo].copy()
    
    if len(df_algo) < 10: 
        print(f"Skipping {algo} (Insufficient data: {len(df_algo)} rows)")
        continue

    formula = (
        "score ~ C(Card_Bin) + C(Str_Bin) + C(n_col_Bin) + C(n_row_Bin) + C(string_diversity_Bin)"
    )
    
    try:
        model = smf.glm(
            formula=formula, 
            data=df_algo, 
            family=sm.families.Gamma(link=sm.families.links.Log())
        ).fit()
        
        # --- Console Output ---
        p_values = model.pvalues
        params = model.params
        for feat in params.index:
            if feat == "Intercept": continue
            if p_values[feat] < 0.05:
                clean_feat = feat.split("[T.")[-1].replace("]", "")
                print(f"{algo:<40} | {clean_feat:<35} | {params[feat]:.4f}   | {p_values[feat]:.4f} *")
                
        # --- Save ---
        today_date = time.strftime("%Y-%m-%d")
        clean_name = algo.replace(" + ", "_").replace(" ", "")
        filename = f"glm_multivariate_{clean_name}_{today_date}.tex"
        
        save_dir = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{TODAYS_FOLDER}/'
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, filename)
        save_custom_latex(model, algo, save_path)
            
    except Exception as e:
        print(f"Error fitting model for {algo}: {e}")

print("-" * 110)
print("Processing complete. Tables generated with updated captions and row names.")



'''
isotropic vs anisotropic characteristic of LLMs
'''

from sklearn.metrics.pairwise import cosine_similarity
from configs.exp_configs import data_list_wide
import re

llm_time_track = pd.read_csv("/data/parietal/store4/soda/gblayer/salts/results/compiled_results/LLM_embedding_timetrack.csv")

llm_list_raw = llm_time_track['method'].drop_duplicates().tolist()

# --- 1. LOAD DATA (Using your exact paths) ---
print("Loading Dataframes...")

# for llm_name in llm_list_raw:

# for df_name in data_list_wide:

df_name = data_list_wide[0]

print(f"\nProcessing dataset: {df_name}...")
# df_name = 'antenna-structure-registration'

try:
    # Path with 'llm_embeding' (single d) as provided
    embeddings_llama8B = pd.read_parquet(
        f'/data/parietal/store4/soda/gblayer/salts/data/llm_embeding/llm-llama-3.1-8b/llm-llama-3.1-8b|{df_name}.parquet'
    )
    
    embeddings_miniLMl6 = pd.read_parquet(
        f'/data/parietal/store4/soda/gblayer/salts/data/llm_embeding/llm-all-MiniLM-L6-v2/llm-all-MiniLM-L6-v2|{df_name}.parquet'
    )
    print("Data loaded successfully.")

except Exception as e:
    print(f"\nCRITICAL ERROR loading files: {e}")
    print("Double check if the folder is 'llm_embedding' (2 d's) or 'llm_embeding' (1 d).")
    # Stop execution if data isn't loaded
    raise e

# Organize into a dictionary for looping
models_data = {
    'LM All-MiniLM-L6-v2': embeddings_miniLMl6,
    'LM LLaMA-3.1-8B': embeddings_llama8B
}

# --- 2. SETUP PLOTS ---
# Try importing UMAP (requires 'pip install umap-learn')
try:
    import umap
except ImportError:
    raise ImportError("UMAP not found. Please run: pip install umap-learn")

fig, axes = plt.subplots(2, 2, figsize=(14, 12)) 

# --- 3. GENERATE PLOTS ---
for i, (model_name, df) in enumerate(models_data.items()):
    
    # Extract embedding columns (X0, X1, ...)
    # Your screenshots show columns are X0, X1, X2...
    emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
    # Sort columns numerically to be safe
    emb_cols.sort(key=lambda x: int(x[1:]))
    
    embs = df[emb_cols].values
    print(f"Processing {model_name} (Shape: {embs.shape})...")

    # --- PLOT A: UMAP (Geometry) ---
    # Subsample 2000 points for speed and clarity
    n_sub = min(2000, len(embs))
    indices = np.random.choice(len(embs), n_sub, replace=False)
    embs_sub = embs[indices]
    
    print(f"  -> Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    proj = reducer.fit_transform(embs_sub)
    
    ax_umap = axes[0, i]
    ax_umap.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.6, c='royalblue')
    ax_umap.set_title(f"{model_name}\nUMAP Projection (Geometry)", fontweight='bold')
    ax_umap.axis('off')
    
    # --- PLOT B: HEATMAP (Anisotropy) ---
    # Use the first 50 samples to visualize self-similarity
    print(f"  -> Generating Heatmap...")
    embs_heat = embs[:50]
    sim_matrix = cosine_similarity(embs_heat)
    
    ax_heat = axes[1, i]
    # vmin=0 forces the scale to show if the baseline is 0 (good) or high (bad)
    sns.heatmap(sim_matrix, ax=ax_heat, cmap='viridis', vmin=0, vmax=1, square=True, cbar=True)
    ax_heat.set_title(f"{model_name}\nCosine Similarity (First 50 samples)", fontweight='bold')
    ax_heat.axis('off')

plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'umap_projection_cosine_similarity_miniLML6_LLama8B_{df_name}_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
Coefficient of Variation
takes 30 minutes to run
'''

from sklearn.metrics.pairwise import cosine_distances

# --- CONFIGURATION ---
# Path used based on your previous logs (single 'd' in embedding)
BASE_PATH = "/data/parietal/store4/soda/gblayer/salts/data/llm_embeding"

# Define models and their "Family" for coloring
# Contrastive = Explicitly trained to separate positive/negative pairs (Clumpy)
# Generative = Trained for next-token prediction (Smooth)
models_config = {
    'LM All-MiniLM-L6-v2': {'folder': 'llm-all-MiniLM-L6-v2', 'family': 'Contrastive (Isotropic)'},
    'LM E5-small-v2':      {'folder': 'llm-e5-small-v2',      'family': 'Contrastive (Anisotropic)'},
    'LM FastText':         {'folder': 'llm-fasttext',         'family': 'Contrastive (Isotropic)'},
    'LM LLaMA-3.1-8B':     {'folder': 'llm-llama-3.1-8b',     'family': 'Generative (Smooth)'},
    'LM Qwen-3-8B':        {'folder': 'llm-qwen3-8b',         'family': 'Generative (Smooth)'},
    'LM Jasper-0.6B':      {'folder': 'llm-jasper-token-comp-0.6b', 'family': 'Generative (Collapsed)'}
}

def get_distance_cv(model_label, config):
    folder = config['folder']
    dir_path = os.path.join(BASE_PATH, folder)
    
    if not os.path.exists(dir_path):
        print(f"Path not found: {dir_path}")
        return []
    
    cv_scores = []
    files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
    
    print(f"Processing {len(files)} datasets for {model_label}...")
    
    for filename in files:
        try:
            file_path = os.path.join(dir_path, filename)
            df = pd.read_parquet(file_path)
            
            # Extract embedding columns (X0, X1...)
            emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
            if not emb_cols: continue
            
            # Subsample for speed (1000 rows is sufficient for robust distance stats)
            n_sample = min(1000, len(df))
            embs = df[emb_cols].sample(n=n_sample, random_state=42).values
            
            # Compute Pairwise Cosine Distances
            dists = cosine_distances(embs)
            
            # Extract upper triangle to get unique pairwise distances (exclude diagonal 0s)
            mask = np.triu_indices_from(dists, k=1)
            dist_values = dists[mask]
            
            # Calculate Coefficient of Variation (Cv = Sigma / Mu)
            mu = np.mean(dist_values)
            sigma = np.std(dist_values)
            
            if mu > 0:
                cv_scores.append(sigma / mu)
            
        except Exception as e:
            continue
            
    return cv_scores

# --- DATA GATHERING ---
all_data = []

for model_name, config in models_config.items():
    scores = get_distance_cv(model_name, config)
    for s in scores:
        all_data.append({
            'Model': model_name, 
            'Coefficient of Variation (Cv)': s,
            'Family': config['family']
        })

plot_df = pd.DataFrame(all_data)

# --- PLOTTING ---
plt.figure(figsize=(12, 7))

# Define color palette for families
palette = {
    'Contrastive (Isotropic)': '#2ecc71',   # Green (Good)
    'Contrastive (Anisotropic)': '#3498db', # Blue (Good/Okay)
    'Generative (Smooth)': '#f1c40f',       # Yellow (Bad)
    'Generative (Collapsed)': '#e74c3c'     # Red (Worst)
}

# Create Boxplot
sns.boxplot(
    x='Model', 
    y='Coefficient of Variation (Cv)', 
    hue='Family',
    data=plot_df, 
    palette=palette, 
    dodge=False,
    showfliers=False
)

# Overlay strip plot for detail
sns.stripplot(
    x='Model', 
    y='Coefficient of Variation (Cv)', 
    data=plot_df, 
    color='black', 
    alpha=0.2, 
    size=2, 
    jitter=True
)

# Styling
plt.title("Signal-to-Noise Ratio of Embedding Geometry\n(Coefficient of Variation of Pairwise Distances)", fontsize=16, fontweight='bold')
plt.ylabel("Distance Contrast ($C_v = \sigma / \mu$)\nHigher is Better for TabPFN", fontsize=12)
plt.xlabel("")
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Model Family / Geometry", loc='upper right')

plt.tight_layout()
plt.show()


'''
boxplot of cosine similarity
takes 30 minutes to run
'''


# --- CONFIGURATION ---
BASE_PATH = "/data/parietal/store4/soda/gblayer/salts/data/llm_embeding" # Note: single 'd' as per your logs

selected_models = [
    'LM All-MiniLM-L6-v2',
    'LM FastText',
    'LM E5-small-v2',
    'LM LLaMA-3.1-8B',
    'LM Qwen-3-8B',
    'LM Jasper-0.6B'
]

folder_map = {
    'LM All-MiniLM-L6-v2': 'llm-all-MiniLM-L6-v2',
    'LM FastText': 'llm-fasttext',
    'LM E5-small-v2': 'llm-e5-small-v2',
    'LM LLaMA-3.1-8B': 'llm-llama-3.1-8b',
    'LM Qwen-3-8B': 'llm-qwen3-8b',
    'LM Jasper-0.6B': 'llm-jasper-token-comp-0.6b' 
}

def get_anisotropy_score(model_label):
    folder = folder_map.get(model_label)
    if not folder: return []
    
    dir_path = os.path.join(BASE_PATH, folder)
    if not os.path.exists(dir_path): return []
    
    scores = []
    
    # Iterate over all parquet files in the folder (datasets)
    files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
    
    # Optional: Limit to first 20 datasets for quick testing, remove limit for final plot
    # files = files[:20] 
    
    print(f"Processing {len(files)} datasets for {model_label}...")
    
    for filename in files:
        try:
            file_path = os.path.join(dir_path, filename)
            df = pd.read_parquet(file_path)
            
            # Extract embedding columns (X0, X1...)
            emb_cols = [c for c in df.columns if re.match(r'^X\d+$', c)]
            if not emb_cols: continue
            
            # Subsample 100 rows to calculate avg similarity (speed optimization)
            # 100 rows is enough to estimate the "background similarity"
            n_sample = min(100, len(df))
            embs = df[emb_cols].sample(n=n_sample, random_state=42).values
            
            # Calculate Cosine Similarity Matrix
            sim_matrix = cosine_similarity(embs)
            
            # Get the average of the off-diagonal elements (similarity of distinct pairs)
            # We want to know: "How similar are two random different rows?"
            mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
            avg_sim = sim_matrix[mask].mean()
            
            scores.append(avg_sim)
            
        except Exception as e:
            continue
            
    return scores

# --- DATA GATHERING ---
all_data = []

for model in selected_models:
    scores = get_anisotropy_score(model)
    for s in scores:
        all_data.append({'Model': model, 'Avg Cosine Similarity': s})

plot_df = pd.DataFrame(all_data)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot Boxplot
sns.boxplot(x='Model', y='Avg Cosine Similarity', data=plot_df, palette="viridis", showfliers=False)
# Optional: Add strip plot to see individual datasets
sns.stripplot(x='Model', y='Avg Cosine Similarity', data=plot_df, color='black', alpha=0.3, size=2, jitter=True)

plt.title("Quantifying Anisotropy: Average Self-Similarity across 108 Datasets", fontsize=14, fontweight='bold')
plt.ylabel("Avg Cosine Similarity\n(Higher = More Anisotropic/Collapsed)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()







# '''
# KendallTau Correlation Plot  for (num+str, num) vs performance
# '''

# # 1. Aggregate: Average score per configuration per dataset/dtype
# # We use 'score' directly as requested
# agg_results = results.groupby(['data_name', 'dtype', 'encoder_learner'])['score'].mean().reset_index()

# plot_data = []
# for name, group in agg_results.groupby('data_name'):
#     # Pivot to align models across dtypes for this specific dataset
#     pivot = group.pivot(index='encoder_learner', columns='dtype', values='score')
    
#     # REQUIREMENT: Skip the dataset if 'Num' or 'Num+Str' columns are missing
#     if 'Num' not in pivot.columns or 'Num+Str' not in pivot.columns:
#         continue
        
#     # Clean up: keep only models that have results for both
#     pivot_clean = pivot[['Num', 'Num+Str']].dropna()
    
#     # Y-axis: Kendall Tau (Correlation between rankings)
#     tau, _ = kendalltau(pivot_clean['Num'], pivot_clean['Num+Str'])
    
#     # X-axis: Mean performance of the 'best' version only
#     best_perf = pivot_clean['Num+Str'].mean()
    
#     plot_data.append({
#         'data_name': name,
#         'kendall_tau': tau,
#         'avg_score_num_str': best_perf
#     })

# df_plot = pd.DataFrame(plot_data)

# fig, ax = plt.subplots(figsize=(5, 4))

# # 1. Scatterplot
# sns.scatterplot(data=df_plot, x='avg_score_num_str', y='kendall_tau', 
#                 ax=ax, alpha=0.6, s=40, color='steelblue', edgecolor='w')

# # 2. Absolute Reference Lines at 0.5
# ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
# ax.axvline(0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

# # 3. Label the Quadrants for clarity in your interpretation
# ax.text(0.25, 0.75, "Numerical\nConsensus", ha='center', va='center', fontsize=9, alpha=0.7)
# ax.text(0.75, 0.75, "Consistent\nWins", ha='center', va='center', fontsize=9, alpha=0.7)
# ax.text(0.25, 0.25, "Unstable\nComplexity", ha='center', va='center', fontsize=9, alpha=0.7)
# ax.text(0.75, 0.25, "Fundamental\nDomain Shift", ha='center', va='center', 
#         fontsize=9, color='darkred', fontweight='bold')
# # 4. Final Polish
# ax.set_title("Absolute Benchmark Sensitivity (Threshold = 0.5)", fontsize=11, fontweight='bold')
# ax.set_xlabel("Average Score (Num+Str)", fontsize=10)
# ax.set_ylabel(r"Ranking Correlation ($\tau$)", fontsize=10)
# ax.set_xlim(0, 1.05)
# ax.set_ylim(-0.1, 1.1)
# ax.grid(True, alpha=0.1)

# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'kendalltau_vs_performance_domain_shift_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()



'''
Strings = Different Domain. If the average $\tau$ is around 0.5–0.6, you can claim: "If strings just reduced to numerical learning, the rankings would be identical. 
Success implies Shift. A downward slope means that the datasets where your new pipelines perform best are also the ones where the ranking changed the most. This proves that high performance is driven by the domain shift, not by accidental numerical improvements.


The Distance from $Y=1.0$: If the preprocessing simply reduced strings to standard numerical learning, every single dot would lie on the $y=1.0$ line (perfect consensus).The "Fundamental Difference" Evidence: The spread of points below the horizontal line represents datasets where the "rules of the game" changed. These are tasks where the numerical leaderboard is effectively irrelevant once string content is introduced.
'''


'''
Quadrant,Suggested Summary Text,Intuition behind the Choice
Top-Right (High Perf / High Stability),"""Numerical Consensus""","Performance is high, and rankings are consistent with the Num baseline. Intuition: String features here act as ""clean"" numerical enhancements. The preprocessing improves the score without changing which models are best, suggesting these datasets are ""numerically grounded""."
Bottom-Right (High Perf / Low Stability),"""Fundamental Domain Shift""","Performance is high, but the Num+Str leaderboard is fundamentally different from Num. Intuition: This is your strongest proof. The best models succeed specifically because they capture non-numerical signals. The preprocessing does not reduce to standard numerical learning; it enables a new type of mastery."
Top-Left (Low Perf / High Stability),"""Consistent Complexity""","Performance is low, but the ranking is relatively stable. Intuition: These are ""stubbornly hard"" datasets. Neither the numerical nor the string-aware models have found the signal yet, so the hierarchy remains unchanged by the addition of text."
Bottom-Left (Low Perf / Low Stability),"""Unstable Complexity""","Both performance and stability are low. Intuition: These represent ""Chaotic"" domains. The task is so difficult that adding strings creates a massive shift in ranking, but since performance is still near-random (0.5), it suggests the current models are ""guessing"" differently rather than ""learning"" differently."
'''

'''
KendallTau Correlation Plot  for (num+str, num) vs improvement of Num+Str over Num
'''

# agg_results = results.groupby(['data_name', 'dtype', 'encoder_learner'])['score'].mean().reset_index()

# plot_data = []
# for name, group in agg_results.groupby('data_name'):
#     pivot = group.pivot(index='encoder_learner', columns='dtype', values='score')
    
#     if 'Num' not in pivot.columns or 'Num+Str' not in pivot.columns:
#         continue
        
#     pivot_clean = pivot[['Num', 'Num+Str']].dropna()
    
#     # Y-axis: Kendall Tau
#     tau, _ = kendalltau(pivot_clean['Num'], pivot_clean['Num+Str'])
    
#     # X-axis: Absolute Improvement
#     improvements = pivot_clean['Num+Str'] - pivot_clean['Num']
#     avg_imp = improvements.mean()
    
#     # NEW: Find the specific model with the highest delta for this dataset
#     best_config = improvements.idxmax()
#     max_delta = improvements.max()
    
#     plot_data.append({
#         'data_name': name,
#         'kendall_tau': tau,
#         'avg_improvement_num_str': avg_imp,
#         'best_model': best_config,
#         'max_delta': max_delta
#     })

# df_plot = pd.DataFrame(plot_data)

# fig, ax = plt.subplots(figsize=(6, 5)) # Slightly larger for labels

# # 1. Scatterplot
# sns.scatterplot(data=df_plot, x='avg_improvement_num_str', y='kendall_tau', 
#                 ax=ax, alpha=0.6, s=50, color='steelblue', edgecolor='w')

# # 2. Reference Lines
# ax.axhline(0.8, color='black', linestyle='--', linewidth=1.2, alpha=0.8) # how should we define the threshold? is 0.5 to strict?
# ax.axvline(0.1, color='black', linestyle='--', linewidth=1.2, alpha=0.8) #how should we define the threshold? is 0.5 to strict?

# # 3. Quadrant Labels
# ax.text(0.6, 0.45, "Tf-Idf - Ridge", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.5, 0.31,  "ContextTab", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.60, 0.28,  "Tf-Idf - Ridge", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.65, 0.2,  "Tf-Idf - Ridge", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.65, 0.07,  "Tf-Idf - ExtraTrees", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.5, 0.015, "ContextTab", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.7, -0.01, "TabPFNv2.5", fontsize=7, color='darkred', fontweight='bold')
# ax.text(0.73, -0.08, "ContextTab", fontsize=7, color='darkred', fontweight='bold')

# # 5. Final Polish
# ax.set_title("STRABLE: Leaders of the Domain Shift", fontsize=12, fontweight='bold')
# ax.set_xlabel("Average Improvement $\Delta{(Num+Str, Num)}$", fontsize=11)
# ax.set_ylabel(r"Ranking Correlation ($\tau$)", fontsize=11)
# ax.set_xlim(0, 1.05)
# ax.set_ylim(-0.1, 1.1)
# ax.grid(True, alpha=0.1)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'kendalltau_vs_performance_domain_shift_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


'''
Domain shift proof
For each model: compute the avg num+str performance across all datasets, the avg num performance across all datasets, and compute:
- Kendalltau correlation —> get one number: need to understand how to justify this - 0.5 threshold? Significant?
- Delta for each model between num+str and num
'''


# subset_a = results[results['method'].str.contains('num-only_tabvec')]

# subset_b = results[results['method'].str.contains('num-only_tabpfn')]

# subset_c = results[results['method'].str.contains('num-only_tabpfn')]


# #for subset a: if it's written tabvec replace tarenc in method column
# for _, row in subset_a.iterrows():
#     if 'tabvec' in row['method']:
#         subset_a['method'] = row['method'].replace('tabvec', 'tarenc')

# #for subset b: if it's written tabvec replace tarenc in method column
# for _, row in subset_b.iterrows():
#     if 'tabpfn' in row['method']:
#         subset_b['method'] = row['method'].replace('tabpfn', 'tabvec', 1)

# #for subset c: if it's written tabvec replace tarenc in method column
# for _, row in subset_c.iterrows():
#     if 'tabpfn' in row['method']:
#         subset_c['method'] = row['method'].replace('tabpfn', 'tarenc', 1)

# # Concatenate all subsets
# results_filtered = pd.concat([subset_a, subset_b, subset_c], ignore_index=True)

# # 4. Concatenate and create the final Pivot Table
# results_aligned = pd.concat([results, results_filtered], ignore_index=True)

# # compute the avg model performance for num and num+str from scratch
# results_avg = results_aligned.groupby(['method_polished', 'dtype'])['score'].mean().reset_index()
# pivot_avg = results_avg.pivot(index='method_polished', columns='dtype', values='score')
# # from each method_polished drop the \n(xx) part
# pivot_avg.index = pivot_avg.index.str.split('\n').str[0]
# # drop Str columns
# pivot_avg = pivot_avg[['Num', 'Num+Str']].reset_index()
# # re-order the results and keep 1 row per method_polished
# final_leaderboard = pivot_avg.groupby('method_polished').first()
# # drop rows with NaN values
# final_leaderboard = final_leaderboard.dropna()
# # give the ranks
# final_leaderboard['Rank_Num'] = final_leaderboard['Num'].rank(ascending=False)
# final_leaderboard['Rank_Num+Str'] = final_leaderboard['Num+Str'].rank(ascending=False)
# # Compute Global Tau
# global_tau_avg, _ = kendalltau(final_leaderboard['Rank_Num'], final_leaderboard['Rank_Num+Str'])
# print(f"Global Kendall Tau between Num and Num+Str across all models (from avg scores): {global_tau_avg:.4f}")
# final_leaderboard['delta'] = final_leaderboard['Num+Str'] - final_leaderboard['Num']
# final_leaderboard['perc_improvement'] = (final_leaderboard['delta'] / final_leaderboard['Num']) * 100
# final_leaderboard['perc_improvement'].mean()


'''
Overall performance per dtypes-encoders
'''

# data_for_input = results[(results['dtype'] == 'Num+Str')]
dtype = 'Num+Str'
llm_subset = 'selected_LMs'
data_for_input = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders))]
algo_representation = 'encoder_learner'


for score in score_list:
    sns.set_context("paper", font_scale=1.7) 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 40)) 
    ax = sns.barplot(
        data=data_for_input,       
        x=score, 
        y=algo_representation, 
        order=data_for_input.groupby(algo_representation)[score].mean().sort_values(ascending=False).index,       
        palette="RdYlBu",      
        errorbar='se',    
        capsize=0.1,           
        edgecolor="black",       
        linewidth=1
    )
    # 3. Explicitly set large font sizes for axis labels
    plt.xlabel("Average Score (R2 & AUC)", fontsize=18, ha='left', x=-0.8)
    plt.ylabel("")
    # Increase tick size - if fonts are still small, increase these values
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=13)
    # Spine and line styling
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    plt.xlim(0.5, 1)
    sns.despine(left=False, bottom=True, top=True, right=True) # Cleaner way to remove spines
    plt.setp(ax.lines, color='black', linewidth=1.5) 


    #save picture
    # format fot the pic name: plot_type + _ + metric + _ + level + date .png
    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'horizontal_barplot_avg_score_method_{score}_{today_date}.{format}'
    plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()



'''
Overall performance per dtypes-encoders - Num+Str
'''

sns.set_context("paper", font_scale=1.8) 
sns.set_theme(style="whitegrid")
plt.figure(figsize=(3, 40)) 
order = results[(results['dtype'] == 'Num+Str')].groupby('encoder_learner')['score'].mean().sort_values(ascending=False).index
ax = sns.barplot(
    data=results[(results['dtype'] == 'Num+Str')],       
    x='score', 
    y='encoder_learner', 
    order=order,       
    palette="RdYlBu",      
    errorbar='se',    
    capsize=0.1,           
    edgecolor="black",       
    linewidth=1
)
# 3. Explicitly set large font sizes for axis labels
plt.xlabel("Average Score (R2 & AUC)", fontsize=18, ha='left', x=-0.5)
plt.ylabel("Encoder (Num+Str)",fontsize=18)
# Increase tick size - if fonts are still small, increase these values
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Spine and line styling
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
plt.xlim(0.5, 0.85)
# sns.despine(left=False, bottom=True, top=True, right=True) # Cleaner way to remove spines
plt.setp(ax.lines, color='black', linewidth=1.5) 


#save picture
# format fot the pic name: plot_type + _ + metric + _ + level + date .png
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'horizontal_barplot_avg_score_method_num_str_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Overall performance per dtypes-encoders - Num+Str - top5 vs last 5
'''

# 1. Setup
sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
sns.set_context("paper", font_scale=1.2)

# Data subsetting (Top 5 / Bottom 5)
df_filtered = results[(results['dtype'] == 'Num+Str') & (results['encoder'].isin(selected_encoders))]
full_order = df_filtered.groupby('encoder_learner')['score'].mean().sort_values(ascending=False).index
subset_order = list(full_order[:5]) + list(full_order[-5:])
plot_df = df_filtered[df_filtered['encoder_learner'].isin(subset_order)]

fig, ax = plt.subplots(figsize=(5, 6))

# 2. Plotting with Refined Error Bars
sns.barplot(
    data=plot_df,
    x='score',
    y='encoder_learner',
    order=subset_order,
    palette="RdYlBu",  # Or "rocket" for colorblind safety
    errorbar='se',
    capsize=0.15,
    edgecolor="black",
    linewidth=0.8,
    width=0.8,
    ax=ax
)

# Refine error bar lines specifically
plt.setp(ax.lines, color='black', linewidth=1.0)

# 3. Labels (NO manual x-shift)
ax.set_xlabel("Avg Score ($R^2$ & AUC)", fontsize=18, labelpad=10)
ax.set_ylabel("Encoder (Num+Str)", fontsize=18, labelpad=8)

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=18)

# 4. Axis limits
ax.set_xlim(0, plot_df['score'].max() * 1.08)

# 5. Reference line
ax.axhline(4.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)

# Annotations (Using data coordinates for stability)
x_pos = ax.get_xlim()[1] * 0.98
ax.text(x_pos, 2, 'Top 5', va='center', ha='right', fontsize=16, fontweight='bold', color='#d73027')
ax.text(x_pos, 7, 'Bottom 5', va='center', ha='right', fontsize=16, fontweight='bold', color='#4575b4')

# 6. Spine styling
sns.despine(left=False, bottom=False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)


#save picture
# format fot the pic name: plot_type + _ + metric + _ + level + date .png
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'horizontal_barplot_avg_score_method_num_str_top5_vs_last5_selectedLLMs_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()
    




'''
 the Wilcoxon signed ranks test for comparison of
two classifiers and the Friedman test with the corresponding post-hoc tests for comparison of more
classifiers over multiple data sets. Results of the latter can also be neatly presented with the newly
introduced CD (critical difference) diagrams.
cite:https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
'''

#clip negative
# results['score'] = results['score'].clip(lower=0)

'''
standard benchmarking methodology (Demšar, 2006):
1. Aggregate folds: Average the scores across folds for each dataset/method pair to ensure samples are independent (pairing by dataset).
2. Calculate Ranks: Compute the average rank of each method across all datasets.
3. Post-hoc Wilcoxon: Run pairwise Wilcoxon signed-rank tests to generate the matrix of p-values.
'''

# We need a matrix where Rows = Datasets (samples) and Columns = Methods.
# We average across 'fold_index' to get one score per dataset per method.
# We use 'score_norm' as it allows comparison across different dataset scales.
pivot_df = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# PAY ATTENTION TO DROP NANS IN THE CORRECT DIMENSION
pivot_df = pivot_df.dropna(axis=1)  

# pivot_df = pivot_df.dropna(axis=0)  


print(f"Data shape after pivoting (Datasets, Methods): {pivot_df.shape}")

# Calculate and Print Average Ranks
# Rank data: for each dataset (row), rank the methods (columns). 
# Ascending=False because higher score (e.g., R2/AUC) is usually better.
ranks = pivot_df.rank(axis=1, ascending=False)
mean_ranks = ranks.mean().sort_values()

print("\n--- Average Ranks (Lower is Better) ---")
print(mean_ranks)

# 4. Friedman Test (Omnibus check)
# It is good practice to check if *any* difference exists before doing pairwise tests.
# stat, p = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
# print(f"\nFriedman Test p-value: {p:.4e}")
# if p < 0.05:
#     print(">> Significant difference detected among methods. Proceeding to post-hoc.")
# else:
#     print(">> No significant difference detected overall.")

# Pairwise Wilcoxon Signed-Rank Test
# This generates the P-value matrix.
# 'holm' correction is recommended to control the Family-Wise Error Rate (FWER).
# pivot_df columns are methods, rows are datasets.
# .values gives a matrix (N_datasets, N_methods).
# .T transposes it to (N_methods, N_datasets).
# This creates a list where each element is the array of scores for one method.
# p_values = sp.posthoc_wilcoxon(pivot_df.values.T, p_adjust='holm')

# # To keep the labels in the result, you can set the index and columns of the output manually:
# p_values.index = pivot_df.columns
# p_values.columns = pivot_df.columns

# print("\n--- Pairwise Wilcoxon P-Values (Holm Corrected) ---")
# print(p_values)








'''
RESIDUAL ANALYSIS
'''

# 1. PREPARE THE DATA
# Extract observed values
obs_n = df_real_agg['N_datasets'].values
obs_tau = df_real_agg['median'].values
obs_sem = df_real_agg['sem'].values

# 2. CALCULATE MEDIAN PARAMETERS FOR EACH MODEL
# We need to compute the median across all successful bootstrap fits (popt)
# Note: Ref1 results did not store popt in your loop, so we'll recalculate the median curve
res_data = {}

for name in ['ref1']:
    if len(results_kendalltau_extrap[name]['curves']) > 0:
        # Convert list of curve arrays to a 2D matrix (bootstraps, x_range_smooth)
        curves = np.array(results_kendalltau_extrap[name]['curves'])
        
        # We need the prediction at exactly the OBSERVED N values, not the smooth range
        # To be precise, we calculate the median prediction for each observed N
        med_curve = np.median(curves, axis=0)
        
        # Map the smooth median curve back to the discrete observed N points
        # interp ensures we compare exactly at the same X-coordinates
        y_pred = np.interp(obs_n, x_range_smooth, med_curve)
        
        # Calculate residuals: Actual - Predicted
        res_data[name] = obs_tau - y_pred

# 3. PLOTTING THE COMPARISON
plt.figure(figsize=(6, 3))
plt.axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

# Style and Color mapping
styles = {
    'ref1': {'label': r'$1 - \frac{a}{\sqrt{x}} e^{-bx}$', 'color': 'green',  'marker': 's'}}

for name, residuals in res_data.items():
    mae = np.mean(np.abs(residuals))
    plt.errorbar(
        obs_n, residuals, yerr=obs_sem, 
        fmt=styles[name]['marker'], color=styles[name]['color'], 
        label=f"{styles[name]['label']} (MAE: {mae:.4f})",
        capsize=3, alpha=0.8, markersize=3
    )

# Formatting
plt.title('Residuals ', fontsize=18)
plt.xlabel('Number of Datasets (N)', fontsize=16)
plt.ylabel('Residual (Actual - Predicted)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_residuals_greenfunction_{today_date}_2.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
the benchmark to converge to the same ranking?
BOOTSTRAPPED EXPONENTIAL SATURATION FITTING (EXTRAPOLATION) - 1 - (a / np.sqrt(x)) * np.exp(-b * x)

by CV fold
'''

# Restructuring Stability Scores by Fold
pivot_df_folds = results.pivot_table(
    index=['data_name', 'fold_index'], 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# pivot_df_folds=pivot_df_folds.dropna(axis=0)

pivot_df_folds=pivot_df_folds.dropna(axis=1)

df_folds = pivot_df_folds[[col for col in pivot_df_folds.columns if 'Num+Str' in col]]

stability_scores_folds = []
unique_folds = df_folds.index.get_level_values('fold_index').unique()

for fold in unique_folds:
    df_fold = df_folds.xs(fold, level='fold_index')
    indices = df_fold.index.tolist()
    mid_point = len(df_fold) // 2
    
    # Run simulation for this specific fold
    for n in range(10, mid_point + 1, 1):
        for i in range(200): # Reduced iterations for fold-wise speed
            random.shuffle(indices)
            # Standard split-half reliability logic
            sub1 = df_fold.loc[indices[:mid_point]].sample(n=n)
            sub2 = df_fold.loc[indices[mid_point:mid_point*2]].sample(n=n)
            
            corr, _ = kendalltau(calculate_rankings(sub1), calculate_rankings(sub2))
            stability_scores_folds.append({'fold': fold, 'N_datasets': n, 'Kendalltau_Correlation': corr})

df_stability_folds = pd.DataFrame(stability_scores_folds)

# Fold-wise Bootstrapping and Extrapolation
fold_results = {}
colors = ['blue', 'orange', 'purple'] # Unique colors for fold 0, 1, 2

for fold in unique_folds:
    fold_data = df_stability_folds[df_stability_folds['fold'] == fold]
    fold_results[fold] = {'curves': [], 'popt': []}
    
    for k in range(500): # Bootstrap iterations per fold
        boot = fold_data.groupby('N_datasets').sample(frac=1.0, replace=True)
        agg = boot.groupby('N_datasets')['Kendalltau_Correlation'].mean()
        
        try:
            popt, _ = curve_fit(model_ref_1, agg.index, agg.values, p0=[0.5, 0.05], bounds=([0,0], [10, np.inf]))
            fold_results[fold]['curves'].append(model_ref_1(x_range_smooth, *popt))
            fold_results[fold]['popt'].append(popt)
        except: pass

#plot
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(6, 3))

for i, fold in enumerate(unique_folds):
    color = colors[i]
    # 1. Plot dots for each CV fold (observed data)
    fold_agg = df_stability_folds[df_stability_folds['fold'] == fold].groupby('N_datasets')['Kendalltau_Correlation'].median()
    ax.scatter(fold_agg.index, fold_agg.values, color=color, s=4, alpha=0.4, label=f'Fold {fold} Obs.')
    
    # 2. Plot extrapolation line (median bootstrap curve)
    if fold_results[fold]['curves']:
        curves = np.array(fold_results[fold]['curves'])
        ax.plot(x_range_smooth, np.median(curves, axis=0), color=color, linewidth=1, label=f'Fold {fold} Extrap.')

# Standard ICML formatting for STRABLE stability
ax.axvline(x=n_datasets, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Number of Datasets (N)', fontsize=9)
ax.set_ylabel('Kendall Tau ($\\tau$)', fontsize=9)
ax.set_title('Stability Extrapolation by CV Fold', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=6, ncol=2, frameon=True)
ax.set_xlim(-5, 200)
ax.set_ylim(0.6, 1.02)
ax.grid(True, alpha=0.15)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_greenfunction_byCVfold_{today_date}_2.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
the benchmark to converge to the same ranking?
BOOTSTRAPPED EXPONENTIAL SATURATION FITTING (EXTRAPOLATION) - 1 - (a / np.sqrt(x)) * np.exp(-b * x)

by number of models
'''

# --- 1. DATA PREPARATION & PIVOTING ---
# Create the pivot table; using 'mean' aggregates scores across CV folds
pivot_df = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# Remove datasets (rows) where any model failed to ensure balanced rankings
# pivot_df = pivot_df.dropna(axis=0)

pivot_df = pivot_df.dropna(axis=1)

# Focus strictly on Num+Str methods for the core benchmark contribution
df = pivot_df[[col for col in pivot_df.columns if 'Num+Str' in col]].copy()

# --- 2. DYNAMIC TIER & INDEX SYNCHRONIZATION ---
available_models = df.columns.tolist()
total_models = len(available_models)

# Define 1/3, 2/3, and Full tiers dynamically based on available models
tiers = {
    'Low (1/3)': max(2, total_models // 3),
    'Med (2/3)': max(3, (2 * total_models) // 3),
    'Full (All)': total_models
}

# Synchronize indices with the filtered dataframe to avoid KeyError
indices = df.index.tolist()
n_datasets = len(indices)
mid_point = n_datasets // 2  # Used for split-half reliability halves

# --- 3. STABILITY SIMULATION LOOP ---
stability_scores_multi_m = []

for tier_name, m_count in tiers.items():
    print(f"Running simulation for {tier_name} ({m_count} models)...")
    
    for n in sample_sizes:
        # Check to ensure n doesn't exceed the half-pool size
        if n > mid_point:
            continue
            
        for i in range(1000):
            # Shuffle synchronized indices for unique disjoint subsets
            random.shuffle(indices)
            
            # Randomly sample models for this iteration tier
            current_models = random.sample(available_models, m_count)

            # Split-half: sub1 and sub2 have zero overlapping datasets
            # Pool retrieval using .loc with synchronized labels
            sub1_pool = df.loc[indices[:mid_point], current_models]
            sub2_pool = df.loc[indices[mid_point:mid_point*2], current_models]
            
            # Subsample exactly n datasets from each independent half
            sub1 = sub1_pool.sample(n=n, replace=False)
            sub2 = sub2_pool.sample(n=n, replace=False)
            
            # Compute rankings and their Kendall Tau correlation
            rank1 = calculate_rankings(sub1)
            rank2 = calculate_rankings(sub2)
            corr, _ = kendalltau(rank1, rank2)
            
            stability_scores_multi_m.append({
                'Tier': tier_name,
                'M_models': m_count,
                'N_datasets': n,
                'Kendalltau_Correlation': corr
            })

# Final stability results for extrapolation
df_multi_m = pd.DataFrame(stability_scores_multi_m)

tier_results = {}

for tier_name in tiers.keys():
    tier_data = df_multi_m[df_multi_m['Tier'] == tier_name]
    tier_results[tier_name] = {'curves': [], 'preds': []}
    
    for k in range(1000): # Bootstrap iterations for the CI
        boot = tier_data.groupby('N_datasets').sample(frac=1.0, replace=True)
        agg = boot.groupby('N_datasets')['Kendalltau_Correlation'].mean()
        
        try:
            popt, _ = curve_fit(model_ref_1, agg.index, agg.values, p0=[0.5, 0.05], bounds=([0,0], [10, np.inf]))
            tier_results[tier_name]['curves'].append(model_ref_1(x_range_smooth, *popt))
        except: pass

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(6, 3))
colors = {'Low (1/3)': '#1f77b4', 'Med (2/3)': '#ff7f0e', 'Full (All)': '#2ca02c'}

for tier_name in tiers.keys():
    if tier_results[tier_name]['curves']:
        curves = np.array(tier_results[tier_name]['curves'])
        med_line = np.median(curves, axis=0)
        low_ci = np.percentile(curves, 2.5, axis=0)
        high_ci = np.percentile(curves, 97.5, axis=0)
        
        m_val = tiers[tier_name]
        # Thicker line for 'Full' to highlight the primary STRABLE result
        lw = 1.5 if tier_name == 'Full (All)' else 1.0
        
        ax.plot(x_range_smooth, med_line, color=colors[tier_name], 
                linewidth=lw, label=f'{tier_name} (M={m_val})')
        # Visible shaded regions confirm high precision of the STRABLE model
        ax.fill_between(x_range_smooth, low_ci, high_ci, 
                        color=colors[tier_name], alpha=0.15)

# --- ANNOTATION FOR ICML READABILITY ---
ax.axvline(x=100, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
ax.text(105, 0.55, 'STRABLE\n(N=100)', color='red', fontweight='bold', fontsize=7)

ax.set_xlabel('Number of Datasets (N)', fontsize=9)
ax.set_ylabel('Kendall Tau ($\\tau$)', fontsize=9)
ax.set_title('STRABLE Stability: Impact of Leaderboard Size', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=7, frameon=True, borderpad=0.5)
ax.set_xlim(0, 300)
ax.set_ylim(0.5, 1.02)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.show()

# Updated for a vertical stack of three (6, 3) plots
fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True, sharey=True)
plt.rcParams.update({'font.size': 8})

for i, (tier_name, ax) in enumerate(zip(tiers.keys(), axes)):
    if tier_results[tier_name]['curves']:
        curves = np.array(tier_results[tier_name]['curves'])
        med_line = np.median(curves, axis=0)
        low_ci = np.percentile(curves, 2.5, axis=0)
        high_ci = np.percentile(curves, 97.5, axis=0)
        
        m_val = tiers[tier_name]
        color = colors[tier_name]
        
        # Plot single tier per subplot
        ax.plot(x_range_smooth, med_line, color=color, linewidth=1.5, 
                label=f'{tier_name} (M={m_val})')
        ax.fill_between(x_range_smooth, low_ci, high_ci, color=color, alpha=0.15)

        # Contextual markers for STRABLE
        ax.axvline(x=100, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.text(105, 0.55, f'{tier_name}\nN=100', color=color, fontweight='bold', fontsize=7)
        
        # Subplot formatting
        ax.set_ylabel('Kendall Tau ($\\tau$)', fontsize=9)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(0.5, 1.02)

axes[-1].set_xlabel('Number of Datasets (N)', fontsize=9)
fig.suptitle('STRABLE Stability: Leaderboard Size Comparison', fontsize=12, fontweight='bold', y=0.99)
fig.text(0.5, 0.93, r'Kendall $\tau$ is independent of Number of Models', ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_greenfunction_byNumModels_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
the benchmark to converge to the same ranking?
BOOTSTRAPPED EXPONENTIAL SATURATION FITTING (EXTRAPOLATION) - 1 - (a / np.sqrt(x)) * np.exp(-b * x)

only the pareto optimal
'''

# 1. Update Data Aggregation for the current metric
agg_cols = ['score', 'run_time_per_10k']
group_cols = ['method_polished', 'encoder', 'learner']
df_agg = results[results['method'].str.contains('num-str_')].groupby(group_cols)[agg_cols].median().reset_index()

pareto_df = get_pareto_front(df_agg, 'run_time_per_10k', 'score', True)
top_methods = pareto_df['method_polished'].unique()
df_plot = df_agg[df_agg['method_polished'].isin(top_methods)].copy()

pivot_df = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# PAY ATTENTION TO DROP NANS IN THE CORRECT DIMENSION
pivot_df = pivot_df.dropna(axis=1)  

# pivot_df = pivot_df.dropna(axis=0)  

df = pivot_df.copy()

# take only those index that are in df_plot['method_polished']
df_pareto = df[[col for col in df.columns if col in df_plot['method_polished'].values]]

indices = df.index.tolist()
mid_point = len(indices) // 2


stability_scores_pareto = []

for n in range(10, mid_point + 1, 1):
    for i in range(1000): # 1000 iterations for statistical precision
        random.shuffle(indices)
        
        # Split-half: sub1 and sub2 have zero overlapping datasets
        sub1 = df_pareto.loc[indices[:mid_point]].sample(n=n, replace=False)
        sub2 = df_pareto.loc[indices[mid_point:mid_point*2]].sample(n=n, replace=False)
        
        # Compute Kendall Tau correlation between the two independent rankings
        corr, _ = kendalltau(calculate_rankings(sub1), calculate_rankings(sub2))
        stability_scores_pareto.append({'N_datasets': n, 'Kendalltau_Correlation': corr})

df_stability_pareto = pd.DataFrame(stability_scores_pareto)

# Setup
n_bootstraps = 2000
target_y = 0.95
# Calculate the disagreement percentage based on the Kendall Tau formula: (1 - tau) / 2
disagreement_pct = ((1 - target_y) / 2) * 100
max_plot_x = 3000
x_range_smooth = np.linspace(1, max_plot_x, 300)

# Storage for results
results_kendalltau_extrap_pareto = {
    'ref1': {'popt': [], 'curves': [], 'preds': []}
}
print(f"Starting {n_bootstraps} bootstrap iterations...")

for k in range(n_bootstraps):
    boot_sample = df_stability_pareto.groupby('N_datasets').sample(frac=1.0, replace=True)
    df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
    X, Y = df_agg['N_datasets'], df_agg['Kendalltau_Correlation']
    
    # --- FIT MODEL REF 1 ---
    try:
        # Fixed asymptote at 1.0
        p_r1, _ = curve_fit(model_ref_1, X, Y, p0=[0.5, 0.05], bounds=([0, 0], [10, np.inf]))
        
        # 1. STORE THE PARAMETERS HERE
        results_kendalltau_extrap_pareto['ref1']['popt'].append(p_r1)
        
        # Finding N requires numerical solver since N is in sqrt and exp
        func = lambda n: model_ref_1(n, *p_r1) - target_y
        req_N = fsolve(func, x0=50)[0]
        if 0 < req_N < 10000:
            results_kendalltau_extrap_pareto['ref1']['preds'].append(req_N)
            results_kendalltau_extrap_pareto['ref1']['curves'].append(model_ref_1(x_range_smooth, *p_r1))
    except: pass

# Convert the list of arrays into a 2D NumPy array (rows=bootstraps, cols=[a, b])
all_popt = np.array(results_kendalltau_extrap_pareto['ref1']['popt'])
# Calculate the optimal (mean) parameters
mean_params = np.mean(all_popt, axis=0)
print(f"Optimal parameter a: {mean_params[0]:.4f}")
print(f"Optimal parameter b: {mean_params[1]:.4f}")

# Calculate 95% Confidence Intervals
ci_lower = np.percentile(all_popt, 2.5, axis=0)
ci_upper = np.percentile(all_popt, 97.5, axis=0)

print(f"95% CI for a: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
print(f"95% CI for b: [{ci_lower[1]:.4f}, {ci_upper[1]:.4f}]")

tau_for_strable_benchmark = round(1 - (mean_params[0]/np.sqrt(n_datasets)) * np.exp(-mean_params[1]*n_datasets),2)
disagreement_pct_strable = ((1 - tau_for_strable_benchmark) / 2) * 100

# --- VISUALIZATION ---
plt.rcParams.update({'font.size': 8}) # Standard academic base size
fig, ax = plt.subplots(figsize=(6, 3))

# 1. Plot Observed Data
df_real_agg = df_stability_pareto.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['median', 'sem'])
ax.errorbar(df_real_agg['N_datasets'], df_real_agg['median'], yerr=df_real_agg['sem'], 
             fmt='o', color='blue', markersize=3, elinewidth=0.8, 
             label='Observed (Median ± SE)', zorder=10)

# 2. Extract and Plot Bootstrap CI Band
if len(results_kendalltau_extrap_pareto['ref1']['curves']) > 0:
    curves = np.array(results_kendalltau_extrap_pareto['ref1']['curves'])
    med_line = np.median(curves, axis=0)
    # Calculate 95% Confidence Interval
    low_ci = np.percentile(curves, 2.5, axis=0)
    high_ci = np.percentile(curves, 97.5, axis=0)
    
    # Render the Model and the Confidence Interval
    # ax.fill_between(x_range_smooth, low_ci, high_ci, color='green', alpha=0.4, 
    #             edgecolor='green', linewidth=0.5, label='95% Bootstrap CI')
    ax.plot(x_range_smooth, med_line, color='green', linewidth=1.5, 
            label=r'$1 - \frac{a}{\sqrt{x}} e^{-bx}$')

# 3. Reference Lines & STRABLE Metrics
ax.axvline(x=n_datasets, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=tau_for_strable_benchmark, color='black', linestyle=':', linewidth=1)

# Shortened annotations for 6x3 readability
ax.text(n_datasets + 5, 0.4, f"STRABLE size: {n_datasets}", 
        color='red', fontweight='bold', fontsize=8)

# Calculate and display disagreement succinctly
annot_text = (f"$\\tau={tau_for_strable_benchmark}$\n"
              f"Disagreement - Pareto Optimal models: {disagreement_pct_strable:.1f}%")
ax.annotate(annot_text, xy=(n_datasets, tau_for_strable_benchmark), 
            xytext=(n_datasets + 25, 0.82),
            # arrowprops=dict(arrowstyle="->", color='gray'),
            fontsize=8, 
            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )


# --- FINAL POLISH ---
ax.set_xlabel('Number of Datasets (N)', fontsize=9)
ax.set_ylabel('Kendall Tau Correlation', fontsize=9)
ax.set_title('Benchmark Stability Analysis - Pareto Optimal Models', fontsize=10)
ax.legend(loc='lower right', fontsize=7, frameon=True)
ax.grid(True, alpha=0.2)
ax.set_ylim(0.0, 1.05)
ax.set_xlim(-5, 300)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_greenfunction_pareto_optimal_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()


'''
LEAVE-ONE-SOURCE-OUT PLOT VERSION 2:
kendalltau correlation between (rank of source_i for all models, the rank of the models for the complementary part of the benchmark)
it should be 2 vectors for each source_i:
- vector 1: rank of each model on source_i
- vector 2: rank of each model on all other sources except source_i (averaged over all other sources)
'''


# add a column that is called source_with_ds_count
source_counts = results.groupby('source')['data_name'].nunique()
source_label_map = {src: f"{src} ({count})" for src, count in source_counts.items()}
results['source_with_ds_count'] = results['source'].map(source_label_map)

# 1. Prepare the Pivot Table
# Ensure we have one score per model per dataset
pivot_source = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)
# Drop any methods/datasets that have NaN values
pivot_source = pivot_source.dropna(axis=1) #drop columns with NaN
# pivot_source = pivot_source.dropna(axis=0) #drop columns with NaN

# select only algos that are Num+Str
pivot_source = pivot_source[[col for col in pivot_source.columns if 'Num+Str' in col]].copy()

valid_datasets = pivot_source.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['source_with_ds_count'].first()

# 2. LOSO Calculation Loop
sources = results['source_with_ds_count'].unique()
loso_correlations = []

for src in sources:
    # Split datasets into current source and all other sources
    datasets_in_source = dataset_to_source[dataset_to_source == src].index
    datasets_other = dataset_to_source[dataset_to_source != src].index
    
    # Filter pivot table
    df_src = pivot_source.loc[datasets_in_source]
    df_others = pivot_source.loc[datasets_other]
    
    # Vector 1: Rank of each model on Source_i (averaged across datasets in that source)
    # Higher score is better -> ascending=False
    ranks_src = df_src.mean().rank(ascending=False)

    print(f"{len(ranks_src)} models ranked on source {src}.")
    
    # Vector 2: Average rank of each model on all other sources
    # We rank per dataset first, then average those ranks
    ranks_others_per_ds = df_others.rank(axis=1, ascending=False)
    avg_ranks_others = ranks_others_per_ds.mean()

    print(f"{len(avg_ranks_others)} models ranked on other sources excluding {src}.")

    # check that both vectors have the same models
    print(f"Source: {src}")
    assert set(ranks_src.index) == set(avg_ranks_others.index), "Model mismatch between source and others"

    # Calculate Kendall Tau between the two ranking vectors
    corr, _ = kendalltau(ranks_src, avg_ranks_others)
    
    loso_correlations.append({
        'Source': src,
        'Correlation': corr,
        'N_datasets': len(datasets_in_source)
    })

df_loso = pd.DataFrame(loso_correlations).sort_values('Correlation', ascending=False)
#drop rows with NaN correlation
df_loso = df_loso.dropna(subset=['Correlation'])

# Visualization - Transposed
# plt.figure(figsize=(6, 5)) # Increased height to accommodate Y-axis labels
# colors = plt.cm.viridis(np.linspace(0, 1, len(df_loso)))

# # Use barh for horizontal bars
# bars = plt.barh(df_loso['Source'], df_loso['Correlation'], color=colors, alpha=0.8)

# # Add value labels for each bar
# for bar in bars:
#     width = bar.get_width()
#     plt.text(
#         width + 0.01,                # Position slightly to the right of the bar
#         bar.get_y() + bar.get_height() / 2, # Center vertically in the bar
#         f'{width:.3f}',             # The Kendall Tau value
#         va='center', 
#         fontsize=10, 
#         fontweight='bold'
#     )

# # Formatting
# plt.title('Leave-One-Source-Out', fontsize=16, pad=20)
# plt.xlabel('Kendall Tau Correlation', fontsize=14)
# plt.ylabel('Excluded Source ($Source_i$)', fontsize=14)
# plt.xlim(0, 1.15) # Extended limit to make room for text labels
# plt.grid(axis='x', alpha=0.3)
# plt.legend(loc='lower right')

# plt.tight_layout()

plt.figure(figsize=(6.75, 5)) 
colors = [plt.cm.RdYlGn(0.15) if x < 0 else plt.cm.RdYlGn(0.85) for x in df_loso['Correlation']]
bars = plt.barh(df_loso['Source'], df_loso['Correlation'], color=colors, 
                alpha=0.8, height=0.7)
for bar in bars:
    width = bar.get_width()
    x_pos = width - 0.01 if width < 0 else width + 0.01
    ha_pos = 'right' if width < 0 else 'left'
    
    plt.text(
        x_pos, 
        bar.get_y() + bar.get_height() / 2, 
        f'{width:.2f}', # Two decimals for space saving
        va='center', 
        ha=ha_pos,
        fontsize=7, 
        fontweight='bold'
    )
plt.title('Leave-One-Source-Out: $\\tau$(Source_i, Benchmark - Source_i)', fontsize=14, pad=15)
plt.xlabel('Kendall Tau Correlation', fontsize=9)
plt.ylabel('Excluded Source ($S_i$)', fontsize=9)
plt.yticks(fontsize=7)
plt.xticks(fontsize=8)
limit = max(abs(df_loso['Correlation'].min()), abs(df_loso['Correlation'].max())) + 0.1
plt.xlim(-0.3, limit)
sns.despine(left=False, bottom=False, top=True, right=True)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
plt.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_source_out_v2_check_transposed_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
VIOLIN PLOT OF THE KENDALLTAU DISTRIBUTION ACROSS SOURCES
'''

# 1. Setup the figure
plt.figure(figsize=(6, 3))
sns.set_style("white")

# 2. Create the Violin Plot
# We set inner=None and density_norm='width' to ensure a clear scale
ax = sns.violinplot(
    data=df_loso, x='Correlation', 
    color='skyblue', alpha=0.3, inner=None,
    density_norm='width' 
)

# 3. Add the points proportional to the source size
sns.scatterplot(
    data=df_loso, x='Correlation', y=[0]*len(df_loso), 
    size='N_datasets', sizes=(50, 500),
    color='navy', alpha=0.6, legend='brief', ax=ax
)

# 5. Formatting Y-Axis Scale
plt.title('Distribution of Kendall Tau Correlations\n(Proportional to Source Size)', fontsize=16, pad=15)
plt.xlabel('Kendall Tau Correlation', fontsize=18)
plt.ylabel('Density (KDE)', fontsize=18)

# Force the Y-axis to show the density scale
ax.set_yticks(np.linspace(-0.5, 0.5, 5)) # Centered at 0 because the violin is symmetric
ax.set_yticklabels([f"{abs(y):.1f}" for y in np.linspace(-0.5, 0.5, 5)]) 
ax.tick_params(axis='y', left=True, labelleft=True)

# 6. Improving the Legend (Larger spacing and no overlap)
plt.legend(
    title='Number of Datasets', 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left',
    labelspacing=1.0,   # Significant spacing to prevent overlap
    borderpad=1.5,
    handletextpad=1.5,
    frameon=True,
    fontsize=12,
    title_fontsize=14
)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_source_out_kendalltau_distribution_violinplot_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Leave one source out V3
ranking of entire benchmark vs ranking of the benchmark without source_i
'''

# --- 1. SETUP & SYNCHRONIZATION ---
# Ensure we have the source label column
source_counts = results.groupby('source')['data_name'].nunique()
source_label_map = {src: f"{src} ({count})" for src, count in source_counts.items()}
results['source_with_ds_count'] = results['source'].map(source_label_map)

# Pivot table for rankings
pivot_source = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# pivot_source=pivot_source.dropna(axis=0)
pivot_source=pivot_source.dropna(axis=1)

# select only algos that are Num+Str
pivot_source = pivot_source[[col for col in pivot_source.columns if 'Num+Str' in col]].copy()

# Synchronize dataset-to-source mapping with the valid pivot index
valid_datasets = pivot_source.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['source_with_ds_count'].first()

# --- 2. CALCULATION OF FULL BENCHMARK RANKING ---
# Compute the consensus ranking (mean of ranks) across ALL datasets
ranks_full_benchmark = pivot_source.rank(axis=1, ascending=False).mean()

# --- 3. LOSO INFLUENCE LOOP ---
sources = dataset_to_source.unique()
loso_influence_results = []

for src in sources:
    # Identify datasets belonging to all sources EXCEPT the current one
    datasets_other = dataset_to_source[dataset_to_source != src].index
    
    # Filter the pivot table to exclude the current source
    df_others = pivot_source.loc[datasets_other]
    
    # Compute the consensus ranking for this restricted benchmark
    ranks_without_source = df_others.rank(axis=1, ascending=False).mean()

    # Calculate Kendall Tau between the Full Ranking and the Restricted Ranking
    # This measures how much the leaderboard changes when source 'src' is gone
    corr, _ = kendalltau(ranks_full_benchmark, ranks_without_source)
    
    loso_influence_results.append({
        'Source': src,
        'Correlation': corr,
        'N_datasets': len(dataset_to_source[dataset_to_source == src])
    })

df_loso_influence = pd.DataFrame(loso_influence_results).sort_values('Correlation', ascending=False)

# --- 4. VISUALIZATION 1: HORIZONTAL BAR CHART ---
sns.set_style("white")
plt.figure(figsize=(5, 6))

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_loso_influence)))
bars = plt.barh(df_loso_influence['Source'], df_loso_influence['Correlation'], 
                color=colors, alpha=0.9, height=0.7)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.001, 
        bar.get_y() + bar.get_height()/2, 
        f'{width:.4f}', 
        va='center', 
        fontsize=9, 
        fontweight='bold'
    )

sns.despine(left=True, bottom=True)

plt.xlabel(r'Kendall Tau Correlation ($\tau$)', fontsize=12, ha='right', x=1.0)
plt.ylabel('Excluded Source', fontsize=12)
plt.xticks(fontsize=8)
min_val = df_loso_influence['Correlation'].min()
plt.xlim(min_val - 0.005, 1.01)
plt.grid(axis='x', color='gray', linestyle='--', alpha=0.3)
plt.title('Leave-One-Source-Out: $\\tau$(Benchmark, Benchmark - Source_i)', fontsize=14, pad=15)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_source_out_v3_kendalltau_distribution_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
VIOLIN PLOT
'''

plt.figure(figsize=(6, 3))
sns.set_style("white")
ax = sns.violinplot(data=df_loso_influence, x='Correlation', color='skyblue', alpha=0.3, inner=None, density_norm='width')

sns.scatterplot(
    data=df_loso_influence, x='Correlation', y=[0]*len(df_loso_influence), 
    size='N_datasets', sizes=(50, 500), color='navy', alpha=0.6, legend='brief', ax=ax
)

plt.xlabel('Kendall Tau Correlation ($\\tau$)', fontsize=18)
plt.ylabel('Density', fontsize=18)
ax.set_yticks([]) # Hide Y-ticks for a cleaner look at influence distribution
plt.legend(title='Datasets in Source', bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing=1.0, borderpad=1.5,
    handletextpad=1.5,
    frameon=True,
    fontsize=12,
    title_fontsize=14)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_source_out_v3_kendalltau_distribution_violinplot_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()





'''
LEAVE-ONE-SOURCE-OUT PLOT VERSION 2:
kendalltau correlation between (rank of source_i for all models, the rank of the models for the complementary part of the benchmark)
it should be 2 vectors for each source_i:
- vector 1: rank of each model on source_i
- vector 2: rank of each model on all other sources except source_i (averaged over all other sources)
'''

# 1. Prepare the Pivot Table
# Ensure we have one score per model per dataset
pivot_source = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# select only algos that are Num+Str
pivot_source = pivot_source[[col for col in pivot_source.columns if 'Num+Str' in col]].copy()

# Drop any methods/datasets that have NaN values
pivot_source = pivot_source.dropna(axis=1) #drop columns with NaN

# pivot_source = pivot_source.dropna(axis=0) #drop columns with NaN

valid_datasets = pivot_source.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['category_with_ds_count'].first()

# 2. LOSO Calculation Loop
sources = results['category_with_ds_count'].unique()
loso_correlations = []

for src in sources:
    # Split datasets into current source and all other sources
    datasets_in_source = dataset_to_source[dataset_to_source == src].index
    datasets_other = dataset_to_source[dataset_to_source != src].index
    
    # Filter pivot table
    df_src = pivot_source.loc[datasets_in_source]
    df_others = pivot_source.loc[datasets_other]
    
    # Vector 1: Rank of each model on Source_i (averaged across datasets in that source)
    # Higher score is better -> ascending=False
    ranks_src = df_src.mean().rank(ascending=False)

    print(f"{len(ranks_src)} models ranked on category {src}.")
    
    # Vector 2: Average rank of each model on all other categories
    # We rank per dataset first, then average those ranks
    ranks_others_per_ds = df_others.rank(axis=1, ascending=False)
    avg_ranks_others = ranks_others_per_ds.mean()

    print(f"{len(avg_ranks_others)} models ranked on other categories excluding {src}.")

    # check that both vectors have the same models
    print(f"Category: {src}")
    assert set(ranks_src.index) == set(avg_ranks_others.index), "Model mismatch between category and others"
    # Calculate Kendall Tau between the two ranking vectors
    corr, _ = kendalltau(ranks_src, avg_ranks_others)
    
    loso_correlations.append({
        'Category': src,
        'Correlation': corr,
        'N_datasets': len(datasets_in_source)
    })

df_loso = pd.DataFrame(loso_correlations).sort_values('Correlation', ascending=False)
#drop rows with NaN correlation
df_loso = df_loso.dropna(subset=['Correlation'])

# Visualization - Transposed
plt.figure(figsize=(6, 5)) # Increased height to accommodate Y-axis labels
colors = plt.cm.viridis(np.linspace(0, 1, len(df_loso)))

# Use barh for horizontal bars
bars = plt.barh(df_loso['Category'], df_loso['Correlation'], color=colors, alpha=0.8)

# Add value labels for each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.01,                # Position slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2, # Center vertically in the bar
        f'{width:.3f}',             # The Kendall Tau value
        va='center', 
        fontsize=10, 
        fontweight='bold'
    )

# Formatting
plt.title('Leave-One-Category-Out: $\\tau$(Category_i, Benchmark - Category_i)', fontsize=16, pad=20)
plt.xlabel('Kendall Tau Correlation', fontsize=14)
plt.ylabel('Excluded Category ($Category_i$)', fontsize=14)
plt.yticks(fontsize=10)
plt.xlim(0, 1.0) # Extended limit to make room for text labels
plt.grid(axis='x', alpha=0.3)
plt.legend(loc='lower right')

plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_v2_check_transposed_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
VIOLIN PLOT OF THE KENDALLTAU DISTRIBUTION ACROSS CATEGORIES
'''

# 1. Setup the figure
plt.figure(figsize=(6, 3))
sns.set_style("white")

# 2. Create the Violin Plot
# We set inner=None and density_norm='width' to ensure a clear scale
ax = sns.violinplot(
    data=df_loso, x='Correlation', 
    color='skyblue', alpha=0.3, inner=None,
    density_norm='width' 
)

# 3. Add the points proportional to the category size
sns.scatterplot(
    data=df_loso, x='Correlation', y=[0]*len(df_loso), 
    size='N_datasets', sizes=(50, 500),
    color='navy', alpha=0.6, legend='brief', ax=ax
)

# 5. Formatting Y-Axis Scale
plt.title('Distribution of Kendall Tau Correlations\n(Proportional to Category Size)', fontsize=16, pad=15)
plt.xlabel('Kendall Tau Correlation', fontsize=18)
plt.ylabel('Density (KDE)', fontsize=18)

# Force the Y-axis to show the density scale
ax.set_yticks(np.linspace(-0.5, 0.5, 5)) # Centered at 0 because the violin is symmetric
ax.set_yticklabels([f"{abs(y):.1f}" for y in np.linspace(-0.5, 0.5, 5)]) 
ax.tick_params(axis='y', left=True, labelleft=True)

# 6. Improving the Legend (Larger spacing and no overlap)
plt.legend(
    title='Number of Datasets', 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left',
    labelspacing=1.0,   # Significant spacing to prevent overlap
    borderpad=1.5,
    handletextpad=1.5,
    frameon=True,
    fontsize=12,
    title_fontsize=14
)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_kendalltau_distribution_violinplot_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Leave one source out V3
ranking of entire benchmark vs ranking of the benchmark without category_i
'''

# Pivot table for rankings
pivot_category = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# pivot_category = pivot_category.dropna(axis=0)

pivot_category = pivot_category.dropna(axis=1)

# select only algos that are Num+Str
pivot_category = pivot_category[[col for col in pivot_category.columns if 'Num+Str' in col]].copy()

# Synchronize dataset-to-source mapping with the valid pivot index
valid_datasets = pivot_category.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['category_with_ds_count'].first()

# --- 2. CALCULATION OF FULL BENCHMARK RANKING ---
# Compute the consensus ranking (mean of ranks) across ALL datasets
ranks_full_benchmark = pivot_category.rank(axis=1, ascending=False).mean()

# --- 3. LOSO INFLUENCE LOOP ---
categories = dataset_to_source.unique()
loso_influence_results = []

for category in categories:
    # Identify datasets belonging to all categories EXCEPT the current one
    datasets_other = dataset_to_source[dataset_to_source != category].index
    
    # Filter the pivot table to exclude the current category
    df_others = pivot_category.loc[datasets_other]
    
    # Compute the consensus ranking for this restricted benchmark
    ranks_without_category = df_others.rank(axis=1, ascending=False).mean()

    # Calculate Kendall Tau between the Full Ranking and the Restricted Ranking
    # This measures how much the leaderboard changes when source 'src' is gone
    corr, _ = kendalltau(ranks_full_benchmark, ranks_without_category)
    
    loso_influence_results.append({
        'Category': category,
        'Correlation': corr,
        'N_datasets': len(dataset_to_source[dataset_to_source == category])
    })

df_loso_influence = pd.DataFrame(loso_influence_results).sort_values('Correlation', ascending=False)

# --- 4. VISUALIZATION 1: HORIZONTAL BAR CHART ---
sns.set_style("white")
plt.figure(figsize=(5, 6))

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_loso_influence)))
bars = plt.barh(df_loso_influence['Category'], df_loso_influence['Correlation'], 
                color=colors, alpha=0.9, height=0.7)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.001, 
        bar.get_y() + bar.get_height()/2, 
        f'{width:.4f}', 
        va='center', 
        fontsize=9, 
        fontweight='bold'
    )

sns.despine(left=True, bottom=True)

plt.xlabel(r'Kendall Tau Correlation ($\tau$)', fontsize=12, ha='right', x=1.0)
plt.ylabel('Excluded Category', fontsize=12)
min_val = df_loso_influence['Correlation'].min()
plt.xlim(0, 1.01)
plt.grid(axis='x', color='gray', linestyle='--', alpha=0.3)
plt.title('Leave-One-Category-Out: $\\tau$(Benchmark, Benchmark - Category_i)', fontsize=14, pad=15)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_v3_kendalltau_distribution_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
VIOLIN PLOT
'''

plt.figure(figsize=(6, 3))
sns.set_style("white")
ax = sns.violinplot(data=df_loso_influence, x='Correlation', color='skyblue', alpha=0.3, inner=None, density_norm='width')

sns.scatterplot(
    data=df_loso_influence, x='Correlation', y=[0]*len(df_loso_influence), 
    size='N_datasets', sizes=(50, 500), color='navy', alpha=0.6, legend='brief', ax=ax
)

plt.xlabel('Kendall Tau Correlation ($\\tau$)', fontsize=18)
plt.ylabel('Density', fontsize=18)
ax.set_yticks([]) # Hide Y-ticks for a cleaner look at influence distribution
plt.legend(title='Datasets in Category', bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing=1.0, borderpad=1.5,
    handletextpad=1.5,
    frameon=True,
    fontsize=12,
    title_fontsize=14)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_v3_kendalltau_distribution_violinplot_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()

'''
Leave one source out V3
ranking of entire benchmark vs ranking of the benchmark without category_i
weighted by category size
'''

# Pivot table for rankings
pivot_category = results.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# pivot_category = pivot_category.dropna(axis=0)

pivot_category = pivot_category.dropna(axis=1)

# select only algos that are Num+Str
pivot_category = pivot_category[[col for col in pivot_category.columns if 'Num+Str' in col]].copy()

# Synchronize dataset-to-source mapping with the valid pivot index
valid_datasets = pivot_category.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['category_with_ds_count'].first()

# --- 2. CALCULATION OF FULL BENCHMARK RANKING ---
# Compute the consensus ranking (mean of ranks) across ALL datasets
ranks_full_benchmark = pivot_category.rank(axis=1, ascending=False).mean()

# --- 3. LOSO INFLUENCE LOOP ---
categories = dataset_to_source.unique()
loso_influence_results = []

for category in categories:
    # Identify datasets belonging to all categories EXCEPT the current one
    datasets_other = dataset_to_source[dataset_to_source != category].index
    
    # Filter the pivot table to exclude the current category
    df_others = pivot_category.loc[datasets_other]
    
    # Compute the consensus ranking for this restricted benchmark
    ranks_without_category = df_others.rank(axis=1, ascending=False).mean()

    # Calculate Kendall Tau between the Full Ranking and the Restricted Ranking
    # This measures how much the leaderboard changes when source 'src' is gone
    corr, _ = kendalltau(ranks_full_benchmark, ranks_without_category)
    
    loso_influence_results.append({
        'Category': category,
        'Correlation': corr,
        'N_datasets': len(dataset_to_source[dataset_to_source == category])
    })

df_loso_influence = pd.DataFrame(loso_influence_results).sort_values('Correlation', ascending=False)

df_loso_influence['Weighted_Sensitivity'] = (1 - df_loso_influence['Correlation']) / df_loso_influence['N_datasets']

# Sort by the new weighted metric
df_loso_influence = df_loso_influence.sort_values('Weighted_Sensitivity', ascending=True)

# 2. VISUALIZATION: Weighted Sensitivity
plt.figure(figsize=(5, 6))
colors = plt.cm.magma(np.linspace(0.3, 0.8, len(df_loso_influence)))

bars = plt.barh(df_loso_influence['Category'], df_loso_influence['Weighted_Sensitivity'], 
                color=colors, alpha=0.9, height=0.7)

for bar in bars:
    width = bar.get_width()
    plt.text(width + (width*0.02), bar.get_y() + bar.get_height()/2, 
             f'{width:.6f}', va='center', fontsize=9, fontweight='bold')

plt.xlabel('Disruption per Dataset ($1-\\tau$)/$N$', fontsize=11)
plt.ylabel('Category', fontsize=11)
plt.title('Weighted Influence: Leading Dataset Categories', fontsize=13, pad=15)
sns.despine()
plt.tight_layout()
plt.show()

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_v3_kendalltau_distribution_weighted_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()





'''
HETEROGENEITY ANALYSIS
Proof heterogeneity of sources. Do columns follow a log-gaussian distribution?
'''

# ==========================================
# TABLE 1: Task Distribution across macro-sources
# ==========================================

# CARTE = [
#     'clear-corpus',
# 'chocolate-bar-ratings',
# 'aijob_ai-ml-ds-salaries',
# 'meta-critic_whisky',
# 'yelp_business',
# 'ramen-ratings',
# 'museums',
# 'journal-ranking_wide'
# ]
# TTB = [
#     'covid-clinical-trials',
# 'it-salary-survey',
# 'insurance-company-complaints',
# 'kickstarter-projects',
# 'lending-club-loan',
# 'osha-accidents',
# 'wine-dataset',
# 'beer-ratings',
# 'california-houses', 
# 'listings-airbnb',
# 'mercari',
# 'sf-building-permits'
# ]

# results['macro_source'] = results['data_name'].apply(lambda x: 'CARTE' if x in CARTE else ('TTB' if x in TTB else 'New'))


# df_datasets = results.drop_duplicates(subset=['data_name']).copy()

# print(f"Analysis performed on {len(df_datasets)} unique datasets across {df_datasets['source'].nunique()} sources and {df_datasets['macro_source'].nunique()} macro-sources.")

# table1 = pd.crosstab(df_datasets['macro_source'], df_datasets['task'])

# # Add Total column
# table1['Total'] = table1.sum(axis=1)

# # save as latex
# table1_final = table1.copy()

# # Add a 'Total' row at the bottom
# table1_final.loc['Total'] = table1_final.sum()

# # Reset index so 'macro_source' becomes a column for the LaTeX table
# table1_latex = table1_final.reset_index()
# table1_latex = table1_latex.rename(columns={'macro_source': 'Macro Source'})

# # 2. Setup file paths
# today_date = time.strftime("%Y-%m-%d")
# filename = f"macro_source_task_contingency_table_{today_date}.tex"
# save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{filename}'

# # 3. Export to LaTeX using the Styler for professional formatting

# table1_latex.style.hide(axis="index").map(lambda x: "font-weight: bold", subset=pd.IndexSlice[table1_latex.index[-1], :]).map(lambda x: "font-weight: bold", subset=pd.IndexSlice[:, 'Total']).to_latex(
#     buf=save_path,
#     caption="Distribution of curated datasets across macro-sources and task types.",
#     label="tab:dataset_distribution_table1",
#     hrules=True,              # Requires \usepackage{booktabs}
#     column_format="l" + "r" * (len(table1_latex.columns) - 1), # Left for name, Right for counts
#     position="ht",
#     position_float="centering"
#     )


# methods = results['method_polished'].unique()
# sources = results['source'].unique()




'''
CARDINALITY AND STRING LENGTH vs performance: GLM and plot
'''

# results[['cardinality', 'string_length']].drop_duplicates().describe()

# 1. Define Binning Function

# 2. Prepare the Data
# df_plot = results[results['dtype'] == 'Num+Str'].copy()
# df_plot['Card_Bin'] = bin_feature(df_plot, 'avg_cardinality')
# df_plot['Str_Bin'] = bin_feature(df_plot, 'avg_string_length_per_cell')

# # 3. Aggregate Performance per Configuration and Bin
# # We use 'score_norm' to ensure fair comparison across different datasets
# pivot_card = df_plot.pivot_table(index='encoder_learner', columns='Card_Bin', 
#                                 values='score_norm', aggfunc='mean')
# pivot_str = df_plot.pivot_table(index='encoder_learner', columns='Str_Bin', 
#                                values='score_norm', aggfunc='mean')

# # 1. Setup the figure with shared Y axis
# # Adjust figsize to accommodate the labels and the single colorbar
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10), sharey=True)

# # 2. Define common performance scale (0.5 to 1.0)
# vmin, vmax = 0.5, 1
# cmap = 'RdYlGn'

# # 3. Plot Heatmap 1: Cardinality
# sns.heatmap(pivot_card, annot=False, cmap=cmap, ax=ax1, 
#             vmin=vmin, vmax=vmax, cbar=False)
# ax1.set_xlabel('Cardinality', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Model', fontsize=12, fontweight='bold')

# # 4. Plot Heatmap 2: String Length
# sns.heatmap(pivot_str, annot=False, cmap=cmap, ax=ax2, 
#             vmin=vmin, vmax=vmax, cbar=False)
# ax2.set_xlabel('String Length', fontsize=12, fontweight='bold')
# ax2.set_ylabel('') # Hidden by sharey, but good practice to clear

# # 5. Create one common performance bar on the right
# # Positioning: [left, bottom, width, height]
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# fig.colorbar(sm, cax=cbar_ax, label='Mean Norm Score')

# # 6. Set the single common title
# fig.suptitle('Performance of the model by Cardinality and String Length', 
#              fontsize=14, fontweight='bold', y=0.98)

# # Adjust subplots to make room for the colorbar and title
# plt.subplots_adjust(wspace=0.05, right=0.9, top=0.92)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'performance_per_cardinality_stringlength_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


'''
CARDINALITY
'''

# Filtering and sorting specifically for Cardinality
# pivot_card_elite = pivot_card[pivot_card.max(axis=1) >= 0.7].copy()
# pivot_card_sorted = pivot_card_elite.sort_values(by='High', ascending=False)

# # 1. Setup Cardinality Figure
# fig1, ax1 = plt.subplots(figsize=(3, 5))
# cmap = sns.diverging_palette(15, 135, s=90, l=50, sep=40, as_cmap=True)
# vmin, vmax, center = 0.7, 1.0, 0.85 

# # 2. Plot Heatmap
# sns.heatmap(pivot_card_sorted, annot=False, cmap=cmap, ax=ax1, 
#             vmin=vmin, vmax=vmax, center=center, 
#             cbar_kws={'label': 'Mean Norm Score (0.7+)'})

# # 3. Formatting
# ax1.set_title('Top Performers by Cardinality Regimes\n(Elite Subset: max score >= 0.7)', 
#               fontsize=12, fontweight='bold', pad=15)
# ax1.set_xlabel('Cardinality Bin', fontsize=10, fontweight='bold')
# ax1.set_ylabel('Model Configuration', fontsize=10, fontweight='bold')

# plt.tight_layout()
# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'performance_per_cardinality_0.7_threshold_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


'''
STRING LENGTH
'''

# Filtering and sorting specifically for String Length
# pivot_str_elite = pivot_str[pivot_str.max(axis=1) >= 0.7].copy()
# pivot_str_sorted = pivot_str_elite.sort_values(by='High', ascending=False)

# # 1. Setup String Length Figure
# fig2, ax2 = plt.subplots(figsize=(3, 5))

# # 2. Plot Heatmap
# sns.heatmap(pivot_str_sorted, annot=False, cmap=cmap, ax=ax2, 
#             vmin=vmin, vmax=vmax, center=center, 
#             cbar_kws={'label': 'Mean Norm Score (0.7+)'})

# # 3. Formatting
# ax2.set_title('Top Performers by String Length Regimes\n(Elite Subset: max score >= 0.7)', 
#               fontsize=12, fontweight='bold', pad=15)
# ax2.set_xlabel('String Length Bin', fontsize=10, fontweight='bold')
# ax2.set_ylabel('Model Configuration', fontsize=10, fontweight='bold')

# plt.tight_layout()
# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'performance_per_string_length_0.7_threshold_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()







# 1. Filter for Num+Str and configurations with observed scaling trends
# target_configs = ['ContextTab - ContextTab', 'CatBoost - CatBoost', 'Tf-Idf - ExtraTrees-tuned', 'Tf-Idf - XGBoost-tuned']
# df_glm = results[(results['dtype'] == 'Num+Str') & 
#                  (results['encoder_learner'].isin(target_configs))].copy()

# # 2. Fit the GLM for Cardinality Scaling
# # Using a log-transform on cardinality to match the log-normal distribution
# model_card = smf.glm(formula="score ~ np.log1p(avg_cardinality) + encoder_learner", 
#                      data=df_glm, family=sm.families.Gamma(link=sm.families.links.Log())).fit()

# # 3. Fit the GLM for String Length Scaling
# model_str = smf.glm(formula="score ~ np.log1p(avg_string_length_per_cell) + encoder_learner", 
#                     data=df_glm, family=sm.families.Gamma(link=sm.families.links.Log())).fit()

# print("--- GLM: Cardinality Scaling Significance ---")
# print(model_card.summary().tables[1])

# # save table in latex
# today_date = time.strftime("%Y-%m-%d")
# filename = f"glm_cardinality_significance_{today_date}.tex"
# save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{filename}'
# results_table = model_card.summary().tables[1]
# latex_str = results_table.as_latex_tabular()
# with open(save_path, 'w') as f:
#     f.write(latex_str)
    
# print("\n--- GLM: String Length Scaling Significance ---")
# print(model_str.summary().tables[1])

# today_date = time.strftime("%Y-%m-%d")
# filename = f"glm_string_length_significance_{today_date}.tex"
# save_path = f'/data/parietal/store4/soda/gblayer/salts/results_tables/{filename}'
# results_table = model_str.summary().tables[1]
# latex_str = results_table.as_latex_tabular()
# with open(save_path, 'w') as f:
#     f.write(latex_str)


'''
Null Hypothesis (12$H_0$): The specific dataset feature (e.g., cardinality) has no effect on the model's ranking or performance.
Alternative Hypothesis ($H_1$): The feature significantly impacts performance.
P>|z| (p-value): The most critical value for significance. A p-value < 0.05 suggests that the predictor has a statistically significant relationship with the performance score.
e.g. cardinality and string length have a significant impact on contexttab performance
'''


'''
GRAVEYARD
'''





# # B. Create Unique Pipeline ID
# df_analysis['pipeline'] = df_analysis['encoder'] + " + " + df_analysis['learner']

# # Filter for selected encoders only
# df_analysis = df_analysis[(df_analysis['encoder'].isin(selected_encoders)) & (df_analysis['dtype'].isin(['Num', 'Num+Str']))]

# df_plot_tau = pd.DataFrame() # Initialize empty DataFrame

# for col in ['Card_Bin', 'Str_Bin']:

#     agg = df_analysis.groupby(['pipeline','learner', col,'dtype'], as_index=False)[score].mean()

#     agg_pivot = agg.pivot(index=['pipeline','learner', col], columns='dtype', values=score)
    
#     #dropna
#     agg_pivot = agg_pivot.dropna(subset=['Num+Str'])

#     #fillna
#     agg_pivot['Num'] = agg_pivot['Num'].fillna(
#         agg_pivot.groupby(level=['learner', col])['Num'].transform('mean')
#     )
#     agg_pivot = agg_pivot.dropna(subset=['Num'])

#     tau_results = []
#     target_bins = [f'Low', 'High']

#     for bin_label in target_bins:
#         try:
#             # 1. Slice the MultiIndex to get only rows for this specific Card_Bin
#             # level='Card_Bin' ensures we filter on the correct index level
#             subset = agg_pivot.xs(bin_label, level=col)
            
#             # 2. Compute Kendall Tau between the two columns
#             # (subset has columns 'Num' and 'Num+Str')
#             corr, _ = kendalltau(subset['Num'], subset['Num+Str'])
            
#             tau_results.append({
#                 'Feature': col, 
#                 'Bin': bin_label, 
#                 'KendallTau': corr
#             })
#             print(f"Computed Tau for {bin_label}: {corr:.4f}")
            
#         except KeyError:
#             print(f"Warning: Bin '{bin_label}' not found in the index.")

#     df_plot_tau = pd.concat([df_plot_tau, pd.DataFrame(tau_results)])


# sns.set_theme(style="whitegrid", context="paper")

# # Synchronize markers based on your STRABLE paper requirements
# encoder_markers = {
#     'ContextTab': 'X', 'TabPFNv2.5': '*', 'TabSTAR': 'h', 
#     'TargetEncoder': 'v', 'Tf-Idf': 'P', 'CatBoost': 'o',
#     'LLM LLaMA-3.1-8B': '^', 'LLM Qwen-3-8B': 's', 'LLM E5-small-v2': 'D',
#     'LLM FastText': 'p', 'LLM All-MiniLM-L6-v2': '<', 'LLM Jasper-Token-Comp-0.6B': '>'
# }

# learner_markers = {
#     'XGBoost': 'D', 'XGBoost-tuned': 'd', 'Ridge': 'P', 
#     'ExtraTrees': 'X', 'CatBoost': 'o', 'TabPFNv2.5': 'v', 'TabSTAR': 'h'
# }

# # --- 2. MODIFIED PLOTTING FUNCTION ---

# # --- 3. GENERATE PLOT ---
# fig, axes = plt.subplots(2, 2, figsize=(6, 6.5), sharex=True, sharey=True)

# # Call function for each quadrant
# plot_regime(card_wide, 'encoder', axes[0,0], encoder_markers)
# plot_regime(card_wide, 'learner', axes[0,1], learner_markers)
# plot_regime(str_wide, 'encoder', axes[1,0], encoder_markers)
# plot_regime(str_wide, 'learner', axes[1,1], learner_markers)

# # --- 4. MANUAL TITLE & LEGEND PLACEMENT ---
# # Titles set manually outside the loop for control
# axes[0,0].set_title("Encoder × Cardinality", fontweight='bold', fontsize=12, pad=60)
# axes[0,1].set_title("Learner × Cardinality", fontweight='bold', fontsize=12, pad=60)
# axes[1,0].set_title("Encoder × String Length", fontweight='bold', fontsize=12, pad=10)
# axes[1,1].set_title("Learner × String Length", fontweight='bold', fontsize=12, pad=10)

# # High-profile legends placed between titles and top plots
# axes[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, fontsize=7, frameon=False, columnspacing=0.2)
# axes[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, fontsize=7, frameon=False, columnspacing=0.2)

# # --- 5. GLOBAL POLISH ---
# fig.text(0.7, 0.04, 'High-bin performance →', ha='center', fontsize=12)
# fig.text(0.01, 0.5, 'Low-bin performance ↑', va='center', rotation='vertical', fontsize=12)

# plt.subplots_adjust(left=0.13, right=1.3)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'performance_per_cardinality_string_length_2by2_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

'''
encoder-learner together
'''

# # Combine names to create unique pipeline identifiers
# card_wide['pipeline'] = card_wide['encoder'] + " + " + card_wide['learner']
# str_wide['pipeline'] = str_wide['encoder'] + " + " + str_wide['learner']


# --- 1. SETTINGS & STYLES ---
# sns.set_theme(style="whitegrid", context="paper")
# bold_markers = [
#     'o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd', 
#     '8', '$\clubsuit$', '$\spadesuit$', '$\u25A3$', '$\u25CF$', '$\u25B2$'
# ]

# # --- 2. GENERATE GLOBAL STYLE MAPPING (Same as your code) ---
# all_pipelines = sorted(list(set(card_wide['pipeline'].unique()) | set(str_wide['pipeline'].unique())))
# palette = sns.color_palette("husl", len(all_pipelines))
# pipeline_style_map = {pipe: (palette[i], bold_markers[i % len(bold_markers)]) 
#                       for i, pipe in enumerate(all_pipelines)}

# # --- 4. FIGURE EXECUTION ---
# fig = plt.figure(figsize=(12, 6)) # Width 12, Height 6 is safer for 2 cols

# gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
# ax0 = fig.add_subplot(gs[0])
# ax1 = fig.add_subplot(gs[1], sharey=ax0)

# # Plotting
# def plot_pipeline_regime(df, ax, title, style_map):
#     # Standard robustness diagonal (y=x)
#     ax.plot([0.5, 1.0], [0.5, 1.0], '--', color='gray', alpha=0.3, zorder=1)
    
#     unique_pipes = sorted(df['pipeline'].unique())
#     for pipe in unique_pipes:
#         subset = df[df['pipeline'] == pipe]
#         color, marker = style_map[pipe]
        
#         ax.scatter(subset['High'], subset['Low'], 
#                    marker=marker, s=110, label=pipe, alpha=0.9, 
#                    color=color, edgecolors='white', linewidth=0.7, zorder=3)
    
#     ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
#     ax.set_xlim(0.48, 1.02)
#     ax.set_ylim(0.48, 1.02)

# plot_pipeline_regime(card_wide, ax0, "Cardinality", pipeline_style_map)
# plot_pipeline_regime(str_wide, ax1, "String Length", pipeline_style_map)

# # --- FIX 1: Correct Label Positioning (Using Axes Coordinates) ---
# # Instead of guessing 'fig.text' coordinates, we anchor text to the axes
# # transform=ax.transAxes means (0,0) is bottom-left of the AXIS, (1,1) is top-right

# for ax in [ax0, ax1]:
#     # "Low-bin" label on the Y-axis
#     ax.set_ylabel("Low-bin $\\uparrow$", fontsize=12, labelpad=10)
    
#     # "High-bin" label inside the plot (bottom center-right)
#     # x=0.5 (center), y=0.02 (just above bottom axis)
#     ax.text(0.5, 0.02, 'High-bin $\\rightarrow$', transform=ax.transAxes, 
#             ha='center', va='bottom', fontsize=12, fontweight='bold')

# # Remove y-label from second plot to avoid clutter (since sharey=True)
# ax1.set_ylabel("") 
# plt.setp(ax1.get_yticklabels(), visible=False)

# # --- FIX 2 & 3: Combined Legend & Variable Name ---
# # Get handles from BOTH plots to ensure we don't miss any pipeline
# h0, l0 = ax0.get_legend_handles_labels()
# h1, l1 = ax1.get_legend_handles_labels()

# # Create a dictionary to remove duplicates (Pipeline A in plot 1 is same as Pipeline A in plot 2)
# unique_legend = dict(zip(l0 + l1, h0 + h1))

# # Sort legend to match your pipeline order
# sorted_labels = sorted(unique_legend.keys())
# sorted_handles = [unique_legend[l] for l in sorted_labels]

# fig.legend(
#     sorted_handles, 
#     sorted_labels,
#     loc='lower center',
#     bbox_to_anchor=(0.5, 0.0), # (0.5, 0) centers it at the bottom of the FIGURE
#     ncol=4,                    # Adjust columns as needed
#     fontsize=10,
#     frameon=False,
#     columnspacing=1.0
# )

# # --- FINAL LAYOUT ADJUSTMENT ---
# # This automatically fits the plots and leaves room at bottom for the legend
# plt.subplots_adjust(bottom=0.25, top=0.9, left=0.08, right=0.98)


    # metric = score_list[0]  # Ensure using the first metric for Y-axis

    # Y_METRIC = metric

    # # 1. Update Data Aggregation for the current metric
    # if 'encoder_learner' in results.columns:
    #     agg_cols = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
    #     group_cols = ['encoder_learner', 'encoder', 'learner']
    #     df_agg = results[(results['method'].str.contains(f'{dtype}_')) & (results['encoder'].isin(selected_encoders))].groupby(group_cols)[agg_cols].median().reset_index()

    # HIGHER_SCORE_IS_BETTER = True

    # # We need explicit dictionaries for every unique value in the dataframe
    # unique_learners = df_agg['learner'].unique()
    # unique_encoders = df_agg['encoder'].unique()

    # # A. Marker Palette (Always based on Learner)
    # # Maps "XGBoost" -> 's', "XGBoost-tuned" -> 's'
    # learner_markers_dict = {L: get_learner_marker(L) for L in unique_learners}

    # # B. Color Palettes
    # # Palette for Right Plot (Hue = Learner)
    # learner_palette_dict = {L: get_learner_color_simple(L) for L in unique_learners}

    # # Palette for Left Plot (Hue = Encoder)
    # encoder_palette_dict = {E: get_encoder_color(E) for E in unique_encoders}

    # # 2. Re-initialize Plotting Parameters
    # sns.set_style("white")
    # # paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    # ROW_METRICS = ['inference_time_per_1k', 'run_time_per_1k']
    # COL_FACTORS = ['encoder', 'learner']
    # COL_TITLES = ['Encoder', 'Learner'] 
    # ROW_LABELS = ['Inference Time per 1K samples (s)', 'Total Run Time per 1K samples (s)']

    # # 3. INDENTED PLOTTING LOOP: Now it runs for EVERY metric
    # for row_idx, x_metric in enumerate(ROW_METRICS):
    #     fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
    #     pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, True)

    #     for col_idx, factor in enumerate(COL_FACTORS):
    #         ax = axes[col_idx]
            
    #         # --- DYNAMIC PALETTE SELECTION ---
    #         if factor == 'encoder':
    #             current_palette = encoder_palette_dict
    #             # Left Plot: Color by Encoder, Shape by Learner
    #             hue_col = 'encoder'
    #             style_col = 'learner' 
    #         else:
    #             current_palette = learner_palette_dict
    #             # Right Plot: Color by Learner, Shape by Learner
    #             hue_col = 'learner'
    #             style_col = 'learner'

    #         # --- PLOTTING ---
    #         sns.scatterplot(
    #             data=df_agg, 
    #             x=x_metric, 
    #             y=Y_METRIC, 
    #             hue=hue_col,       # Color controlled by column factor
    #             style=style_col,   # Shape ALWAYS controlled by Learner
    #             palette=current_palette,
    #             markers=learner_markers_dict, 
    #             s=80, 
    #             alpha=0.8, 
    #             ax=ax, 
    #             legend='full'
    #         )
            
    #         # Add Pareto Line
    #         ax.step(
    #             pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
    #             linestyle='--', color='black', linewidth=1.2, zorder=0
    #         )
            
    #         # --- FORMATTING (Kept from your code) ---
    #         ax.set_box_aspect(1) 
    #         ax.set_xscale('log')
    #         ax.tick_params(axis='both', which='major', labelsize=9)
            
    #         ax.set_title(COL_TITLES[col_idx], fontsize=12, fontweight='bold', pad=105) 
    #         ax.set_xlabel('')
            
    #         # Adjusted Legend: filter out redundant labels if needed
    #         # (Seaborn might generate a complex legend here because hue != style on the left)
    #         ax.legend(
    #             loc='lower center', 
    #             bbox_to_anchor=(0.5, 1.05), 
    #             ncol=2, 
    #             fontsize=8, # Slightly smaller to fit combined encoder/learner keys
    #             frameon=False,
    #             handletextpad=0.1,
    #             columnspacing=0.5
    #         )
            
    #         if col_idx == 0:
    #             ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=10)
    #         else:
    #             ax.set_ylabel('')
                
    #         sns.despine(ax=ax) 

    #     fig.text(0.5, 0.12, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=10)
    #     plt.subplots_adjust(bottom=0.10, top=0.75, wspace=0.55, left=0.10, right=0.99)

    # metric = score_list[0]  # Ensure using the first metric for Y-axis

    # Y_METRIC = metric

    # # 1. Update Data Aggregation for the current metric
    # if 'encoder_learner' in results.columns:
    #     agg_cols = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
    #     group_cols = ['encoder_learner', 'encoder', 'learner']
    #     df_agg = results[(results['method'].str.contains(f'{dtype}_')) & (results['encoder'].isin(selected_encoders))].groupby(group_cols)[agg_cols].median().reset_index()

    # HIGHER_SCORE_IS_BETTER = True

    # # We need explicit dictionaries for every unique value in the dataframe
    # unique_learners = df_agg['learner'].unique()
    # unique_encoders = df_agg['encoder'].unique()

    # # A. Marker Palette (Always based on Learner)
    # # Maps "XGBoost" -> 's', "XGBoost-tuned" -> 's'
    # learner_markers_dict = {L: get_learner_marker(L) for L in unique_learners}

    # # B. Color Palettes
    # # Palette for Right Plot (Hue = Learner)
    # learner_palette_dict = {L: get_learner_color_simple(L) for L in unique_learners}

    # # Palette for Left Plot (Hue = Encoder)
    # encoder_palette_dict = {E: get_encoder_color(E) for E in unique_encoders}

    # # 2. Re-initialize Plotting Parameters
    # sns.set_style("white")
    # # paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    # ROW_METRICS = ['inference_time_per_1k', 'run_time_per_1k']
    # COL_FACTORS = ['encoder', 'learner']
    # COL_TITLES = ['Encoder', 'Learner'] 
    # ROW_LABELS = ['Inference Time per 1K samples (s)', 'Total Run Time per 1K samples (s)']

    # # 3. INDENTED PLOTTING LOOP: Now it runs for EVERY metric
    # for row_idx, x_metric in enumerate(ROW_METRICS):
    #     # Maintain your 6x5 figsize
    #     fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
    #     pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, True)

    #     for col_idx, factor in enumerate(COL_FACTORS):
    #         ax = axes[col_idx]
            
    #         sns.scatterplot(
    #             data=df_agg, x=x_metric, y=Y_METRIC, hue=factor, style=factor,
    #             s=80, alpha=0.8, ax=ax, palette=paper_palette, legend='full'
    #         )
            
    #         ax.step(
    #             pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
    #             linestyle='--', color='black', linewidth=1.2, zorder=0
    #         )
            
    #         # 1. Force the physical square shape
    #         ax.set_box_aspect(1) 
            
    #         ax.set_xscale('log')
    #         ax.tick_params(axis='both', which='major', labelsize=9)
            
    #         # 2. Adjust title/legend relative to the square top
    #         ax.set_title(COL_TITLES[col_idx], fontsize=12, fontweight='bold', pad=105) 
    #         ax.set_xlabel('')
            
    #         # Move legend precisely above the square box
    #         ax.legend(
    #             loc='lower center', 
    #             bbox_to_anchor=(0.5, 1.05), 
    #             ncol=2, 
    #             fontsize=9, 
    #             frameon=False,
    #             handletextpad=0.1,
    #             columnspacing=0.7
    #         )
            
    #         if col_idx == 0:
    #             ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=10)
    #         else:
    #             ax.set_ylabel('')
                
    #         sns.despine(ax=ax) 

    #     # 3. FIX LABEL POSITION: Use 'y' coordinate to lock it near the bottom ticks
    #     fig.text(0.5, 0.12, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=10)
        
    #     # 4. COMPACT LAYOUT: Push the subplots toward the bottom
    #     # 'bottom' provides room for the fig.text; 'top' brings the title down
    #     plt.subplots_adjust(bottom=0.10, top=0.75, wspace=0.55, left=0.10, right=0.99)

        # today_date = time.strftime("%Y-%m-%d")
        # format = 'pdf'
        # PIC_NAME = f'comparative_pareto_optimality_plot_1Ksample_scale_{Y_METRIC}_{x_metric}_{today_date}.{format}'
        # plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
        # plt.show()



# '''
# COMPARATIVE PARETO PLOTS - split into 2 - no legend
# '''

# dtype = 'num-str'

# for metric in score_list:
    
#     metric = score_list[0]  # Ensure using the first metric for Y-axis
#     Y_METRIC = metric

#     # 1. Update Data Aggregation for the current metric
#     if 'encoder_learner' in results.columns:
#         agg_cols = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
#         group_cols = ['encoder_learner', 'encoder', 'learner']
#         df_agg = results[results['method'].str.contains(f'{dtype}_')].groupby(group_cols)[agg_cols].median().reset_index()

#     HIGHER_SCORE_IS_BETTER = True

#     # 2. Re-initialize Plotting Parameters
#     sns.set_style("white")
#     paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
#     ROW_METRICS = ['inference_time_per_1k', 'run_time_per_1k']
#     COL_FACTORS = ['encoder', 'learner']
#     COL_TITLES = ['Encoder', 'Learner'] 
#     ROW_LABELS = ['Inference Time per 1K samples (s)', 'Total Run Time per 1K samples (s)']

#     # 3. INDENTED PLOTTING LOOP: Now it runs for EVERY metric
#     for row_idx, x_metric in enumerate(ROW_METRICS):
#         fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
        
#         # Calculate Pareto front for current metric and current X-axis
#         pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, HIGHER_SCORE_IS_BETTER)

#         # --- NEW: Save the Pareto Front data to CSV/LaTeX ---
#         today_date = time.strftime("%Y-%m-%d")
#         base_name = f"pareto_data_{Y_METRIC}_{x_metric}_{today_date}"
        
#         # Save the Pareto front points specifically
#         pareto_df.to_csv(f'/data/parietal/store4/soda/gblayer/salts/results_tables/{base_name}_front.csv', index=False)
        
#         # Save as LaTeX table string for quick copy-paste
#         # This selects the key columns for your paper
#         latex_cols = ['encoder_learner', 'encoder', 'learner', x_metric, Y_METRIC]
#         pareto_df[latex_cols].to_latex(
#             f'/data/parietal/store4/soda/gblayer/salts/results_tables/{base_name}_front.tex', 
#             index=False, 
#             float_format="%.4f",
#             caption=f"Pareto Front points for {x_metric} vs {Y_METRIC_LABELS[Y_METRIC]}",
#             label=f"tab:pareto_{x_metric}"
#         )
#         # ---------------------------------------------------

#         for col_idx, factor in enumerate(COL_FACTORS):
#             ax = axes[col_idx]
            
#             sns.scatterplot(
#                 data=df_agg, x=x_metric, y=Y_METRIC, hue=factor, style=factor,
#                 s=100, alpha=0.8, ax=ax, palette=paper_palette, legend=False 
#             )
            
#             ax.step(
#                 pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
#                 linestyle='--', color='black', linewidth=1.5, zorder=0
#             )
            
#             ax.set_xscale('log')
#             ax.tick_params(axis='both', which='major', labelsize=12)
#             ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', pad=10) 
#             ax.set_xlabel('')
            
#             if col_idx == 0:
#                 ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=14, labelpad=10)
#             else:
#                 ax.set_ylabel('')
                
#             sns.despine(ax=ax) 

#         fig.text(0.5, -0.05, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=14)
#         plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.15, left=0.12, right=0.95)

#         today_date = time.strftime("%Y-%m-%d")
#         format = 'pdf'
#         PIC_NAME = f'comparative_pareto_optimality_plot_1Ksample_scale_{Y_METRIC}_{x_metric}_{today_date}.{format}'
#         plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
#         plt.show()

# '''
# COMPARATIVE PARETO PLOTS - split into 2 - with legend
# fewer points
# '''

# for metric in score_list:
    
#     metric = score_list[0]  # Ensure using the first metric for Y-axis
#     Y_METRIC = metric

#     # 1. Update Data Aggregation for the current metric
#     if 'encoder_learner' in results.columns:
#         agg_cols = [Y_METRIC, 'run_time_per_1k']
#         group_cols = ['encoder_learner', 'encoder', 'learner']
#         df_agg = results[results['method'].str.contains('num-str_')].groupby(group_cols)[agg_cols].median().reset_index()

#     HIGHER_SCORE_IS_BETTER = True

#     # 2. Re-initialize Plotting Parameters
#     sns.set_style("white")
#     paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
#     ROW_METRICS = ['run_time_per_1k']
#     COL_FACTORS = ['encoder', 'learner']
#     COL_TITLES = ['Encoder', 'Learner'] 
#     ROW_LABELS = ['Total Run Time per 1K samples (s)']

#     FONT_AXIS_LABEL = 11
#     FONT_TITLE = 14
#     FONT_TICK = 9
#     FONT_LEGEND = 9.5 
#     # 3. INDENTED PLOTTING LOOP: Now it runs for EVERY metric
#     for row_idx, x_metric in enumerate(ROW_METRICS):
#         # We use a slightly taller figure to accommodate the stacked legend/title
#         fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
        
#         pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, HIGHER_SCORE_IS_BETTER)
#         top_methods = pareto_df['method_polished'].unique()
#         df_plot = df_agg[df_agg['method_polished'].isin(top_methods)].copy()

#         # --- FIX 1: GLOBAL ROW CALCULATION ---
#         # Calculate max rows across both factors to keep titles level and avoid overlap
#         num_encoders = len(df_plot['encoder'].unique())
#         num_learners = len(df_plot['learner'].unique())
#         ncol = 2
#         # Determine the maximum height required by either legend
#         max_rows = math.ceil(max(num_encoders, num_learners) / ncol)
        
#         # Calculate dynamic pad: legend height + buffer
#         # We increase the buffer to 30 for safety in academic PDF rendering
#         dynamic_title_pad = (max_rows * FONT_LEGEND * 1.6) + 30

#         for col_idx, factor in enumerate(COL_FACTORS):
#             ax = axes[col_idx]
            
#             sns.scatterplot(
#                 data=df_plot, x=x_metric, y=Y_METRIC, hue=factor, style=factor,
#                 s=80, alpha=0.9, ax=ax, palette=paper_palette, edgecolor='black', linewidth=0.6
#             )
            
#             ax.step(
#                 pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
#                 linestyle='--', color='black', linewidth=1.2, zorder=0
#             )
            
#             ax.set_xscale('log')
#             ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
            
#             # --- FIX 2: ALIGNED LEGEND BOXES ---
#             handles, labels = ax.get_legend_handles_labels()
#             # Force both legends to have the same capacity to ensure identical height
#             total_capacity = ncol * max_rows
#             while len(labels) < total_capacity:
#                 handles.append(plt.Line2D([0], [0], color='none', label=''))
#                 labels.append('')
            
#             ax.legend(
#                 handles, labels,
#                 loc='lower center', 
#                 bbox_to_anchor=(0.5, 1.02), # Anchored just above the axis
#                 ncol=ncol,                     
#                 frameon=False,
#                 fontsize=FONT_LEGEND,
#                 handletextpad=0.2,
#                 columnspacing=1,
#                 labelspacing=1.2,           
#                 markerscale=0.8
#             )

#             # Apply the synchronized dynamic title padding
#             ax.set_title(COL_TITLES[col_idx], fontsize=FONT_TITLE, fontweight='bold', pad=dynamic_title_pad) 

#             ax.set_xlabel('')
#             if col_idx == 0:
#                 ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=FONT_AXIS_LABEL)
#             else:
#                 ax.set_ylabel('')
                
#             sns.despine(ax=ax) 

#         # --- FIX 3: LAYOUT SPREAD ---
#         # Adjust 'top' dynamically if max_rows is very high
#         # For 3-4 rows, top=0.6 is usually safe for a 5-inch tall figure
#         top_margin = max(0.55, 0.75 - (max_rows * 0.05)) 
        
#         fig.text(0.5, 0.04, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=FONT_AXIS_LABEL)
#         plt.subplots_adjust(bottom=0.15, top=top_margin, wspace=0.25, left=0.12, right=0.95)

#         today_date = time.strftime("%Y-%m-%d")
#         format = 'pdf'
#         PIC_NAME = f'comparative_pareto_optimality_plot_10Ksample_scale_filtered_{Y_METRIC}_{x_metric}_{today_date}.{format}'
#         plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
#         plt.show()


# '''
# Critical Difference Diagram - Jun's implementation BIG ONE
# '''

# dtype = 'num-str'
# score = score_list[0]  # Change index to select different score types


# # Load results
# df_score = results.copy()
# df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)

# count_results = df_score.method.value_counts()
# keep_method_list = count_results[count_results == 324].index.tolist()

# df_score = df_score[df_score.method.isin(keep_method_list)].reset_index(drop=True)

# # Change namings
# # df_score.method = df_score.method.str.replace('num-str_', '')
# # df_score.method = df_score.method.str.replace('tabvec_', 'TabVec|')
# # df_score.method = df_score.method.str.replace('tarenc_', 'TarEnc|')
# # df_score.method = df_score.method.str.replace('llm-', 'LLM-')
# # df_score.method = df_score.method.str.replace('ridge', 'Ridge')
# # df_score.method = df_score.method.str.replace('xgb', 'XGB')
# # df_score.method = df_score.method.str.replace('extrees', 'ExTrees')
# # df_score.method = df_score.method.str.replace('catboost_catboost', 'CatBoost')
# # df_score.method = df_score.method.str.replace('_default', '-Default')
# # df_score.method = df_score.method.str.replace('_tune', '-Tune')
# # df_score.method = df_score.method.str.replace('LLM-e5-base-v2_', 'LLM-e5-base|')

# df_score['method'] = df_score['method'].str.replace(f'{dtype}_', '', regex=False)

# # Apply the cleaning logic
# df_score['method'] = df_score['method'].apply(clean_method_name)

# df_score.method.value_counts()

# df_score_temp = df_score[['r2', 'roc_auc']].copy()
# df_score_temp.fillna(0, inplace=True)
# df_score[score] = df_score_temp.sum(axis=1)

# df_score_final = _generate_marker(df_score)
# df_score_final["rank"] = df_score_final.groupby(["marker"], group_keys=True).score.rank(
#     pct=False, ascending=False
# )

# df_score_final.method.value_counts()

# # Ranks and test results
# avg_rank = (
#     df_score_final.groupby(["marker"], group_keys=True)  # marker
#     .score.rank(pct=False, ascending=False)
#     .groupby(df_score_final.method)
#     .mean()
# )
# avg_rank = -1 * avg_rank

# test_results = sp.posthoc_conover_friedman(
#     df_score_final,
#     melted=True,
#     block_col="marker",
#     block_id_col="marker",
#     group_col="method",
#     y_col=score,
# )

# # test_results = sp.posthoc_wilcoxon(
# #     df_score_final,
# #     val_col = "rank",
# #     group_col="marker",
# #     p_adjust='holm'
# # )
# test_results = test_results.replace(0, 1e-100)  # Required for visualization


# # Lines
# # line_style = {model: "-" for model in models}
# models = df_score_final.method.unique()
# line_style = {model: "-" for model in models}
# for model in models:
#     if "TargetEncoder" in model:
#         line_style[model] = "--"
#     if "LLM" in model:
#         line_style[model] = "-."
# #     elif "tm" in model:
# #         line_style[model] = "--"
# #     if "ernie" in model:
# #         line_style[model] = "--"
# #     if 'llm-col' in model:
# #         line_style[model] = "-."

# # Colors
# # color_palette = dict()
# # for model in models:
# #     if 'XGBoost' in model:
# #         color_palette[model] = "C0"
# #     elif 'ExtraTrees' in model:
# #         color_palette[model] = "C1"
# #     elif 'CatBoost' in model:
# #         color_palette[model] = "C2"
# #     elif 'Ridge' in model:
# #         color_palette[model] = "C3"

# # unique_learners = sorted(list(set([m.split(' - ')[-1] for m in models])))

# # # 2. Generate a large, distinct color palette
# # # For many learners, 'colorblind' or 'husl' are best for accessibility
# # num_learners = len(unique_learners)
# # colors = sns.color_palette("colorblind", n_colors=num_learners)
# # learner_color_map = dict(zip(unique_learners, colors))

# # # 3. Create the final color_palette for all 77+ models
# # color_palette = {}
# # for model in models:
# #     # Identify the learner suffix
# #     learner = model.split(' - ')[-1]
# #     # Assign the color belonging to that learner group
# #     color_palette[model] = learner_color_map[learner]

# # Check how many unique learner groups you have

# # print(f"Unique learner groups: {len(unique_learners)}")

# # # color_palette = model_color_palette

# # sns.set_theme(style="white", font_scale=1)

# # # Plot
# # fig, axes = plt.subplots(1, 1, figsize=(5,6))
# # ax = axes

# # cdd = critical_difference_diagram(
# #     ranks=avg_rank,
# #     sig_matrix=test_results,
# #     label_fmt_left="{label} ",
# #     label_fmt_right=" {label}",
# #     label_props={
# #         "fontsize": 10,
# #     },
# #     crossbar_props={"color": "black", "linewidth": 1},
# #     marker_props={"marker": ""},
# #     elbow_props={"linewidth": 1.5},
# #     text_h_margin=1.2,
# #     color_palette=color_palette,
# #     line_style=line_style,
# #     bold_control=True,
# #     v_space=4,
# #     ax=ax,
# # )

# # ax.set_xticklabels([None, 10, 8, 6, 4, 2], fontsize=15)

# print(f"Total models: {len(models)}")
# palette_by_learner = {}
# palette_by_encoder = {}

# for model in models:
#     # Assuming format is "Encoder - Learner" based on your previous split code
#     parts = model.split(' - ')
#     encoder_part = parts[0]
#     learner_part = parts[-1]
    
#     # 1. Learner Palette
#     palette_by_learner[model] = get_learner_color_simple(learner_part)
    
#     # 2. Encoder Palette
#     palette_by_encoder[model] = get_encoder_color(encoder_part)

# # ==========================================
# # 4. PLOT VERSION 1: COLORED BY LEARNER
# # ==========================================
# sns.set_theme(style="white", font_scale=1)

# fig1, ax1 = plt.subplots(1, 1, figsize=(3, 4))

# critical_difference_diagram(
#     ranks=avg_rank,
#     sig_matrix=test_results,
#     label_fmt_left="{label} ",
#     label_fmt_right=" {label}",
#     label_props={"fontsize": 10},
#     crossbar_props={"color": "black", "linewidth": 1},
#     marker_props={"marker": ""},
#     elbow_props={"linewidth": 1.5},
#     text_h_margin=1.2,
    
#     color_palette=palette_by_learner,  # <--- APPLIED HERE
#     line_style=line_style,
    
#     bold_control=True,
#     v_space=4,
#     ax=ax1,
# )
# n_models = len(models)

# # 1. Round up to the nearest multiple of 5 to get the maximum tick
# max_tick = math.ceil(n_models / 5) * 5

# # 2. Generate the descending range with a step of 5
# #    range(start, stop, step) -> stop is exclusive, so we go down to 0 to include 5
# tick_range = list(range(max_tick, 0, -5))

# # 3. Add None at the beginning as requested
# xticklabels = [None] + tick_range
# # Adjust ticks based on your rank range (e.g. if max rank is 40)
# ax1.set_xticklabels(xticklabels, fontsize=15)
# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'critical_difference_diagram_custom_Jun_selectedLLMs_friedman_colorbylearner_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()



# # ==========================================
# # 5. PLOT VERSION 2: COLORED BY ENCODER
# # ==========================================
# fig2, ax2 = plt.subplots(1, 1, figsize=(3, 4))

# critical_difference_diagram(
#     ranks=avg_rank,
#     sig_matrix=test_results,
#     label_fmt_left="{label} ",
#     label_fmt_right=" {label}",
#     label_props={"fontsize": 10},
#     crossbar_props={"color": "black", "linewidth": 1},
#     marker_props={"marker": ""},
#     elbow_props={"linewidth": 1.5},
#     text_h_margin=1.2,
    
#     color_palette=palette_by_encoder,  # <--- APPLIED HERE
#     line_style=line_style,
    
#     bold_control=True,
#     v_space=4,
#     ax=ax2,
# )
# n_models = len(models)

# # 1. Round up to the nearest multiple of 5 to get the maximum tick
# max_tick = math.ceil(n_models / 5) * 5

# # 2. Generate the descending range with a step of 5
# #    range(start, stop, step) -> stop is exclusive, so we go down to 0 to include 5
# tick_range = list(range(max_tick, 0, -5))

# # 3. Add None at the beginning as requested
# xticklabels = [None] + tick_range
# ax2.set_xticklabels(xticklabels, fontsize=15)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'critical_difference_diagram_custom_Jun_selectedLLMs_friedman_colorbyencoder_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'critical_difference_diagram_custom_Jun_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()



# '''
# Average performance when us-
# ing numerical-only (dashed
# lines), or combined (Num+Str) features.
# Encoders, e2e models and biggest LLMs
# '''

# # ============================================
# # 1. DATA PREPARATION & SUBSETTING
# # ============================================
# df = results.copy()

# plot_data = df[(df['dtype'] == 'Num+Str') & (df['encoder'].isin(selected_encoders))].copy()
# unique_learners = sorted(df['learner'].unique())

# # Pre-calculate baseline means for the vertical dashed lines
# baseline_data = df[(df['dtype'] == 'Num') & (df['encoder'].isin(selected_encoders))]
# baseline_means = baseline_data.groupby('learner')['score'].mean()

# #Filter and establish the decreasing global order
# encoder_order = (
#     plot_data
#     .groupby('encoder')['score']
#     .mean()
#     .sort_values(ascending=False)
#     .index
# )

# # ============================================
# # 2. SPACING & THEME CONFIGURATION
# # ============================================

# plt.rcParams['font.sans-serif'] = ["Arial", "Helvetica", "DejaVu Sans"]
# plt.rcParams['font.family'] = "sans-serif" 
# sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
# sns.set_context("paper")

# total_group_height = 1.0  # Space allocated to each encoder group
# intra_group_sep = 0.07    # Separation between bars within a group
# inter_encoder_sep = 0.6   # Space between different encoder groups

# fig, ax = plt.subplots(figsize=(3, 6))
# current_y = 0
# yticks_locs = []

# # Define consistent color mapping
# palette = sns.color_palette([
#     "#4E79A7",  # blue
#     "#F28E2B",  # orange
#     "#59A14F",  # green
#     "#E15759",  # red
#     "#76B7B2",  # teal
#     "#EDC948",  # yellow
#     "#B07AA1",  # purple
#     "#FF9DA7",  # pink
#     "#9C755F",  # brown
#     "#BAB0AC",  # gray

#     "#1F77B4",  # strong blue
#     "#FF7F0E",  # strong orange
#     "#2CA02C",  # strong green
#     "#D62728",  # strong red
#     "#9467BD",  # strong purple
# ])
# color_map = dict(zip(unique_learners, palette))
# ordered_learners = list(color_map.keys())

# # ============================================
# # 3. DYNAMIC ITERATIVE PLOTTING
# # ============================================
# for encoder in encoder_order:
#     enc_df = plot_data[plot_data['encoder'] == encoder]
#     present_learners = [l for l in ordered_learners if not enc_df[enc_df['learner'] == l].empty]
#     num_learners = len(present_learners)
    
#     # Calculate bar height so the group always fills 'total_group_height'
#     # dynamic_bar_height = (total_group_height / num_learners) - intra_group_sep
#     adaptive_sep = max(0.07, 0.18 - 0.015 * num_learners)
#     dynamic_bar_height = (total_group_height - adaptive_sep * (num_learners - 1)) / num_learners
#     encoder_top_y = current_y
    
#     for i, learner in enumerate(present_learners):
#         learner_data = enc_df[enc_df['learner'] == learner]['score']
#         mean_score = learner_data.mean()
#         sem = learner_data.sem()
        
#         bar_y = current_y - (i * (dynamic_bar_height + intra_group_sep))
        
#         # Bars at zorder 3
#         ax.barh(bar_y, mean_score, height=dynamic_bar_height, 
#                 color=color_map[learner], edgecolor='none', 
#                 linewidth=0, zorder=3)
        
#         # Error bars at zorder 4
#         if num_learners >= 6:
#             ax.errorbar(
#                 mean_score, bar_y,
#                 xerr=sem,
#                 fmt='none',
#                 ecolor='black',
#                 elinewidth=1.0,
#                 capsize=1.5,
#                 zorder=4
#             )
#         else:
#             ax.errorbar(
#                 mean_score, bar_y,
#                 xerr=sem,
#                 fmt='none',
#                 ecolor='black',
#                 elinewidth=1.2,
#                 capsize=2.5,
#                 zorder=4
#             )
    
#     # Track midpoint for Y-ticks
#     yticks_locs.append(encoder_top_y - (total_group_height / 2) + (dynamic_bar_height / 2))

#     if encoder != encoder_order[-1]:
#         ax.axhline(
#             current_y + inter_encoder_sep / 2,
#             color='gray',
#             alpha=0.15,
#             linewidth=0.5,
#             zorder=1
#         )

#     current_y -= (total_group_height + inter_encoder_sep)
    

# # ============================================
# # 4. COLORED DASHED BASELINES (ON TOP)
# # ============================================
# for learner, score in baseline_means.items():
#     # Setting zorder=5 ensures these lines pass ON TOP of bars and error bars
#     ax.axvline(x=score, color=color_map[learner], linestyle='--', 
#                linewidth=1.5, alpha=0.8, zorder=5)

# # ============================================
# # 5. FINAL ICML POLISH & LEGEND
# # ============================================
# ax.set_yticks(yticks_locs)
# ax.set_yticklabels(encoder_order, fontsize=16)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
# ax.tick_params(axis='x', labelsize=16)
# ax.set_xlabel(f"Avg Performance ($R^2$ & AUC)", fontsize=18)
# ax.set_ylabel("Encoder (Num+Str)", fontsize=18)
# ax.set_xlim(0.35, 0.75)


# # Styling grid (behind bars)
# ax.xaxis.grid(True, linestyle='-', alpha=0.2)
# ax.set_axisbelow(True)

# # Legend setup
# learner_handles = [
#     mlines.Line2D(
#         [], [], color=color_map[l], marker='s',
#         linestyle='', markersize=7, label=l
#     )
#     for l in unique_learners
# ]

# baseline_style_handle = mlines.Line2D(
#     [], [], color='black', linestyle='--',
#     linewidth=1.5, label='Num-only baseline'
# )

# legend = ax.legend(
#     handles=learner_handles + [baseline_style_handle],
#     title='Tabular Learner',
#     title_fontsize=9,
#     fontsize=7.5,
#     loc='center left',
#     bbox_to_anchor=(1.02, 0.5),  # OUTSIDE, vertically centered
#     frameon=True,
#     borderpad=0.4,
#     handletextpad=0.4,
#     labelspacing=0.35
# )

# # Transparent legend box
# legend.get_frame().set_alpha(0.0)
# legend.get_frame().set_edgecolor('none')

# sns.despine(left=False, bottom=False)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'avg_performance_dtypes_vs_numerical_top3biggestLLMs_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# ==========================================
# VISUAL PROOF OF HETEROGENEITY (LOG-NORMALITY)
# ==========================================
# We plot 4 key metrics in a 4x2 grid.
# LEFT Column: Linear Scale (Shows the "Heavy Tail" / Skew)
# RIGHT Column: Log Scale (Shows the "Bell Curve" / Log-Normality)

# fig, axes = plt.subplots(2, 2, figsize=(5, 6))

# -------------------------------------------------------
# ROW 1: Number of Rows (Dataset Size)
# -------------------------------------------------------
# Linear
# sns.histplot(df_datasets['num_rows'], kde=True, log_scale=False, ax=axes[0, 0], color='salmon')
# axes[0, 0].set_title("Number of Rows (Linear: Heavy Tail)", fontsize=18)
# axes[0, 0].set_xlabel("Count", fontsize=16)

# # Log Scale
# # sns.histplot(df_datasets['num_rows'], kde=True, log_scale=True, ax=axes[0, 1], color='salmon')
# # axes[0, 1].set_title("Number of Rows (Log-Scale: Bell Curve)", fontsize=18)
# # axes[0, 1].set_xlabel("Count (Log Scale)", fontsize=16)

# # -------------------------------------------------------
# # ROW 2: Number of Columns (Dimensionality)
# # -------------------------------------------------------
# # Linear
# sns.histplot(df_datasets['num_columns'], kde=True, log_scale=False, ax=axes[0, 1], color='skyblue')
# axes[0, 1].set_title("Number of Columns (Linear: Heavy Tail)", fontsize=18)
# axes[0, 1].set_xlabel("Count", fontsize=16)

# # Log Scale
# # sns.histplot(df_datasets['num_columns'], kde=True, log_scale=True, ax=axes[1, 1], color='skyblue')
# # axes[1, 1].set_title("Number of Columns (Log-Scale: Bell Curve)", fontsize=18)
# # axes[1, 1].set_xlabel("Count (Log Scale)", fontsize=16)

# # -------------------------------------------------------
# # ROW 3: Cardinality (Feature Complexity)
# # -------------------------------------------------------
# # Linear
# sns.histplot(df_datasets['cardinality'], kde=True, log_scale=False, ax=axes[1, 0], color='purple')
# axes[1, 0].set_title("Avg Cardinality per Column (Linear: Heavy Tail)", fontsize=18)
# axes[1, 0].set_xlabel("Unique Values", fontsize=16)

# # Log Scale
# # sns.histplot(df_datasets['cardinality'], kde=True, log_scale=True, ax=axes[2, 1], color='purple')
# # axes[2, 1].set_title("Avg Cardinality per Column (Log-Scale: Bell Curve)", fontsize=18)
# # axes[2, 1].set_xlabel("Unique Values (Log Scale)", fontsize=16)

# # -------------------------------------------------------
# # ROW 4: Text Length (Content Complexity)
# # -------------------------------------------------------
# # Linear
# sns.histplot(df_datasets['string_length'], kde=True, log_scale=False, ax=axes[1, 1], color='green')
# axes[1, 1].set_title("Avg Text Length (Linear: Heavy Tail)", fontsize=18)
# axes[1, 1].set_xlabel("Characters", fontsize=16)

# Log Scale
# sns.histplot(df_datasets['string_length'], kde=True, log_scale=True, ax=axes[3, 1], color='green')
# axes[3, 1].set_title("Avg Text Length (Log-Scale: Bell Curve)", fontsize=18)
# axes[3, 1].set_xlabel("Characters (Log Scale)", fontsize=16)

# -------------------------------------------------------
# Final Formatting
# -------------------------------------------------------
# plt.suptitle("Proof of Heterogeneity: Log-Normal Distributions Across All Dimensions", fontsize=20, y=1.02)


'''
test
'''

# def kendall_w(rank_matrix):
#     """
#     Calculates Kendall's Coefficient of Concordance (W).
#     Input: DataFrame where Index=Algorithms, Columns=Scenarios (Dropped Sources)
#     Values must be integer ranks (1, 2, 3...), not float average ranks.
#     """
#     m = rank_matrix.shape[1]  # Number of raters (scenarios)
#     n = rank_matrix.shape[0]  # Number of items (algorithms)
    
#     if n <= 1: return 1.0 # Trivial perfect agreement
    
#     # Sum of ranks for each algorithm
#     R = rank_matrix.sum(axis=1)
#     R_mean = R.mean()
#     S = ((R - R_mean) ** 2).sum()
    
#     # Formula for W
#     denom = (m**2) * (n**3 - n)
#     if denom == 0: return 0.0
#     W = (12 * S) / denom
#     return W

# print(f"Running Leave-One-Source-Out analysis on {len(sources)} sources...")

# def calculate_rankings(data_frame):
#     """
#     Calculates the average rank of each algorithm across the given datasets.
#     Output: Series of ranks (e.g., XGB: 1.2, Ridge: 3.4)
#     """
#     # Rank per dataset
#     ranks_per_dataset = data_frame.rank(axis=1, ascending=False, method='min')
#     # Average across datasets
#     avg_ranks = ranks_per_dataset.mean(axis=0)
#     return avg_ranks

# loso_data = []

# for source_drop in sources:
#     # A. Filter OUT the specific source
#     subset = results[results['source'] != source_drop]

#     print(f"Analyzing without source: {source_drop}")
#     print(f"Number of datasets per source: {results[results['source'] == source_drop]['data_name'].nunique()}")
#     print(f"Number of datasets dropped: {results[results['source'] == source_drop]['data_name'].nunique()}")
#     print(f"number of datasets left: {results['data_name'].nunique() - results[results['source'] == source_drop]['data_name'].nunique()})")
    
#     # B. Pivot the subset (Rows=Datasets, Cols=Methods)
#     # We group by mean just in case there are duplicates, though there shouldn't be
#     subset_pivot = subset.groupby(['data_name', 'method_polished'])['score'].mean().unstack()
    
#     # C. Calculate Average Ranks (Float values, e.g., 1.5)
#     avg_ranks = calculate_rankings(subset_pivot)
    
#     # D. Convert to Integer Ranks (1, 2, 3) for the Plot/Stats
#     # This tells us: "In this scenario, who came 1st, 2nd, 3rd?"
#     final_ranks = rankdata(avg_ranks, method='min')
    
#     # Store for analysis
#     for method, rank_val in zip(avg_ranks.index, final_ranks):
#         loso_data.append({
#             'Method': method,
#             'Dropped_Source': source_drop,
#             'Rank': rank_val
#         })

# df_loso = pd.DataFrame(loso_data)

# # Pivot to shape: Index=Method, Cols=Dropped_Source, Values=Rank
# rank_matrix = df_loso.pivot(index='Method', columns='Dropped_Source', values='Rank')

# W_score = kendall_w(rank_matrix)

# print("-" * 40)
# print(f"Kendall's W (Stability Score): {W_score:.4f}")
# print("  (1.0 = Perfect Stability, 0.0 = Random)")

# if W_score > 0.9:
#     print("✅ VERDICT: Robust. Source selection does not alter algorithm rankings.")
# else:
#     print("⚠️ VERDICT: Unstable. Rankings depend on specific data sources.")
# print("-" * 40)



# # 1. PREPARE THE DATA
# # results columns: 'encoder', 'learner', 'method', 'score_centred', 'data_name'
# df_filtered = results[results['method'].str.contains('num-only|num-str_')].copy()

# # 2. FILTER FOR ENCODERS WITH MULTIPLE LEARNERS
# # Count unique learners per encoder to ensure we have a ranking vector to correlate
# learner_counts = df_filtered.groupby('encoder')['learner'].nunique()
# valid_encoders = learner_counts[learner_counts > 1].index.tolist()
# df_valid = df_filtered[df_filtered['encoder'].isin(valid_encoders)]

# encoder_correlations = []

# pivot_num = df_valid[df_valid['dtype'] == 'Num'].pivot_table(
#     index='data_name', columns='learner', values='score'
# ).dropna()
# print(f"  NUM modality datasets: {len(pivot_num)}")

# # 3. COMPUTE CORRELATION PER ENCODER
# for enc in valid_encoders:
#     print(f"Processing Encoder: {enc}")
#     df_enc = df_valid[df_valid['encoder'] == enc]
    
#     pivot_both = df_enc[df_enc['dtype'] == 'Num+Str'].pivot_table(
#         index='data_name', columns='learner', values='score'
#     ).dropna()
#     print(f"  NUM+STR modality datasets: {len(pivot_both)}")
    
#     # Align datasets
#     common_idx = pivot_num.index.intersection(pivot_both.index)
#     if len(common_idx) < 2: continue
    
#     # Compute average learner ranks across the common datasets
#     rank_num = pivot_num.loc[common_idx].rank(axis=1, ascending=False).mean()
#     rank_both = pivot_both.loc[common_idx].rank(axis=1, ascending=False).mean()

#     #Align algorithms
#     common_learners = rank_num.index.intersection(rank_both.index)
#     print(f"  Common learners: {len(common_learners)}")
#     if len(common_learners) < 2: continue

#     rank_num = rank_num.loc[common_learners]
#     rank_both = rank_both.loc[common_learners]
    
#     # Compute Kendall Tau correlation between the two ranking vectors
#     tau, _ = kendalltau(rank_num, rank_both)
#     encoder_correlations.append({'Encoder': enc, 'KendallTau': tau})

# df_plot = pd.DataFrame(encoder_correlations).sort_values('KendallTau', ascending=False)

# # 4. VISUALIZATION
# plt.figure(figsize=(10, 6))
# sns.set_style("white")

# # Horizontal Bar Plot
# ax = sns.barplot(
#     data=df_plot, x='KendallTau', y='Encoder', 
#     palette='viridis', hue='Encoder', legend=False
# )

# # Add target threshold line (e.g., 0.5 for stability)
# plt.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Domain Shift Threshold (0.5)')

# # Formatting for paper-ready quality
# plt.title('Learner Ranking Correlation: NUM vs. NUM+STR', fontsize=18, pad=20)
# plt.xlabel('Kendall Tau Correlation', fontsize=16)
# plt.ylabel('Encoder', fontsize=16)
# plt.xlim(0, 1.1)
# plt.grid(axis='x', linestyle=':', alpha=0.7)
# plt.legend(loc='lower right', fontsize=16)
# plt.tight_layout()
# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'kendalltau_encoder_learner_domain_shift_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# ==========================================
# 5. STATISTICAL PROOF: KS Test for Log-Normality
# ==========================================
# H0: The distribution of columns follows a Log-Normal distribution.
# H1: The distribution of columns does not follow a Log-Normal distribution.
# Reject H0 if p-value < 0.05. If pvalue < 0.05 the data does NOT follow a Log-Normal distribution.

# def test_log_normal(data, feature_name):
#     # Ensure data is strictly positive for log-fitting
#     data_clean = data[data > 0].dropna()
    
#     # 1. Fit the parameters of a log-normal distribution to the data
#     # shape (s), loc, scale
#     shape, loc, scale = stats.lognorm.fit(data_clean)
    
#     # 2. Perform Kolmogorov-Smirnov Test
#     # Compare empirical data against the theoretical CDF of the fitted log-normal
#     ks_stat, p_value = stats.kstest(data_clean, 'lognorm', args=(shape, loc, scale))
    
#     print(f"--- KS Test for {feature_name} ---")
#     # print the H0 and H1
#     print("H0: The distribution follows a Log-Normal distribution.")
#     print("H1: The distribution does not follow a Log-Normal distribution.")
#     print("-" * 50)
#     print(f"KS Statistic: {ks_stat:.4f}")
#     print(f"P-value:      {p_value:.4f}")
    
#     if p_value > 0.05:
#         print(f"Result: > 0.05. We CANNOT reject H0. \nEvidence supports that '{feature_name}' follows a Log-Normal distribution (Heterogeneous).")
#     else:
#         print(f"Result: < 0.05. We reject H0. Distribution is statistically distinct from Log-Normal.")
#     print("-" * 50)

# # Run the test
# test_log_normal(df_datasets['num_columns'], 'Number of Columns')
# test_log_normal(df_datasets['num_rows'], 'Number of Rows')
# test_log_normal(df_datasets['cardinality'], 'Average Cardinality per Column')
# test_log_normal(df_datasets['string_length'], 'Average Text Length')


'''
KendallTau correlation for domain shift - e2e models
'''

# # 1. Filter for E2E Models (Those where the encoder is the same as the learner)
# # e.g., 'num-str_tarte_tarte', 'num-str_contexttab_contexttab', etc.
# e2e_models = ['catboost', 'tarte', 'contexttab', 'tabstar']
# df_e2e = results[results['learner'].isin(e2e_models)].copy()

# # Define modalities
# df_e2e['modality'] = df_e2e['method'].apply(
#     lambda x: 'NUM+STR' if 'num-str' in x else 'NUM'
# )

# # 2. Calculate Lift per Dataset
# # We pivot to have NUM and NUM+STR side-by-side for each dataset/model pair
# pivot_lift = df_e2e.pivot_table(
#     index=['data_name', 'learner'], 
#     columns='modality', 
#     values='score_centred'
# ).reset_index()

# # Calculate Lift: (Combined Score) - (Numerical Score)
# pivot_lift['Lift'] = pivot_lift['NUM+STR'] - pivot_lift['NUM']

# # 3. Aggregate for Plotting
# df_plot = pivot_lift.groupby('learner')['Lift'].agg(['mean', 'std']).reset_index()
# df_plot = df_plot.sort_values('mean', ascending=False)

# # 4. Visualization
# plt.figure(figsize=(10, 6))
# sns.set_style("white")

# ax = sns.barplot(
#     data=df_plot, x='mean', y='learner', 
#     hue='learner', palette='magma', legend=False
# )

# # Add Error Bars manually (Standard Deviation across datasets)
# plt.errorbar(
#     x=df_plot['mean'], y=df_plot['learner'], 
#     xerr=df_plot['std'], fmt='none', c='black', capsize=5
# )

# # Add a vertical line at 0 (No improvement)
# plt.axvline(0, color='black', linestyle='-', alpha=0.3)

# # Formatting
# plt.title('Predictive Lift: Adding Strings to E2E Models', fontsize=18, pad=20)
# plt.xlabel('$\Delta$ Mean Centred Score (NUM+STR - NUM)', fontsize=16)
# plt.ylabel('End-to-End Model', fontsize=16)
# plt.grid(axis='x', linestyle=':', alpha=0.6)

# plt.tight_layout()
# plt.show()


'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
the benchmark to converge to the same ranking?
BOOTSTRAPPED EXPONENTIAL SATURATION FITTING (EXTRAPOLATION) - comparison with 2 other functions 
'''


# # 1. DEFINE THE THREE FUNCTIONS
# def model_exp(x, a, b, c):
#     """Original Exponential Saturation: a - b * exp(-cx)"""
#     return a - b * np.exp(-c * x)

# def model_ref_1(x, a, b):
#     """Reference 1: 1 - (a/sqrt(x)) * exp(-bx)"""
#     # Note: 1.0 is the fixed asymptote here
#     return 1 - (a / np.sqrt(x)) * np.exp(-b * x)

# def model_ref_2(x, a, b, c):
#     """Reference 2: a - (b/sqrt(x)) * exp(-cx)"""
#     return a - (b / np.sqrt(x)) * np.exp(-c * x)

# # Setup
# n_bootstraps = 2000
# target_y = 0.95
# disagreement_pct = ((1 - target_y) / 2) * 100
# max_plot_x = 3000
# x_range_smooth = np.linspace(1, max_plot_x, 300)

# # Storage for results
# results = {
#     'exp': {'curves': [], 'preds': []},
#     'ref1': {'curves': [], 'preds': []},
#     'ref2': {'curves': [], 'preds': []}
# }

# print(f"Starting {n_bootstraps} bootstrap iterations...")

# for k in range(n_bootstraps):
#     boot_sample = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
#     df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
#     X, Y = df_agg['N_datasets'], df_agg['Kendalltau_Correlation']
    
#     # --- FIT MODEL EXP ---
#     try:
#         # Increase upper bound of 'a' slightly to 1.05 to help the optimizer
#         p_exp, _ = curve_fit(model_exp, X, Y, p0=[0.98, 0.3, 0.05], 
#                              bounds=([0.5, 0, 0], [1.05, 1.0, np.inf]))
        
#         # ALWAYS store the curve
#         results['exp']['curves'].append(model_exp(x_range_smooth, *p_exp))
        
#         # ONLY store the prediction if mathematically reachable
#         if p_exp[0] > target_y:
#             # Check for positive log argument
#             log_arg = (p_exp[0] - target_y) / p_exp[1]
#             if log_arg > 0:
#                 req_N_exp= -np.log(log_arg) / p_exp[2]
#                 if 0 < req_N_exp < 10000:
#                     results['exp']['preds'].append(req_N_exp)
#     except: pass

#     # --- FIT MODEL REF 1 (Fixed Asymptote 1.0) ---
#     try:
#         p_r1, _ = curve_fit(model_ref_1, X, Y, p0=[0.5, 0.05], bounds=([0, 0], [10, np.inf]))
#         results['ref1']['curves'].append(model_ref_1(x_range_smooth, *p_r1))
        
#         func = lambda n: model_ref_1(n, *p_r1) - target_y
#         req_N_1 = fsolve(func, x0=50)[0]
#         if 0 < req_N_1 < 10000:
#             results['ref1']['preds'].append(req_N_1)
#     except: pass

#     # --- FIT MODEL REF 2 ---
#     try:
#         p_r2, _ = curve_fit(model_ref_2, X, Y, p0=[0.98, 0.5, 0.05], 
#                              bounds=([0.5, 0, 0], [1.05, 10, np.inf]))
#         results['ref2']['curves'].append(model_ref_2(x_range_smooth, *p_r2))
        
#         if p_r2[0] > target_y:
#             func = lambda n: model_ref_2(n, *p_r2) - target_y
#             req_N_2 = fsolve(func, x0=50)[0]
#             if 0 < req_N_2 < 10000:
#                 results['ref2']['preds'].append(req_N_2)
#     except: pass

# # --- VISUALIZATION ---
# plt.figure(figsize=(14, 8))

# # Plot Observed Data
# df_real_agg = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['mean', 'sem'])
# plt.errorbar(df_real_agg['N_datasets'], df_real_agg['mean'], yerr=df_real_agg['sem'], 
#              fmt='o', color='blue', label='Observed (Mean ± SE)', zorder=10)

# # Plotting Helper
# def plot_model(name, label, color):
#     if len(results[name]['curves']) > 0:
#         curves = np.array(results[name]['curves'])
#         med_line = np.median(curves, axis=0)
        
#         # Calculate median prediction if it exists
#         if len(results[name]['preds']) > 0:
#             med_N = int(np.median(results[name]['preds']))
#             legend_label = f'{label} (N≈{med_N})'
#         else:
#             legend_label = f'{label} (Target Unreachable)'
            
#         plt.plot(x_range_smooth, med_line, color=color, linewidth=2, label=legend_label)
        
#         # Confidence bands
#         low = np.percentile(curves, 2.5, axis=0)
#         high = np.percentile(curves, 97.5, axis=0)
#         plt.fill_between(x_range_smooth, low, high, color=color, alpha=0.1)

# plot_model('exp', 'Exponential (Original)', 'red')
# plot_model('ref1', r'$1 - \frac{a}{\sqrt{x}} e^{-bx}$', 'green')
# plot_model('ref2', r'$a - \frac{b}{\sqrt{x}} e^{-cx}$', 'purple')
# plt.axhline(y=target_y, color='black', linestyle=':', label=f'Target Tau {target_y}')
# plt.axvline(x=n_datasets, color='purple', linestyle='--', alpha=0.6)
# plt.text(n_datasets + 10, 0.6, 
#          f"Current N: {n_datasets}", fontsize=16,
#          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
# plt.xlabel('Number of Datasets (N)', fontsize=16)
# plt.ylabel('Kendall Tau Correlation', fontsize=16)
# plt.title('Comparison of Stability Models', fontsize=18)
# plt.legend(loc='lower right', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.ylim(0.5, 1.02)
# plt.xlim(-5, 500) # Zoomed in to see model differences
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_comparison_{today_date}_2.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# '''
# RESIDUAL ANALYSIS
# '''

# # 1. PREPARE THE DATA
# # Extract observed values
# obs_n = df_real_agg['N_datasets'].values
# obs_tau = df_real_agg['mean'].values
# obs_sem = df_real_agg['sem'].values

# # 2. CALCULATE MEDIAN PARAMETERS FOR EACH MODEL
# # We need to compute the median across all successful bootstrap fits (popt)
# # Note: Ref1 results did not store popt in your loop, so we'll recalculate the median curve
# res_data = {}

# for name in ['exp', 'ref1', 'ref2']:
#     if len(results[name]['curves']) > 0:
#         # Convert list of curve arrays to a 2D matrix (bootstraps, x_range_smooth)
#         curves = np.array(results[name]['curves'])
        
#         # We need the prediction at exactly the OBSERVED N values, not the smooth range
#         # To be precise, we calculate the median prediction for each observed N
#         med_curve = np.median(curves, axis=0)
        
#         # Map the smooth median curve back to the discrete observed N points
#         # interp ensures we compare exactly at the same X-coordinates
#         y_pred = np.interp(obs_n, x_range_smooth, med_curve)
        
#         # Calculate residuals: Actual - Predicted
#         res_data[name] = obs_tau - y_pred

# # 3. PLOTTING THE COMPARISON
# plt.figure(figsize=(14, 7))
# plt.axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

# # Style and Color mapping
# styles = {
#     'exp':  {'label': 'Original Exp', 'color': 'red',    'marker': 'o'},
#     'ref1': {'label': 'Ref 1 (1 - a/√x * e^-bx)', 'color': 'green',  'marker': 's'},
#     'ref2': {'label': 'Ref 2 (a - b/√x * e^-cx)', 'color': 'purple', 'marker': '^'}
# }

# for name, residuals in res_data.items():
#     mae = np.mean(np.abs(residuals))
#     plt.errorbar(
#         obs_n, residuals, yerr=obs_sem, 
#         fmt=styles[name]['marker'], color=styles[name]['color'], 
#         label=f"{styles[name]['label']} (MAE: {mae:.4f})",
#         capsize=3, alpha=0.8, markersize=8
#     )

# # Formatting
# plt.title('Residuals Comparison', fontsize=18)
# plt.xlabel('Number of Datasets (N)', fontsize=16)
# plt.ylabel('Residual (Actual - Predicted)', fontsize=16)
# plt.grid(True, alpha=0.3)
# plt.legend(loc='upper right', fontsize=12)
# plt.tight_layout()


# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_residuals_comparison_{today_date}_2.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# Y_metric_list = ['score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']
# for metric in Y_metric_list:
#     Y_METRIC = metric
#     Y_METRIC_LABELS = {
#         'score_norm': 'Normalized Score',
#         'score_norm_clip': 'Clipped Normalized Score',
#         'score_norm_max1': 'Max-1 Normalized Score',
#         'score_centred': 'Mean-Centred Score'
#     }

#     if 'method_polished' in results.columns:
#         agg_cols = [Y_METRIC, 'inference_time_per_10k', 'run_time_per_10k']
#         group_cols = ['method_polished', 'encoder', 'learner']
#         df_agg = results[results['method'].str.contains('num-str_')].groupby(group_cols)[agg_cols].median().reset_index()

#     HIGHER_SCORE_IS_BETTER = True


#     def get_pareto_front(df, x_col, y_col, maximize_y=True):
#         sorted_df = df.sort_values(x_col, ascending=True)
#         pareto_points = []
#         if maximize_y:
#             current_best_y = -float('inf')
#             for _, row in sorted_df.iterrows():
#                 if row[y_col] > current_best_y:
#                     pareto_points.append(row)
#                     current_best_y = row[y_col]
#         else:
#             current_best_y = float('inf')
#             for _, row in sorted_df.iterrows():
#                 if row[y_col] < current_best_y:
#                     pareto_points.append(row)
#                     current_best_y = row[y_col]
#         return pd.DataFrame(pareto_points)

#     sns.set_style("white")
#     paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']

#     # Font configurations
#     FONT_AXIS_LABEL = 22
#     FONT_TITLE = 24
#     FONT_TICK = 22
#     FONT_LEGEND = 22 

#     ROW_METRICS = ['inference_time_per_10k', 'run_time_per_10k']
#     COL_FACTORS = ['encoder', 'learner']
#     COL_TITLES = ['Encoder', 'Learner'] 
#     ROW_LABELS = ['Inference Time per 10K samples (s)', 'Total Run Time per 10K samples (s)']

#     for row_idx, x_metric in enumerate(ROW_METRICS):
#         # Create a new figure for each metric (Inference vs Run Time)
#         fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
        
#         pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, HIGHER_SCORE_IS_BETTER)
        
#         # --- NEW: Save the Pareto Front data to CSV/LaTeX ---
#         today_date = time.strftime("%Y-%m-%d")
#         base_name = f"pareto_data_{Y_METRIC}_{x_metric}_{today_date}"
        
#         # Save the Pareto front points specifically
#         pareto_df.to_csv(f'/data/parietal/store4/soda/gblayer/salts/results_tables/{base_name}_front.csv', index=False)
        
#         # Save as LaTeX table string for quick copy-paste
#         # This selects the key columns for your paper
#         latex_cols = ['method_polished', 'encoder', 'learner', x_metric, Y_METRIC]
#         pareto_df[latex_cols].to_latex(
#             f'/data/parietal/store4/soda/gblayer/salts/results_tables/{base_name}_front.tex', 
#             index=False, 
#             float_format="%.4f",
#             caption=f"Pareto Front points for {x_metric} vs {Y_METRIC_LABELS[Y_METRIC]}",
#             label=f"tab:pareto_{x_metric}"
#         )
#         # ---------------------------------------------------

#         for col_idx, factor in enumerate(COL_FACTORS):
#             ax = axes[col_idx]
            
#             # Scatter Plot & Pareto Front
#             sns.scatterplot(
#                 data=df_agg, x=x_metric, y=Y_METRIC, hue=factor, style=factor,
#                 s=180, alpha=0.8, ax=ax, palette=paper_palette
#             )
#             ax.step(
#                 pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
#                 linestyle='--', color='black', linewidth=1.5, zorder=0
#             )
#             ax.set_xscale('log')
#             ax.tick_params(axis='both', which='major', labelsize=12)
            
#             # 1. Spacing between Title and Plot
#             ax.set_title(COL_TITLES[col_idx], fontsize=14, pad=10) 
            
#             ax.set_xlabel('')
#             # 2. Legend Formatting
#             # handles, labels = ax.get_legend_handles_labels()
            
#             # Custom column count: 2 for Encoder, 3 for Learner
#             # num_cols = 5 #if factor == 'encoder' else 3
            
#             # ax.legend(
#             #     handles, labels,
#             #     loc='lower center', 
#             #     bbox_to_anchor=(0.5, 1.05), # Positioned above plot
#             #     ncol=num_cols,
#             #     frameon=False,
#             #     fontsize=FONT_LEGEND,
#             #     handletextpad=0.5,          # Space between icon and text
#             #     columnspacing=1.5,          # Increased space between names in legend
#             #     labelspacing=0.3,
#             #     markerscale=1.5
#             # )
            
#             # 3. Increase space between Y label and Y axis
#             if col_idx == 0:
#                 ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=14, labelpad=10)
#             else:
#                 ax.set_ylabel('')

#             sns.despine(ax=ax) 

#         # Global X-Axis Label repeated once per figure
#         fig.text(0.5, -0.02, f'{ROW_LABELS[row_idx]} (Log Scale)', 
#                 ha='center', fontsize=14)

#         # Adjust layout to accommodate multi-row legends and global label
#         plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.15, left=0.12, right=0.95)
        
#         today_date = time.strftime("%Y-%m-%d")
#         format = 'pdf'
#         PIC_NAME = f'comparative_pareto_optimality_plot_10Ksample_scale_{Y_METRIC}_{x_metric}_{today_date}.{format}'
#         plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
#         plt.show()


'''
PARETO OPTIMALITY PLOT
'''

# # Settings
# HIGHER_SCORE_IS_BETTER = True  # Set to False if Y-axis is RMSE/Loss
# X_METRIC = 'inference_time'
# Y_METRIC = 'score_norm'

# # 2. Aggregation
# # We calculate the median time and mean score for each method
# agg_df = results.groupby('method_polished')[[X_METRIC, Y_METRIC]].median().reset_index()

# # 3. Identify the Pareto Front
# # Sort by speed (fastest to slowest)
# agg_df = agg_df.sort_values(X_METRIC, ascending=True)

# pareto_front = []
# if HIGHER_SCORE_IS_BETTER:
#     # Logic: Keep point if it has a higher score than all previous (faster) points
#     current_best_score = -float('inf')
#     for index, row in agg_df.iterrows():
#         if row[Y_METRIC] > current_best_score:
#             pareto_front.append(row)
#             current_best_score = row[Y_METRIC]
# else:
#     # Logic: Keep point if it has a lower error than all previous (faster) points
#     current_best_score = float('inf')
#     for index, row in agg_df.iterrows():
#         if row[Y_METRIC] < current_best_score:
#             pareto_front.append(row)
#             current_best_score = row[Y_METRIC]

# pareto_df = pd.DataFrame(pareto_front)

# # 4. Visualization
# plt.figure(figsize=(14, 10))
# sns.set_style("whitegrid")

# # A. Plot all methods
# sns.scatterplot(
#     data=agg_df, 
#     x=X_METRIC, 
#     y=Y_METRIC, 
#     hue='method_polished', 
#     style='method_polished', 
#     s=150, # Marker size
#     alpha=0.7,
#     legend=False # Hiding legend if too many methods, relying on labels instead
# )

# # B. Draw the Pareto Front (Stepped Line)
# # 'where="post"' draws the line horizontally then vertically, 
# # representing that you maintain the previous performance until you pay the cost for the next tier.
# plt.step(
#     pareto_df[X_METRIC], 
#     pareto_df[Y_METRIC], 
#     where='post', 
#     linestyle='--', 
#     color='black', 
#     linewidth=2, 
#     label='Pareto Front'
# )

# # C. Add Labels to Pareto Points
# # We only label points on the frontier to avoid clutter
# for line in range(0, pareto_df.shape[0]):
#     plt.text(
#         pareto_df[X_METRIC].iloc[line], 
#         pareto_df[Y_METRIC].iloc[line], 
#         pareto_df['method_polished'].iloc[line], 
#         horizontalalignment='left', 
#         size='medium', 
#         color='black', 
#         weight='semibold'
#     )

# # D. Formatting
# plt.xscale('log') # Log scale is standard for time/latency plots
# plt.title('Pareto Optimality: Performance vs. Inference Time', fontsize=16)
# plt.xlabel('Inference Time (s) [Log Scale]', fontsize=12)
# plt.ylabel('Normalized Score (Higher is Better)', fontsize=12)
# plt.grid(True, which="both", ls="-", alpha=0.5)
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'pareto_optimality_plot_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

'''
COMPARATIVE PARETO PLOTS
(style retrieve merge predict)
'''

# if 'method_polished' in results.columns:
#     agg_cols = ['score_norm', 'inference_time', 'run_time']
#     group_cols = ['method_polished', 'encoder', 'learner']
#     df_agg = results.groupby(group_cols)[agg_cols].median().reset_index()

# HIGHER_SCORE_IS_BETTER = True
# Y_METRIC = 'score_norm'
# ROW_METRICS = ['inference_time', 'run_time']
# ROW_LABELS = ['Inference Time (s)', 'Total Run Time (s)']
# COL_FACTORS = ['encoder', 'learner']
# COL_TITLES = ['Encoder', 'Learner'] 
# paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']

# def get_pareto_front(df, x_col, y_col, maximize_y=True):
#     sorted_df = df.sort_values(x_col, ascending=True)
#     pareto_points = []
#     if maximize_y:
#         current_best_y = -float('inf')
#         for _, row in sorted_df.iterrows():
#             if row[y_col] > current_best_y:
#                 pareto_points.append(row)
#                 current_best_y = row[y_col]
#     else:
#         current_best_y = float('inf')
#         for _, row in sorted_df.iterrows():
#             if row[y_col] < current_best_y:
#                 pareto_points.append(row)
#                 current_best_y = row[y_col]
#     return pd.DataFrame(pareto_points)


# sns.set_style("white") # Clean background
# paper_palette = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e'] # Green, Blue, Red, Orange

# # Font configurations for Paper
# FONT_AXIS_LABEL = 18
# FONT_TITLE = 22
# FONT_TICK = 16
# FONT_LEGEND = 16

# fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharey=True, sharex=False)

# for row_idx, x_metric in enumerate(ROW_METRICS):
    
#     pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, HIGHER_SCORE_IS_BETTER)
    
#     for col_idx, factor in enumerate(COL_FACTORS):
#         ax = axes[row_idx, col_idx]
        
#         # A. SCATTER PLOT
#         sns.scatterplot(
#             data=df_agg, x=x_metric, y=Y_METRIC, hue=factor, style=factor,
#             s=150, alpha=0.8, ax=ax, palette=paper_palette, legend='brief'
#         )
        
#         # B. PARETO LINE
#         ax.step(
#             pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
#             linestyle='--', color='black', linewidth=2, zorder=0
#         )
        
#         # C. TICKS & GRID
#         ax.set_xscale('log')
#         ax.grid(False)
#         ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
        
#         # D. REMOVE DEFAULT LABELS (We will add shared ones later)
#         ax.set_xlabel('')
#         ax.set_ylabel('')

#         # E. LEGEND & TITLES (Top Row Only)
#         if row_idx == 0:
#             ax.set_title(COL_TITLES[col_idx], fontsize=FONT_TITLE, pad=65) 
            
#             handles, labels = ax.get_legend_handles_labels()
#             ax.legend(
#                 handles, labels,
#                 loc='lower center', 
#                 bbox_to_anchor=(0.5, 1.02),
#                 ncol=len(labels),
#                 frameon=False,
#                 fontsize=FONT_LEGEND,
#                 handletextpad=0.5,
#                 columnspacing=1.5,
#                 markerscale=1.5
#             )
#         else:
#             if ax.get_legend() is not None:
#                 ax.get_legend().remove()

# fig.text(0.04, 0.5, 'Normalized Score', va='center', rotation='vertical', fontsize=FONT_AXIS_LABEL)
# fig.text(0.5, 0.5, 'Inference Time (s) [Log Scale]', ha='center', fontsize=FONT_AXIS_LABEL)
# fig.text(0.5, 0.1, 'Total Run Time (s) [Log Scale]', ha='center', fontsize=FONT_AXIS_LABEL)
# plt.subplots_adjust(top=0.82, bottom=0.15, left=0.12, right=0.92, hspace=0.5, wspace=0.15)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'comparative_pareto_optimality_plot_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# # 6. Visualization: Significance Heatmap
# plt.figure(figsize=(10, 8))

# # Mask the upper triangle to reduce visual clutter (since the matrix is symmetric)
# # Note: For signed-rank tests, results are symmetric.
# mask = p_values.isnull()

# sns.heatmap(
#     p_values, 
#     annot=True, 
#     fmt=".3f", 
#     cmap="viridis_r", # Purple = High p-value (No diff), Yellow = Low p-value (Sig diff)
#     cbar_kws={'label': 'P-Value'},
#     linewidths=0.5,
#     vmin=0, vmax=0.05 # Highlight significant range
# )

# # Remove the axis labels "method_polished"
# plt.ylabel('')  # Removes Y-axis label
# plt.xlabel('')  # Removes X-axis label

# plt.title('Pairwise Wilcoxon Test P-Values (Holm Corrected)', fontsize=16)
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'pairwise_wilcoxon_test_pvalues_holm_corrected_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

'''
Critical Difference Diagram
'''

# If you want the classic "lines connecting non-significant groups" diagram:
# Note: This requires the 'critical_difference_diagram' function often found in 
# recent versions of scikit-posthocs or implemented manually via orange-data-mining logic.
# try:
#     plt.figure(figsize=(6, 6))
#     sp.critical_difference_diagram(mean_ranks, p_values, text_h_margin=0.5)
#     today_date = time.strftime("%Y-%m-%d")
#     format = 'pdf'
#     PIC_NAME = f'critical_difference_diagram_{today_date}.{format}'
#     plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
#     plt.show()
# except AttributeError:
#     print("Critical Difference Diagram function not available in this version of scikit-posthocs.")

'''
Critical difference diagram (Gioia custom implementation)
'''

# def print_clique_details(cliques, mean_ranks, sig_matrix):
#     """
#     Prints the start and end points of each clique (black line),
#     plus the clique length and the p-value between the start and end methods.
#     """
#     # 1. Sort the list of cliques so they print in order (left to right on the plot)
#     # We use the minimum rank in the clique to determine the order
#     sorted_cliques = sorted(cliques, key=lambda c: min([mean_ranks[m] for m in c]))
    
#     print("-" * 110)
#     print(f"{'CLIQUE':<8} | {'START (Left)':<30} | {'END (Right)':<30} | {'SIZE':<5} | {'P-VAL (Start vs End)'}")
#     print("-" * 110)
    
#     for i, clique in enumerate(sorted_cliques):
#         # Sort methods within this specific clique by their rank
#         members = sorted(clique, key=lambda m: mean_ranks[m])
        
#         # The line connects the first (best) and last (worst) method in this sorted list
#         start_method = members[0]
#         end_method = members[-1]
        
#         start_rank = mean_ranks[start_method]
#         end_rank = mean_ranks[end_method]

#         try:
#             p_val = sig_matrix.loc[start_method, end_method]
#         except KeyError:
#             p_val = np.nan
        
#         # Clean up names for printing (optional)
#         s_name = start_method.replace('\n', ' ')
#         e_name = end_method.replace('\n', ' ')
        
#         line_str = (f"Line {i+1:<3} | {s_name} ({start_rank:.1f})".ljust(41) + 
#                     f"| {e_name} ({end_rank:.1f})".ljust(33) + 
#                     f"| {len(members):<5} | {p_val:.4f}")
#         print(line_str)

#     print("-" * 110)

# def get_cliques(p_values_df, mean_ranks, alpha=0.05):
#     """
#     Calculates cliques based on p-values.
#     Removes strictly duplicate (Start, End) couples, but KEEPS subsets 
#     (e.g., keeps 6.1->6.7 even if 6.0->6.7 exists).
#     """
#     sorted_methods = sorted(mean_ranks.keys(), key=lambda x: mean_ranks[x])
#     n = len(sorted_methods)
#     raw_cliques = []

#     # 1. Greedy approach (Find all continuous non-significant groups)
#     i = 0
#     while i < n:
#         j = i + 1
#         current_clique = [sorted_methods[i]]
#         while j < n:
#             p_val = p_values_df.loc[sorted_methods[i], sorted_methods[j]]
#             if p_val > alpha:
#                 current_clique.append(sorted_methods[j])
#                 j += 1
#             else:
#                 break
#         if len(current_clique) > 1:
#             raw_cliques.append(current_clique)
#         i += 1

#     # 2. Deduplicate based on Unique Couples (Start, End)
#     # This ensures we don't have two identical lines for the EXACT SAME pair,
#     # but allows different pairs (e.g. 6.0->6.7 vs 6.1->6.7) to coexist.
#     unique_couples = {}
    
#     for clique in raw_cliques:
#         members = sorted(clique, key=lambda m: mean_ranks[m])
#         start_method = members[0]
#         end_method = members[-1]
        
#         couple_key = (start_method, end_method)
        
#         # Keep the largest group for this specific start-end pair
#         if couple_key not in unique_couples:
#             unique_couples[couple_key] = clique
#         else:
#             if len(clique) > len(unique_couples[couple_key]):
#                 unique_couples[couple_key] = clique

#     # 3. Return all unique couples (Removed the 'is_subset' filtering)
#     final_cliques = list(unique_couples.values())
    
#     return final_cliques
# # ---------------------------------------------------------
# # 2. MAIN PLOTTING FUNCTION (Adapted)
# # ---------------------------------------------------------
# def critical_difference_diagram(
#     ranks: Union[dict, Series],
#     sig_matrix: DataFrame,
#     *,
#     ax: plt.Axes = None,
#     label_fmt_left: str = "{label} [{rank:.1f}]",
#     label_fmt_right: str = "[{rank:.1f}] {label}",
#     color_palette: Union[Dict[str, str], List] = None,
#     text_h_margin: float = 0.2,
#     label_props: dict = None,
#     # New parameters for control
#     width: float = 12, 
#     height: float = None,
#     fontsize: float = 16,
#     linewidth: float = 2.5
# ) -> Dict[str, list]:
    
#     # 1. DATA PREPARATION
#     if isinstance(ranks, dict):
#         ranks = Series(ranks)
        
#     # Sort ranks (1 is best)
#     ranks = ranks.sort_values()
#     methods = ranks.index.tolist()
    
#     cliques = get_cliques(sig_matrix, ranks, alpha=0.05)

#     # Sort for consistent display
#     cliques.sort(key=lambda x: ranks[x].min())

#     print_clique_details(cliques, ranks, sig_matrix)

#     # 2. SETUP PLOT
#     if ax is None:
#         # Dynamic height if not provided
#         if height is None:
#             height = 3 + len(methods) * 0.4
#         fig, ax = plt.subplots(figsize=(width, height))
        
#     # Default styling
#     if color_palette is None:
#         # Bright high-contrast palette
#         colors_list = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd', 
#                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#         color_palette = {m: colors_list[i % len(colors_list)] for i, m in enumerate(methods)}
    
#     label_props = {"weight": "bold", "fontsize": fontsize, **(label_props or {})}

#     # 3. SPLIT DATA (LEFT vs RIGHT)
#     n = len(methods)
#     mid = n // 2
    
#     # Split: Top half (Best ranks) -> Left, Bottom half (Worst ranks) -> Right
#     left_methods = methods[:mid]
#     right_methods = methods[mid:]
    
#     min_axis, max_axis = 1, 20
#     ax.plot([min_axis, max_axis], [0, 0], color='black', linewidth=3, zorder=1)
    
#     # Ticks
#     for r in range(min_axis, max_axis + 1):
#         ax.plot([r, r], [-0.02, 0.02], color='black', linewidth=2)
#         ax.text(r, 0.05, str(r), ha='center', va='bottom', fontweight='bold', fontsize=fontsize)

#     # 5. LAYOUT LOGIC (The "Waterfall")
#     # To avoid intersection: Outer ranks (1, 10) must be at TOP (short lines). 
#     # Inner ranks (5, 6) must be at BOTTOM (long lines).
    
#     # Anchor text exactly at the axis limits
#     x_text_left = min_axis 
#     x_text_right = max_axis
    
#     # Vertical spacing
#     y_start = -2
#     y_step = 1  # Controls distance between names
    
#     artists = {'markers': [], 'lines': [], 'texts': []}

#     # --- LEFT STACK (Best methods) ---
#     # Sort: Outer (Rank 1) -> Inner (Rank 5). 
#     # Outer must be plotted first (Top). So we iterate ranks Ascending (1..5).
#     # Since left_methods is already sorted [1..5], we just iterate.
#     for i, method in enumerate(left_methods):
#         rank = ranks[method]
#         color = color_palette[method]
        
#         y_pos = y_start - (i * y_step)
        
#         # Draw
#         # 1. Dot on axis
#         ax.scatter(rank, 0, color=color, s=150, zorder=4, edgecolors='white', linewidth=1.5)
#         # 2. Vertical Drop
#         ax.plot([rank, rank], [0, y_pos], color=color, linewidth=linewidth, zorder=2)
#         # 3. Horizontal to Text
#         ax.plot([rank, x_text_left], [y_pos, y_pos], color=color, linewidth=linewidth, zorder=2)
#         # 4. Text
#         label_str = label_fmt_left.format(label=method, rank=rank)
#         ax.text(x_text_left, y_pos, label_str, 
#                 ha='left', va='center', color=color, **label_props, zorder=30,
#                 bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=5))

#     # --- RIGHT STACK (Worst methods) ---
#     # Sort: Outer (Rank 10) -> Inner (Rank 6).
#     # Outer must be plotted first (Top). So we iterate ranks Descending (10..6).
#     # right_methods is [6..10]. We need to reverse it.
#     for i, method in enumerate(reversed(right_methods)):
#         rank = ranks[method]
#         color = color_palette[method]
        
#         y_pos = y_start - (i * y_step)
        
#         # Draw
#         ax.scatter(rank, 0, color=color, s=150, zorder=4, edgecolors='white', linewidth=1.5)
#         ax.plot([rank, rank], [0, y_pos], color=color, linewidth=linewidth, zorder=2)
#         ax.plot([rank, x_text_right], [y_pos, y_pos], color=color, linewidth=linewidth, zorder=2)
        
#         label_str = label_fmt_right.format(label=method, rank=rank)
#         ax.text(x_text_right, y_pos, label_str, 
#                 ha='right', va='center', color=color, **label_props, zorder=30,
#                 bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=5))

#     # 6. DRAW CLIQUES (Simple Stacking - One Level Per Clique)
#     # This guarantees 1 distinct black line for every clique found.
    
#     # Sort cliques by their starting rank so the ladder looks orderly
#     cliques.sort(key=lambda x: ranks[x].min()) 
    
#     # Start drawing just below the axis
#     clique_y_start = -0.5 
#     clique_step = 0.15   # Vertical distance between black lines
    
#     for i, clique in enumerate(cliques):
#         c_ranks = ranks[clique]
#         min_r = c_ranks.min()
#         max_r = c_ranks.max()
        
#         # Assign a unique Y level for every single clique
#         y_c = clique_y_start - (i * clique_step)
        
#         # Draw thick black bar
#         ax.plot([min_r, max_r], [y_c, y_c], color='black', linewidth=4, solid_capstyle='butt', zorder=20)
        
#     # DYNAMIC Y-LIMITS UPDATE
#     # Ensure the plot extends far enough down to show the lowest black line
#     lowest_black_line = clique_y_start - (len(cliques) * clique_step)
    
#     # Calculate the bottom of the text columns
#     text_bottom = y_start - (max(len(left_methods), len(right_methods)) * y_step) - 0.5
    
#     # Set the limit to whichever is lower (text or black lines)
#     bottom_limit = min(text_bottom, lowest_black_line - 0.2)
    
#     ax.set_ylim(bottom_limit, 1.0)

#     # Final Adjustments
#     ax.axis('off')
#     # Set Y limits to include text and bars
#     # ax.set_ylim(y_start - (max(len(left_methods), len(right_methods)) * y_step) - 0.5, 
#     #             clique_y_start + (len(cliques) * clique_step) + 0.2)
#     ax.set_ylim(bottom_limit, 1.0)
#     # ax.set_xlim(0, 11)
#     padding = 1.0 
#     ax.set_xlim(min_axis - padding, max_axis + padding)
    
#     return artists

# # ---------------------------------------------------------
# # EXAMPLE USAGE
# # ---------------------------------------------------------
# # Assume 'mean_ranks' and 'p_values' are already loaded from your dataframe
# critical_difference_diagram(mean_ranks, p_values, width=40, height=16,fontsize=27)
# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'critical_difference_diagram_custom_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# # We fix 'a' to 1.0 because a perfect benchmark must yield correlation=1.0
# def learning_curve_fixed(x, a, b, c, d):
#     return a - b * np.power(x + d, -c)

# n_bootstraps = 2000  # High number for smooth histograms
# prediction_results = []
# curve_fits = []

# # Create a smooth X-axis for plotting lines
# max_plot_x = 3000
# x_range_smooth = np.linspace(5, max_plot_x, 200)

# print(f"Starting {n_bootstraps} bootstrap iterations...")

# for k in range(n_bootstraps):
#     # A. RESAMPLING (The Magic Step)
#     # For each N, we sample with replacement from the scores you already observed.
#     # This simulates: "What if the experiment had slight random variations?"
#     boot_sample = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
    
#     # B. Compute the Mean/SEM for this synthetic reality
#     df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
    
#     # C. Fit the Curve
#     try:
#         # We weigh the fit by the variance (SEM) if available, or just standard fit
#         popt, _ = curve_fit(
#             learning_curve_fixed,
#             df_agg['N_datasets'],
#             df_agg['Kendalltau_Correlation'],
#             p0 = [0.98, 10.0, 1.2, 22.0], 
#             bounds = ([0.9, 0.0, 0.5, 5.0], [1.0, np.inf, 3.0, 100.0]),
#             maxfev=2000
#         )
        
#         a_fit, b_fit, c_fit, d_fit = popt

        
#         # D. Extrapolate to Target (0.95)
#         target_y = 0.95
#         if a_fit > 0 and b_fit > 0 and c_fit > 0 and d_fit >= 0:
#             required_N = (((a_fit - target_y) / b_fit) ** (-1 / c_fit)) - d_fit
            
#             # Filter out crazy outliers (e.g., flat lines predicting N=1,000,000)
#             if required_N < 10000:
#                 prediction_results.append(required_N)
#                 # Save the curve line for the plot
#                 curve_fits.append(learning_curve_fixed(x_range_smooth, *popt))
                
#     except (RuntimeError, ValueError):
#         continue

# # ---------------------------------------------------------
# # 3. CALCULATE STATISTICS
# # ---------------------------------------------------------
# predictions = np.array(prediction_results)
# median_N = np.median(predictions)
# ci_lower = np.percentile(predictions, 2.5)  # Lower 95% bound
# ci_upper = np.percentile(predictions, 97.5) # Upper 95% bound

# print(f"--- Results ---")
# print(f"Median Datasets Required: {int(median_N)}")
# print(f"95% Confidence Interval:  [{int(ci_lower)} - {int(ci_upper)}]")

# # ---------------------------------------------------------
# # 4. VISUALIZATION
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7))

# # A. Plot the Uncertainty Band (The "Cloud" of curves)
# # We compute the 2.5% and 97.5% boundaries of the curves at every X point
# curves_array = np.array(curve_fits)
# lower_bound_line = np.percentile(curves_array, 2.5, axis=0)
# upper_bound_line = np.percentile(curves_array, 97.5, axis=0)
# median_line = np.median(curves_array, axis=0)

# plt.fill_between(x_range_smooth, lower_bound_line, upper_bound_line, color='red', alpha=0.15, label='95% Confidence Band')
# plt.plot(x_range_smooth, median_line, 'r--', linewidth=2, label=f'Median Fit (N ≈ {int(median_N)})')

# # B. Plot the Observed Data
# # Aggregate the original data for plotting the blue dots
# df_real_agg = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['mean', 'sem'])
# plt.errorbar(
#     df_real_agg.index, 
#     df_real_agg['mean'], 
#     yerr=df_real_agg['sem'], 
#     fmt='o', color='blue', ecolor='blue', capsize=4, 
#     label='Observed (Mean ± SE)', zorder=5
# )

# # C. Formatting
# plt.axhline(y=0.95, color='green', linestyle=':', linewidth=2, label='Stability Target (0.95)')
# plt.axvline(x=median_N, color='k', linestyle='--', alpha=0.4)

# plt.text(median_N + 50, 0.8, 
#          f"Median N: {int(median_N)}\n95% CI: [{int(ci_lower)} - {int(ci_upper)}]", 
#          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# plt.xlabel('Number of Datasets (N)', fontsize=12)
# plt.ylabel('Kendall Tau Correlation', fontsize=12)
# plt.title('Required Benchmark Size', fontsize=14)
# plt.legend(loc='lower right', fontsize=11)
# plt.grid(True, alpha=0.3)
# plt.ylim(0.5, 1.05) # Focus on the top half
# plt.xlim(-25, median_N * 1.5) # Dynamic X-axis

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_curvefit_{today_date}_2.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# '''
# KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
# the benchmark to converge to the same ranking?
# BOOTSTRAPPED MICHAELIS-MENTEN FITTING (EXTRAPOLATION)
# '''

# def michaelis_menten(x, tau_inf, K):
#     return tau_inf * (x / (x + K))

# n_bootstraps = 2000
# prediction_results = []
# curve_fits = []
# max_plot_x = 3000
# x_range_smooth = np.linspace(1, max_plot_x, 500)
# target_y = 0.95

# print(f"Starting {n_bootstraps} bootstrap iterations...")

# for k in range(n_bootstraps):
#     boot_sample = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
#     df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
    
#     try:
#         # Guesses: tau_inf near 0.98, K=2.0 (stability is high quickly)
#         # CRITICAL: tau_inf must be > target_y to have a mathematical solution
#         popt, _ = curve_fit(
#         michaelis_menten,
#         df_agg['N_datasets'],
#         df_agg['Kendalltau_Correlation'],
#         p0=[0.9, 2.0], # Start guess closer to observed ~0.89
#         bounds=([target_y + 0.001, 0.0], [1.0, 500.0]), # tau_inf is bounded by 1.0 correlation
#         maxfev=5000
#     )
    
#         tau_inf_fit, K_fit = popt
        
#         # 2. Correct Extrapolation Formula
#         # Solving target_y = tau_inf * (N / (N + K)) for N:
#         if tau_inf_fit > target_y:
#             # Curve reaches the target
#             required_N = (target_y * K_fit) / (tau_inf_fit - target_y)
#             # Store results...
#         else:
#             # Curve saturates before reaching the target
#             # You can record this as np.inf or simply ignore it
#             pass
            
#         if 0 < required_N < 10000:
#             prediction_results.append(required_N)
#             curve_fits.append(michaelis_menten(x_range_smooth, *popt))
                
#     except (RuntimeError, ValueError):
#         continue

# # --- SAFE STATISTICS ---
# predictions = np.array(prediction_results)

# if len(predictions) == 0:
#     print("\nCRITICAL ERROR: Zero successful fits.")
#     print(f"The data is saturating below your target of {target_y}.")
#     print("Try lowering the Stability Target to 0.90 to verify.")
# else:
#     median_N = np.median(predictions)
#     ci_lower = np.percentile(predictions, 2.5)
#     ci_upper = np.percentile(predictions, 97.5)
    
#     print(f"--- Results ---")
#     print(f"Median Datasets Required: {int(median_N)}")
#     print(f"95% Confidence Interval:  [{int(ci_lower)} - {int(ci_upper)}]")

# # ---------------------------------------------------------
# # 4. VISUALIZATION
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7))

# # A. Plot the Uncertainty Band (The "Cloud" of curves)
# # We compute the 2.5% and 97.5% boundaries of the curves at every X point
# curves_array = np.array(curve_fits)
# lower_bound_line = np.percentile(curves_array, 2.5, axis=0)
# upper_bound_line = np.percentile(curves_array, 97.5, axis=0)
# median_line = np.median(curves_array, axis=0)

# plt.fill_between(x_range_smooth, lower_bound_line, upper_bound_line, color='red', alpha=0.15, label='95% Confidence Band')
# plt.plot(x_range_smooth, median_line, 'r--', linewidth=2, label=f'Median Fit (N ≈ {int(median_N)})')

# # B. Plot the Observed Data
# # Aggregate the original data for plotting the blue dots
# df_real_agg = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['mean', 'sem'])
# plt.errorbar(
#     df_real_agg.index, 
#     df_real_agg['mean'], 
#     yerr=df_real_agg['sem'], 
#     fmt='o', color='blue', ecolor='blue', capsize=4, 
#     label='Observed (Mean ± SE)', zorder=5
# )

# # C. Formatting
# plt.axhline(y=0.95, color='green', linestyle=':', linewidth=2, label='Stability Target (0.95)')
# plt.axvline(x=median_N, color='k', linestyle='--', alpha=0.4)

# plt.text(median_N + 50, 0.8, 
#          f"Median N: {int(median_N)}\n95% CI: [{int(ci_lower)} - {int(ci_upper)}]", 
#          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# plt.xlabel('Number of Datasets (N)', fontsize=12)
# plt.ylabel('Kendall Tau Correlation', fontsize=12)
# plt.title('Required Benchmark Size - Michaelis-Menten approximation', fontsize=14)
# plt.legend(loc='lower right', fontsize=11)
# plt.grid(True, alpha=0.3)
# plt.ylim(0.5, 1.05) # Focus on the top half
# plt.xlim(-25, median_N * 1.5) # Dynamic X-axis

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_michaelis_menten_{today_date}_2.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets do I need for 
the benchmark to converge to the same ranking?
BOOTSTRAPPED EXPONENTIAL SATURATION FITTING (EXTRAPOLATION)
'''

# def exponential_saturation(x, a, b, c):
#     """
#     a: The asymptotic maximum (ceiling)
#     b: The gap between the start and the ceiling
#     c: The growth rate
#     """
#     return a - b * np.exp(-c * x)

# n_bootstraps = 2000
# prediction_results = []
# curve_fits = []
# all_popt = []
# max_plot_x = 3000
# x_range_smooth = np.linspace(1, max_plot_x, 300)
# target_y = 0.95

# print(f"Starting {n_bootstraps} bootstrap iterations with Exponential model...")

# for k in range(n_bootstraps):
#     # A. RESAMPLING
#     boot_sample = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
#     df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
    
#     # B. Fit the Curve
#     try:
#         # Guesses: a=0.98 (ceiling), b=0.3 (starting at ~0.68), c=0.05 (rate)
#         # Bounds: a must be at least the target 0.95 to have a solution
#         popt, _ = curve_fit(
#             exponential_saturation,
#             df_agg['N_datasets'],
#             df_agg['Kendalltau_Correlation'],
#             p0=[0.98, 0.3, 0.05],
#             bounds=([0.5, 0, 0], [1.0, 1.0, np.inf]),
#             maxfev=5000
#         )
        
#         a_fit, b_fit, c_fit = popt

#         # We check if a_fit > target_y, otherwise no solution exists
#         if a_fit > target_y:
#             ratio = (a_fit - target_y) / b_fit
#             if ratio > 0:
#                 required_N = -np.log(ratio) / c_fit
                
#                 if 0 < required_N < 10000:
#                     prediction_results.append(required_N)
#                     curve_fits.append(exponential_saturation(x_range_smooth, *popt))
#                     all_popt.append(popt)
#     except (RuntimeError, ValueError):
#         continue

# # ---------------------------------------------------------
# # 3. CALCULATE STATISTICS
# # ---------------------------------------------------------
# predictions = np.array(prediction_results)

# if len(predictions) == 0:
#     print(f"\nCRITICAL ERROR: No successful fits reached {target_y}.")
#     print("This means the data saturates below your target.")
# else:
#     # Correct calculation of the Median Asymptote using all_popt
#     all_popt_array = np.array(all_popt)
#     median_asymptote = np.median(all_popt_array[:, 0]) # Column 0 is 'a'
    
#     median_N = np.median(predictions)
#     ci_lower = np.percentile(predictions, 2.5)
#     ci_upper = np.percentile(predictions, 97.5)

#     print(f"\n--- Results ---")
#     print(f"Maximum achievable stability (Median Asymptote): {median_asymptote:.3f}")
#     print(f"Median Datasets Required for {target_y}: {int(median_N)}")
#     print(f"95% Confidence Interval: [{int(ci_lower)} - {int(ci_upper)}]")


# # ---------------------------------------------------------
# # 4. VISUALIZATION
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7))

# # Uncertainty Band
# if len(curve_fits) > 0:
#     curves_array = np.array(curve_fits)
#     lower_bound_line = np.percentile(curves_array, 2.5, axis=0)
#     upper_bound_line = np.percentile(curves_array, 97.5, axis=0)
#     median_line = np.median(curves_array, axis=0)

#     plt.fill_between(x_range_smooth, lower_bound_line, upper_bound_line, color='red', alpha=0.15, label='95% Confidence Band')
#     plt.plot(x_range_smooth, median_line, 'r--', linewidth=2, label=f'Median Fit (N ≈ {int(median_N)})')


# # Important fix: plot against 'N_datasets' column values, not the index
# df_real_agg = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['mean', 'sem'])
# plt.errorbar(
#     df_real_agg['N_datasets'], 
#     df_real_agg['mean'], 
#     yerr=df_real_agg['sem'], 
#     fmt='o', color='blue', ecolor='blue', capsize=4, 
#     label='Observed (Mean ± SE)', zorder=5
# )

# # Formatting
# plt.axhline(y=target_y, color='green', linestyle=':', linewidth=2, label=f'Target ({target_y})')
# if len(predictions) > 0:
#     plt.axvline(x=median_N, color='k', linestyle='--', alpha=0.4)
#     plt.text(median_N - 70, 0.57, 
#              f"95% CI: [{int(ci_lower)} - {int(ci_upper)}]", fontsize=16,
#              bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
# # add vertical line withe our number of datasets: n_datasets
# plt.axvline(x=n_datasets, color='purple', linestyle='--', alpha=0.6)
# plt.text(n_datasets + 20, 0.6, 
#          f"Current N: {n_datasets}", fontsize=16,
#          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
# plt.xlabel('Number of Datasets (N)', fontsize=16)
# plt.ylabel('Kendall Tau Correlation', fontsize=16)
# plt.title('Required Benchmark Size (Exponential Saturation)', fontsize=16)
# plt.legend(loc='lower right', fontsize=16)
# plt.grid(True, alpha=0.3)
# plt.ylim(0.5, 1.05) 
# plt.xlim(-5, median_N * 1.5 if len(predictions) > 0 else 100)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_{today_date}_2.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()




# ==========================================
# 1. MANUAL CONFIGURATION (EDIT HERE)
# ==========================================

# A. DEFINE THE LABELS TO WRITE (Top to Bottom Order)
# Keys must match the Method Name and Rank exactly.
# manual_annotations = {
#     # List 1: TargetEncoder - XGBoost (Str) @ Rank 5
#     ('TargetEncoder - XGBoost\n(Str)', 5): [
#         'worldbankfinancesone (15)', 
#         'energydata.info (3)', 
#         'OHCA (6)', 
#         'Medicaid (5)', 
#         'HRSA (8)'
#     ],
    
#     # List 2: Tf-Idf - Ridge (Str) @ Rank 4
#     ('Tf-Idf - Ridge\n(Str)', 4): [
#         'worldbankfinancesone (15)', 
#         'HRSA (8)', 
#         'Medicaid (5)', 
#         'OHCA (6)', 
#         'energydata.info (3)'
#     ],

#     # List 3: Tf-Idf - Ridge (Str) @ Rank 6
#     ('Tf-Idf - Ridge\n(Str)', 6): ['fda (12)'],

#     # List 4: Tf-Idf - XGBoost (Num) @ Rank 5
#     ('Tf-Idf - XGBoost\n(Num)', 5): ['fda (12)'],

#     # List 5: Tf-Idf - XGBoost (Num) @ Rank 7
#     ('Tf-Idf - XGBoost\n(Num)', 7): ['European-Medicines-Agency (3)'],

#     # List 6: Tf-Idf - XGBoost (Num) @ Rank 8
#     ('Tf-Idf - XGBoost\n(Num)', 8): ['HIFLD (18)'],

#     # List 7: Tf-Idf - Ridge (Num+Str) @ Rank 6
#     ('Tf-Idf - Ridge\n(Num+Str)', 6): ['European-Medicines-Agency (3)'],

#     # List 8: TargetEncoder - Ridge (Str) @ Rank 6
#     ('TargetEncoder - Ridge\n(Str)', 6): ['HIFLD (18)']
# }

# # B. DEFINE POSITIONS (X_Offset, Y_Offset)
# # Units are in "points" (pixels). 
# # Positive X = Right, Negative X = Left.
# # Positive Y = Up, Negative Y = Down.
# manual_positions = {
#     ('TargetEncoder - XGBoost\n(Str)', 5):   (15, 25),   # Move Right & Up
#     ('Tf-Idf - Ridge\n(Str)', 4):     (-20, 25), # Move Left & Up
#     ('Tf-Idf - Ridge\n(Str)', 6):     (20, 0),    # Right, Center
#     ('Tf-Idf - XGBoost\n(Num)', 5):   (-20, 0),   # Left, Center
#     ('Tf-Idf - XGBoost\n(Num)', 7):   (0, -20),    # Right, Center
#     ('Tf-Idf - XGBoost\n(Num)', 8):   (0, 20),    # Right, Center
#     ('Tf-Idf - Ridge\n(Num+Str)', 6): (-15, 0),    # Right, Center
#     ('TargetEncoder - Ridge\n(Str)', 6):     (25, 0),    # Right, Center
# }


# # ==========================================
# # 2. SETUP DATA & COLORS
# # ==========================================
# # (Standard Setup)
# source_counts = results.groupby('source')['data_name'].nunique()
# total_sources = len(source_counts)
# label_map = {src: f"{src} ({count})" for src, count in source_counts.items()}
# if 'Label_With_Count' not in df_loso.columns:
#     df_loso['Label_With_Count'] = df_loso['Dropped_Source'].map(label_map)

# # Master Alphabetical Order (for Legend & Color Assignment)
# # master_order = sorted(df_loso['Label_With_Count'].unique(), key=lambda x: x.lower())

# # Color Mapping
# unique_labels = len(df_loso['Label_With_Count'].unique())
# color_palette = sns.color_palette("husl", unique_labels)
# color_dict = dict(zip(df_loso['Label_With_Count'].unique(), color_palette))

# # Identify Median Ranks
# consensus_ranks = df_loso.groupby('Method')['Rank'].median()
# method_order = consensus_ranks.sort_values().index
# y_mapping = {method: i for i, method in enumerate(method_order)}


# # ==========================================
# # 3. PLOT GENERATION
# # ==========================================
# plt.figure(figsize=(20, 16))

# # Swim Lanes
# for y in np.arange(0.5, len(method_order), 1):
#     plt.axhline(y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# # Swarmplot
# ax = sns.swarmplot(
#     data=df_loso,
#     y='Method',
#     x='Rank',
#     order=method_order,
#     hue='Label_With_Count',
#     hue_order=df_loso['Label_With_Count'].unique(),
#     palette=color_dict,
#     size=12,
#     marker='D',
#     alpha=0.9,
#     edgecolor='gray',
#     linewidth=0.5
# )


# # ==========================================
# # 4. MANUAL ANNOTATION LOOP
# # ==========================================
# print("Applying Manual Annotations...")

# for (method_key, rank_key), label_list in manual_annotations.items():
    
#     # 1. Get Base Coordinates
#     if method_key not in y_mapping:
#         print(f"Warning: Method '{method_key}' not found in Y-axis mapping. Skipping.")
#         continue
        
#     base_x = rank_key
#     base_y = y_mapping[method_key]
    
#     # 2. Get User-Defined Offset (or default)
#     x_off, y_off = manual_positions.get((method_key, rank_key), (50, 0))
    
#     # 3. Alignment Logic
#     # If x_off is negative (left side), align text 'right' so it grows leftwards
#     ha_align = 'right' if x_off < 0 else 'left'
    
#     # 4. Draw Each Label (Stacking Downwards)
#     step_size = 14 # Spacing between lines
    
#     for i, label_text in enumerate(label_list):
#         # Calculate Y position for this specific line
#         # We start at y_off and move down
#         current_y_pos = y_off - (i * step_size)
        
#         ax.annotate(
#             label_text,
#             xy=(base_x, base_y),
#             xytext=(x_off, current_y_pos),
#             textcoords='offset points',
#             fontsize=12,
#             color='#333333',
#             # weight='bold',
#             ha=ha_align,
#             va='center'
#             # Removed arrowprops for cleaner look as requested previously
#         )

# # ==========================================
# # 5. LEGEND & FINAL FORMATTING
# # ==========================================
# legend_handles = []
# for label in df_loso['Label_With_Count'].unique():
#     patch = mlines.Line2D([], [], color='white', marker='D', markersize=10, 
#                           markerfacecolor=color_dict[label], label=label)
#     legend_handles.append(patch)

# plt.legend(
#     handles=legend_handles,
#     bbox_to_anchor=(1.02, 1),
#     loc='upper left',
#     fontsize=18,
#     title_fontsize=20,
#     ncol=1,
#     title=f"Excluded Source (Total: {total_sources})"
# )

# plt.title(f"Selection Bias Check: Ranking Stability (W = {W_score:.3f})", fontsize=22, pad=20)
# plt.xlabel("Rank (Lower is Better)", fontsize=20, labelpad=15)
# plt.ylabel("")
# ax.tick_params(axis='y', labelsize=20)
# plt.grid(False, axis='y')
# plt.xticks(np.arange(df_loso['Rank'].min(), df_loso['Rank'].max() + 1))
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'selection_bias_source_check_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

# '''
# PERMUTATION TEST FOR DOMAIN SHIFT
# Does the global hierarchy of algorithms change when we add string features?
# adding strings creates a mathematically distinct problem space?
# '''

# results = pd.read_csv('/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison.csv')

# results['score'] = results['r2'].fillna(results['roc_auc'])

# meta = results['method'].str.split('_', expand=True, n=2)

# results['dtype'] = meta[0]
# results['encoder'] = meta[1]
# results['learner'] = meta[2]

# # 2. DEFINE YOUR MAPPINGS
# dtype_map = {
#     'num-str': 'Num+Str',
#     'num-only': 'Num',
#     'str-only': 'Str'
# }

# encoder_map = {
#     'tabvec': 'Tf-Idf',
#     'tarenc': 'TargetEncoder',
#     'catboost': 'CatBoost'
# }

# learner_map = {
#     'ridge': 'Ridge', 
#     'xgb': 'XGBoost', 
#     'tabpfn': 'TabPFNv2.5',
#     'extrees': 'ExtraTrees',
#     'realmlp': 'RealMLP',
#     'catboost': 'CatBoost'
# }

# # 3. APPLY MAPPINGS
# results['dtype'] = results['dtype'].replace(dtype_map)
# results['encoder'] = results['encoder'].replace(encoder_map)
# results['learner'] = results['learner'].replace(learner_map)

# def robust_minmax(x):
#     x_clipped = x.clip(lower=0.0)
#     if x_clipped.max() == x_clipped.min():
#         return 1.0
#     return (x_clipped - x_clipped.min()) / (x_clipped.max() - x_clipped.min())

# # Apply this function to each group
# results['score_norm'] = results.groupby(['data_name', 'task'])['score'].transform(robust_minmax)

# results['method_polished'] = results['encoder'] + ' - ' + results['learner'] + '\n(' + results['dtype'] + ')'


# # Aggregate Mean Scores
# df_agg = results.groupby(['data_name', 'dtype', 'method_polished'])['score'].mean().reset_index()

# # ==========================================
# # 2. CREATE PAIRED DOMAINS (THE MATRICES)
# # ==========================================

# # Pivot to create Matrix A (Num+Str) and Matrix B (Num)
# # Rows = Datasets, Columns = Algorithms
# df_A = df_agg[df_agg['dtype'] == 'Num+Str'].pivot(index='data_name', columns='method_polished', values='score')
# df_B = df_agg[df_agg['dtype'] == 'Num'].pivot(index='data_name', columns='method_polished', values='score')

# # --- BROADCASTING (CRITICAL STEP) ---
# # In the 'Num' domain, TargetEncoder and Tf-Idf are identical.
# # But to compare rankings of ALL algorithms, we must ensure df_B has the same columns as df_A.
# # We map "TargetEncoder - X" in df_B to be the scores of "Tf-Idf - X"
# for col in df_A.columns:
#     if "TargetEncoder" in col and col not in df_B.columns:
#         # Find the equivalent Tf-Idf column (the 'proxy')
#         proxy_col = col.replace("TargetEncoder", "Tf-Idf")
#         if proxy_col in df_B.columns:
#             df_B[col] = df_B[proxy_col]

# # --- ALIGNMENT ---
# # Keep only common datasets and common methods
# common_data = df_A.index.intersection(df_B.index)
# common_cols = df_A.columns.intersection(df_B.columns)

# df_A = df_A.loc[common_data, common_cols]
# df_B = df_B.loc[common_data, common_cols]

# # Drop any datasets with NaNs (Ranking requires complete data)
# valid_rows = df_A.notna().all(axis=1) & df_B.notna().all(axis=1)
# df_A = df_A[valid_rows]
# df_B = df_B[valid_rows]

# print(f"Analysis Dimensions: {df_A.shape[0]} Datasets x {df_A.shape[1]} Algorithms")
# print("Algorithms included:", list(df_A.columns))

# # ==========================================
# # 3. RANKING DISTANCE FUNCTION
# # ==========================================
# def calculate_ranking_distance(d1, d2):
#     """
#     Computes distance between the leaderboards of domain d1 vs domain d2.
#     """
#     # REUSE the core function to ensure consistency
#     r1 = calculate_rankings(d1)
#     r2 = calculate_rankings(d2)
    
#     # Safety Check: If variance is 0 (all algorithms tied), correlation is undefined
#     if np.std(r1) == 0 or np.std(r2) == 0:
#         return 0.0 if np.array_equal(r1.values, r2.values) else 1.0
        
#     # Compare the two rank vectors
#     corr, _ = kendalltau(r1, r2)
#     return (1.0 - corr) / 2.0

# # ==========================================
# # 4. PERMUTATION TEST
# # ==========================================
# n_permutations = 2000
# obs_dist = calculate_ranking_distance(df_A, df_B)

# null_dists = []
# A_vals = df_A.values
# B_vals = df_B.values
# n_rows = A_vals.shape[0]

# print(f"Running {n_permutations} permutations...")

# for _ in range(n_permutations):
#     # Randomly swap rows between A and B
#     # Null Hypothesis: The label 'Num+Str' vs 'Num' is meaningless for ranking
#     mask = np.random.rand(n_rows, 1) > 0.5
    
#     # Create synthetic matrices
#     perm_A = np.where(mask, B_vals, A_vals)
#     perm_B = np.where(mask, A_vals, B_vals)
    
#     # Convert back to DF for ranking function
#     d_null = calculate_ranking_distance(pd.DataFrame(perm_A), pd.DataFrame(perm_B))
#     null_dists.append(d_null)

# null_dists = np.array(null_dists)
# p_value = (null_dists >= obs_dist).mean()

# # ==========================================
# # 5. VISUALIZATION & VERDICT
# # ==========================================

# plt.figure(figsize=(10, 6))
# sns.histplot(null_dists, kde=True, color="lightgray", label="Null Distribution (Random Noise)")
# plt.axvline(obs_dist, color='red', linestyle='--', linewidth=2, label=f"Observed Shift ({obs_dist:.3f})")

# plt.title(f"Are 'Num' and 'Num+Str' Distinct Problem Spaces?\n(Permutation Test P={p_value:.5f})", fontsize=14)
# plt.xlabel("Ranking Distance (1 - Spearman Correlation)\n0.0 = Identical Rankings | 1.0 = Distinct Rankings")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'impact_domain_shift_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# print("-" * 40)
# print(f"Observed Ranking Distance: {obs_dist:.4f}")
# print(f"P-Value: {p_value:.5f}")
# print("-" * 40)

# if p_value < 0.05:
#     print("✅ VERDICT: YES, distinct problem space.")
#     print("The relative ranking of algorithms (Ridge vs XGB vs etc.) changes SIGNIFICANTLY")
#     print("when string features are added. The 'Num+Str' leaderboard is statistically different")
#     print("from the 'Num' leaderboard.")
# else:
#     print("❌ VERDICT: NO, same problem space.")
#     print("Adding strings does not significantly change the winner. If XGBoost wins on numbers,")
#     print("it likely still wins on numbers+strings.")


# # ==========================================
# # 1. DATA PREPARATION
# # ==========================================

# results = pd.read_csv('/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison.csv')

# results['score'] = results['r2'].fillna(results['roc_auc'])

# meta = results['method'].str.split('_', expand=True, n=2)

# results['dtype'] = meta[0]
# results['encoder'] = meta[1]
# results['learner'] = meta[2]

# # 2. DEFINE YOUR MAPPINGS
# dtype_map = {
#     'num-str': 'Num+Str',
#     'num-only': 'Num',
#     'str-only': 'Str'
# }

# encoder_map = {
#     'tabvec': 'Tf-Idf',
#     'tarenc': 'TargetEncoder'
# }

# learner_map = {
#     'ridge': 'Ridge', 
#     'xgb': 'XGBoost', 
#     'tabpfn': 'TabPFNv2.5',
#     'extrees': 'ExtraTrees',
#     'realmlp': 'RealMLP'
# }

# # 3. APPLY MAPPINGS
# results['dtype'] = results['dtype'].replace(dtype_map)
# results['encoder'] = results['encoder'].replace(encoder_map)
# results['learner'] = results['learner'].replace(learner_map)
 

# def robust_minmax(x):
#     x_clipped = x.clip(lower=0.0)
#     if x_clipped.max() == x_clipped.min():
#         return 1.0
#     return (x_clipped - x_clipped.min()) / (x_clipped.max() - x_clipped.min())

# # Apply this function to each group
# results['score_norm'] = results.groupby(['data_name', 'task'])['score'].transform(robust_minmax)

# results['method_polished'] = results['encoder'] + ' - ' + results['learner'] + '\n(' + results['dtype'] + ')'

# #clip negative
# results['score'] = results['score'].clip(lower=0)


# # ==========================================
# # 1. DATA PREPARATION
# # ==========================================

# # Assuming 'df' is your raw dataframe
# print("Processing raw data...")

# results['base_method'] = results['method_polished'].astype(str).apply(lambda x: x.split('\n')[0])
# df_agg = results.groupby(['data_name', 'dtype', 'base_method'])['score'].mean().reset_index()

# # Create Domains
# df_A = df_agg[df_agg['dtype'] == 'Num+Str'].pivot(
#     index='data_name', columns='base_method', values='score'
# )
# df_B = df_agg[df_agg['dtype'] == 'Num'].pivot(
#     index='data_name', columns='base_method', values='score'
# )

# # Broadcast Missing Columns (The Fix)
# for col in df_A.columns:
#     if "TargetEncoder" in col and col not in df_B.columns:
#         proxy_col = col.replace("TargetEncoder", "Tf-Idf")
#         if proxy_col in df_B.columns:
#             df_B[col] = df_B[proxy_col]

# # Align
# common_datasets = df_A.index.intersection(df_B.index)
# common_methods = df_A.columns.intersection(df_B.columns)
# df_A = df_A.loc[common_datasets, common_methods]
# df_B = df_B.loc[common_datasets, common_methods]

# # Drop NaNs
# valid_mask = df_A.notna().all(axis=1) & df_B.notna().all(axis=1)
# df_A = df_A[valid_mask]
# df_B = df_B[valid_mask]

# # ==========================================
# # 2. ROBUST RANKING FUNCTION (THE FIX)
# # ==========================================

# def get_ranking_distance(d1, d2):
#     """
#     Calculates Distance between domains.
#     Handles edge case where one domain has a 'Perfect Tie' (Zero Variance).
#     """
#     # 1. Compute Average Ranks
#     r1 = d1.rank(axis=1, ascending=False, method='min').mean(axis=0)
#     r2 = d2.rank(axis=1, ascending=False, method='min').mean(axis=0)
    
#     # 2. Check Variance (Standard Deviation)
#     std1 = np.std(r1)
#     std2 = np.std(r2)
    
#     # CASE A: Both domains are perfect ties (e.g., both say "It's a draw")
#     # Result: They are Identical. Distance = 0.
#     if std1 == 0 and std2 == 0:
#         return 0.0
        
#     # CASE B: One domain is a tie, the other has a clear winner
#     # Result: They are Totally Different. Distance = 1.0 (Max difference)
#     # (Because Correlation is mathematically undefined here, but logically they are opposite)
#     if std1 == 0 or std2 == 0:
#         return 1.0
        
#     # CASE C: Both have rankings. Calculate standard Spearman.
#     corr, _ = spearmanr(r1, r2)
#     return 1 - corr

# # ==========================================
# # 3. PERMUTATION TEST
# # ==========================================

# def run_permutation_test(df_A, df_B, n_permutations=1000):
#     # Calculate True Difference with the new robust function
#     obs_distance = get_ranking_distance(df_A, df_B)
    
#     null_distances = []
#     arr_A = df_A.values
#     arr_B = df_B.values
#     n_rows = arr_A.shape[0]
    
#     for _ in range(n_permutations):
#         # Swap logic
#         mask = np.random.rand(n_rows, 1) > 0.5
#         mask_broad = np.tile(mask, (1, arr_A.shape[1]))
        
#         perm_A = np.where(mask_broad, arr_B, arr_A)
#         perm_B = np.where(mask_broad, arr_A, arr_B)
        
#         # Calculate random difference
#         d_null = get_ranking_distance(pd.DataFrame(perm_A), pd.DataFrame(perm_B))
#         null_distances.append(d_null)
    
#     # Convert to numpy array
#     null_distances = np.array(null_distances)
    
#     # Calculate P-Value
#     p_value = (null_distances >= obs_distance).mean()
    
#     return obs_distance, p_value, null_distances

# # Run the test
# print("Running Permutation Test with Robust Distance Metric...")
# obs_dist, p_val, null_dists = run_permutation_test(df_A, df_B)

# # ==========================================
# # 4. RESULTS
# # ==========================================

# plt.figure(figsize=(10, 6))
# sns.histplot(null_dists, kde=True, color="lightgray", label="Random Noise (Null)")
# plt.axvline(obs_dist, color='red', linestyle='--', linewidth=2, label=f"Observed Diff ({obs_dist:.3f})")
# plt.title(f"Domain Shift Test (P={p_val:.5f})", fontsize=14)
# plt.xlabel("Ranking Distance (0=Same, 1=Different)")
# plt.legend()
# plt.show()

# print("-" * 30)
# print(f"Observed Distance: {obs_dist:.4f}")
# print(f"P-Value: {p_val:.5f}")

# if obs_dist == 1.0:
#     print("\nℹ️ NOTE: Observed Distance is 1.0.")
#     print("This confirms that the Numerical Domain is a 'Tie' (Method A = Method B),")
#     print("while the Num+Str Domain has a clear winner.")
#     print("This is the strongest possible proof of a domain shift.")



# # ==========================================
# # 1. PREPARE THE PAIRS
# # ==========================================

# # We assume df_A (Num+Str) and df_B (Num) are already aligned from the previous step.
# # If they are not in memory, run the "Data Preparation" block from the previous code first.

# # Extract the scores for the specific comparison
# # We want to compare: TargetEncoder (in Num+Str) vs. Tf-Idf (in Num)
# # (Recall: In 'Num', Tf-Idf == TargetEncoder mathematically)

# # Verify column names exist (Adjust string if your column name is different)
# target_col = [c for c in df_A.columns if "TargetEncoder" in c][0]
# baseline_col = [c for c in df_B.columns if "Tf-Idf" in c][0]

# print(f"Comparing Improvement:\n  {target_col}\n  vs\n  {baseline_col}")

# scores_target = df_A[target_col]
# scores_baseline = df_B[baseline_col] # This is your "Control"

# # Calculate the "Delta" (Improvement) per dataset
# deltas = scores_target - scores_baseline

# # ==========================================
# # 2. STATISTICAL TEST (Wilcoxon)
# # ==========================================
# # This tests if the median of the differences is non-zero
# stat, p_val = wilcoxon(deltas, alternative='two-sided')

# print("-" * 30)
# print(f"Mean Improvement: {deltas.mean():.4f}")
# print(f"Median Improvement: {deltas.median():.4f}")
# print(f"Wilcoxon P-Value: {p_val:.10f}")

# if p_val < 0.05:
#     print("✅ VERDICT: The domain shift is REAL. Adding strings significantly improves performance.")
# else:
#     print("⚠️ VERDICT: No significant change detected.")
    
# # ==========================================
# # 3. THE "BETTER" PLOT
# # ==========================================
# plt.figure(figsize=(10, 6))

# # Plot the distribution of Score Improvements (Deltas)
# sns.histplot(deltas, kde=True, color="forestgreen", bins=20, alpha=0.6)

# # Add reference line at 0 (No Improvement)
# plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Zero Impact (Baseline)")

# # Add Mean line
# plt.axvline(deltas.mean(), color='darkgreen', linestyle='-', linewidth=2, label=f"Mean Lift (+{deltas.mean():.3f})")

# plt.title(f"Impact of Domain Shift (Strings)\n(Wilcoxon Test P={p_val:.2e})", fontsize=14)
# plt.xlabel("Score Improvement (TargetEncoder - Baseline)", fontsize=12)
# plt.ylabel("Count of Datasets", fontsize=12)
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'performance_gain_from_adding_strings_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# Assuming 'deltas' is the variable from the previous step
# deltas = scores_target - scores_baseline

# plt.figure(figsize=(10, 6))

# # 1. Clean Histogram (No wobbly KDE line, just clear bars)
# sns.histplot(deltas, color="#2ecc71", edgecolor="white", bins=25, alpha=0.7, label="Dataset Distribution")

# # 2. Add Reference Line (Baseline)
# plt.axvline(0, color='#e74c3c', linestyle='--', linewidth=2.5, label="No Improvement (Baseline)")

# # 3. Add Mean with Confidence Interval Look
# mean_val = deltas.mean()
# plt.axvline(mean_val, color='#27ae60', linestyle='-', linewidth=3, label=f"Average Lift (+{mean_val:.3f})")

# # Add a text annotation for the P-Value (Scientific Standard)
# plt.text(0.7, 0.5, f"Wilcoxon $p < 0.001$", transform=plt.gca().transAxes, 
#          fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# # 4. Professional Labels
# plt.title("Performance Gain from Including String Features", fontsize=16, pad=20)
# plt.xlabel("Score Improvement ($R^2$ Increase)", fontsize=13)
# plt.ylabel("Number of Datasets", fontsize=13)
# plt.legend(loc='upper right', frameon=True)
# plt.grid(axis='y', alpha=0.3)
# plt.xlim(deltas.min() - 0.05, deltas.max() + 0.05)
# plt.tight_layout()


# def sign_array(p_values: Union[List, np.ndarray], alpha: float = 0.05) -> np.ndarray:
#     p_values = np.array(p_values)
#     p_values[p_values > alpha] = 0
#     p_values[(p_values < alpha) & (p_values > 0)] = 1
#     np.fill_diagonal(p_values, 1)
#     return p_values

# def _find_maximal_cliques(adj_matrix: DataFrame) -> List[Set]:
#     if (adj_matrix.index != adj_matrix.columns).any():
#         raise ValueError("adj_matrix must be symmetric")
#     result = []
#     _bron_kerbosch(set(), set(adj_matrix.index), set(), adj_matrix, result)
#     return result

# def _bron_kerbosch(current_clique: Set, candidates: Set, visited: Set, adj_matrix: DataFrame, result: List[Set]):
#     while candidates:
#         v = candidates.pop()
#         _bron_kerbosch(
#             current_clique | {v},
#             {n for n in candidates if adj_matrix.loc[v, n]},
#             {n for n in visited if adj_matrix.loc[v, n]},
#             adj_matrix,
#             result,
#         )
#         visited.add(v)
#     if not visited:
#         result.append(current_clique)


# ---------------------------------------------------------
# 3. PLOTTING
# ---------------------------------------------------------
# sns.set_style("white")

# fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

# for row_idx, x_metric in enumerate(ROW_METRICS):
    
#     # Calculate Global Pareto Frontier for this row
#     pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, HIGHER_SCORE_IS_BETTER)
    
#     for col_idx, factor in enumerate(COL_FACTORS):
#         ax = axes[row_idx, col_idx]
        
#         # A. SCATTER PLOT
#         # We save the plot object to extract handles for the legend later
#         sns.scatterplot(
#             data=df_agg, x=x_metric, y=Y_METRIC, hue=factor, style=factor,
#             s=120, alpha=0.8, ax=ax, palette=paper_palette, legend='brief'
#         )
        
#         # B. PARETO LINE
#         ax.step(
#             pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
#             linestyle='--', color='black', linewidth=1.5, zorder=0
#         )
        
#         # C. FORMATTING
#         ax.set_xscale('log')
#         ax.grid(False) # Explicitly disable grid
        
#         # Axis Labels
#         ax.set_xlabel(f"{ROW_LABELS[row_idx]} [Log Scale]", fontsize=12, fontweight='bold')
#         if col_idx == 0:
#             ax.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
#         else:
#             ax.set_ylabel('')

#         # D. TITLES & LEGENDS (The Paper Style)
#         # We only add titles and legends to the TOP ROW
#         if row_idx == 0:
#             # 1. Reduce padding (was 35, now 20) to bring title down closer to legend
#             ax.set_title(COL_TITLES[col_idx], fontsize=16, fontweight='bold', pad=45)
            
#             # 2. Position legend closer to plot
#             handles, labels = ax.get_legend_handles_labels()
#             ax.legend(
#                 handles, labels,
#                 loc='lower center', 
#                 bbox_to_anchor=(0.5, 1.01), # Slightly lower anchor (was 1.02)
#                 ncol=len(labels),
#                 frameon=False,
#                 fontsize=12,
#                 handletextpad=0.5,
#                 columnspacing=1.5
#             )
#         else:
#             if ax.get_legend() is not None:
#                 ax.get_legend().remove()
# plt.subplots_adjust(top=0.88, hspace=0.25, wspace=0.1)
# plt.show()
# plt.close()

# df = pivot_df.copy()

# if df.isna().sum().sum() > 0:
#     print(f"Warning: Dropping {df.isna().any(axis=1).sum()} datasets containing NaN values.")
#     df = df.dropna()

# n_datasets = len(df)
# print(f"Total valid datasets for analysis: {n_datasets}")


# # Step A: Establish "True" Ranking (Ground Truth using ALL available data)
# true_rankings = calculate_rankings(df)

# print("\nTrue Ranking (Top 3 based on all data):")
# print(true_rankings.sort_values().head(3))

# # Parameters
# # We start at N=10 and go up to the full dataset count in steps of 5
# sample_sizes = range(10, n_datasets + 1, 5) 
# n_iterations = 100 
# stability_scores = []

# print(f"\nRunning simulation ({n_iterations} iterations per step)...")

# # Step B: Bootstrapping Loop
# for n in sample_sizes:
#     for i in range(n_iterations):
#         # 1. Subsample N datasets (randomly select N rows)
#         subset = df.sample(n=n, replace=False)
        
#         # 2. Compute rankings on this subset
#         subset_rankings = calculate_rankings(subset)
        
#         # 3. Compare subset ranking vs. true ranking
#         # We use Spearman correlation to see if the ordering is preserved
#         corr, _ = spearmanr(subset_rankings, true_rankings)
        
#         stability_scores.append({
#             'N_datasets': n,
#             'Spearman_Correlation': corr
#         })

# df_stability = pd.DataFrame(stability_scores)

# # ==========================================
# # 3. VISUALIZATION
# # ==========================================

# plt.figure(figsize=(10, 6))

# # Boxplot showing the distribution of correlations at each N
# sns.boxplot(x='N_datasets', y='Spearman_Correlation', data=df_stability, color='#a1c9f4')

# # Add threshold line
# plt.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='Stability Threshold (0.95)')

# plt.title('Benchmark Stability: How many datasets do you need?', fontsize=14)
# plt.ylabel('Correlation with Full Ranking (Spearman)', fontsize=12)
# plt.xlabel('Number of Datasets Used (N)', fontsize=12)
# plt.legend(loc='lower right')
# plt.grid(axis='y', alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# # ==========================================
# # 4. AUTOMATED CONCLUSION
# # ==========================================
# # Check statistics at the max N (closest to your current count)
# max_n = sample_sizes[-1]
# final_stats = df_stability[df_stability['N_datasets'] == max_n]['Spearman_Correlation']

# mean_corr = final_stats.mean()
# std_corr = final_stats.std()

# print("-" * 40)
# print(f"At N={max_n} datasets:")
# print(f"  - Mean Correlation: {mean_corr:.4f}")
# print(f"  - Stability (Std Dev): {std_corr:.4f}")

# if mean_corr > 0.95 and std_corr < 0.05:
#     print("\n✅ VERDICT: Your benchmark is STABLE.")
#     print("Adding more datasets is unlikely to change the relative ranking of these algorithms.")
# else:
#     print("\n⚠️ VERDICT: Your benchmark is still VOLATILE.")
#     print("The ranking heavily depends on which specific datasets are included.")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # 1. PREPARE DATA: Average the 100 iterations FIRST
# df_stability = pd.DataFrame(stability_scores)
# summary_stats = df_stability.groupby('N_datasets')['Kendalltau_Correlation'].agg(['mean', 'sem']).reset_index()

# x_data = summary_stats['N_datasets']
# y_data = summary_stats['mean']
# y_err = summary_stats['sem'] # We use this to weight the fit!

# # 2. DEFINE MODEL
# def learning_curve(x, a, b, c):
#     return a - b * np.power(x, -c)

# # 3. FIT WITH WEIGHTS (The "Gold Standard" way)
# # sigma=y_err tells the solver: "If the error bar is wide, don't worry about missing this point."
# # absolute_sigma=True tells it these are real standard errors.
# p0 = [1.0, 1.0, 0.5]

# try:
#     popt, pcov = curve_fit(
#         learning_curve, 
#         x_data, 
#         y_data, 
#         p0=p0, 
#         sigma=y_err, 
#         absolute_sigma=True, 
#         maxfev=10000
#     )
#     a_fit, b_fit, c_fit = popt
#     print(f"Fitted Parameters: Max Potential (a) = {a_fit:.3f}")
# except RuntimeError:
#     print("Fit failed even on averaged data.")
#     popt = None

# # 4. EXTRAPOLATE
# if popt is not None:
#     target_y = 0.95
#     if a_fit < target_y:
#         print(f"Result: The benchmark is predicted to saturate at {a_fit:.3f}, never reaching 0.95.")
#     else:
#         # Solve for N
#         required_N = ((a_fit - target_y) / b_fit) ** (-1 / c_fit)
#         print(f"--- FINAL RESULT ---")
#         print(f"To reach 0.95 stability, you need approximately: {int(required_N)} datasets.")

#     # 5. VISUALIZE
#     plt.figure(figsize=(10, 6))
    
#     # Plot the averaged data with error bars
#     plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='blue', 
#                  ecolor='lightblue', elinewidth=2, capsize=4, label='Mean Correlation ± SEM')
    
#     # Plot the fit
#     max_x = max(100, required_N * 1.2) if 'required_N' in locals() else 100
#     x_range = np.linspace(5, max_x, 100)
#     y_fit = learning_curve(x_range, *popt)
#     plt.plot(x_range, y_fit, 'r--', label='Weighted Fit Extrapolation')
    
#     # Thresholds
#     plt.axhline(0.95, color='green', linestyle=':', label='Target (0.95)')
#     if 'required_N' in locals():
#         plt.axvline(required_N, color='k', linestyle='--', alpha=0.5)
#         plt.text(required_N, 0.75, f'  N ≈ {int(required_N)}', verticalalignment='center')
        
#     plt.xlabel('Number of Datasets')
#     plt.ylabel('Kendall Tau Correlation')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()

# mid_point = n_datasets // 2
# indices = df.index.tolist()
# sample_sizes = range(10, mid_point + 1, 5) 
# n_iterations = 100
# required_N_list = []
# failed_fits = 0
# # Step B: Bootstrapping Loop
# for i in range(n_iterations):
    
#     print(f"  Iteration {i+1}/{n_iterations}...")

#     # print(f"Running simulations for N={n} datasets...")
#     random.shuffle(indices)
#     df_subsample1 = df.loc[indices[:mid_point], :]
#     df_subsample2 = df.loc[indices[mid_point:mid_point*2], :]

#     # print("sizes of subsamples:", df_subsample1.shape, df_subsample2.shape)

#     # check that sampled datasets in each subsample are different
#     assert len(set(df_subsample1.index).intersection(set(df_subsample2.index))) == 0, "Subsamples overlap!"
    
#     current_iteration_data = []
#     for n in sample_sizes:
    
#         # 1. Subsample N datasets (randomly select N rows)
#         subset1 = df_subsample1.sample(n=n, replace=False)
#         subset2 = df_subsample2.sample(n=n, replace=False)
        
#         # 2. Compute rankings on this subset
#         subset_rankings_1 = calculate_rankings(subset1)
#         subset_rankings_2 = calculate_rankings(subset2)
        
#         # 3. Compare subset ranking vs. true ranking
#         # We use kendall-tau correlation to see if the ordering is preserved
#         corr, _ = kendalltau(subset_rankings_1, subset_rankings_2)
        
#         current_iteration_data.append({
#             'N_datasets': n,
#             'Kendalltau_Correlation': corr
#         })
#     #aggregate per dataset size
#     df_stability = pd.DataFrame(current_iteration_data)
#     # df_stability = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['mean', 'sem']).reset_index()

#     # 3. Fit the curve to your data
#     x_data = df_stability['N_datasets']
#     y_data = df_stability['Kendalltau_Correlation']


#     # Initial parameter guesses (help the solver):
#     # a=1 (max correlation), b=1 (scaling), c=0.5 (decay)
#     p0 = [1.0, 1.0, 0.5] 

#     try:
#         popt, pcov = curve_fit(learning_curve, x_data, y_data, p0=[1.0, 1.0, 0.5], maxfev=5000)
#         a_fit, b_fit, c_fit = popt
        
#         # 5. Extrapolate
#         target_y = 0.95
#         if a_fit > target_y:
#             required_N = ((a_fit - target_y) / b_fit) ** (-1 / c_fit)
#             required_N_list.append(required_N)
#         else:
#             # If this specific simulation says you can't reach 0.95, 
#             # you might want to log it as a failure or None
#             required_N_list.append(None)
            
#     except RuntimeError:
#         failed_fits += 1

# # 2. Filter: Separate "Reasonable" estimates from "Failures/Explosions"
# # Let's define "Reasonable" as N < 5000 (You can adjust this threshold)
# clean_values = []
# failures = 0

# for val in required_N_list:
#     if val is None:
#         failures += 1
#     else:
#         clean_values.append(val)

# clean_array = np.array(clean_values)

# # 3. Calculate Robust Statistics
# median_N = np.median(clean_array)
# p95_N = np.percentile(clean_array, 95)
# success_rate = (len(clean_values) / len(required_N_list)) * 100

# print(f"--- Analysis of Bootstrap ---")
# print(f"Total Iterations: {len(required_N_list)}")
# print(f"Successful Convergences (N < 5000): {len(clean_values)} ({success_rate:.1f}%)")
# print(f"Failed/Infinite Predictions: {failures}")
# print(f"-----------------------------")
# print(f"Estimated Required N (Median): {median_N:.0f}")
# print(f"Conservative Estimate (95th Percentile): {p95_N:.0f}")


# # Setup for the visualization
# plt.figure(figsize=(14, 6))

# # ==========================================
# # PLOT 1: SIGNIFICANT RESULT (Ideal Case)
# # ==========================================
# plt.subplot(1, 2, 1)

# # Simulate "Random Noise" (Null Distribution) - mostly small differences
# null_dist_sig = np.random.normal(0.1, 0.05, 1000) 
# # Simulate "Real Result" (Observed) - huge difference
# obs_val_sig = 0.8 

# sns.histplot(null_dist_sig, kde=True, color="lightgray", label="Random Noise (Null)")
# plt.axvline(obs_val_sig, color='red', linestyle='--', linewidth=3, label=f"Your Result ({obs_val_sig})")

# plt.title("Scenario A: Validated Domain Shift\n(Significant)", fontsize=14, color='green')
# plt.xlabel("Ranking Distance", fontsize=12)
# plt.legend()
# plt.xlim(0, 1.0)

# # ==========================================
# # PLOT 2: NOT SIGNIFICANT (Null Result)
# # ==========================================
# plt.subplot(1, 2, 2)

# # Simulate "Random Noise" - wider spread
# null_dist_fail = np.random.normal(0.1, 0.1, 1000)
# # Simulate "Real Result" - buried in the noise
# obs_val_fail = 0.15 

# sns.histplot(null_dist_fail, kde=True, color="lightgray", label="Random Noise (Null)")
# plt.axvline(obs_val_fail, color='red', linestyle='--', linewidth=3, label=f"Your Result ({obs_val_fail})")

# plt.title("Scenario B: No Domain Difference\n(Not Significant)", fontsize=14, color='red')
# plt.xlabel("Ranking Distance", fontsize=12)
# plt.legend()
# plt.xlim(0, 1.0)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))

# sns.pointplot(
#     x='N_datasets', 
#     y='Kendalltau_Correlation', 
#     data=df_stability, 
#     color='#a1c9f4',
#     markers='o',       # Use circle markers
#     linestyles='-',    # Connect with a line
#     errorbar='se',     # Show Standard Error as error bars (optional)
#     capsize=0.1
# )

# plt.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='Stability Threshold (0.95)')

# plt.title('Kendall-Tau Correlation vs. Number of Datasets', fontsize=14)
# plt.ylabel('Kendall $\\tau$', fontsize=12)
# plt.xlabel('Number of Datasets (N)', fontsize=12)
# plt.legend(loc='lower right')
# plt.grid(axis='y', alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()


# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'kendalltau_correlation_vs_datasets_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# '''
# EXTRAPOLATION
# # Fit a curve to the kendall-tau data and extrapolate to find N where correlation reaches 0.95
# why not linear: it ignores the bounds of the correlation metric [-1,1].
# x-->inf,  x^{-c}-->0. So y-->a, which is the upper bound of the correlation.

# a (Asymptote): Enforces a "ceiling." We know correlation cannot exceed 1.0. This parameter finds the theoretical max stability your method could ever achieve.
# x (Datasets): This is your sample size $n$.
# -c (Decay): This allows the model to align with the Central Limit Theorem ($c \approx 0.5$) or adjust if your specific data has different noise properties (scaling laws often find $c$ between 0.07 and 0.35 in deep learning tasks).
# '''

# #aggregate per dataset size
# df_stability = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['mean', 'sem']).reset_index()

# # 2. Define the asymptotic function (Inverse Power Law)
# def learning_curve(x, b, c):
#     return 1.0 - b * np.power(x, -c)

# # 3. Fit the curve to your data
# x_data = df_stability['N_datasets']
# y_data = df_stability['mean']
# y_error = df_stability['sem']

# # Initial parameter guesses (help the solver):
# p0 = [1.0, 0.5] 

# try:
#     # bounds are optional but recommended: b > 0, c > 0
#     popt, pcov = curve_fit(
#         learning_curve, 
#         x_data, y_data, 
#         p0=p0, 
#         sigma=y_error,       # <--- USE THE ERROR BARS
#         absolute_sigma=True, # <--- Treat sigma as absolute values
#         bounds=([0, 0], [np.inf, np.inf]),
#         maxfev=10000
#     )
#     b_fit, c_fit = popt
#     print(f"Fitted Parameters: b={b_fit:.3f}, c={c_fit:.3f} (Fixed Asymptote a=1.0)")
# except RuntimeError:
#     print("Could not find optimal fit.")
#     popt = None

# # 4. Extrapolate to find N where correlation >= 0.95
# if popt is not None:
#     target_y = 0.95
#     required_N = ((1.0 - target_y) / b_fit) ** (-1 / c_fit)
#     print(f"Estimated datasets needed for 0.95 stability: {required_N:.1f}")

#     # 5. Visualizing the extrapolation
#     plt.figure(figsize=(10, 6))
    
#     # Plot original points
#     # plt.scatter(x_data, y_data, label='Actual Data', color='blue', zorder=5)
    
#     plt.errorbar(
#         x_data, 
#         y_data, 
#         yerr=y_error,     # <--- Add the error bars here
#         fmt='o',          # 'o' means plot dots (markers)
#         color='blue', 
#         ecolor='blue', # Color of the error bar lines
#         elinewidth=2,     # Thickness of the error bar lines
#         capsize=4,        # Width of the caps at the end of the bars
#         label='Kendall-Tau (Mean ± SE)', 
#         zorder=5
#     )

#     # Plot Fit
#     max_x_plot = max(300, required_N * 1.2) # Dynamic range
#     x_range = np.linspace(5, max_x_plot, 100)
#     y_fit = learning_curve(x_range, *popt)
    
#     plt.plot(x_range, y_fit, 'r--', label='Fitted Extrapolation (Fixed a=1.0)')
    
#     # Draw Thresholds
#     plt.axhline(y=0.95, color='green', linestyle=':', label='Target (0.95)')
#     plt.axvline(x=required_N, color='k', linestyle='--', alpha=0.5)
#     plt.text(required_N, 0.75, f'  N ≈ {int(required_N)}', verticalalignment='center')
#     plt.xlabel('Number of Datasets')
#     plt.ylabel('Kendall Tau Correlation')
#     plt.title('How Many Datasets are Needed for a robust Benchmark?')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

'''
Performance per dtypes-encoders subplots
'''

# # 1. Establish the high-legibility context
# sns.set_context("paper", font_scale=1.7) 
# sns.set_theme(style="whitegrid")

# # 2. Orders generated dynamically (Alphabetical)
# encoder_order = sorted(results['encoder'].unique().tolist())
# learner_order = sorted(results['learner'].unique().tolist())
# dtype_order = results['dtype'].unique().tolist()

# # 3. Adjusted figsize: Wider for three panels plus legend
# fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=True)

# for ax, d_type in zip(axes, dtype_order):
#     subset = results[results['dtype'] == d_type]
#     if subset.empty:
#         ax.set_visible(False)
#         continue

#     sns.barplot(
#         data=subset,
#         y='encoder',
#         x='score',
#         hue='learner',
#         order=encoder_order,
#         hue_order=learner_order,
#         palette='tab10',
#         errorbar='se',
#         capsize=0.1,
#         edgecolor='black',
#         linewidth=0.8,
#         ax=ax
#     )
    
#     # # Professional Spine Styling
#     # ax.spines['left'].set_linewidth(2.0)
#     # ax.spines['left'].set_color('black')
#     # sns.despine(ax=ax, bottom=True, top=True, right=True)

#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_edgecolor('black')
#         spine.set_linewidth(1.5) # Adjust thickness as desired
    
#     # 2. X-AXIS FIX: Force the right wall to be exactly at 1.0
#     ax.set_xlim(0, 1.0)
    
#     # 4. Labeling and Formatting
#     ax.set_title(d_type.upper(), fontsize=20, pad=15)
#     ax.set_xlabel("")
#     ax.set_xticks(np.arange(0, 1.01, 0.5))
#     ax.tick_params(axis='x', labelsize=16) 
    
#     # Handle Y-Axis visibility
#     if d_type == 'Num':
#         ax.set_ylabel("")
#         ax.set_yticklabels([])
#         # ax.set_yticks([1]) 
#         # ax.set_yticklabels(["No Encoder"])
#         # ax.tick_params(axis='y', labelsize=20)
#     elif d_type == 'Str':   
#         ax.set_ylabel("") 
#         ax.tick_params(axis='y', labelsize=20) 
#     else:
#         ax.set_ylabel("")
#         ax.set_yticklabels([])

#     if ax.get_legend():
#         ax.get_legend().remove()

# # 5. CLOSER VERTICAL LEGEND
# # Adjusting bbox_to_anchor to be closer to the rightmost plot
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, title="Learner", loc='center left', 
#            bbox_to_anchor=(0.84, 0.5), # Anchor shifted to match the 'right' margin
#            fontsize=20, title_fontsize=24, frameon=True)


# # 6. SINGLE X-AXIS LABEL (Centered)
# fig.text(0.5, 0.05, "Avg. Score (R2 & AUC)", ha='center', fontsize=22)

# # 7. CRITICAL SPACING FIX
# # Reducing wspace slightly and increasing 'right' to pull plots toward the legend
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2, right=0.82, bottom=0.25, left=0.18)


# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'horizontal_barplot_perdtype_avg_score_encoder_learner_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()

# plt.tight_layout(rect=[0, 0, 0.8, 1])

# # ============================================
# # 3. PLOT MAIN BARS
# # ============================================
# ax = sns.barplot(
#     data=plot_data,
#     y='y_label',
#     x='score',
#     hue='learner',
#     order=order,
#     palette=color_map, # Use our fixed color map
#     errorbar='se',     # Standard deviation black lines
#     capsize=0,
#     edgecolor='black',
#     linewidth=1.5,      # Thickness of error bars
#     width=1.0,  # Increases the thickness of the whole group (0.1 to 1.0)
#     gap=0,
#     dodge=True
# )

# # Ensure error bars are black
# plt.setp(ax.lines, color='black', linewidth=1)

# # ============================================
# # 4. PLOT BASELINE DASHED LINES
# # ============================================
# # Iterate through the calculated baselines and draw vertical lines
# for learner, score in baseline_means.items():
#     line_color = color_map.get(learner, 'black')
#     if learner == 'Ridge': line_color = '#0033CC' # Electric Blue per your style

#     plt.axvline(
#         x=score, 
#         color=line_color, 
#         linestyle='--', 
#         linewidth=1.5, 
#         alpha=0.8,
#         zorder=0 # Ensure lines are behind the error bars
#     )

# # ============================================
# # 5. STYLING & AESTHETICS (Matching the reference)
# # ============================================
# # Heavier left spine, hide others
# ax.spines['left'].set_linewidth(2)
# ax.spines['left'].set_color('black')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.tick_params(axis='y', which='major', left=True, length=5, width=1.5, color='black')
# plt.xlabel("Average Score (R2 & AUC)", fontsize=14, labelpad=15, x=-0.25, ha='left')
# plt.ylabel("Encoder (Num+Str)", fontsize=14) # Y-labels are self-explanatory
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# learner_handles = []
# for learner in unique_learners:
#     color = color_map[learner]
#     # Create a handle with ONLY the square marker (representing the bars)
#     # linestyle='' removes the line, marker='s' adds the square
#     handle = mlines.Line2D([], [], color=color, marker='s', linestyle='', 
#                           markersize=10, label=learner)
#     learner_handles.append(handle)

# # Draw the FIRST legend (Learners)
# # Position it slightly higher (y=0.18) to leave room for the second one below
# first_legend = plt.legend(
#     handles=learner_handles,
#     title="Tabular Learner",
#     loc='center left',
#     bbox_to_anchor=(1.05, 0.5), 
#     fontsize=10,
#     title_fontsize=11,
#     frameon=True
# )

# # CRITICAL: Add this legend manually to the plot so it isn't deleted 
# # when we create the second one.
# ax.add_artist(first_legend)

# # ============================================
# # 2. CREATE SECOND LEGEND (NUM-ONLY DASHED)
# # ============================================
# # Create a single handle for the black dashed line
# dash_handle = mlines.Line2D([], [], color='black', linestyle='--', 
#                            linewidth=2, label='Num-only')

# # Draw the SECOND legend below the first (y=0.05)
# plt.legend(
#     handles=[dash_handle],
#     loc='center left',
#     bbox_to_anchor=(1.05, 0.5),
#     fontsize=10,
#     frameon=True
# )
# plt.tight_layout(rect=[0, 0, 0.75, 1])

# today_date = time.strftime("%Y-%m-%d")
# format = 'pdf'
# PIC_NAME = f'avg_performance_dtypes_vs_numerical_{today_date}.{format}'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
# plt.show()


# results = pd.read_csv('/data/parietal/store4/soda/gblayer/salts/results/compiled_results/result_comparison.csv')

# results['score'] = results['r2'].fillna(results['roc_auc'])

# meta = results['method'].str.split('_', expand=True, n=2)

# results['dtype'] = meta[0]
# results['encoder'] = meta[1]
# results['learner'] = meta[2]

# # 2. DEFINE YOUR MAPPINGS
# dtype_map = {
#     'num-str': 'Num+Str',
#     'num-only': 'Num',
#     'str-only': 'Str'
# }

# encoder_map = {
#     'tabvec': 'Tf-Idf',
#     'tarenc': 'TargetEncoder',
#     'catboost': 'CatBoost'
# }

# learner_map = {
#     'ridge': 'Ridge', 
#     'xgb': 'XGBoost', 
#     'tabpfn': 'TabPFNv2.5',
#     'extrees': 'ExtraTrees',
#     'realmlp': 'RealMLP',
#     'catboost': 'CatBoost'
# }

# # 3. APPLY MAPPINGS
# results['dtype'] = results['dtype'].replace(dtype_map)
# results['encoder'] = results['encoder'].replace(encoder_map)
# results['learner'] = results['learner'].replace(learner_map)

# def robust_minmax(x):
#     x_clipped = x.clip(lower=0.0)
#     if x_clipped.max() == x_clipped.min():
#         return 1.0
#     return (x_clipped - x_clipped.min()) / (x_clipped.max() - x_clipped.min())

# # Apply this function to each group
# results['score_norm'] = results.groupby(['data_name', 'task'])['score'].transform(robust_minmax)

# results['method_polished'] = results['encoder'] + ' - ' + results['learner'] + '\n(' + results['dtype'] + ')'


# 1. Prepare Data for Regression (R2)
# subset_r2 = results[
#     (results['dtype'] == 'Num+Str') &
#     (results['encoder'] == 'Tf-Idf') &
#     (results['learner'] == 'Ridge') &
#     (results['task'] == 'regression')
# ].groupby(['data_name'], as_index=False)['score_norm'].mean()

# # 2. Prepare Data for Classification (AUC)
# # We select everything that is NOT regression (assuming others are classification tasks)
# subset_auc = results[
#     (results['dtype'] == 'Num+Str') &
#     (results['encoder'] == 'Tf-Idf') &
#     (results['learner'] == 'Ridge') &
#     (results['task'] != 'regression') 
# ].groupby(['data_name'], as_index=False)['score_norm'].mean()

# # 3. Draw Distribution Plot
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(8, 6))

# # Plot R2 Distribution (Blue)
# sns.histplot(
#     data=subset_r2,
#     x='score_norm',
#     bins=15,
#     kde=True,
#     color='skyblue',
#     edgecolor='black',
#     alpha=0.6,               # Transparency helps visibility when they overlap
#     label='Regression (R2)'  # Label for the legend
# )

# # Plot AUC Distribution (Orange)
# sns.histplot(
#     data=subset_auc,
#     x='score_norm',
#     bins=15,
#     kde=True,
#     color='orange',
#     edgecolor='black',
#     alpha=0.6,
#     label='Classification (AUC)'
# )

# # 4. Styling
# ax = plt.gca()
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['left'].set_color('black')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

# plt.xlabel("Normalized Score", fontsize=12) # Generic label since we plot both
# plt.ylabel("Frequency", fontsize=12)
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)

# # Add Legend
# plt.legend(fontsize=12, frameon=True)

# plt.tight_layout()
# plt.show()



# '''
# #### NOT-NORMALISED SCORE PLOTS
# '''

# '''
# Overall ridge performance per dtypes-encoders
# '''

# order = results.groupby('method')['score'].mean().sort_values(ascending=False).index
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(8, 6))
# ax = sns.barplot(
#     data=results,       
#     x='score', 
#     y='method', 
#     order=order,       
#     palette="RdYlBu",      
#     errorbar='sd',    
#     capsize=0.0,           
#     edgecolor="none"       
# )
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['left'].set_color('black')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.xlabel("Average Normalised Score (AUC and R2)", fontsize=12)
# plt.ylabel("")
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
# plt.setp(ax.lines, color='black', linewidth=1.5) 
# plt.tight_layout()


# #save picture
# # format fot the pic name: plot_type + _ + metric + _ + level + date .png
# today_date = time.strftime("%Y-%m-%d")
# PIC_NAME = f'horizontal_barplot_avg_minmax_score_method_{today_date}.png'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/' + PIC_NAME, dpi=300, bbox_inches='tight')
# plt.show()

# '''
# Performance per dtypes-encoders subplots
# '''

# dtype_order = results['dtype'].unique().tolist()

# fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

# sns.set_theme(style="whitegrid")

# for ax, d_type in zip(axes, dtype_order):
    
#     # Filter data
#     subset = results[results['dtype'] == d_type]
    
#     if subset.empty:
#         ax.set_visible(False) # Hide empty plots
#         continue

#     # Determine sorting order (Best encoders on top)
#     order = subset.groupby('encoder')['score'].mean().sort_values(ascending=False).index

#     # Create the Bar Plot
#     sns.barplot(
#         data=subset,
#         y='encoder',
#         x='score',
#         hue='learner',
#         order=order,
#         palette='tab10',
#         errorbar='sd',
#         capsize=0.05,
#         edgecolor='none',
#         ax=ax  # <--- Important: Plot on the specific subplot axis
#     )
    
#     # 1. Clean Spines (Borders)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['left'].set_color('black')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
    
#     # 2. Titles and Labels
#     ax.set_title(d_type, fontsize=16, pad=15, fontweight='bold')
#     ax.set_xlabel("Avg Normalized Score", fontsize=12)
    
#     # 3. Handle Y-Axis Labels (Encoders)
#     if d_type == 'num-only':
#         # specific request: drop labels for num-only
#         ax.set_ylabel("")
#         ax.set_yticklabels([]) 
#         ax.tick_params(axis='y', length=0) # remove tick marks
#     else:
#         ax.set_ylabel("") # Remove the word "encoder", keep the names
#         ax.tick_params(axis='y', labelsize=11)

#     if ax.get_legend():
#         ax.get_legend().remove()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, title="Learner", loc='center right', bbox_to_anchor=(1.08, 0.5), fontsize=12)

# plt.tight_layout()


# today_date = time.strftime("%Y-%m-%d")
# PIC_NAME = f'horizontal_barplot_perdtype_avg_minmax_score_encoder_learner_{today_date}.png'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/' + PIC_NAME, dpi=300, bbox_inches='tight')
# plt.show()

# '''
# Average performance when us-
# ing text-only (Str), numerical-only (dashed
# lines), or combined (Num+Str) features.
# '''


# df = results.copy()


# # 1b. Separate data for Bars vs. Baselines (Dashed lines)
# # Data for the main bars (num-str and str-only)
# plot_data = df[df['dtype'].isin(['num-str', 'str-only'])].copy()

# # Create the combined Y-axis label: e.g., "num-str (tabvec)"
# # We capitalize dtype for simpler presentation like the example images
# plot_data['y_label'] = plot_data['dtype'].str.title() + " (" + plot_data['encoder'] + ")"

# # Data for the baselines (num-only)
# baseline_data = df[df['dtype'] == 'num-only']

# # 1c. Calculate Baseline Averages per Learner
# # We average across all encoders/folds for num-only to get one line per learner
# baseline_means = baseline_data.groupby('learner')['score'].mean()

# # ============================================
# # 2. PLOTTING SETUP
# # ============================================
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(10, 7))

# # Define a consistent color palette for the learners
# unique_learners = sorted(df['learner'].unique())
# palette = sns.color_palette("tab10", n_colors=len(unique_learners))
# color_map = dict(zip(unique_learners, palette))

# # Determine sorting order for Y-axis based on average performance
# order = plot_data.groupby('y_label')['score'].mean().sort_values(ascending=False).index

# # ============================================
# # 3. PLOT MAIN BARS
# # ============================================
# ax = sns.barplot(
#     data=plot_data,
#     y='y_label',
#     x='score',
#     hue='learner',
#     order=order,
#     palette=color_map, # Use our fixed color map
#     errorbar='sd',     # Standard deviation black lines
#     capsize=0.05,
#     edgecolor='none',
#     linewidth=1.5      # Thickness of error bars
# )

# # ============================================
# # 4. PLOT BASELINE DASHED LINES
# # ============================================
# # Iterate through the calculated baselines and draw vertical lines
# for learner, score in baseline_means.items():
#     color = color_map.get(learner, 'black') # Get the matching color
#     plt.axvline(
#         x=score, 
#         color=color, 
#         linestyle='--', 
#         linewidth=2, 
#         alpha=0.8,
#         zorder=3 # Ensure lines are behind the error bars
#     )

# # ============================================
# # 5. STYLING & AESTHETICS (Matching the reference)
# # ============================================
# # Heavier left spine, hide others
# ax.spines['left'].set_linewidth(2)
# ax.spines['left'].set_color('black')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

# plt.xlabel("Average Normalized Score (AUC & R2)", fontsize=13, fontweight='bold', labelpad=15)
# plt.ylabel("", fontsize=12) # Y-labels are self-explanatory
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# # Ensure error bars are black
# plt.setp(ax.lines, color='black')

# # ============================================
# # 6. CUSTOM LEGEND
# # ============================================
# # Create custom legend handles to clearly show what bars vs lines mean
# legend_handles = []

# # Header handle (invisible) just for the text
# legend_handles.append(mpatches.Patch(color='none', label='Learner (Bars vs. num-only dashed)'))

# for learner in unique_learners:
#     color = color_map[learner]
#     # Create a combined handle: A colored patch representing the bar, 
#     # and a colored dashed line representing the baseline.
#     # We use Line2D for this custom look.
#     handle = mlines.Line2D([], [], color=color, marker='s', markersize=10, 
#                            linestyle='--', linewidth=2, label=learner)
#     legend_handles.append(handle)

# # Remove default legend and add custom one
# if ax.get_legend():
#     ax.get_legend().remove()

# plt.legend(
#     handles=legend_handles,
#     loc='center left',
#     bbox_to_anchor=(1.02, 0.5),
#     frameon=True,
#     fontsize=11
# )

# plt.tight_layout()


# today_date = time.strftime("%Y-%m-%d")
# PIC_NAME = f'avg_performance_dtypes_vs_numerical_minmax_{today_date}.png'
# plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/' + PIC_NAME, dpi=300, bbox_inches='tight')
# plt.show()


# ============================================
# 6. CUSTOM LEGEND
# ============================================
# Create custom legend handles to clearly show what bars vs lines mean
# legend_handles = []

# # Header handle (invisible) just for the text
# legend_handles.append(mpatches.Patch(color='none', label='Learner (Bars vs. num-only dashed)'))

# for learner in unique_learners:
#     color = color_map[learner]
#     # Create a combined handle: A colored patch representing the bar, 
#     # and a colored dashed line representing the baseline.
#     # We use Line2D for this custom look.
#     handle = mlines.Line2D([], [], color=color, marker='s', markersize=10, 
#                            linestyle='--', linewidth=2, label=learner)
#     legend_handles.append(handle)

# # Remove default legend and add custom one
# if ax.get_legend():
#     ax.get_legend().remove()

# plt.legend(
#     handles=legend_handles,
#     loc='lower right',
#     bbox_to_anchor=(1.0, 0.05),
#     fontsize=12,
#     title="Learner",
#     title_fontsize=13,
#     frameon=True,
# )
