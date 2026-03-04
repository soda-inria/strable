"""Shared setup, styling, and data loading for all plotting scripts."""

import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import hashlib
from matplotlib.font_manager import FontProperties
from strable.configs.path_configs import path_configs
import os
pd.set_option('display.max_columns', None)


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

TODAYS_FOLDER = time.strftime("%Y-%m-%d")
os.makedirs(path_configs["base_path"] + '/results_pics/'+ TODAYS_FOLDER, exist_ok=True)
os.makedirs(path_configs["base_path"] + '/results_tables/'+ TODAYS_FOLDER, exist_ok=True)
