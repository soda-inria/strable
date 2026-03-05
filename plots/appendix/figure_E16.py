import pandas as pd
import numpy as np
import re
import nltk
import os
from scipy.io import arff
from skrub import TableVectorizer
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from strable.configs.path_configs import path_configs
from strable.configs.exp_configs import wide_datasets
from strable.scripts.analysis_setup import TODAYS_FOLDER
import time


# Setup NLTK (Using your local folder)
nltk.data.path.insert(0, '/data/parietal/store4/soda/gblayer/salts/nltk_data')
from nltk.corpus.reader import WordListCorpusReader

# Load dictionary manually
try:
    reader = WordListCorpusReader('/data/parietal/store4/soda/gblayer/salts/nltk_data/corpora/words', ['en'])
    english_vocab = set(reader.words())
    print(f"✅ Loaded {len(english_vocab)} words from local NLTK.")
except Exception as e:
    print(f"⚠️ Could not load local NLTK: {e}. Attempting default download...")
    nltk.download('words')
    from nltk.corpus import words
    english_vocab = set(words.words())

# --- 1. METRIC FUNCTIONS ---

def get_naturalness_metrics(text_series):
    """Computes Dictionary Hit Rate and Symbol Density for a column."""
    # Clean and Sample (First 1000 rows)
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000: sample = sample.iloc[:1000]
    full_text = " ".join(sample.tolist())
    
    if not full_text: return 0.0, 0.0

    # A. Dictionary Hit Rate
    tokens = [t for t in re.split(r'[^a-zA-Z]+', full_text.lower()) if len(t) > 1]
    
    if tokens:
        hit_rate = sum(1 for t in tokens if t in english_vocab) / len(tokens)
    else:
        hit_rate = 0.0

    # B. Symbol Density
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', full_text))
    symbol_density = special_chars / len(full_text) if len(full_text) > 0 else 0.0

    return hit_rate, symbol_density

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

def get_vocab_size(text_series):
    """Count unique tokens across all values in a column."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000: sample = sample.iloc[:1000]
    full_text = " ".join(sample.tolist())
    tokens = [t.lower() for t in re.split(r'[^a-zA-Z]+', full_text) if len(t) > 1]
    return len(set(tokens))

def get_avg_token_count(text_series):
    """Average number of whitespace-separated tokens per string."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000: sample = sample.iloc[:1000]
    return sample.apply(lambda x: len(re.split(r'\s+', x.strip()))).mean()

def get_ngram_diversity(text_series, ngram_range=(2, 4)):
    """
    Computes the number of unique character n-grams (averaged per row)
    and the total unique n-gram vocabulary size across the column.
    """
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000: sample = sample.iloc[:1000]
    
    unique_ngrams_global = set()
    per_row_unique_counts = []
    
    for text in sample:
        row_ngrams = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            row_ngrams.update([text[i:i+n] for i in range(len(text) - n + 1)])
        per_row_unique_counts.append(len(row_ngrams))
        unique_ngrams_global.update(row_ngrams)
    
    avg_ngrams_per_row = np.mean(per_row_unique_counts) if per_row_unique_counts else 0.0
    total_unique_ngrams = len(unique_ngrams_global)
    
    return avg_ngrams_per_row, total_unique_ngrams

def append_metrics(results_list, results_complexity, results_structure,
                   benchmark, dataset, col, series, df, text_cols):
    hr, sd = get_naturalness_metrics(series)
    results_list.append({
        "dataset": dataset, "benchmark": benchmark,
        "dict_hit_rate": hr, "symbol_density": sd
    })
    uniq, ent = calculate_complexity(series)
    results_complexity.append({
        'benchmark': benchmark, 'dataset': dataset,
        'col': col, 'uniqueness': uniq, 'entropy': ent
    })
    vocab = get_vocab_size(series)
    avg_tokens = get_avg_token_count(series)
    text_col_ratio = len(text_cols) / len(df.columns) if len(df.columns) > 0 else 0.0
    avg_ngrams_per_row, total_unique_ngrams = get_ngram_diversity(series)  # NEW
    results_structure.append({
        'benchmark': benchmark, 'dataset': dataset, 'col': col,
        'vocab_size': vocab,
        'avg_token_count': avg_tokens,
        'text_col_ratio': text_col_ratio,
        'avg_ngrams_per_row': avg_ngrams_per_row,        # NEW
        'total_unique_ngrams': total_unique_ngrams        # NEW
    })

results_complexity = []
results_list = []
results_structure = []


def get_df_structure(cache: bool = True):
    """Compute (or load from cache) the structure DataFrame and associated frames.

    Returns:
        (df_structure, df_comp, df_results)
    """
    # If a cache is requested and exists, load and return it
    cache_dir = os.path.join(path_configs["base_path"], "__cache__")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"df_structure_{TODAYS_FOLDER}.parquet")
    if cache and os.path.exists(cache_file):
        df_structure = pd.read_parquet(cache_file)
        # Recreate df_comp and df_results from df_structure if needed (minimal)
        df_comp = pd.DataFrame(results_complexity)
        df_results = pd.DataFrame(results_list)
        return df_structure, df_comp, df_results

    # --- 2. PROCESSING ---

    # --- PART A: VSE DATASETS (Logic: TableVectorizer @ Threshold 30) ---
    # Download the datasets following the guidelines on https://github.com/LeoGrin/lm_tab/tree/reproduce
    vse_path = os.path.join(path_configs["base_path"], "VSE_datasets")

    vse_datasets = ["bikewale.parquet", "clear_corpus.parquet", "company_employees.parquet", 
                    "employee_salaries.arff", "employee-remuneration-and-expenses-earning-over-75000.parquet", 
                    "goodreads.parquet", "journal_jcr_cls.parquet", "spotify.parquet", 
                    "us_accidents_counts.parquet", "us_accidents_severity.parquet", 
                    "us_presidential.parquet", "ramen_ratings.parquet", 
                    "wine_review.parquet", "zomato.parquet"]

    print("--- Processing VSE Datasets ---")
    string_lengths_vse = []
    for filename in vse_datasets:
        try:
            path = os.path.join(vse_path, filename)
            if filename.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif filename.endswith(".arff"):
                # Try loading with explicit encoding
                try:
                    data, meta = arff.loadarff(path)
                except Exception:
                    # Fallback for encoding issues which often trigger the "not supported" error
                    with open(path, 'r', encoding='latin1') as f:
                        data, meta = arff.loadarff(f)
                df = pd.DataFrame(data)
                for col in df.select_dtypes([object]):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            else:
                continue

            cleaner = TableVectorizer(cardinality_threshold=30, high_cardinality="passthrough")
            df_processed = cleaner.fit_transform(df)

            text_cols = [c for c in df_processed.select_dtypes(['object', 'string']).columns]
            avg_length = df_processed[text_cols].astype(str).applymap(len).values.flatten().mean()
            string_lengths_vse.append(avg_length)

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'VSE', filename, col, df_processed[col], df_processed, text_cols)
        except Exception as e:
            print(f"Skipping VSE {filename}: {e}")

    # --- PART B: STRABLE DATASETS (Logic: TableVectorizer @ Threshold 0) ---
    print("--- Processing STRABLE Datasets ---")
    strable_path = path_configs["path_data_processed"]
    string_lengths_strable = []
    for data_name in wide_datasets:
        try:
            path = f"{strable_path}/{data_name}/data.parquet"
            df = pd.read_parquet(path)
            cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
            df_processed = cleaner.fit_transform(df)
            text_cols = [c for c in df_processed.select_dtypes(['object', 'string', 'category']).columns]
            avg_length = df_processed[text_cols].astype(str).applymap(len).values.flatten().mean()
            string_lengths_strable.append(avg_length)

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'STRABLE', data_name, col, df_processed[col], df_processed, text_cols)
        except Exception as e:
            print(f"Skipping STRABLE {data_name}: {e}")

    # --- PART C: CARTE DATASETS ---
    print("--- Processing CARTE Datasets ---")
    carte_path = os.path.join(path_configs["base_path"], "CARTE_datasets")
    carte_datasets = [f for f in os.listdir(carte_path) if f.endswith('.csv') or f.endswith('.json')]

    for filename in carte_datasets:
        print(f"Processing CARTE dataset: {filename}")
        try:
            path = os.path.join(carte_path, filename)
            if filename.endswith(".csv"):
                df = pd.read_csv(path, on_bad_lines='skip', engine='python', encoding='latin1')
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(path)
                except ValueError:
                    df = pd.read_json(path, lines=True)
            else:
                continue

            cleaner = TableVectorizer(cardinality_threshold=40, low_cardinality=OneHotEncoder(handle_unknown='ignore', sparse_output=False), high_cardinality="passthrough")
            df_processed = cleaner.fit_transform(df)
            text_cols = [c for c in df_processed.select_dtypes(['object', 'string']).columns]
            avg_length = df_processed[text_cols].astype(str).applymap(len).values.flatten().mean()

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'CARTE', filename, col, df_processed[col], df_processed, text_cols)
        except Exception as e:
            print(f"Skipping CARTE {filename}: {e}")

    # --- PART D: TEXTTABBENCH DATASETS ---
    print("--- Processing TextTabBench Datasets ---")
    ttb_path = os.path.join(path_configs["base_path"], "TTB_datasets", "paper_datasets")
    classif_path = os.path.join(ttb_path, 'classification')
    regress_path = os.path.join(ttb_path, 'regression')

    ttb_datasets_clf = [os.path.join(classif_path, f) for f in os.listdir(classif_path) if f.endswith('.csv')]
    ttb_datasets_reg = [os.path.join(regress_path, f) for f in os.listdir(regress_path) if f.endswith(('.csv', '.tsv'))]

    for filename in ttb_datasets_clf + ttb_datasets_reg:
        print(f"Processing TextTabBench dataset: {filename}")
        try:
            path = filename
            if filename.endswith(".csv"):
                df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            elif filename.endswith(".tsv"):
                df = pd.read_csv(path, sep='\t', encoding='latin1', on_bad_lines='skip')
            else:
                continue

            cleaner = TableVectorizer(high_cardinality="passthrough")
            df_processed = cleaner.fit_transform(df)
            text_cols = [c for c in df_processed.select_dtypes(['object', 'string']).columns]

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'TTB', filename, col, df_processed[col], df_processed, text_cols)
        except Exception as e:
            print(f"Skipping TTB {filename}: {e}")

    df_comp = pd.DataFrame(results_complexity)
    df_results = pd.DataFrame(results_list)
    df_structure = pd.DataFrame(results_structure)  # NEW

    # Cache df_structure so other scripts can reuse it without re-running heavy processing
    try:
        df_structure.to_parquet(cache_file)
        print(f"Cached df_structure to {cache_file}")
    except Exception as e:
        print(f"Warning: could not write df_structure cache: {e}")

    return df_structure, df_comp, df_results


if __name__ == "__main__":
    # Ensure we have the computed dataframes (this will compute + cache if necessary)
    df_structure, df_comp, df_results = get_df_structure()

    plt.figure(figsize=(5, 3))

    # VSE Density
    sns.kdeplot(data=df_comp[df_comp['benchmark']=='VSE'], x='entropy', 
                label='VSE (med=0.89)', fill=True, color='#1f77b4', linewidth=2)

    # STRABLE Density
    sns.kdeplot(data=df_comp[df_comp['benchmark']=='STRABLE'], x='entropy', 
                label='STRABLE (med=0.73)', fill=True, color='#ff7f0e', linewidth=2)

    # CARTE Density
    sns.kdeplot(data=df_comp[df_comp['benchmark']=='CARTE'], x='entropy', 
                label='CARTE (med=0.82)', fill=True, color='#2ca02c', linewidth=2)

    # TTB Density
    sns.kdeplot(data=df_comp[df_comp['benchmark']=='TTB'], x='entropy', 
                label='TTB (med=0.79)', fill=True, color='#d62728', linewidth=2)

    plt.xlabel("Normalized Entropy (0=Category, 1=Unique ID)", fontsize=12)
    plt.ylabel("Density of Columns", fontsize=12)
    plt.legend()
    plt.xlim(-0.1, 1.1)

    fmt = 'pdf'
    # Use the central path configuration so a fresh clone will save figures inside the repo
    PIC_NAME = f'VSE_STRABLE_CARTE_TTB_entropy_dist_{TODAYS_FOLDER}.{fmt}'
    out_dir = os.path.join(path_configs["base_path"], "results_pics", TODAYS_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, PIC_NAME), bbox_inches='tight')
    plt.show()