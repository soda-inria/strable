"""Compare string features of STRABLE vs VSE / CARTE / TTB.

Per text column we compute a metrics (dictionary hit rate, symbol density, stopword density, ngram diversity,
multi-word ratio, datetime hit rate, …), then classify each column into
one of six taxonomy buckets:
``categorical | name | structured_code | free_text | identifier | datetime``.

Outputs (in order they are produced):
* ``df_complexity_VSE_STRABLE_CARTE_TTB.csv`` — per-column complexity
  (uniqueness, entropy, n_unique, n_rows). 
* ``df_structure_VSE_STRABLE_CARTE_TTB.csv`` — per-column structural /
  semantic metrics including ``col_type_heuristic``. 
* ``VSE_STRABLE_CARTE_TTB_uniqueness_<date>.pdf`` — KDE of per-column
  uniqueness across the four benchmarks (paper figure
  ``fig:uniqueness_ratio_comparison_4_benchmarks``).
* ``4_benchmark_comparison_<date>.tex`` — median structural
  characteristics LaTeX table (paper appendix
  ``tab:four_benchmark_comparison``).
* STRABLE column-type distribution LaTeX (paper
  ``tab:strable_column_types``), printed and saved alongside.

Benchmark folders (configured via ``configs.path_configs``):
* STRABLE  → ``path_configs['path_data_processed']``
* VSE      → ``path_configs['vse_datasets']``
* CARTE    → ``path_configs['carte_datasets']``
* TTB      → ``path_configs['ttb_datasets']``
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import os
import re
import time
import zlib

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
from nltk.corpus import stopwords
from nltk.corpus.reader import WordListCorpusReader
from scipy.io import arff
from skrub import TableVectorizer
from sklearn.preprocessing import OneHotEncoder

from configs.path_configs import path_configs
from scripts.datasets_metadata_recap import wide_datasets


# ---------------------------------------------------------------------------
# 0. NLTK setup
# ---------------------------------------------------------------------------
nltk.data.path.insert(0, f"{path_configs['nltk_data']}")

try:
    reader = WordListCorpusReader(f"{path_configs['nltk_data']}/corpora/words", ['en'])
    english_vocab = set(reader.words())
    print(f"✅ Loaded {len(english_vocab)} words from local NLTK.")
except Exception as e:
    print(f"⚠️ Could not load local NLTK: {e}. Attempting default download...")
    nltk.download('words')
    from nltk.corpus import words
    english_vocab = set(words.words())

try:
    stop_words = set(stopwords.words('english'))
    print(f"✅ Loaded {len(stop_words)} stopwords.")
except Exception as e:
    print(f"⚠️ Could not find stopwords: {e}. Downloading now...")
    nltk.download('stopwords', download_dir=f"{path_configs['nltk_data']}")
    stop_words = set(stopwords.words('english'))
    print(f"✅ Successfully downloaded and loaded {len(stop_words)} stopwords.")


# ---------------------------------------------------------------------------
# 1. Per-column metric functions (sampling cap = 1000 rows everywhere)
# ---------------------------------------------------------------------------
DATETIME_PATTERNS = [
    r'\d{4}-\d{2}-\d{2}',                          # 2024-03-28  (ISO date)
    r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}',           # 2024-03-28T14:30 (ISO datetime)
    r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',        # 28/03/2024, 3-28-24
    r'\d{2}:\d{2}(:\d{2})?(\s?[APap][Mm])?',        # 14:30:00, 2:30 PM
    r'\d{1,2}\s+\w{3,9}\s+\d{4}',                  # 28 March 2024
    r'\w{3,9}\s+\d{1,2},?\s+\d{4}',                # March 28, 2024
]
_COMPILED = [re.compile(p) for p in DATETIME_PATTERNS]


def get_datetime_proportion(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    hits = non_null.astype(str).apply(
        lambda v: any(r.search(v) for r in _COMPILED)
    )
    return hits.mean()


def get_semantic_depth_metrics(text_series):
    """Stopword Density and Compression Ratio on the first 1000 rows.
    High semantic richness = high stopword density, low compression ratio.
    """
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]

    full_text = " ".join(sample.tolist())
    if not full_text:
        return 0.0, 1.0

    tokens = [t.lower() for t in re.split(r'[^a-zA-Z]+', full_text) if len(t) > 0]
    stop_density = (
        sum(1 for t in tokens if t in stop_words) / len(tokens) if tokens else 0.0
    )

    bytes_text = full_text.encode('utf-8')
    compression_ratio = (
        min(len(zlib.compress(bytes_text)) / len(bytes_text), 1.0)
        if len(bytes_text) >= 100
        else 1.0
    )
    return stop_density, compression_ratio


def compute_prose_score(text_series):
    """Prose score = stopword_density × avg_words_per_cell.
    Captures combined grammatical structure + length signal.
    """
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]

    word_counts = sample.apply(lambda x: len(x.split()))
    avg_words = word_counts.mean()

    full_text = " ".join(sample.tolist())
    tokens = [t.lower() for t in re.split(r'[^a-zA-Z]+', full_text) if len(t) > 0]
    stop_density = (
        sum(1 for t in tokens if t in stop_words) / len(tokens) if tokens else 0.0
    )
    return stop_density * avg_words, stop_density, avg_words


def get_vowel_density(text_series):
    """Phonological vowel density (proxy for natural-language strings)."""
    sample_series = text_series.dropna().astype(str)
    if len(sample_series) > 1000:
        sample_series = sample_series.iloc[:1000]
    sample = " ".join(sample_series.tolist()).lower()
    if not sample:
        return 0.0
    vowels = len(re.findall(r'[aeiou]', sample))
    alphas = len(re.findall(r'[a-z]', sample))
    return vowels / alphas if alphas > 0 else 0.0


def get_naturalness_metrics(text_series):
    """Dictionary Hit Rate and Symbol Density."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    full_text = " ".join(sample.tolist())
    if not full_text:
        return 0.0, 0.0

    MIN_TOKEN_LEN = 1
    tokens = [t for t in re.split(r'[^a-zA-Z]+', full_text.lower()) if len(t) >= MIN_TOKEN_LEN]
    hit_rate = (
        sum(1 for t in tokens if t in english_vocab) / len(tokens) if tokens else 0.0
    )
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', full_text))
    symbol_density = special_chars / len(full_text) if full_text else 0.0
    return hit_rate, symbol_density


def calculate_complexity(series):
    """Returns (uniqueness, normalized_entropy, n_unique, n_rows)."""
    s = series.dropna().astype(str)
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0, 0
    counts = s.value_counts()
    n_unique = len(counts)
    uniqueness = n_unique / n
    if n_unique <= 1:
        norm_entropy = 0.0
    else:
        probs = counts / n
        norm_entropy = scipy.stats.entropy(probs) / np.log(n_unique)
    return uniqueness, norm_entropy, n_unique, n


def get_global_vocab_diversity(text_series):
    """Ratio of unique alphabetic tokens to total tokens in the first-1000 sample."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    full_text = " ".join(sample.tolist())
    tokens = [t for t in re.split(r'[^a-zA-Z]+', full_text.lower()) if len(t) >= 1]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def get_avg_unique_vocab(text_series):
    """Average number of unique tokens per individual cell."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]

    def count_unique(text):
        tokens = [t for t in re.split(r'[^a-zA-Z]+', text.lower()) if len(t) >= 1]
        return len(set(tokens))

    return sample.apply(count_unique).mean()


def get_avg_token_count(text_series):
    """Average number of whitespace-separated tokens per cell."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    return sample.apply(lambda x: len(re.split(r'\s+', x.strip()))).mean()


def get_ngram_diversity(text_series, ngram_range=(2, 4)):
    """Average unique n-grams per cell, and global unique-to-total n-gram ratio."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]

    unique_ngrams_global = set()
    per_row_unique_counts = []
    total_ngrams_generated = 0

    for text in sample:
        row_ngrams = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            cleaned = re.sub(r'[^a-zA-Z]', '', text).lower()
            ngrams = [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]
            total_ngrams_generated += len(ngrams)
            row_ngrams.update(ngrams)
        per_row_unique_counts.append(len(row_ngrams))
        unique_ngrams_global.update(row_ngrams)

    avg_ngrams_per_row = np.mean(per_row_unique_counts) if per_row_unique_counts else 0.0
    global_ngram_ratio = (
        len(unique_ngrams_global) / total_ngrams_generated
        if total_ngrams_generated > 0
        else 0.0
    )
    return avg_ngrams_per_row, global_ngram_ratio


def get_prop_multiword(text_series):
    """Proportion of cells with at least one whitespace (multi-word entries)."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    return sample.str.contains(r'\s', regex=True).mean()


def get_prop_numeric_chars(text_series):
    """Proportion of all characters that are digits."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    full_text = "".join(sample.tolist())
    if not full_text:
        return 0.0
    return sum(c.isdigit() for c in full_text) / len(full_text)


def get_prop_uppercase(text_series):
    """Proportion of cells with ≥ 70% of alphabetic characters in uppercase."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]

    def is_mostly_upper(s):
        alpha = [c for c in s if c.isalpha()]
        if len(alpha) < 2:
            return False
        return sum(c.isupper() for c in alpha) / len(alpha) >= 0.7

    return sample.apply(is_mostly_upper).mean()


_PATTERN_RE = re.compile(
    r'\b\d{4}-\d{2}-\d{2}\b'           # ISO date
    r'|\b\d{5}(-\d{4})?\b'             # US zip code
    r'|\$[\d,]+(\.\d+)?'               # currency
    r'|\b\d+(\.\d+)?%'                 # percentage
    r'|\b[A-Z]\d{2}(\.\d+)?\b'         # ICD-style codes (e.g. J45.5)
    r'|\b[A-Z]{2,}-\d+\b'              # alphanumeric codes (e.g. NDC-12345)
)


def get_prop_has_pattern(text_series):
    """Proportion of cells matching a structured pattern (date, zip, currency, …)."""
    sample = text_series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    return sample.apply(lambda x: bool(_PATTERN_RE.search(x))).mean()


# ---------------------------------------------------------------------------
# 2. Aggregation helpers
# ---------------------------------------------------------------------------
def append_metrics(results_list, results_complexity, results_structure,
                   benchmark, dataset, col, series, df, text_cols):
    """Compute every metric and append one row each to the three accumulators."""
    hr, sd                           = get_naturalness_metrics(series)
    uniq, ent, n_unique, n_rows      = calculate_complexity(series)
    vocab_ratio                      = get_global_vocab_diversity(series)
    avg_unique_vocab                 = get_avg_unique_vocab(series)
    avg_tokens                       = get_avg_token_count(series)
    avg_ngrams_per_row, global_ngram_ratio = get_ngram_diversity(series)
    cell_lexical_density             = avg_unique_vocab / avg_tokens if avg_tokens > 0 else 0.0
    sample = series.dropna().astype(str)
    if len(sample) > 1000:
        sample = sample.iloc[:1000]
    avg_chars                        = sample.str.len().mean()
    stop_density, compress_ratio     = get_semantic_depth_metrics(series)
    v_density                        = get_vowel_density(series)
    text_col_ratio                   = len(text_cols) / len(df.columns) if len(df.columns) > 0 else 0.0
    prop_multiword                   = get_prop_multiword(series)
    prop_numeric_chars               = get_prop_numeric_chars(series)
    prop_uppercase                   = get_prop_uppercase(series)
    prop_has_pattern                 = get_prop_has_pattern(series)
    prop_is_datetime                 = get_datetime_proportion(series)
    prose_score, _, _                = compute_prose_score(series)

    results_list.append({
        "dataset": dataset, "benchmark": benchmark,
        "dict_hit_rate": hr, "symbol_density": sd,
    })
    results_complexity.append({
        'benchmark': benchmark, 'dataset': dataset, 'col': col,
        'uniqueness': uniq, 'entropy': ent, 'n_unique': n_unique, 'n_rows': n_rows,
    })
    results_structure.append({
        'benchmark': benchmark, 'dataset': dataset, 'col': col,
        'avg_words_per_cell':            avg_tokens,
        'avg_chars_per_cell':            avg_chars,
        'global_vocab_diversity_ratio':  vocab_ratio,
        'avg_unique_words_per_cell':     avg_unique_vocab,
        'cell_lexical_density':          cell_lexical_density,
        'avg_unique_ngrams_per_cell':    avg_ngrams_per_row,
        'global_ngram_diversity_ratio':  global_ngram_ratio,
        'text_col_ratio':                text_col_ratio,
        'stopword_density':              stop_density,
        'compression_ratio':             compress_ratio,
        'vowel_density':                 v_density,
        'dict_hit_rate':                 hr,
        'symbol_density':                sd,
        'uniqueness':                    uniq,
        'entropy':                       ent,
        'n_unique':                      n_unique,
        'n_rows_col':                    n_rows,
        'prop_multiword':                prop_multiword,
        'prop_numeric_chars':            prop_numeric_chars,
        'prop_uppercase':                prop_uppercase,
        'prop_has_pattern':              prop_has_pattern,
        'prop_is_datetime':              prop_is_datetime,
        'prose_score':                   prose_score,
    })


def classify_column_heuristic(uniqueness, entropy, dict_hit_rate, symbol_density,
                              avg_words_per_cell, avg_chars_per_cell,
                              prop_multiword, prop_numeric_chars, prop_uppercase,
                              prop_has_pattern, prop_is_datetime,
                              stopword_density=0.0, compression_ratio=1.0):
    """Classify a string column into one of:
    ``datetime | structured_code | identifier | free_text | name | categorical``.
    Order of rules matters — most specific first.
    """
    # Rule 0: datetime
    if prop_is_datetime >= 0.7:
        return 'datetime'

    # Rule 1: structured code — patterns / digits / symbols dominate.
    # symbol_density >= 0.15 alone catches URLs (high symbol but possibly high
    # dict_hit_rate). Guard with uniqueness > 0.01 so 2-4-value categoricals
    # like "100-161 / UNDER 100 / 220 AND ABOVE" don't get pulled in.
    is_structured = (
        prop_has_pattern >= 0.2
        or prop_numeric_chars >= 0.25
        or symbol_density >= 0.15
        or (symbol_density >= 0.1 and dict_hit_rate < 0.3)
    )
    if is_structured and uniqueness >= 0.02:
        return 'structured_code'

    # Rule 2: identifier — near-unique, short, not natural language
    if uniqueness >= 0.8 and dict_hit_rate < 0.3 and avg_words_per_cell < 3:
        return 'identifier'

    # Rule 3: free text — prose with real stopwords (not just multi-word names)
    if (avg_words_per_cell >= 4
            and dict_hit_rate >= 0.4
            and prop_multiword >= 0.8
            and stopword_density >= 0.1):
        return 'free_text'

    # Rule 4: name — high dict_hit_rate + ~2 words = person/place name even at
    # low uniqueness (e.g. senators repeated across many rows still look like
    # names); fall back to original moderate-uniqueness rule.
    if (dict_hit_rate >= 0.5
            and 1.5 <= avg_words_per_cell <= 4
            and prop_multiword >= 0.5
            and prop_numeric_chars < 0.1):
        return 'name'

    if uniqueness >= 0.3 and dict_hit_rate >= 0.3 and avg_words_per_cell < 4:
        return 'name'

    return 'categorical'


def _classify_last(results_structure, last_idx=-1):
    """Run the heuristic on the most recently appended structure row and write
    the result back as ``col_type_heuristic``. Centralized here so all four
    benchmark loops route through the same kwargs and the same field name.
    """
    last = results_structure[last_idx]
    col_type = classify_column_heuristic(
        uniqueness=last['uniqueness'],
        entropy=last['entropy'],
        dict_hit_rate=last['dict_hit_rate'],
        symbol_density=last['symbol_density'],
        avg_words_per_cell=last['avg_words_per_cell'],
        avg_chars_per_cell=last['avg_chars_per_cell'],
        prop_multiword=last['prop_multiword'],
        prop_numeric_chars=last['prop_numeric_chars'],
        prop_uppercase=last['prop_uppercase'],
        prop_has_pattern=last['prop_has_pattern'],
        prop_is_datetime=last['prop_is_datetime'],
        stopword_density=last['stopword_density'],
        compression_ratio=last['compression_ratio'],
    )
    results_structure[last_idx]['col_type_heuristic'] = col_type


def compute_dataset_level_features(results_structure_df):
    """Aggregate per-column metrics to dataset-level features.
    Returns one row per dataset with column-type proportions and a composite
    ``semantic_score``.
    """
    df = results_structure_df.copy()

    type_dummies = pd.get_dummies(df['col_type_heuristic'], prefix='prop')
    df = pd.concat([df, type_dummies], axis=1)
    type_cols = [c for c in df.columns if c.startswith('prop_')]

    agg_dict = {
        **{c: 'mean' for c in type_cols},
        'stopword_density':            'mean',
        'dict_hit_rate':               'mean',
        'avg_words_per_cell':          'mean',
        'compression_ratio':           'mean',
        'prop_multiword':              'mean',
        'prop_numeric_chars':          'mean',
        'prop_uppercase':              'mean',
        'prop_has_pattern':            'mean',
        'global_vocab_diversity_ratio':'mean',
        'uniqueness':                  'mean',
    }

    dataset_df = df.groupby(['benchmark', 'dataset']).agg(agg_dict).reset_index()

    dataset_df['avg_words_norm'] = dataset_df['avg_words_per_cell'].clip(upper=20) / 20
    dataset_df['semantic_score'] = (
        dataset_df['stopword_density']
        + dataset_df['dict_hit_rate']
        + dataset_df['avg_words_norm']
    ) / 3.0
    return dataset_df


# ---------------------------------------------------------------------------
# 3. Per-benchmark processing
# ---------------------------------------------------------------------------
def process_strable(wide_datasets, strable_path, results_list, results_complexity,
                    results_structure):
    """STRABLE uses ``TableVectorizer(cardinality_threshold=0)`` and the raw
    parquet for each dataset. Returns a list of mean string-lengths (one per
    dataset) for downstream stats."""
    print("--- Processing STRABLE Datasets ---")
    string_lengths = []

    for data_name in wide_datasets:
        print(f"Processing STRABLE dataset: {data_name}")
        try:
            df = pd.read_parquet(f"{strable_path}/{data_name}/data.parquet")

            cleaner = TableVectorizer(
                cardinality_threshold=0,
                low_cardinality="passthrough",
                high_cardinality="passthrough",
                numeric="passthrough",
                datetime="passthrough",
            )
            cleaner.fit(df)
            text_cols = [
                c for c, kind in cleaner.column_to_kind_.items()
                if kind == 'high_cardinality'
            ]

            if not text_cols:
                continue

            avg_length = df[text_cols].astype(str).map(len).values.flatten().mean()
            string_lengths.append(avg_length)

            for col in text_cols:
                print(f"-> Inspecting text column: {col}")
                append_metrics(results_list, results_complexity, results_structure,
                               'STRABLE', data_name, col, df[col], df, text_cols)
                _classify_last(results_structure)
        except Exception as e:
            print(f"Skipping STRABLE {data_name}: {e}")

    return string_lengths


def process_vse(vse_path, vse_files, results_list, results_complexity,
                results_structure):
    """VSE uses ``TableVectorizer(cardinality_threshold=30)``. Iterates over a
    fixed file list (parquet + arff)."""
    print("--- Processing VSE Datasets ---")
    string_lengths = []

    for filename in vse_files:
        print(f"Processing VSE dataset: {filename}")
        try:
            path = os.path.join(vse_path, filename)
            if filename.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif filename.endswith(".arff"):
                try:
                    data, _meta = arff.loadarff(path)
                except Exception:
                    with open(path, 'r', encoding='latin1') as f:
                        data, _meta = arff.loadarff(f)
                df = pd.DataFrame(data)
                for col in df.select_dtypes([object]):
                    df[col] = df[col].apply(
                        lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                    )
            else:
                continue

            cleaner = TableVectorizer(cardinality_threshold=30, high_cardinality="passthrough")
            df_processed = cleaner.fit_transform(df)
            text_cols = [c for c in df_processed.select_dtypes(['object', 'string']).columns]
            if not text_cols:
                continue

            avg_length = df_processed[text_cols].astype(str).map(len).values.flatten().mean()
            string_lengths.append(avg_length)

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'VSE', filename, col, df_processed[col], df_processed, text_cols)
                _classify_last(results_structure)
        except Exception as e:
            print(f"Skipping VSE {filename}: {e}")

    return string_lengths


def process_carte(carte_path, results_list, results_complexity, results_structure):
    """CARTE uses ``TableVectorizer(cardinality_threshold=40,
    low_cardinality=OneHotEncoder, high_cardinality='passthrough')``."""
    print("--- Processing CARTE Datasets ---")
    carte_files = [
        f for f in os.listdir(carte_path)
        if f.endswith('.csv') or f.endswith('.json')
    ]
    string_lengths = []

    for filename in carte_files:
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

            cleaner = TableVectorizer(
                cardinality_threshold=40,
                low_cardinality=OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                high_cardinality="passthrough",
            )
            df_processed = cleaner.fit_transform(df)
            text_cols = [c for c in df_processed.select_dtypes(['object', 'string']).columns]
            if not text_cols:
                continue

            avg_length = df_processed[text_cols].astype(str).map(len).values.flatten().mean()
            string_lengths.append(avg_length)

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'CARTE', filename, col, df_processed[col], df_processed, text_cols)
                _classify_last(results_structure)
        except Exception as e:
            print(f"Skipping CARTE {filename}: {e}")

    return string_lengths


def process_ttb(ttb_path, results_list, results_complexity, results_structure):
    """TextTabBench uses ``TableVectorizer(high_cardinality='passthrough')``.
    Walks the ``classification/`` and ``regression/`` sub-folders."""
    print("--- Processing TextTabBench Datasets ---")
    classif_path = os.path.join(ttb_path, 'classification')
    regress_path = os.path.join(ttb_path, 'regression')
    ttb_files_clf = [
        os.path.join(classif_path, f)
        for f in os.listdir(classif_path)
        if f.endswith('.csv')
    ]
    ttb_files_reg = [
        os.path.join(regress_path, f)
        for f in os.listdir(regress_path)
        if f.endswith(('.csv', '.tsv'))
    ]
    string_lengths = []

    for path in ttb_files_clf + ttb_files_reg:
        print(f"Processing TextTabBench dataset: {path}")
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            elif path.endswith(".tsv"):
                df = pd.read_csv(path, sep='\t', encoding='latin1', on_bad_lines='skip')
            else:
                continue

            cleaner = TableVectorizer(high_cardinality="passthrough")
            df_processed = cleaner.fit_transform(df)
            text_cols = [c for c in df_processed.select_dtypes(['object', 'string']).columns]
            if not text_cols:
                continue

            avg_length = df_processed[text_cols].astype(str).map(len).values.flatten().mean()
            string_lengths.append(avg_length)

            for col in text_cols:
                append_metrics(results_list, results_complexity, results_structure,
                               'TTB', path, col, df_processed[col], df_processed, text_cols)
                _classify_last(results_structure)
        except Exception as e:
            print(f"Skipping TTB {path}: {e}")

    return string_lengths


VSE_FILES = [
    "bikewale.parquet", "clear_corpus.parquet", "company_employees.parquet",
    "employee_salaries.arff", "employee-remuneration-and-expenses-earning-over-75000.parquet",
    "goodreads.parquet", "journal_jcr_cls.parquet", "spotify.parquet",
    "us_accidents_counts.parquet", "us_accidents_severity.parquet",
    "us_presidential.parquet", "ramen_ratings.parquet",
    "wine_review.parquet", "zomato.parquet",
]


# ---------------------------------------------------------------------------
# 4. Output / paper-artifact builders
# ---------------------------------------------------------------------------
def export_strable_taxonomy_table(df_structure, out_path):
    """Print the STRABLE column-type distribution and save it as
    ``tab:strable_column_types`` LaTeX. ``df_structure`` is the column-level
    metrics dataframe (must contain ``benchmark`` and ``col_type_heuristic``).
    """
    type_counts = (
        df_structure.groupby(['benchmark', 'col_type_heuristic'])['col'].count()
        / df_structure.groupby(['benchmark'])['dataset'].nunique()
    )
    print("\n=== Column Type Distribution (per dataset, by benchmark) ===")
    print(type_counts)

    df_to_export = type_counts.to_frame(name='Count')
    df_to_export.index.name = None
    latex_str = df_to_export.to_latex(
        caption='Distribution of Column Types per dataset, by benchmark.',
        label='tab:strable_column_types',
        position='ht',
        bold_rows=True,
        column_format='lc',
        escape=False,
        index=True,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(latex_str)
    print(f"✅ Saved {out_path}")


def plot_uniqueness_kde(df_comp, save_path):
    """KDE of per-column uniqueness across the four benchmarks.
    Produces the paper figure ``fig:uniqueness_ratio_comparison_4_benchmarks``.
    Median is computed dynamically from ``df_comp``.
    """
    benchmarks = [
        ('VSE',     '#1f77b4'),
        ('STRABLE', '#ff7f0e'),
        ('CARTE',   '#2ca02c'),
        ('TTB',     '#d62728'),
    ]
    plt.figure(figsize=(5, 3))
    for name, color in benchmarks:
        sub = df_comp[df_comp['benchmark'] == name]
        if sub.empty:
            continue
        median_uniq = sub['uniqueness'].median()
        sns.kdeplot(
            data=sub, x='uniqueness',
            label=f'{name} (med={median_uniq:.3f})',
            fill=True, color=color, linewidth=2,
        )
    plt.xlabel(
        "Proportion of Unique Values\n"
        "0.0 = highly repetitive category / constants\n"
        "1.0 = every row unique",
        fontsize=12,
    )
    plt.ylabel("Density of Columns", fontsize=12)
    plt.legend()
    plt.xlim(0, 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {save_path}")


def export_4benchmark_median_table(df_structure, df_comp, save_path):
    """Build the median structural-characteristics LaTeX table
    (paper appendix ``tab:four_benchmark_comparison``)."""
    df_merged = pd.merge(
        df_structure, df_comp, how='left', on=['benchmark', 'dataset', 'col'],
    )
    df_median = df_merged.groupby('benchmark')[
        ['avg_words_per_cell', 'avg_chars_per_cell',
         'avg_unique_words_per_cell', 'avg_unique_ngrams_per_cell',
         'uniqueness', 'text_col_ratio']
    ].median()

    df_median.columns = [
        '\\shortstack{Avg Tokens\\\\/ Cell}',
        '\\shortstack{Avg Char\\\\/ Cell}',
        '\\shortstack{Avg Unique\\\\Words / Cell}',
        '\\shortstack{Avg Unique\\\\N-grams / Cell}',
        '\\shortstack{Prop. Unique\\\\Values}',
        '\\shortstack{Text Col\\\\Ratio}',
    ]
    df_median = df_median.round(2)

    latex_str = df_median.to_latex(
        float_format="%.2f",
        caption=(
            'Median structural characteristics of text columns across the four benchmarks. '
            'Avg Tokens / Cell: average number of whitespace-separated tokens per cell within a sample of 1000 rows. '
            'Avg Char / Cell: average number of characters per cell for a sample of 1000 rows. '
            r'Avg Unique Alphabetic Words / Cell: average number of unique, case-insensitive alphabetic sequences (length $\geq$ 2) per cell, computed over a random sample of 1000 rows. This excludes numbers, punctuation, and single-letter characters. '
            'Avg Unique N-grams / Cell: average number of unique character n-grams (length 2--4) per cell within a sample of 1000 rows. '
            'Proportion of Unique Values: the ratio of unique values to total values in a column, indicating how repetitive vs. unique the entries are. 0.0 means all values are the same (e.g., a column of constants), while 1.0 means every value is unique (e.g., unique IDs or rich text). '
            'Text Col Ratio: proportion of columns that are text. '
            'STRABLE shows the lowest metrics. In this regime, simple frequency-based methods such as Tf-Idf are well-suited and sufficient, as the discriminative signal is concentrated in a small set of recurring tokens rather than in semantic context — which explains the strong performance of Tf-Idf on the Pareto plot.'
        ),
        label='tab:four_benchmark_comparison',
        position='ht',
        bold_rows=True,
        escape=False,
        column_format='lcccccc',
    )
    latex_str = latex_str.replace(
        '\\begin{tabular}',
        '\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}',
    )

    # Bold the STRABLE row's numeric cells.
    lines = latex_str.split('\n')
    for i, line in enumerate(lines):
        if 'STRABLE' in line:
            lines[i] = re.sub(r'(\d+\.\d+)', r'\\textbf{\1}', line)
    latex_str = '\n'.join(lines)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(latex_str)
    print(f"✅ Saved {save_path}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results_complexity = []
    results_list = []
    results_structure = []

    process_strable(
        wide_datasets, path_configs['path_data_processed'],
        results_list, results_complexity, results_structure,
    )
    process_vse(
        path_configs['vse_datasets'], VSE_FILES,
        results_list, results_complexity, results_structure,
    )
    process_carte(
        path_configs['carte_datasets'],
        results_list, results_complexity, results_structure,
    )
    process_ttb(
        f"{path_configs['ttb_datasets']}/paper_datasets",
        results_list, results_complexity, results_structure,
    )

    df_comp      = pd.DataFrame(results_complexity)
    df_results   = pd.DataFrame(results_list)
    df_structure = pd.DataFrame(results_structure)

    # Drop datetime columns from downstream analyses (they aren't text).
    df_structure_no_dt = df_structure[df_structure['col_type_heuristic'] != 'datetime']

    # CSV exports consumed by figures/posthoc_analysis.py
    df_comp.to_csv(
        f"{path_configs['base_path']}/df_complexity_VSE_STRABLE_CARTE_TTB.csv",
        index=False,
    )
    df_structure.to_csv(
        f"{path_configs['base_path']}/df_structure_VSE_STRABLE_CARTE_TTB.csv",
        index=False,
    )
    print(f"✅ Saved df_complexity_VSE_STRABLE_CARTE_TTB.csv "
          f"and df_structure_VSE_STRABLE_CARTE_TTB.csv to {path_configs['base_path']}")

    today = time.strftime("%Y-%m-%d")
    pics_dir   = f"{path_configs['results_pics']}/{today}"
    tables_dir = f"{path_configs['results_tables']}/{today}"

    # Paper figure: uniqueness KDE across the four benchmarks
    plot_uniqueness_kde(
        df_comp,
        save_path=f"{pics_dir}/VSE_STRABLE_CARTE_TTB_uniqueness_{today}.pdf",
    )

    # Paper appendix: 4-benchmark median structural characteristics
    export_4benchmark_median_table(
        df_structure_no_dt, df_comp,
        save_path=f"{tables_dir}/4_benchmark_comparison_{today}.tex",
    )

    # Paper appendix: STRABLE column-type distribution
    export_strable_taxonomy_table(
        df_structure_no_dt,
        out_path=f"{tables_dir}/strable_column_types_{today}.tex",
    )
