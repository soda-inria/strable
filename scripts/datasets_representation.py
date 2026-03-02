
'''
CREATE SUMMARY_DF
'''

import os
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
pd.set_option('display.max_columns', None)
from configs.path_configs import path_configs


def get_text_column_names(pf: pq.ParquetFile) -> list[str]:
    """
    Return names of columns that are treated as text:
    - string / large_string
    - dictionary-encoded strings (categoricals)
    """
    text_cols = []
    schema = pf.schema_arrow
    for field in schema:
        t = field.type
        if (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_dictionary(t)
        ):
            text_cols.append(field.name)
    return text_cols


def compute_text_stats(pf: pq.ParquetFile, text_cols: list[str]) -> tuple[int, float, int, float, float, float, float]:
    """
    Compute text/categorical column statistics (operates on provided text_cols):
      - number of text columns
      - average string length over all non-null cells of those columns
      - total characters across those columns
      - cardinality: average number of unique values per text column
      - string similarity: average pairwise cosine similarity between rows
      - proportion missing: fraction of text cells that are missing
      - proportion unique: average (nunique / n_rows) across text columns

    Notes:
      - TF-IDF / cosine similarity is computed on the per-row concatenation of
        all text columns. For large tables we sample up to a fixed number of
        rows to control memory/time.
    """
    if not text_cols:
        return 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

    total_chars = 0
    total_cells = 0
    unique_counts = []
    total_missing = 0

    # Collect per-row concatenated text for TF-IDF (use pandas for convenience)
    concatenated_text = []
    num_rows = pf.metadata.num_rows

    for col in text_cols:
        # read the entire column into pandas Series (handles dictionary decoding)
        ser = pf.read(columns=[col]).to_pandas()[col]

        # Cardinality (unique non-null values)
        # sample_size = min(1024, len(ser))
        nunique = int(ser.nunique(dropna=True))
        unique_counts.append(nunique)

        # String length statistics via PyArrow for robustness on chunks
        # arr = ser._data if hasattr(ser, "_data") else None
        # Fallback: compute lengths in pandas
        # Use pandas to compute total characters and non-null counts
        ser_str = ser.dropna().astype(str)
        chars_sum = ser_str.map(len).sum()
        non_null = ser_str.shape[0]

        total_chars += int(chars_sum)
        total_cells += int(non_null)

        # missing count for this column
        missing = int(num_rows - non_null)
        total_missing += missing

        # Accumulate per-row text (as string; keep NaNs as empty strings)
        # We'll build a temporary list of strings per column and merge later
        if not concatenated_text:
            concatenated_text = ser.fillna("").astype(str).tolist()
        else:
            # append this column's text to existing per-row text
            col_text = ser.fillna("").astype(str).tolist()
            for i, t in enumerate(col_text):
                concatenated_text[i] = concatenated_text[i] + " " + t

    avg_len = float(total_chars / total_cells) if total_cells else 0.0

    # Cardinality = average number of unique values per text column
    avg_cardinality = float(sum(unique_counts) / len(unique_counts)) if unique_counts else 0.0

    # Proportion of missing values across all text cells
    total_text_cells = int(num_rows * len(text_cols)) if text_cols else 0
    prop_missing = float(total_missing / total_text_cells) if total_text_cells else 0.0

    # Proportion of unique values (average per-column unique fraction)
    if num_rows:
        prop_unique = float(sum((c / num_rows) for c in unique_counts) / len(unique_counts))
    else:
        prop_unique = 0.0

    # ---------------------------------------------------------
    # SUBSAMPLE FIRST 5000 ROWS FOR COSINE SIMILARITY, AND USE IT ALSO FOR THE N_GRAM
    # ---------------------------------------------------------
    n_all = len(concatenated_text)
    
    # Defaults (small dataset case)
    tfidf_texts = concatenated_text
    diversity_texts = concatenated_text
    
    # Random Number Generator
    rng = np.random.default_rng(12345)

    if n_all > 5000:
        # Case 1: Large Dataset (> 5000 rows)
        # Sample 5000 randomly for TF-IDF
        idx = rng.choice(n_all, size=5000, replace=False)
        tfidf_texts = [concatenated_text[i] for i in idx]
        
        # Subsample the first 1000 from those 5000 for Diversity
        # (Since tfidf_texts is already shuffled/random, taking the first 1000 is valid)
        diversity_texts = tfidf_texts[:1000]

    elif n_all > 1000:
        # Case 2: Medium Dataset (1000 < N <= 5000)
        # Use ALL rows for TF-IDF (no sampling needed)
        tfidf_texts = concatenated_text
        
        # But we still need a RANDOM 1000 for Diversity (to avoid sorting bias)
        idx_div = rng.choice(n_all, size=1000, replace=False)
        diversity_texts = [concatenated_text[i] for i in idx_div]

    # Case 3: Small Dataset (<= 1000) -> Use full data for both (already set by defaults)

    # ---------------------------------------------------------
    # ### NEW: String Diversity (N-grams per 1000 samples)
    # ---------------------------------------------------------
    string_diversity = 0
    try:
        if diversity_texts:
            
            # 1. Use Character analyzer with range 2-4 (as per paper)
            div_vec = CountVectorizer(
                analyzer='char',       # "based on characters"
                ngram_range=(2, 4),    # "between lengths 2 and 4"
                min_df=1,              # Keep ALL unique n-grams
                strip_accents='unicode' # Optional: handles accents consistently
            )
            
            # 2. Fit on the 1000 sampled rows
            div_vec.fit(diversity_texts)
            
            # 3. Calculate the TOTAL number of unique n-grams (Vocabulary Size)
            # This matches "number of unique n-grams for 1000 rows"
            string_diversity = len(div_vec.vocabulary_)
    except Exception:
        string_diversity = 0

    # ---------------------------------------------------------
    # TF-IDF Cosine Similarity
    # ---------------------------------------------------------
    avg_cosine = 0.0
    try:
        n = len(tfidf_texts)
        if n < 2:
            avg_cosine = 0.0
        else:
            # TF-IDF (keep features reasonably bounded)
            vec = TfidfVectorizer(max_features=20000)
            X = vec.fit_transform(tfidf_texts)  # shape (n, d), sparse

            # Ensure rows are L2-normalized (TfidfVectorizer does this by default)
            # Sum of all pairwise dot-products (including diagonal) = sum_k (sum_i X_ik)^2
            col_sums = np.asarray(X.sum(axis=0)).ravel()
            total_dot = float((col_sums ** 2).sum())

            # sum of diagonal entries = n (since rows are unit-norm)
            # average pairwise (off-diagonal) cosine similarity:
            avg_cosine = float((total_dot - n) / (n * (n - 1)))
    except Exception:
        # If sklearn or scipy missing or something else fails, leave as 0.0
        avg_cosine = 0.0

    return (len(text_cols),
        avg_len,
        total_chars,
        avg_cardinality,
        avg_cosine,
        prop_missing,
        prop_unique,
        string_diversity)

def compute_missingness_stats(pf, text_cols):
    """
    Returns:
      - prop_missing_total: dataset-wide missing ratio
      - prop_rows_affected: ratio of rows with at least one missing value
      - prop_missing_text: missing ratio specifically for string columns
    """
    num_rows = pf.metadata.num_rows
    num_cols = pf.metadata.num_columns
    total_cells = num_rows * num_cols
    
    # Read the whole table (or chunks if memory is an issue)
    # Using pandas is often easiest for complex row-wise null checks
    df = pf.read().to_pandas()
    
    # 1. Dataset-level total
    total_missing = df.isnull().sum().sum()
    prop_missing_total = float(total_missing / total_cells) if total_cells > 0 else 0.0
    
    # 2. Row-level impact (The "Sparsity" factor)
    rows_with_null = df.isnull().any(axis=1).sum()
    prop_rows_affected = float(rows_with_null / num_rows) if num_rows > 0 else 0.0
    
    # 3. Domain-specific (Text vs Num)
    if text_cols:
        text_missing = df[text_cols].isnull().sum().sum()
        prop_missing_text = float(text_missing / (num_rows * len(text_cols)))
    else:
        prop_missing_text = 0.0
        
    return prop_missing_total, prop_rows_affected, prop_missing_text

BASE_DIR = Path(path_configs["base_path"] + "/data/data_processed")

records = []

for root, dirs, files in os.walk(BASE_DIR):
    root_path = Path(root)
    files_set = set(files)
    print("Checking:", root_path, files_set)

    # Only process folders containing BOTH config.json and data.parquet
    if "config.json" in files_set and "data.parquet" in files_set:

        config_path = root_path / "config.json"
        parquet_path = root_path / "data.parquet"

        print("Processing:", parquet_path)

        # Dataset name = path after /data
        dataset_name = root_path.name

        # ---------------------------
        # Read CONFIG
        # ---------------------------
        with open(config_path, "r") as f:
            cfg = json.load(f)

        task = cfg.get("task")
        panel = (cfg.get("task_type") == "panel")
        target_column = cfg.get("target_name")
        source_column = cfg.get("source")

        # ---------------------------
        # Read PARQUET METADATA
        # ---------------------------
        pf = pq.ParquetFile(parquet_path)

        num_rows = pf.metadata.num_rows
        num_columns = pf.metadata.num_columns

        # text stats
        text_cols = get_text_column_names(pf)
        n_cols, avg_len, _, avg_card, avg_cos, prop_missing, prop_unique, str_div = compute_text_stats(pf, text_cols)
        prop_missing_total, prop_rows_affected, prop_missing_text = compute_missingness_stats(pf, text_cols)
        # ---------------------------
        # Save record
        # ---------------------------
        records.append(
            {
                "data_name": dataset_name,
                "data_path": str(parquet_path),
                "target_column": target_column,
                "source":source_column,
                "task": task,
                "panel": panel,
                "num_rows": num_rows,
                "num_columns": num_columns,
                "num_text_columns": n_cols,
                "avg_string_length_per_cell": avg_len,
                "avg_cardinality": avg_card,
                "avg_tfidf_cosine_similarity": avg_cos,
                "prop_missing_text_cells": prop_missing,
                "prop_unique_text_cells": prop_unique,
                "prop_missing_total": prop_missing_total,
                "prop_rows_affected": prop_rows_affected,
                "prop_missing_text": prop_missing_text,
                "string_diversity": str_div
            }
        )

# Final dataframe
summary_df = pd.DataFrame(records)

# check that columns are at dataset level (no duplicates per name)

#save
summary_path = path_configs["base_path"] + "/dataset_summary_wide.parquet"
summary_df.to_parquet(summary_path, index=False, engine="fastparquet")
print("Saved to:", summary_path)