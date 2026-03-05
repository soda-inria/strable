import os
import pandas as pd
import numpy as np
# --- 0. SETUP & IMPORTS ---
from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import TODAYS_FOLDER

# Obtain df_structure via explicit API exposed by the plotting module.
try:
    from strable.plots.appendix.figure_E16 import get_df_structure
except Exception as e:
    raise RuntimeError(f"Required function get_df_structure not available from figure_E16: {e}")

# This will load from cache if present, otherwise compute and cache the result.
df_structure, df_comp, df_results = get_df_structure()

# --- 5. Export to LaTeX ---
filename = f"4_benchmark_comparison_{TODAYS_FOLDER}.tex"
out_dir = os.path.join(path_configs["base_path"], "results_tables", TODAYS_FOLDER)
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, filename)

# Compute the median table
df_median = df_structure.groupby('benchmark')[
    ['vocab_size', 'avg_token_count', 'text_col_ratio',
     'avg_ngrams_per_row', 'total_unique_ngrams']
].median()

# Rename columns for LaTeX readability
df_median.columns = [
    'Vocab Size',
    'Avg Tokens / Row',
    'Text Col Ratio',
    'Avg N-grams / Row',
    'Total Unique N-grams'
]

# Round for cleanliness
df_median = df_median.round(2)

# Generate LaTeX
latex_str = df_median.to_latex(
    caption=(
        'Median structural characteristics of text columns across the four benchmarks. '
        'Vocab Size: number of unique word tokens per column. '
        'Avg Tokens / Row: average number of whitespace-separated tokens per string. '
        'Text Col Ratio: proportion of columns that are text. '
        'Avg N-grams / Row: average number of unique character n-grams (length 2--4) per string. '
        'Total Unique N-grams: total unique character n-gram vocabulary per column. '
        'STRABLE text columns are concise (1.3 tokens), with lower amount of unique words (20) and compact character-level structure (688 n-grams), reflecting the status of strings-in-the-wild. In this regime, simple frequency-based methods such as Tf-Idf are sufficient, as the discriminative signal is concentrated in a small set of recurring tokens rather than in semantic context — which explains the strong performance of Tf-Idf on the Pareto plot.'
    ),
    label='tab:benchmark_comparison',
    position='ht',
    bold_rows=True,
    escape=True,
)

# Save
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    f.write(latex_str)

print(f"✅ LaTeX table saved to {save_path}")
print(latex_str)