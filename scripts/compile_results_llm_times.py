"""Compile per-(model, dataset) LLM embedding-extraction runtimes into the
``tab:runtime_summary`` LaTeX table (Table C.3 of the paper).

Walks ``data/llm_embed_time_backup/**/*.npy`` — each file holds a single
scalar runtime, with filename ``<method>|<dataset>.npy``. Aggregates to
median (IQR) per encoder, joins with HuggingFace IDs and a hardcoded MTEB
English-mean score, and writes a ``tabularx`` table to
``results_tables/<today>/llm_compiles_results_<today>.tex``.

The MTEB scores are pinned to the values reported on the public MTEB
leaderboard at submission time and kept hardcoded so the table stays
reproducible without a network call.
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import datetime
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from configs.path_configs import path_configs
from figures._main import encoder_map


# HuggingFace identifiers per local method tag. Used as the second column
# of the table; '-' for non-HuggingFace baselines (TabPFN, FastText, etc.).
HF_MAP = {
    # Encoders / baselines (not on HF)
    'tabvec':                          '-',
    'tarenc':                          '-',
    'catboost':                        '-',
    'tabpfn':                          '-',
    'tabstar':                         '-',
    'contexttab':                      '-',
    'tarte':                           '-',
    'llm-fasttext':                    '-',

    # Sentence Transformers / BERT family
    'llm-all-MiniLM-L6-v2':            'sentence-transformers/all-MiniLM-L6-v2',
    'llm-all-MiniLM-L12-v2':           'sentence-transformers/all-MiniLM-L12-v2',
    'llm-all-mpnet-base-v2':           'sentence-transformers/all-mpnet-base-v2',
    'llm-e5-base-v2':                  'intfloat/e5-base-v2',
    'llm-e5-large-v2':                 'intfloat/e5-large-v2',
    'llm-e5-small-v2':                 'intfloat/e5-small-v2',
    'llm-roberta-base':                'FacebookAI/roberta-base',
    'llm-roberta-large':               'FacebookAI/roberta-large',
    'llm-modernbert-base':             'answerdotai/ModernBERT-base',
    'llm-modernbert-large':            'answerdotai/ModernBERT-large',
    'llm-deberta-v3-xsmall':           'microsoft/deberta-v3-xsmall',
    'llm-deberta-v3-small':            'microsoft/deberta-v3-small',
    'llm-deberta-v3-base':             'microsoft/deberta-v3-base',
    'llm-deberta-v3-large':            'microsoft/deberta-v3-large',

    # BGE
    'llm-bge-large':                   'BAAI/bge-large-en-v1.5',
    'llm-bge-small':                   'BAAI/bge-small-en-v1.5',
    'llm-bge-base':                    'BAAI/bge-base-en-v1.5',

    # LLaMA
    'llm-llama-3.1-8b':                'meta-llama/Llama-3.1-8B',
    'llm-llama-3.2-1b':                'meta-llama/Llama-3.2-1B',
    'llm-llama-3.2-3b':                'meta-llama/Llama-3.2-3B',
    'llm-llama-nemotron-embed-1b-v2':  'nvidia/llama-nemotron-embed-1b-v2',

    # Qwen
    'llm-qwen3-8b':                    'Qwen/Qwen3-Embedding-8B',
    'llm-qwen3-4b':                    'Qwen/Qwen3-Embedding-4B',
    'llm-qwen3-0.6b':                  'Qwen/Qwen3-Embedding-0.6B',

    # OPT (note: 0.1b -> 125m, 0.3b -> 350m).
    'llm-opt-0.1b':                    'facebook/opt-125m',
    'llm-opt-0.3b':                    'facebook/opt-350m',
    'llm-opt-1.3b':                    'facebook/opt-1.3b',
    'llm-opt-2.7b':                    'facebook/opt-2.7b',
    'llm-opt-6.7b':                    'facebook/opt-6.7b',

    # F2LLM
    'llm-f2llm-0.6b':                  'codefuse-ai/F2LLM-0.6B',
    'llm-f2llm-1.7b':                  'codefuse-ai/F2LLM-1.7B',
    'llm-f2llm-4b':                    'codefuse-ai/F2LLM-4B',

    # T5
    'llm-t5-small':                    'google-t5/t5-small',
    'llm-sentence-t5-base':            'sentence-transformers/sentence-t5-base',
    'llm-sentence-t5-large':           'sentence-transformers/sentence-t5-large',
    'llm-sentence-t5-xl':              'sentence-transformers/sentence-t5-xl',
    'llm-sentence-t5-xxl':             'sentence-transformers/sentence-t5-xxl',

    # Others
    'llm-gemma-0.3b':                  'google/gemma-3-270m',
    'llm-uae-large':                   'WhereIsAI/UAE-Large-V1',
    'llm-kalm-embed':                  'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5',
    'llm-jasper-token-comp-0.6b':      'infgrad/Jasper-Token-Compression-600M',
}

# MTEB English-mean scores per HF model id, pinned to the public MTEB
# leaderboard at submission time. Models without a leaderboard entry are
# omitted and surface as "-" in the final table.
MTEB_SCORES = {
    'sentence-transformers/all-MiniLM-L6-v2':                       56.03,
    'intfloat/e5-base-v2':                                          61.67,
    'intfloat/e5-large-v2':                                         62.79,
    'intfloat/e5-small-v2':                                         61.32,
    'BAAI/bge-large-en-v1.5':                                       65.89,
    'BAAI/bge-base-en-v1.5':                                        65.14,
    'BAAI/bge-small-en-v1.5':                                       64.30,
    'Qwen/Qwen3-Embedding-8B':                                      75.23,
    'Qwen/Qwen3-Embedding-4B':                                      74.61,
    'Qwen/Qwen3-Embedding-0.6B':                                    70.47,
    'codefuse-ai/F2LLM-0.6B':                                       70.03,
    'codefuse-ai/F2LLM-1.7B':                                       72.01,
    'codefuse-ai/F2LLM-4B':                                         73.67,
    'WhereIsAI/UAE-Large-V1':                                       66.40,
    'infgrad/Jasper-Token-Compression-600M':                        74.75,
    'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5':       71.29,
    'sentence-transformers/sentence-t5-base':                       60.30,
    'sentence-transformers/sentence-t5-large':                      77.67,
    'sentence-transformers/sentence-t5-xl':                         76.58,
    'sentence-transformers/sentence-t5-xxl':                        66.13,
}


def _process_runtime_file(file_path):
    """Load one ``<method>|<dataset>.npy`` runtime scalar."""
    try:
        parts = file_path.stem.split('|')
        if len(parts) == 2:
            method_name, dataset_name = parts
        else:
            method_name = file_path.parent.name
            dataset_name = file_path.stem
        runtime = float(np.load(file_path))
        return {'method': method_name, 'dataset': dataset_name, 'runtime': runtime}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def _median_iqr(x):
    """``Median [Q1, Q3]`` formatted as a string for the LaTeX table."""
    return f"{x.median():.0f} [{x.quantile(0.25):.0f}, {x.quantile(0.75):.0f}]"


def _load_runtime_dataframe():
    score_dir = Path(f"{path_configs['base_path']}/data/llm_embed_time_backup")
    score_files = list(score_dir.glob("**/*.npy"))
    print(f"Found {len(score_files)} files to process.")

    rows = Parallel(n_jobs=-1)(
        delayed(_process_runtime_file)(f) for f in score_files
    )
    df = pd.DataFrame([r for r in rows if r is not None])
    print(f"Successfully loaded {len(df)} rows.")
    return df


def _build_summary_df(df_runs):
    summary = (
        df_runs.groupby('method')['runtime']
        .apply(_median_iqr)
        .reset_index()
    )
    summary['Hugging Face'] = summary['method'].map(HF_MAP).fillna('-')
    summary['Language Model'] = summary['method'].replace(encoder_map)
    summary['MTEB (En) Score'] = (
        summary['method'].map(HF_MAP).map(MTEB_SCORES).round(2)
    )
    summary = summary[
        ['Language Model', 'Hugging Face', 'runtime', 'MTEB (En) Score']
    ].rename(columns={'runtime': 'Median (IQR) of Runtime [s]'})
    return summary


def _summary_to_latex(summary):
    body = summary.to_latex(index=False, na_rep='-', float_format='%.2f')
    body = re.sub(
        r"\\begin\{tabular\}\{.*?\}",
        r"\\begin{tabularx}{\\textwidth}{l X c c}",
        body,
    )
    body = body.replace(r"\end{tabular}", r"\end{tabularx}")
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Median (IQR) of Runtime and MTEB Performance per Language Model}\n"
        "\\label{tab:runtime_summary}\n"
        f"{body}"
        "\\end{table}\n"
    )


def main():
    df_runs = _load_runtime_dataframe()

    runs_csv = f"{path_configs['compiled_results']}/LLM_embedding_timetrack.csv"
    os.makedirs(os.path.dirname(runs_csv), exist_ok=True)
    df_runs.to_csv(runs_csv, index=False)

    summary = _build_summary_df(df_runs)
    print("Summary of LLM Embedding Runtimes:")
    print(summary.head())

    today = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = f"{path_configs['results_tables']}/{today}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/llm_compiles_results_{today}.tex"
    with open(out_path, 'w') as f:
        f.write(_summary_to_latex(summary))
    print(f"Table saved to {out_path}")


if __name__ == "__main__":
    main()
