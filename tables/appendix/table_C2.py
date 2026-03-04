import os
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import TODAYS_FOLDER

# 1. Setup paths
score_dir = Path(path_configs["base_path"]) / "data" / "llm_embed_time"
# Get all .npy files recursively
score_files = list(score_dir.glob("**/*.npy"))

print(f"Found {len(score_files)} files to process.")

# 2. Define the processing function
def process_file(file_path):
    try:
        # Extract metadata from filename: "model|dataset.npy"
        # file_path.stem removes the ".npy" extension
        filename_parts = file_path.stem.split('|')
        
        if len(filename_parts) == 2:
            model_name = filename_parts[0]
            dataset_name = filename_parts[1]
        else:
            # Fallback for unexpected naming
            model_name = file_path.parent.name
            dataset_name = file_path.stem

        # Load the scalar value (runtime)
        data = np.load(file_path)
        runtime = float(data) # Convert 0-d array to float

        return {
            "method": model_name,
            "dataset": dataset_name,
            "runtime": runtime
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# 3. Run in Parallel
llm_results = Parallel(n_jobs=-1)(delayed(process_file)(f) for f in score_files)

df_llm_runs = pd.DataFrame(llm_results)

print(f"Successfully loaded {len(df_llm_runs)} rows.")
print(df_llm_runs.head())

# compute median and ICR for each method
def median_iqr(x):
    """Returns string formatted as 'Median (IQR)'"""
    med = x.median()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return f"{med:.0f} [{q1:.0f}, {q3:.0f}]"

encoder_map = {
    'tabvec': 'Tf-Idf + SVD',
    'tarenc': 'TargetEncoder',
    'catboost': 'CatBoostEncoder',
    'tabpfn': 'TabPFN-2.5',
    'tabstar': 'TabSTAR',
    'contexttab': 'ContextTab',
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
    'llm-llama-nemotron-embed-1b-v2': 'LM LLaMA-Nemotron-Embed-1B-v2'
}


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
    'llm-llama-nemotron-embed-1b-v2': 'nvidia/llama-nemotron-embed-1b-v2',
    
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

summary_df = df_llm_runs.groupby('method')['runtime'].apply(median_iqr).reset_index()

# Create 'Hugging Face' column using the raw 'method' column
summary_df['Hugging Face'] = summary_df['method'].map(hf_map).fillna('-')

# Create 'Language Model' column using the encoder_map
summary_df['Language Model'] = summary_df['method'].replace(encoder_map)

def get_mteb_scores(hf_model_map):
    """
    Fetches the MTEB leaderboard data and extracts the 'Mean (Task)' score
    for the English benchmark for each model in the provided map.
    """
    print("Fetching MTEB Leaderboard data...")
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
    
    # Map your models
    model_scores = {}
    for local_name, hf_id in hf_model_map.items():
        if hf_id in mteb_scores:
            model_scores[local_name] = mteb_scores[hf_id]
        else:
            model_scores[local_name] = np.nan # Not found
            
    return model_scores

# --- Integration into your Main Script ---

# 1. Get the scores
mteb_map = get_mteb_scores(hf_map)

# 2. Add to DataFrame
# Map the 'method' column to the score using the mteb_map
summary_df['MTEB (En) Score'] = summary_df.index.map(lambda x: mteb_map.get(summary_df.loc[x, 'method'], np.nan))

# If 'method' is not in the index (it was reset), map on the column
if 'method' in summary_df.columns:
     summary_df['MTEB (En) Score'] = summary_df['method'].map(mteb_map)

#round the MTEB score to integer
summary_df['MTEB (En) Score'] = summary_df['MTEB (En) Score'].round(2)

# Reorder columns: Language Model | Hugging Face | Median (IQR)
summary_df = summary_df[['Language Model', 'Hugging Face', 'runtime', 'MTEB (En) Score']]
summary_df = summary_df.rename(columns={'runtime': 'Median (IQR) of Runtime [s]'})

print("Summary of LLM Embedding Runtimes:")
print(summary_df.head())

latex_code = summary_df.to_latex(
    index=False,                  
    na_rep="-",
    float_format="%.2f"  
)

latex_code = re.sub(
    r"\\begin\{tabular\}\{.*?\}",
    r"\\begin{tabularx}{\\textwidth}{l X c c}",
    latex_code
)
latex_code = latex_code.replace(r"\end{tabular}", r"\end{tabularx}")

today_date = time.strftime("%Y-%m-%d")
filename = f"llm_compiles_results_{today_date}.tex"
save_path = os.path.join(
    path_configs["base_path"],
    "results_tables",
    TODAYS_FOLDER,
    filename,
)

# 4. Manually write the exact structure you want
with open(save_path, 'w') as f:
    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n") 
    f.write("\\small\n") # Added the missing \small tag
    f.write("\\caption{Median (IQR) of Runtime and MTEB Performance per Language Model}\n")
    f.write("\\label{tab:runtime_summary}\n")
    
    f.write(latex_code) # Insert the modified pandas tabularx part
    
    f.write("\\end{table}\n")

print(f"✅ Table successfully generated and saved to {save_path}")
