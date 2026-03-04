import os
import re
import time

import pandas as pd

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import TODAYS_FOLDER

def generate_llm_summary_table():
    # 1. Define the exact text and LaTeX commands for each cell
    # Note: Using standard "--" for en-dashes as in your target LaTeX
    data = [
        # Statistical Baselines
        [r"\multirow{4}{*}{Statistical Baselines}", "StringEncoder (Tf-Idf + SVD)", "-", "N/A", "N/A"],
        ["", "TargetEncoder", "-", "N/A", "N/A"],
        ["", "CatBoostEncoder", "-", "N/A", "N/A"],
        ["", "FastText", "300", "N/A", "N/A"],
        
        # Embedders (Contrastive)
        [r"\multirow{5}{*}{Embedders (Contrastive)}", "E5 (Small/Base/Large)", "384--1024", "33M--335M", "512"],
        ["", "BGE (Small/Base/Large)", "384--1024", "0.1B--33.5B", "512"],
        ["", "UAE-Large", "1024", "335M", "512"],
        ["", "All-MiniLM (L6/L12)", "384", "22M--33.4M", "256"],
        ["", "All-MPNet-Base-v2", "768", "110M", "384"],
        ["", "KALM (Embed)", "896", "0.5B", "512"],
        ["", "Tarte", "768", "25M", "N/A"],
        
        # Encoder-only (MLM)
        [r"\multirow{3}{*}{Encoder-only (MLM)}", "RoBERTa (Base/Large)", "768--1024", "125M--355M", "512"],
        ["", "DeBERTa-v3 (XS/S/B/L)", "384--1024", "22M--304M", "512"],
        ["", "ModernBERT (Base/Large)", "768--1024", "149M--395M", "8192"],
        
        # Encoder-Decoder
        [r"\multirow{1}{*}{Encoder-Decoder}", "Sentence-T5 (Base/L/XL/XXL)", "768", "0.5B--5B", "512"],
        
        # Decoder-only (Causal) - Updated with LLaMA-Nemotron
        [r"\multirow{7}{*}{Decoder-only (Causal)}", "LLaMA-3.1 / 3.2", "2048--4096", "1B--8B", "128k"],
        ["", "LLaMA-Nemotron-Embed-1B-v2", "2048", "1B", "128k"],
        ["", "Qwen-3 (0.6B/4B/8B)", "1024--4096", "0.6B--8B", "32k"],
        ["", "OPT (0.1B to 6.7B)", "768--4096", "125M--6.7B", "2048"],
        ["", "Gemma-0.3B", "768", "300M", "128k"],
        ["", "F2LLM (0.6B/1.7B/4B)", "1024--2560", "0.6B--4B", "1024"],
        ["", "Jasper-0.6B", "2048", "0.6B", "2048"],
        
        # End-to-End Architectures
        [r"\multirow{2}{*}{End-to-End Architectures}", "ContextTab", "768", "172M", "256"],
        ["", "TabSTAR", "384", "47.2M", "512"]
    ]

    # Define the pre-bolded column headers
    columns = [
        r"\textbf{Category}", 
        r"\textbf{Model Name}", 
        r"\textbf{Dim ($d$)}", 
        r"\textbf{Params}", 
        r"\textbf{Context}"
    ]

    df = pd.DataFrame(data, columns=columns)

    # 2. Generate Base LaTeX with Pandas
    styler = df.style.hide(axis="index")
    
    latex_output = styler.to_latex(
        environment="table*", # Use table* for full page width if needed
        position="t",
        caption=r"Summary of Encoders and Architectures used in the STRABLE benchmark. Embedding dimensions, parameters and context length are taken from the English Massive Text Embedding Benchmark Leaderboard \citep{enevoldsen2025mmtebmassivemultilingualtext}",
        label="tab:llm_summary",
        column_format="llccc",
        hrules=True
    )

    # 3. Inject structural commands
    # Inject \footnotesize and \centering right after \begin{table*}[t]
    latex_output = latex_output.replace(
        r"\begin{table*}[t]",
        r"\begin{table*}[t]" + "\n" + r"\footnotesize" + "\n" + r"\centering"
    )

    # Force a line break after \multirow declarations to match your exact snippet style
    latex_output = re.sub(
        r"(\\multirow\{\d+\}\{.*?\}\{.*?\}) \s*&", 
        r"\1 \n&", 
        latex_output
    )

    # 4. Inject \midrule dynamically at specific row boundaries
    midrule_targets = [
        r"(.*FastText.*\\\\)",
        r"(.*Tarte.*\\\\)",
        r"(.*ModernBERT.*\\\\)",
        r"(.*Sentence-T5.*\\\\)",
        r"(.*Jasper-0.6B.*\\\\)"
    ]

    for target in midrule_targets:
        latex_output = re.sub(target, r"\1\n\\midrule", latex_output)

    # 5. Save the output
    today_date = time.strftime("%Y-%m-%d")
    filename = f"table_llm_summary_{today_date}.tex"
    save_path = os.path.join(
        path_configs["base_path"],
        "results_tables",
        TODAYS_FOLDER,
        filename,
    )
    with open(save_path, "w") as f:
        f.write(latex_output)

    print(f"✅ Table successfully generated and saved to {save_path}")

if __name__ == "__main__":
    generate_llm_summary_table()