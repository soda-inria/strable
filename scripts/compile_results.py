"""Script to compile all benchmark results into a single CSV."""

import os
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from configs.path_configs import path_configs

def compile_results():
    # 1. Dynamically locate the benchmark results directory
    base_path = Path(path_configs["base_path"])
    score_dir = base_path / "results" / "benchmark"
    
    if not score_dir.exists():
        print(f"❌ Directory not found: {score_dir}")
        return

    print(f"Scanning for result files in: {score_dir}...")
    
    # 2. Find all CSVs located inside any 'score' subfolder
    score_files = list(score_dir.rglob("score/*.csv"))
    
    # Safety check: Prevent crash if no files have been generated yet
    if not score_files:
        print("⚠️ No score files found to compile. Run the evaluation script first.")
        return
        
    print(f"Found {len(score_files)} result files. Compiling...")

    # 3. Read and concatenate all files in parallel
    df_score_ = Parallel(n_jobs=-1)(delayed(pd.read_csv)(file) for file in score_files)
    
    # ignore_index=True does the exact same thing as your reset_index(drop=True)
    df_score_runs = pd.concat(df_score_, axis=0, ignore_index=True)

    # 4. Ensure the compiled output folder exists
    compiled_results_dir = base_path / "results" / "compiled_results"
    compiled_results_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save the final compiled CSV
    save_path = compiled_results_dir / "result_comparison.csv"
    df_score_runs.to_csv(save_path, index=False)
    
    print(f"✅ Successfully compiled {len(score_files)} files into:\n{save_path}")

if __name__ == "__main__":
    compile_results()