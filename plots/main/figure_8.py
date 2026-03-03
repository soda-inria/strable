"""
Figure 8: PCA 30 vs PCA 60 for LLaMA-3.1-8B + TabPFN-2.5.

This script expects precomputed PCA-60 ablation runs for LLaMA-3.1-8B stored under
`<base_path>/results/pca-ablation/**/score/*.csv`. These runs are *not* generated
by this plotting script; they must be produced by running the dedicated
PCA-60 experiment pipeline before calling this figure.
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    dtype_map,
    encoder_map,
    learner_map,
    results,
    score_list,
)


# Root folder where PCA-60 ablation CSVs live (must be created by a prior experiment)
pca_ablation_path = os.path.join(path_configs["base_path"], "results", "pca-ablation")

# Ensure the PCA-60 ablation directory exists so users are aware of the requirement
score_dir = Path(pca_ablation_path)
if not score_dir.exists():
    raise FileNotFoundError(
        f"PCA-60 ablation results not found at '{score_dir}'. "
        "Run the LLaMA-3.1-8B + TabPFN PCA-60 experiment to generate "
        "score CSVs before running this figure."
    )

# Compile PCA-60 results from all experiment score CSVs
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

# Filter baseline PCA-30 results for LLaMA-3.1-8B and TabPFN-2.5 for Num+Str
df_score_pca_30 = results[
    (results["dtype"] == "Num+Str")
    & (results["encoder"] == "LM LLaMA-3.1-8B")
    & (results["learner"] == "TabPFN-2.5")
]

score = score_list[0]

avg_score_llama_tabpfn_pca30 = df_score_pca_30.groupby(['data_name'], as_index=False)[score].mean()

avg_score_llama_tabpfn_pca60 = df_score_pca_60.groupby(['data_name'], as_index=False)[score].mean()

# merge both on data_name
merged_scores = pd.merge(avg_score_llama_tabpfn_pca30, avg_score_llama_tabpfn_pca60, on='data_name', suffixes=('_pca30', '_pca60'))

# add num_rows per data_name
num_rows_per_data = results[['data_name', 'num_rows']].drop_duplicates()
merged_scores = pd.merge(merged_scores, num_rows_per_data, on='data_name', how='left')

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
fmt = 'pdf'
PIC_NAME = f'boxplot_pca30_vs_60_delta_per_sample_size_{today_date}.{fmt}'
out_path = os.path.join(path_configs["base_path"], "results_pics", TODAYS_FOLDER, PIC_NAME)
plt.savefig(out_path, bbox_inches='tight')
plt.show()

'''
RUNTIME ANALYSIS PCA-30 VS PCA-60
'''

pca_60_avg_runtime_per_dataset = df_score_pca_60.groupby(['data_name'], as_index=False)['run_time'].mean()

pca_30_avg_runtime_per_dataset = df_score_pca_30.groupby(['data_name'], as_index=False)['run_time'].mean()

#merge both on data_name
merged_runtimes = pd.merge(pca_30_avg_runtime_per_dataset, pca_60_avg_runtime_per_dataset, on='data_name', suffixes=('_pca30', '_pca60'))

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
fmt = 'pdf'
PIC_NAME = f'runtime_boxplot_pca30_vs_60_delta_per_sample_size_{today_date}.{fmt}'
out_path = os.path.join(path_configs["base_path"], "results_pics", TODAYS_FOLDER, PIC_NAME)
plt.savefig(out_path, bbox_inches='tight')

plt.show()