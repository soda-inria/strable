'''
BARPLOT: KENDALLTAU of NUM+STR BY HIGH/LOW BIN of CARDINALITY, STRING LENGTH and STRING DIVERSITY
'''

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    bin_feature_33_66,
    results,
    score_list,
    selected_encoders,
)

# --- 1. CONFIGURATION ---
score = score_list[0]
percentile = 33 

# --- 2. DATA PREPARATION ---
df_analysis = results.copy()
df_analysis = df_analysis[(df_analysis['dtype'].isin(['Num+Str'])) & 
                          (df_analysis['encoder'].isin(selected_encoders))].copy()

# Apply binning
df_analysis['Card_Bin'] = bin_feature_33_66(df_analysis, 'avg_cardinality')
df_analysis['Str_Bin'] = bin_feature_33_66(df_analysis, 'avg_string_length_per_cell')
df_analysis['n_col_Bin'] = bin_feature_33_66(df_analysis, 'num_columns')
df_analysis['n_row_Bin'] = bin_feature_33_66(df_analysis, 'num_rows')
df_analysis['string_diversity_Bin'] = bin_feature_33_66(df_analysis, 'string_diversity')

heterogeneity_dimensions = ['Card_Bin', 'Str_Bin', 'n_col_Bin', 'n_row_Bin', 'string_diversity_Bin']
heterogeneity_feature_map = {
    'Card_Bin': 'Cardinality',
    'Str_Bin': 'String Length',
    'n_col_Bin': 'Num Columns',
    'n_row_Bin': 'Num Rows',
    'string_diversity_Bin': 'String Diversity'
}

# --- 3. CALCULATE METRICS ---
metrics_data = []

for dim in heterogeneity_dimensions:
    # Group by learner and bin to get average scores per pipeline first
    df_pipeline = df_analysis.groupby(['encoder_learner', dim], as_index=False)[score].mean()
    df_pivot = df_pipeline.pivot(index='encoder_learner', columns=dim, values=score)
    
    # A. Kendall Tau (Stability)
    tau, _ = kendalltau(df_pivot['Low'], df_pivot['High'])
    
    # B. Average Performance (High vs Low)
    # We Average the performance across all learners for the specific bin
    # (This gives the general difficulty/impact of that bin)
    avg_high = df_pivot['High'].mean()
    avg_low = df_pivot['Low'].mean()
    
    metrics_data.append({
        'Feature': dim,
        'FeatureName': heterogeneity_feature_map.get(dim, dim),
        'KendallTau': tau,
        'High': avg_high,
        'Low': avg_low
    })

df_metrics = pd.DataFrame(metrics_data)

# Sort by Kendall Tau (to align both plots by stability)
df_metrics = df_metrics.sort_values('KendallTau', ascending=True)

# Reshape for the Right Plot (Long format for hue plotting)
df_long = df_metrics.melt(id_vars=['FeatureName', 'KendallTau'], 
                          value_vars=['High', 'Low'], 
                          var_name='Bin', value_name='AvgScore')

# --- 4. PLOTTING ---
sns.set_theme(style="whitegrid", context="paper") # Reduced font size context
fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True)

# --- LEFT PLOT: STABILITY ---
# We use a neutral or distinct palette for the features
palette_left = sns.color_palette("tab10", n_colors=len(df_metrics))

sns.barplot(
    data=df_metrics,
    x='KendallTau',
    y='FeatureName',
    palette=palette_left,
    edgecolor='black',
    linewidth=0.8,
    ax=axes[0]
)

axes[0].set_xlabel('Stability of Model Ranks\n(Kendall $\\tau$ High vs Low)', fontsize=16, labelpad=11, ha='right', x=0.9)
axes[0].set_ylabel('', )
axes[0].set_xlim(0.45, 0.65) # Adjust x-limits to fit your specific data range
axes[0].tick_params(axis='y', labelsize=16)  # <--- Change 14 to your desired font size
axes[0].tick_params(axis='x', labelsize=14)  # Keep X axis size separate if needed
axes[0].grid(True, axis='x', alpha=0.5)

# --- RIGHT PLOT: IMPACT (High vs Low) ---
# Custom colors: Light Red (High) and Light Green (Low)
custom_palette = {'High': '#ff9999', 'Low': '#99ff99'} # Light red and light green

sns.barplot(
    data=df_long,
    x='AvgScore',
    y='FeatureName',
    hue='Bin',
    palette=custom_palette,
    edgecolor='black',
    linewidth=0.8,
    ax=axes[1]
)

axes[1].set_xlabel(f'Impact on Avg {Y_METRIC_LABELS[score]}\n($R^2$ & AUC)', fontsize=16, labelpad=11, ha='center', x=0.7, multialignment='center')
axes[1].set_ylabel('', fontsize=18)
# axes[1].tick_params(axis='y', labelsize=14)  # <--- Change 14 to your desired font size
axes[1].tick_params(axis='x', labelsize=14)  # Keep X axis size separate if needed
axes[1].grid(True, axis='x', alpha=0.5)
axes[1].set_xlim(0.6, 0.8) 

# Legend settings
axes[1].legend(title=None, bbox_to_anchor=(0.59, 0.3), loc='upper left', borderaxespad=0, frameon=False, fontsize=16)

sns.despine(left=True, bottom=False)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)

today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'kendalltau_heterogeneity_axis_barplot_{percentile}_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()
