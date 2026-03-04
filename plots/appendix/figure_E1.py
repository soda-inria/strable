'''
ranking change of learners for different encoders between numeric and num+str
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
    get_learner_color_simple,
    get_learner_marker,
    results,
    selected_encoders,
)

dtype = 'Num+Str'
score = 'score'
encoder_performance_byavg = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(['encoder'], as_index=False)[score].mean()

encoder_performance_bymedian = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(['encoder'], as_index=False)[score].median()
encoder_performance_bymedian.rename(columns={score: 'score_median'}, inplace=True)

encoder_performance_bymax = (
    results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(['encoder','learner'], as_index=False)[score].mean()
    .sort_values(by=['encoder',score], ascending=[True,False])
    .groupby('encoder').head(1)[['encoder','learner', score]]
    .sort_values(by=score, ascending=False)
    .reset_index(drop=True)
)

merge_df = pd.merge(encoder_performance_byavg, encoder_performance_bymax, on='encoder', suffixes=('_avg', '_max'))
merge_df = pd.merge(merge_df, encoder_performance_bymedian, on='encoder')
merge_df.rename(columns={'learner':'learner_max'}, inplace=True)

df_pivot = results[(results['dtype'].isin(['Num', 'Num+Str'])) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].pivot_table(
    index=['encoder', 'learner'],
    columns='dtype',
    values=score,
    aggfunc='mean'  # Average if there are multiple runs/folds
)

def calculate_tau(group):
    # Drop learners that don't have BOTH scores (avoids errors)
    valid_data = group[['Num', 'Num+Str']]
            
    tau, _ = kendalltau(valid_data['Num'], valid_data['Num+Str'])
    return tau

# show all rows
pd.set_option('display.max_rows', None)

df_pivot['Num'] = df_pivot['Num'].fillna(
    df_pivot.groupby(level='learner')['Num'].transform('mean')
)

df_pivot.dropna(subset=['Num+Str'], inplace=True)

merge_df['kendall_tau'] = df_pivot.groupby('encoder', as_index=False).apply(calculate_tau)

merge_df.sort_values(by='score_max', ascending=False)

merge_df.sort_values(by=['kendall_tau','score_max'], ascending=[True,False])

# drop E2E
merge_df.dropna(axis=0, inplace = True)
merge_df = merge_df[merge_df['encoder'] != 'CatBoost']
merge_df.reset_index(drop=True, inplace=True)



# 1. Prepare Data: Calculate Ranks for each (Encoder, Learner) pair
# We utilize the pivot table 'df_pivot' from your previous code
# Rows: (Encoder, Learner), Cols: [Num, Num+Str]

# Reset index to work with a flat dataframe
df_ranks = df_pivot.reset_index()

# Define ranking function (Higher score = Lower Rank #1)
def get_learner_ranks(df, score_col):
    df[f'rank_{score_col}'] = df[score_col].rank(ascending=False)
    return df

# Apply ranking within each encoder group for the 'Num+Str' score
df_ranks = df_ranks.groupby('encoder', group_keys=False).apply(lambda x: get_learner_ranks(x, 'Num+Str'))

# For 'Num' column, the ranking is theoretically identical across all encoder groups 
# (since the encoder doesn't affect the Num-only score). 
# We calculate the global 'Num' ranking once to ensure consistency.
numeric_scores = df_ranks[['learner', 'Num']].drop_duplicates('learner').set_index('learner')['Num']
numeric_ranks = numeric_scores.rank(ascending=False)
df_ranks['rank_Num'] = df_ranks['learner'].map(numeric_ranks)

# 2. Select Representative Encoders to Plot
# We contrast "Stable" (High Tau) vs "Shifty" (Low Tau) encoders

# drop TF-IDF for clarity AND USE PARETO OPTIMALITY LLM ORDER
encoders_to_plot = results[(results['dtype'].isin(['Num+Str'])) & (results['encoder'].isin(['TargetEncoder', 'Tf-Idf','LM FastText', 'LM E5-small-v2', 'LM All-MiniLM-L6-v2', 'LM Jasper-0.6B', 'LM Qwen-3-8B', 'LM LLaMA-3.1-8B', 'Tarte']))].groupby(['encoder'], as_index=False)['run_time_per_1k'].mean().sort_values(by='run_time_per_1k', ascending=True)['encoder'].to_list()
# (results['learner']!='ExtraTrees-tuned') & 
fig, axes = plt.subplots(1, 9, figsize=(24, 10), sharey=True)

# Define X-axis points
x_points = [0, 1]
x_labels = ['Numeric\nOnly', 'Numeric\n+ String']

# Iterate over the selected encoders
for idx, enc in enumerate(encoders_to_plot):
    ax = axes[idx]
    
    # Get data for this specific encoder
    data = df_ranks[df_ranks['encoder'] == enc].copy()
    
    # Plot lines for each learner
    for _, row in data.iterrows():
        learner = row['learner']
        # if learner == 'ExtraTrees-tuned':
        #     continue  # Skip ExtraTrees-tuned
        rank_start = row['rank_Num']
        rank_end = row['rank_Num+Str']
        
        # Style using your existing maps
        color = get_learner_color_simple(learner)
        marker = get_learner_marker(learner)
        
        # Plot Line connecting Start Rank -> End Rank
        ax.plot(x_points, [rank_start, rank_end], 
                color=color, marker=marker, markersize=10, 
                linewidth=2.5, alpha=0.8)
        
        # Add text labels to the left of the first plot for learner names
        if idx == 0: 
            ax.text(-0.15, rank_start+0.25, learner, ha='right', va='center', 
                    fontsize=15, color=color, fontweight='bold')
            
    # Styling the Subplot
    ax.set_xticks(x_points)
    ax.set_xticklabels(x_labels, fontsize=16, fontweight='bold')
    ax.set_title(f"{enc}", fontsize=16, fontweight='bold', pad=15)
    
    # Invert Y axis so Rank 1 is at the top
    if idx == 0:
        ax.invert_yaxis()
        ax.set_ylabel("Learner Rank (1 = Best)", fontsize=20, labelpad=45)
        # Set integer ticks only
        max_rank = int(data['rank_Num'].max())
        ax.set_yticks(range(1, max_rank + 1))
    
    # Add vertical grid lines only
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.grid(axis='y', linestyle='-', alpha=0.1) # Faint horizontal guide
    
    # Add Tau value to title/annotation
    # (Assuming merge_df exists from your previous code)
    if 'merge_df' in locals():
        try:
            tau = merge_df[merge_df['encoder'] == enc]['kendall_tau'].values[0]
            ax.text(0.5, 0.98, f"Stability $\\tau$ = {tau:.2f}", 

                    transform=ax.transAxes, ha='center', color='dimgray', fontsize=14, style='italic')
        except:
            pass

    sns.despine(left=True, bottom=True)


plt.tight_layout(w_pad=1.0)

# Save & Show
today_date = time.strftime("%Y-%m-%d")
PIC_NAME = f'learner_rank_bump_chart_{today_date}.pdf'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()