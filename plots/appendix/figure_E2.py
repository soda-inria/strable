'''
AVERAGE PERFORMANCE OF ENCODER (NUM+STR) VS KENDALLTAU CORRELATION(KENDALLTAU CORRELATION(NUM+STR VS NUM))
'''

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import (
    TODAYS_FOLDER,
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

# compute difference between Num+Str and Num
# df_pivot['difference'] = df_pivot['Num+Str'] - df_pivot['Num']

# merge_df['avg_difference'] = df_pivot.groupby('encoder', as_index=False)['difference'].mean()

df_pivot.dropna(subset=['Num+Str'], inplace=True)

merge_df['kendall_tau'] = df_pivot.groupby('encoder', as_index=False).apply(calculate_tau)

merge_df.sort_values(by='score_max', ascending=False)

merge_df.sort_values(by=['kendall_tau','score_max'], ascending=[True,False])

# drop E2E
merge_df.dropna(axis=0, inplace = True)
merge_df = merge_df[merge_df['encoder'] != 'CatBoost']
merge_df.reset_index(drop=True, inplace=True)


#plot BY MEDIAN
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(4, 3))

# Scatter Plot
sns.scatterplot(
    data=merge_df, 
    x='score_median', 
    y='kendall_tau', 
    s=200,          # Size of markers
    color='#1f77b4', # Standard blue
    edgecolor='black', 
    alpha=0.8,
    ax=ax
)

#check the table
# merge_df[['encoder', 'score_median', 'kendall_tau']].sort_values(by=['kendall_tau'], ascending=False)


# 3. Add Labels for each point (Encoder names)
ax.text(0.63, 0.9, 'TargetEncoder', fontsize=9, weight='bold',color='#333333')
ax.text(0.75, 0.9, 'Tf-Idf', fontsize=9, weight='bold',color='#333333')
ax.text(0.63, 0.45, 'Tarte', fontsize=9, weight='bold',color='#333333')
ax.text(0.65, 0.70, 'LM FastText', fontsize=9, weight='bold',color='#333333')
ax.text(0.7, 0.75, 'LM All-MiniLM-L6-v2', fontsize=9, weight='bold',color='#333333')
ax.text(0.73, 0.70, 'LM E5-small-v2', fontsize=9, weight='bold',color='#333333')
ax.text(0.696320+0.002, 0.0 + 0.15, 'LM Jasper-0.6B', fontsize=9, weight='bold',color='#333333')
ax.text(0.718, -0.1, 'LM LLaMA-3.1-8B', fontsize=9, weight='bold',color='#333333')
ax.text(0.718, -0.25, 'LM Qwen-3-8B', fontsize=9, weight='bold',color='#333333')


# 4. Styling
ax.set_xlabel('Median Score Achieved by Encoder', labelpad=10, fontsize=12)
ax.set_ylabel('Kendall $\\tau$(Num+Str,Num)', labelpad=10, fontsize=12)

# Optional: Add reference lines
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(-0.3,1)
ax.set_xlim(0.6,0.85)
sns.despine()

today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'kendalltau_per_median_score_encoder_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()