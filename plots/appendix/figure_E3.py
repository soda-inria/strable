'''
PIPELINE PERFORMANCE PER NUM VS NUM+STR
'''

import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    get_encoder_color,
    get_learner_marker,
    results,
    score_list,
    selected_encoders,
)

score = score_list[0]
dtype = ['Num+Str', 'Num']  

df = results.copy()
df = df[df['dtype'].isin(dtype)]
df = df[df['encoder'].isin(selected_encoders)]

# drop TabPFN-2.5 encoder
df = df[(df['method'] != 'num-str_tabpfn_tabpfn_default')]



## by encoder-learner
avg_performance_per_encoder_learner_dtype = df.groupby(['encoder_learner','learner','dtype'], as_index=False)[score].mean()

avg_performance_per_encoder_learner_dtype = avg_performance_per_encoder_learner_dtype.pivot_table(
    index=['encoder_learner','learner'],
    columns='dtype',
    values=score
)

avg_performance_per_encoder_learner_dtype['Num'] = avg_performance_per_encoder_learner_dtype['Num'].fillna(
    avg_performance_per_encoder_learner_dtype.groupby(level='learner')['Num'].transform('mean')
)

avg_performance_per_encoder_learner_dtype.reset_index(inplace=True)

avg_performance_per_encoder_learner_dtype.drop(columns=['learner'], inplace=True)

df_plot = avg_performance_per_encoder_learner_dtype.copy()

# Recover Encoder and Learner from the string "Encoder - Learner"
# We assume the format is consistently "Encoder - Learner"
df_plot[['encoder', 'learner']] = df_plot['encoder_learner'].str.split(' - ', expand=True)

fig, ax = plt.subplots(figsize=(8, 5)) 

# 1. Background Zones
lims = [df_plot[['Num', 'Num+Str']].min().min() * 0.98, df_plot[['Num', 'Num+Str']].max().max() * 1.02]
ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0) # Diagonal

# Red Zone (Better on Num Only)
ax.fill_between(lims, [0,0], lims, color='#d62728', alpha=0.05, zorder=0)
ax.text(lims[1]*0.95, lims[0]*1.05, "Better on\nNUMERIC ONLY", color='darkred', ha='right', va='bottom', fontweight='bold')

# Green Zone (Better with Strings)
ax.fill_between(lims, lims, [2,2], color='#2ca02c', alpha=0.05, zorder=0)
ax.text(lims[0]*1.60, lims[1]*0.95, "Better with\nSTRINGS", color='darkgreen', ha='left', va='top', fontweight='bold')

# 2. Plotting Loop
for _, row in df_plot.iterrows():
    # Style Logic
    color = get_encoder_color(row['encoder'])
    marker = get_learner_marker(row['learner'])
    is_tuned = 'tuned' in row['learner']
    
    style_kwargs = {
        'marker': marker,
        'color': color,
        'markersize': 12,
        'linestyle': '',
        'label': '_nolegend_' # Prevent automatic legend creation
    }
    
    if is_tuned:
        style_kwargs.update({
            'fillstyle': 'left',
            'markerfacecoloralt': 'white',
            'markeredgecolor': 'black',
            'markeredgewidth': 1.0
        })
    else:
        style_kwargs.update({
            'fillstyle': 'full',
            'markeredgecolor': 'black',
            'markeredgewidth': 0.5
        })
        
    ax.plot(row['Num'], row['Num+Str'], **style_kwargs)

# 3. Aesthetics
ax.set_xlim(0.32, 0.75)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel(f'{Y_METRIC_LABELS[score]} on Numeric Only', fontsize=14)
ax.set_ylabel(f'{Y_METRIC_LABELS[score]} on Numeric + String', fontsize=14)

# --- 4. LEGEND GENERATION (Side-by-Side) ---

# A. Legend for Learners (Shapes) - LEFT COLUMN
unique_learners = sorted(df_plot['learner'].unique())
learner_handles = []

for lrn in unique_learners:
    m = get_learner_marker(lrn)
    is_tuned = 'tuned' in lrn
    
    # Create line handle to match scatter style
    if is_tuned:
        h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                   fillstyle='left', markerfacecoloralt='white', 
                   markeredgewidth=1.0, markersize=10)
    else:
        h = Line2D([], [], color='black', marker=m, linestyle='', label=lrn,
                   fillstyle='full', markeredgewidth=0.5, markersize=10)
    learner_handles.append(h)

# Place immediately to the right of the plot axis (x=1.02)
leg_learners = plt.legend(
    handles=learner_handles, 
    title=r"$\bf{Learner}$" + "\n" + r"$\it{(shape)}$",
    loc='upper left', 
    bbox_to_anchor=(0.98, 1.0), 
    frameon=False,
    ncol=1, 
    fontsize=12,
    title_fontsize=14,
    labelspacing=0.8,
    handletextpad=0.4
)
ax.add_artist(leg_learners) # Manually add to prevent overwrite

# B. Legend for Encoders (Colors) - RIGHT COLUMN
unique_encoders = sorted(df_plot['encoder'].unique())
encoder_handles = []

for enc in unique_encoders:
    c = get_encoder_color(enc)
    h = mpatches.Patch(facecolor=c, edgecolor='black', label=enc, linewidth=0.5)
    encoder_handles.append(h)

# Place further to the right (x=1.35) so it sits next to the first legend
plt.legend(
    handles=encoder_handles, 
    title=r"$\bf{Encoder}$" + "\n" + r"$\it{(color)}$",
    loc='upper left', 
    bbox_to_anchor=(1.49, 1.0), 
    ncol=1,
    frameon=False,
    fontsize=12,
    title_fontsize=14,
    labelspacing=0.8,
    handletextpad=0.4
)

# 5. Save and Show
sns.despine()
plt.subplots_adjust(right=0.6)

today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'avg_{score}_performance_by_encoder-learner_num+str_num_selectedLLMs_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()