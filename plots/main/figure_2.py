'''
LEARNER PERFORMANCE PER NUM VS NUM+STR (averaged over all datasets)
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import dataset_summary_wide

score = score_list[0]
dtype_filter = ['Num+Str', 'Num']  

df_raw = results.copy()
df_raw = df_raw[df_raw['dtype'].isin(dtype_filter)]
df_raw = df_raw[df_raw['encoder'].isin(selected_encoders)]

# drop TabPFN-2.5 encoder
df_raw = df_raw[(df_raw['method'] != 'num-str_tabpfn_tabpfn_default')]

# Calculate Sort Order
avg_perf = df_raw.groupby(['learner', 'dtype'])[score].mean().unstack()
sort_order = avg_perf.sort_values('Num+Str', ascending=False).index

# Calculate Counts for Annotation
counts = df_raw.groupby(['learner', 'dtype'])[score].count().unstack().fillna(0)

# 2. PLOTTING
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(6, 5))

sns.barplot(
    data=df_raw, 
    y='learner', 
    x=score, 
    hue='dtype', 
    order=sort_order,
    palette={'Num': '#1f77b4', 'Num+Str': '#d62728'},
    edgecolor='black',
    linewidth=1,
    errorbar=('ci', 95), 
    capsize=0.1,         
    err_kws={'linewidth': 1.5, 'color': 'black'}, 
    ax=ax
)

# 3. HATCHING LOGIC
for i, learner_name in enumerate(sort_order):
    hatch = get_learner_hatch(learner_name)
    if hatch:
        for container in ax.containers:
            if isinstance(container[i], mpatches.Rectangle): 
                bar = container[i]
                rect = mpatches.Rectangle(
                    (bar.get_x(), bar.get_y()), 
                    bar.get_width(), 
                    bar.get_height(), 
                    fill=False, 
                    hatch=hatch, 
                    edgecolor='white', 
                    linewidth=0, 
                    alpha=0.6
                )
                ax.add_patch(rect)
                border = mpatches.Rectangle(
                    (bar.get_x(), bar.get_y()), 
                    bar.get_width(), 
                    bar.get_height(), 
                    fill=False, 
                    edgecolor='black', 
                    linewidth=1
                )
                ax.add_patch(border)


# 5. STYLING & LEGEND
ax.set_xlabel(f'Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)\nwith 95% CI', fontsize=16)
ax.set_xlim(0.2, 0.95)
ax.set_ylabel('')

handles, labels = ax.get_legend_handles_labels()
tuned_handle = mpatches.Patch(facecolor='gray', edgecolor='white', hatch='///', label='Tuned Model')
handles.append(tuned_handle)
labels.append("Tuned")

ax.legend(
    handles=handles, 
    labels=labels,
    title='Features & Model', 
    loc='lower left', 
    bbox_to_anchor=(0.6, -0.02), 
    fontsize=10, 
    title_fontsize=16,
    framealpha=0.9
)

sns.despine(left=False, bottom=False)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'avg_{score}_performance_by_learner_num+str_num_selectedLLMs_{today_date}.{format}'
plt.savefig(path_configs["base_path"] + '/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()