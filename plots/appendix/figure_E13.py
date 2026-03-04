'''
AVG PERFORMANCE PER ENCODER
'''

import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    get_encoder_color,
    results,
    score_list,
)

score = score_list[0]
dtype = 'Num+Str'

# plot_data = results[(results['dtype'] == dtype) & (results['encoder'].isin(selected_encoders))].copy()
plot_data = results[(results['dtype'] == dtype)].copy()

# drop TabPFN-2.5 encoder
plot_data = plot_data[(plot_data['method'] != 'num-str_tabpfn_tabpfn_default')]

plot_data

# 1. Prepare Data (using your provided snippet)
encoder_performance = plot_data.groupby(['encoder'], as_index=False)[score].mean().sort_values(by=[score], ascending=False)

# 2. Create Plot
plt.figure(figsize=(5, 15)) # Width, Height

# Generate palette list matching the sorted order of encoders
palette_list = [get_encoder_color(enc) for enc in encoder_performance['encoder']]

ax = sns.barplot(
    data=encoder_performance,
    y='encoder',
    x=score,
    palette=palette_list,
    edgecolor='black',
    linewidth=0.5
)

# 3. Add Value Labels
# fmt='%.3f' matches the precision in your screenshot
# ax.bar_label(ax.containers[0], fmt='%.3f', padding=5, fontsize=16)

# 4. Styling
ax.set_xlabel(f'Average {Y_METRIC_LABELS[score]} ($R^2$ & AUC)', fontsize=18, ha='right', x=1.0)
ax.set_ylabel('') # Remove y-label as the names are self-explanatory
ax.set_xlim(0.6, 0.75) # Add room for labels
sns.despine()

today_date = time.strftime("%Y-%m-%d")
PIC_NAME = f'barplot_encoder_performance_{today_date}.pdf'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()