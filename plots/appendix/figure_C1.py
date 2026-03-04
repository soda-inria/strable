'''
MACRO-SOURCE ANALYSIS: map each source to a field (education, finance, health, etc.)
repeat the transposed histogram plot and the violin plot for macro-sources
'''

import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    results,
    score_list,
)

# Build the same mapping but derive entries from the existing `sources_list`

category_counts = results.groupby('category')['data_name'].nunique()
category_label_map = {cat: f"{cat} ({count})" for cat, count in category_counts.items()}
results['category_with_ds_count'] = results['category'].map(category_label_map)

hist_df = results.groupby('category', as_index=False)['data_name'].nunique()

# Plotting
# Swapped figsize dimensions slightly to accommodate horizontal layout
plt.figure(figsize=(5, 4))

# Use barh instead of bar
bars = plt.barh(hist_df['category'], hist_df['data_name'], color='#5da5da', edgecolor='#333333', linewidth=1.2)

# Add the count labels to the right of each bar
for bar in bars:
    width = bar.get_width() # Get the length of the bar
    plt.text(
        width + 0.5,        # x-position: end of bar + small offset
        bar.get_y() + bar.get_height()/2, # y-position: center of bar
        int(width), 
        va='center',        # Vertically align to center
        ha='left',          # Horizontally align to left of the text point
        fontsize=16, 
        fontweight='bold'
    )

# Formatting
plt.xlabel('Number of Datasets', fontsize=16) # Swapped label
plt.ylabel('Application Field', fontsize=20)           # Swapped label
plt.xlim(0, hist_df['data_name'].max() + 1)   # Changed ylim to xlim
plt.yticks(fontsize=16)                       # Removed rotation, usually easier to read horizontally
plt.grid(axis='x', linestyle='--', alpha=0.4) # Changed grid to x-axis

# Optional: Invert y-axis if you want the first category at the top
# plt.gca().invert_yaxis()

# Adjust layout
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'histogram_macro_category_ds_count_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()



'''
Distribution plots of R2 score for Num+Str 
'''


score = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

subset_r2 = results[
    (results['dtype'] == 'Num+Str') &
    (results['task'].isin(['regression'])) &
    (results['encoder'] != 'TabPFN-2.5') # drop TabPFN encoder
].groupby(['data_name'],as_index=False)[score].mean()

# all the negative r2 are clipped to zero
# subset_r2['r2'] = subset_r2['r2'].clip(lower=0)

# draw distribution plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=subset_r2,
    x=score,
    bins=15,
    kde=True,
    color='skyblue',
    edgecolor='black'
)
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xlabel(f"R2 {Y_METRIC_LABELS[score]}", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# for each bin of the histogram, print the upper and lower limit and the count,
# and annotate the count above each bar
for patch in ax.patches:
    bin_left = patch.get_x()
    bin_right = bin_left + patch.get_width()
    count = int(round(patch.get_height()))
    print(f"Bin lower limit: {bin_left}, Bin upper limit: {bin_right}, Count: {count}")
    if count > 0:
        ax.text(bin_left + patch.get_width() / 2, patch.get_height(),
                str(count), ha='center', va='bottom', fontsize=10)

#save picture
# format fot the pic name: plot_type + _ + metric + _ + level + date .png
today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'distribution_plot_{score}_r2_regression_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()


'''
Distribution plots of ROC_AUC score for Num+Str 
'''

score = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

subset_roc_auc = results[
    (results['dtype'] == 'Num+Str') &
    (~results['task'].isin(['regression'])) &
    (results['encoder'] != 'TabPFN-2.5') # drop TabPFN encoder
].groupby(['data_name'],as_index=False)[score].mean()

# draw distribution plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=subset_roc_auc,
    x=score,
    bins=15,
    kde=True,
    color='orange',
    edgecolor='black'
)
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xlabel(f"AUC {Y_METRIC_LABELS[score]}", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# for each bin of the histogram, print the upper and lower limit and the count,
# and annotate the count above each bar
for patch in ax.patches:
    bin_left = patch.get_x()
    bin_right = bin_left + patch.get_width()
    count = int(round(patch.get_height()))
    print(f"Bin lower limit: {bin_left}, Bin upper limit: {bin_right}, Count: {count}")
    if count > 0:
        ax.text(bin_left + patch.get_width() / 2, patch.get_height(),
                str(count), ha='center', va='bottom', fontsize=10)

#save picture
# format fot the pic name: plot_type + _ + metric + _ + level + date .png
today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'distribution_plot_{score}_roc_auc_classification_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()