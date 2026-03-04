'''
DISTRIBUTION OF PROPORTION OF UNIQUE TEXT CELLS
MEDIAN+MODE
'''

import os
import time

import matplotlib.legend_handler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    dataset_summary_wide,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
feature_name = 'prop_unique_text_cells'
plot_title = 'Proportion Unique Text Cells'
plot_color = 'teal' 

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))

# 1. Extract Data
# Assuming 'dataset_summary_wide' is already loaded in your environment
data = dataset_summary_wide[feature_name].dropna()

# 2. Plot Histogram + KDE (Log Scale)
sns.histplot(data, kde=True, log_scale=True, 
             ax=ax, color=plot_color, alpha=0.6, line_kws={'linewidth': 1.5},
             edgecolor='black')

# 3. Calculate Mode (Empirical from Data bins)
# Handle log(0) explicitly if data contains absolute zeros, though 'unique' implies > 0
log_data = np.log10(data[data > 0]) 
counts, bin_edges = np.histogram(log_data, bins='auto')
max_idx = np.argmax(counts)
mode_log = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
mode_val = 10**mode_log

# 4. Calculate Median
median_val = data.median()

# 5. Add Mode Line and Label (Red)
ax.axvline(x=mode_val, color='red', linestyle='-', linewidth=2, zorder=5)

# Place text slightly offset
ax.text(mode_val, 0.9, f'{mode_val:.2f}', 
        transform=ax.get_xaxis_transform(), 
        color='red', fontsize=12, fontweight='bold', 
        ha='left', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 6. Add Median Line and Label (Blue)
ax.axvline(x=median_val, color='blue', linestyle='-', linewidth=2, zorder=5)

# Place text lower (y=0.75) to avoid overlap with Mode label
ax.text(median_val-0.09, 0.75, f'{median_val:.2f}', 
        transform=ax.get_xaxis_transform(), 
        color='blue', fontsize=12, fontweight='bold', 
        ha='left', va='bottom', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 7. Formatting
ax.set_title(plot_title, fontsize=14, fontweight='bold')
ax.set_xlabel("Proportion (Log Scale)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.grid(True, alpha=0.15)

# ---------------------------------------------------------
# LEGEND
# ---------------------------------------------------------
# Combined handle for Hist + KDE
strable_dist_handle = (
    Patch(facecolor=plot_color, alpha=0.6, edgecolor='black'),
    Line2D([0], [0], color=plot_color, linewidth=1.5)
)
mode_handle = Line2D([0], [0], color='red', linestyle='-', linewidth=2)
median_handle = Line2D([0], [0], color='blue', linestyle='-', linewidth=2) # New median handle

fig.legend(
    handles=[strable_dist_handle, mode_handle, median_handle],
    labels=['STRABLE\nDistribution', 'STRABLE\nMode', 'STRABLE\nMedian'], # Added Median label
    loc='upper right', 
    bbox_to_anchor=(0.65, 0.85),
    fontsize=10,
    handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)}
)

plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'prop_unique_text_cells_distribution_median_mode_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()