'''
DATASETS DISTRIBUTION - HETEROGENEITY
MEDIAN+MODE
'''
import openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import dataset_summary_wide

# ---------------------------------------------------------
# 1. LOAD OPENML DATA
# ---------------------------------------------------------
# Assuming 'dataset_summary_wide' is already loaded in your environment
print("Fetching OpenML metadata...")
openml_datasets = openml.datasets.list_datasets(output_format="dataframe")

# ---------------------------------------------------------
# 2. PLOTTING
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

metrics = [
    ('num_rows', 'Rows', 'salmon', (0, 0)),
    ('num_columns', 'Columns', 'skyblue', (0, 1)),
    ('avg_cardinality', 'Cardinality', 'purple', (1, 0)),
    ('avg_string_length_per_cell', 'String Length (Avg # char)', 'green', (1, 1))
]

for col_name, title, color, (r, c) in metrics:
    ax = axes[r, c]
    
    # Extract data for this column
    data = dataset_summary_wide[col_name].dropna()
    
    # --- A. Plot STRABLE Benchmark (Your Data) ---
    sns.histplot(data, kde=True, log_scale=True, 
                 ax=ax, color=color, alpha=0.6, line_kws={'linewidth': 1.5},
                 edgecolor='black')
    
    # Lock the scale to your dataset's range
    x_min, x_max = ax.get_xlim()

    # --- B. CALCULATE & PLOT MODE (Red Line) ---
    log_data = np.log10(data[data > 0]) # Handle log(0)
    counts, bin_edges = np.histogram(log_data, bins='auto')
    max_idx = np.argmax(counts)
    
    # Estimate mode from bin center
    mode_log = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
    mode_val = 10**mode_log
    
    ax.axvline(x=mode_val, color='red', linestyle='-', linewidth=2, zorder=5)

    # --- C. CALCULATE & PLOT MEDIAN (Blue Line) ---
    median_val = data.median()
    
    # Draw vertical blue line
    ax.axvline(x=median_val, color='blue', linestyle='-', linewidth=2, zorder=5)

    # --- D. Plot OpenML Distribution (Overlay) ---
    if col_name == 'num_rows':
        ax2 = ax.twinx()
        openml_rows = openml_datasets[openml_datasets["NumberOfInstances"] > 0]["NumberOfInstances"]
        sns.kdeplot(openml_rows, ax=ax2, log_scale=True, 
                    color=color, linestyle='--', linewidth=2.5, warn_singular=False,
                    zorder=10)
        ax2.set_yticks([]) 
        ax2.set_ylabel("")
        ax2.set_xlim(x_min, x_max)
        
    elif col_name == 'num_columns':
        ax2 = ax.twinx()
        openml_cols = openml_datasets[openml_datasets["NumberOfFeatures"] > 0]["NumberOfFeatures"]
        sns.kdeplot(openml_cols, ax=ax2, log_scale=True, 
                    color=color, linestyle='--', linewidth=2.5, warn_singular=False,
                    zorder=10)
        ax2.set_yticks([]) 
        ax2.set_ylabel("")
        ax2.set_xlim(x_min, x_max)
    
    # --- LABELS ---
    transform = ax.get_xaxis_transform()
    
    # Mode Label (Top, Red)
    # Special handling for 'Rows' to avoid cutting off text on the left edge
    if col_name == 'num_rows':
        mode_x_pos = mode_val/6
        median_x_pos = median_val/4
    elif col_name == 'num_columns':
        mode_x_pos = mode_val
        median_x_pos = median_val-5
    elif col_name == 'avg_cardinality':
        mode_x_pos = mode_val
        median_x_pos = median_val-1000
    else:
        mode_x_pos = mode_val-5
        median_x_pos = median_val
            
    ax.text(mode_x_pos, 0.9, f'{mode_val:.0f}', 
            transform=transform,
            color='red', fontsize=12, fontweight='bold', 
            ha='left', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Median Label (Slightly Lower, Blue)
    # We position it at y=0.75 to minimize overlap with the Mode label
    ax.text(median_x_pos, 0.75, f'{median_val:.0f}', 
            transform=transform, 
            color='blue', fontsize=12, fontweight='bold', 
            ha='left', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # --- Formatting ---
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.15)

# ---------------------------------------------------------
# 3. LEGEND & LAYOUT
# ---------------------------------------------------------
plt.subplots_adjust(hspace=0.4, wspace=0.3, right=0.85)

fig.supxlabel("Value (Log Scale)", fontsize=16, y=0.01)
fig.supylabel("Frequency", fontsize=16)

# Custom Legend with Median Added
legend_elements = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='OpenML\nDistribution'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='STRABLE\nMode'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='STRABLE\nMedian'), # Added this
    Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='STRABLE\nDistribution')
]

fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.1, 0.83), fontsize=12)

today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'metadata_distribution_median_mode_{today_date}.{format}'
plt.savefig(path_configs["base_path"] + '/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()