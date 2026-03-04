'''
SAMPLING DIAGRAM
Integrate Marine's comments
'''

import os
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import TODAYS_FOLDER

def draw_schema():
    # Increased height to fit the new step comfortably
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # --- Styles ---
    # Blue: Data Pools
    box_style = dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1565C0", lw=2)
    # Green: Sampling
    sample_style = dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#2E7D32", lw=2)
    # Purple: Evaluation
    eval_style = dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec="#7B1FA2", lw=2)
    # Orange: Ranking (Processing)
    proc_style = dict(boxstyle="ellipse,pad=0.3", fc="#FFF3E0", ec="#EF6C00", lw=2)
    # Red: Final Metric
    tau_style = dict(boxstyle="darrow,pad=0.3", fc="#FFEBEE", ec="#C62828", lw=2)
    
    # Added shrinkA=0, shrinkB=0 to ensure the arrow doesn't stop short of the coordinates
    arrow_props = dict(arrowstyle="->", color="#555555", lw=2, mutation_scale=15, shrinkA=0, shrinkB=0)

    # --- Y-Coordinates for Layers (Centers) ---
    y_main = 9.8
    y_part = 8.5
    y_sub = 6.8
    y_eval = 4.8
    y_rank = 2.8
    y_tau = 1.0

    # --- Half-Heights (Approximate distance from center to edge) ---
    h_main = 0.45
    h_part = 0.45
    h_sub = 0.45
    h_eval = 0.55  # Taller because of 3 lines
    h_rank = 0.45
    h_tau = 0.5

    # --- 1. Main Pool ---
    ax.text(5, y_main+0.5, "Total Datasets Pool\n($M$ Datasets)", ha="center", va="center", size=12, bbox=box_style)

    # --- 2. Partitions (Disjoint) ---
    # Arrows: Start at Bottom of Main, End at Top of Partitions
    ax.annotate("", xy=(3, y_part + h_part), xytext=(5, y_main - h_main+0.5), arrowprops=arrow_props, zorder=0)
    ax.annotate("", xy=(7, y_part + h_part), xytext=(5, y_main - h_main+0.5), arrowprops=arrow_props, zorder=0)
    
    ax.text(3, y_part, "Partition A\n($M/2$ datasets)", ha="center", va="center", size=11, bbox=box_style)
    ax.text(7, y_part, "Partition B\n($M/2$ datasets)", ha="center", va="center", size=11, bbox=box_style)
    
    ax.text(5, 9.3, "Random Split (Disjoint)", ha="center", va="center", size=9, color="#555555", 
            bbox=dict(fc="white", ec="none"), zorder=0)

    # --- LOOP BOX ---
    rect = patches.FancyBboxPatch((0.5, 1.8), 9, 5.8, boxstyle="round,pad=0.2", 
                                  linewidth=2, edgecolor='gray', facecolor='none', linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(0.5, 7.0, "Bootstrap \n $K$ times", size=10, color="gray", weight="bold")

    # --- 3. Sampling (Green) ---
    # Start at Bottom of Partitions, End at Top of Subsamples
    ax.annotate("", xy=(3, y_sub + h_sub), xytext=(3, y_part - h_part), arrowprops=arrow_props)
    ax.annotate("", xy=(7, y_sub + h_sub), xytext=(7, y_part - h_part), arrowprops=arrow_props)
    
    ax.text(3, y_sub, "Subsample $S_1$\n(Size $N$)", ha="center", va="center", size=11, bbox=sample_style)
    ax.text(7, y_sub, "Subsample $S_2$\n(Size $N$)", ha="center", va="center", size=11, bbox=sample_style)

    # --- 4. Evaluation (Purple) ---
    # Start at Bottom of Subsamples, End at Top of Eval
    ax.annotate("", xy=(3, y_eval + h_eval), xytext=(3, y_sub - h_sub), arrowprops=arrow_props)
    ax.annotate("", xy=(7, y_eval + h_eval), xytext=(7, y_sub - h_sub), arrowprops=arrow_props)
    
    eval_text = "Evaluate Avg Score\n(R2/AUC) per Model\nacross $N$ datasets"
    ax.text(3, y_eval, eval_text, ha="center", va="center", size=10, bbox=eval_style)
    ax.text(7, y_eval, eval_text, ha="center", va="center", size=10, bbox=eval_style)

    # --- 5. Rankings (Orange) ---
    # Start at Bottom of Eval, End at Top of Rankings
    ax.annotate("", xy=(3, y_rank + h_rank), xytext=(3, y_eval - h_eval), arrowprops=arrow_props)
    ax.annotate("", xy=(7, y_rank + h_rank), xytext=(7, y_eval - h_eval), arrowprops=arrow_props)
    
    ax.text(3, y_rank, "Compute Rankings\n$R_1$", ha="center", va="center", size=11, bbox=proc_style)
    ax.text(7, y_rank, "Compute Rankings\n$R_2$", ha="center", va="center", size=11, bbox=proc_style)

    # --- 6. Kendall Tau (Red) ---
    # Start at Bottom of Rankings, End at Top of Kendall
    ax.annotate("", xy=(5, y_tau + h_tau), xytext=(3, y_rank - h_rank), arrowprops=arrow_props, zorder=1)
    ax.annotate("", xy=(5, y_tau + h_tau), xytext=(7, y_rank - h_rank), arrowprops=arrow_props, zorder=1)
    
    ax.text(5, y_tau, "Kendall $\\tau(R_1, R_2)$", ha="center", va="center", size=12, bbox=tau_style)

    plt.tight_layout()
    today_date = time.strftime("%Y-%m-%d")
    fmt = 'pdf'
    PIC_NAME = f'sampling_diagram_two_benchmarks_convergence_{today_date}.{fmt}'
    out_path = os.path.join(
        path_configs["base_path"],
        "results_pics",
        TODAYS_FOLDER,
        PIC_NAME,
    )
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()

draw_schema()