'''
PERFORMANCE BY CARDINALITY AND STRING LENGTH REGIMES
'''

import os
import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RANSACRegressor

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    bin_feature_33_66,
    get_encoder_color,
    get_learner_marker,
    results,
    score_list,
    selected_encoders,
)


# --- 1. SETTINGS & STYLES ---
sns.set_theme(style="whitegrid", context="paper")

# --- 4. PLOTTING FUNCTION ---
def plot_regime_custom(df, ax, title, feature_type, set_xlim_min=0.45, set_xlim_max=1.0, set_ylim_min=0.45, set_ylim_max=1.0):
    # 1. Diagonal & Backgrounds
    lims = [0.0, 1.0]
    ax.plot(lims, lims, ls='--', c='grey', alpha=0.5, zorder=0)
    
    # Regions
    # Below Diagonal (High Bin > Low Bin) -> Red
    ax.fill_between(lims, [0,0], lims, color='red', alpha=0.05, transform=ax.transData)
    # Above Diagonal (Low Bin > High Bin) -> Green
    ax.fill_between(lims, lims, [1.1, 1.1], color='green', alpha=0.05, transform=ax.transData)
    
    # ---------------------------------------------------------
    # NEW: RANSAC REGRESSOR LOGIC (CLIPPED)
    # ---------------------------------------------------------
    if len(df) > 5: # Only fit if we have enough points
        # Prepare Data
        X = df[['High-bin']].values
        y = df['Low-bin'].values
        
        # Fit RANSAC
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        
        # 1. Generate dense prediction line across X range
        # (Use 500 points to ensure smooth clipping)
        line_X = np.linspace(set_xlim_min, set_xlim_max, 500).reshape(-1, 1)
        line_y = ransac.predict(line_X)
        
        # 2. Filter: Keep only points where Y is between 0.7 and 1.0
        mask = (line_y >= 0.4) & (line_y <= 1.0)
        
        line_X_clipped = line_X[mask]
        line_y_clipped = line_y[mask]
        
        # 3. Plot clipped line
        if len(line_X_clipped) > 1:
            ax.plot(line_X_clipped, line_y_clipped, color='cornflowerblue', linewidth=3, 
                    alpha=0.8, zorder=2, label='_nolegend_')
    # ---------------------------------------------------------

    # 2. Plot Points Loop
    unique_pipelines = df['pipeline'].unique()
    
    for pipe in unique_pipelines:
        subset = df[df['pipeline'] == pipe]
        if len(subset) == 0: continue
        
        # Parse info
        enc = subset['encoder'].iloc[0]
        lrn = subset['learner'].iloc[0]
        
        color = get_encoder_color(enc)
        marker = get_learner_marker(lrn)
        
        # Tuning Logic
        is_tuned = 'tuned' in lrn
        
        style_kwargs = {
            'color': color,
            'marker': marker,
            'markersize': 10,
            'linestyle': '',
            'alpha': 0.9,
            'zorder': 3 
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
                'markeredgecolor': 'white',
                'markeredgewidth': 0.5
            })
            
        # Plot
        ax.plot(subset['High-bin'], subset['Low-bin'], label=pipe, **style_kwargs)
        ax.set_xlim(set_xlim_min, set_xlim_max)
        ax.set_ylim(set_ylim_min, set_ylim_max)

    # 3. Styling & Annotations
    ax.set_title(title+' '+title_tag, fontsize=18, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 4. Descriptive Text Annotations (Same as before...)
    ax.text(0.05, 0.98, 
            f"The algorithm is better on datasets\nwith LOW {feature_type}\nfeatures", 
            transform=ax.transAxes, fontsize=16, color='darkgreen', 
            va='top', ha='left', weight='bold')
    ax.text(0.15, 0.82, "↑", transform=ax.transAxes, fontsize=20, color=f'darkgreen', weight='bold', rotation=45)
    
    # Axis Labels
    if feature_type == 'cardinality':
        ax.set_xlabel('Score on high cardinality datasets', fontsize=20, labelpad=10)
        ax.set_ylabel('Score on low cardinality datasets', fontsize=20, labelpad=10)
        if upper_33:
            ax.text(0.95, 0.05, 
                    f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
                    transform=ax.transAxes, fontsize=16, color='darkred', 
                    va='bottom', ha='right', weight='bold')
            ax.text(0.8, 0.25, "↓", transform=ax.transAxes, fontsize=24, color='darkred', weight='bold', rotation=45)
        # else:
        #      ax.text(0.95, 0.05, 
        #             f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
        #             transform=ax.transAxes, fontsize=11, color='darkred', 
        #             va='bottom', ha='right', weight='bold')
        #      ax.text(0.85, 0.2, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)
    else:
        ax.set_xlabel('Score on high string length datasets', fontsize=20, labelpad=10)
        ax.set_ylabel('Score on low string length datasets', fontsize=20, labelpad=10)
        if upper_33:
            ax.text(0.95, 0.05, 
                    f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
                    transform=ax.transAxes, fontsize=16, color='darkred', 
                    va='bottom', ha='right', weight='bold')
            ax.text(0.8, 0.25, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)
        # else:
        #     ax.text(0.95, 0.05, 
        #             f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\nfeatures", 
        #             transform=ax.transAxes, fontsize=11, color='darkred', 
        #             va='bottom', ha='right', weight='bold')
        #     ax.text(0.85, 0.2, "↓", transform=ax.transAxes, fontsize=20, color='darkred', weight='bold', rotation=45)



def apply_annotations(ax, df, position_dict):
    for target, offset in position_dict.items():
        # 1. Find the point
        row = df[df['pipeline'] == target]
        if len(row) == 0: continue
            
        x = row['High-bin'].values[0]
        y = row['Low-bin'].values[0]
        
        # 2. Determine alignment based on offset direction
        # If pushing Right (x>0), align text Left. If pushing Left (x<0), align text Right.
        ha = 'left' if offset[0] >= 0 else 'right'
        
        # If pushing Up (y>0), align Bottom. If pushing Down (y<0), align Top.
        va = 'bottom' if offset[1] >= 0 else 'top'

        # 3. Draw Annotation
        ax.annotate(
            target, 
            xy=(x, y), 
            xytext=offset,
            textcoords='offset points',
            fontsize=12, 
            fontweight='bold', 
            color='black',
            ha=ha, 
            va=va
        )


score = score_list[0] # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

# task = 'all_task'

for task in ['regression', 'classification', 'all_task']:

    if task == 'regression':
        title_tag = '($R^2$)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] == 'regression')].copy()
    elif task == 'classification':
        title_tag = '(AUC)'
        df_plot = results[(results['dtype'] == 'Num+Str') & (results['task'] != 'regression')].copy()
    else:
        title_tag = '(Avg $R^2$ & AUC)'
        df_plot = results[results['dtype'] == 'Num+Str'].copy()

    upper_33 = True # keep 33 percentile (drop median)

    if upper_33:
        percentile = 33
        df_plot['Card_Bin'] = bin_feature_33_66(df_plot, 'avg_cardinality')
        df_plot['Str_Bin'] = bin_feature_33_66(df_plot, 'avg_string_length_per_cell')
    # else:
    #     percentile = 50
    #     df_plot['Card_Bin'] = bin_feature_median(df_plot, 'avg_cardinality')
    #     df_plot['Str_Bin'] = bin_feature_median(df_plot, 'avg_string_length_per_cell')


    df_plot = df_plot[df_plot['encoder'].isin(selected_encoders)].copy()

    df_plot = df_plot[(df_plot['method'] != 'num-str_tabpfn_tabpfn_default')]

    # show all rows
    pd.set_option('display.max_rows', None)

    # Aggregate scores
    card_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'Card_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    str_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'Str_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    # Merge Low / High into columns
    card_wide = card_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='Card_Bin',
        values=score
    ).reset_index()

    str_wide = str_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='Str_Bin',
        values=score
    ).reset_index()

    rename_map = {'High': 'High-bin', 'Low': 'Low-bin'}
    card_wide = card_wide.rename(columns=rename_map)
    str_wide = str_wide.rename(columns=rename_map)

    # Combine names to create unique pipeline identifiers
    card_wide['pipeline'] = card_wide['encoder'] + " + " + card_wide['learner']
    str_wide['pipeline'] = str_wide['encoder'] + " + " + str_wide['learner']

    # --- 5. MAIN EXECUTION ---
    fig = plt.figure(figsize=(16, 8)) # Increased size for better readability

    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # --- APPLY TO PLOTS ---

    # Run Plots
    # Assumes 'High-bin' and 'Low-bin' are column names in card_wide/str_wide
    if upper_33:
        plot_regime_custom(card_wide, ax0, "Cardinality", "cardinality", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)
        plot_regime_custom(str_wide, ax1, "String Length", "string length", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)

        cardinality_positions = {
        "TabPFN-2.5 + TabPFN-2.5":       (10, 10),   # Push Right & Up
        "ContextTab + ContextTab":       (-3, -5), # Push Left & Down (avoid top edge)
        "Tf-Idf + TabPFN-2.5":     (-5, 10)  # Push Left & Down
        }

        # --- Configuration for Plot 2 (String Length) ---
        string_positions = {
            "TabPFN-2.5 + TabPFN-2.5":       (-1, 10),   # Push Right & Up
            "ContextTab + ContextTab":       (-20, -1),   # Push Left (avoid right spine)
            "Tf-Idf + TabPFN-2.5":     (-20, 2),    # Push Left harder (avoid edge)
            # "LM Qwen-3-8B + TabPFN-2.5":     (-10, -10),
            # "LM LLaMA-3.1-8B + TabPFN-2.5":   (-15, -1)
        }

        # --- APPLY SEPARATELY ---
        # apply_annotations(ax0, card_wide, cardinality_positions)
        # apply_annotations(ax1, str_wide, string_positions)

    # --- 6. LEGEND GENERATION ---
    # We build a custom legend to handle the "Tuned" look nicely
    handles, labels = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()

    # Merge
    temp_dict = dict(zip(labels + l1, handles + h1))
    sorted_labels = sorted(temp_dict.keys())
    sorted_handles = [temp_dict[k] for k in sorted_labels]

    # Place Legend
    fig.legend(
        sorted_handles, 
        sorted_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=5, 
        fontsize=12,
        frameon=False,
        columnspacing=0.05,
        handletextpad=0.1
    )


    # Layout adjustments
    plt.subplots_adjust(bottom=0.5, top=1.15, left=0.08, right=0.98)

    today_date = time.strftime("%Y-%m-%d")
    fmt = 'pdf'
    PIC_NAME = f'performance_per_cardinality_string_length_1by2_percentile_{percentile}_{task}_{today_date}.{fmt}'
    out_path = os.path.join(
        path_configs["base_path"],
        "results_pics",
        TODAYS_FOLDER,
        PIC_NAME,
    )
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()