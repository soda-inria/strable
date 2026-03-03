'''
PERFORMANCE BY N_GRAM (STRING DIVERSITY)
'''

import os
import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from sklearn.linear_model import RANSACRegressor

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import (
    TODAYS_FOLDER,
    bin_feature_33_66,
    get_encoder_color,
    get_learner_marker,
    results,
    score_list,
    selected_encoders,
)

score = score_list[0] # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

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
        
        # 2. Filter: Keep only points where Y is between 0.4 and 0.9
        mask = (line_y >= 0.4) & (line_y <= 0.9)
        
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
            f"The algorithm is better on datasets\nwith LOW {feature_type}\n  ", 
            transform=ax.transAxes, fontsize=10, color='darkgreen', 
            va='top', ha='left', weight='bold')
    ax.text(0.15, 0.72, "↑", transform=ax.transAxes, fontsize=20, color=f'darkgreen', weight='bold', rotation=45)
    
    # Axis Labels
    ax.set_xlabel('Score on high string diversity datasets', fontsize=16, labelpad=10)
    ax.set_ylabel('Score on low string diversity datasets', fontsize=16, labelpad=10)
    ax.text(0.95, 0.05, 
            f"The algorithm is\nbetter on datasets\nwith HIGH {feature_type}\n", 
            transform=ax.transAxes, fontsize=10, color='darkred', 
            va='bottom', ha='right', weight='bold')
    ax.text(0.4, 0.2, "↓", transform=ax.transAxes, fontsize=24, color='darkred', weight='bold', rotation=45)

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
        df_plot['string_diversity_Bin'] = bin_feature_33_66(df_plot, 'string_diversity')


    df_plot = df_plot[df_plot['encoder'].isin(selected_encoders)].copy()

    df_plot = df_plot[(df_plot['method'] != 'num-str_tabpfn_tabpfn_default')]

    # Aggregate scores
    string_diversity_plot = (
        df_plot
        .groupby(['encoder', 'learner', 'string_diversity_Bin'], as_index=False)[score]
        .mean()
        .dropna()
    )

    # Merge Low / High into columns
    string_diversity_wide = string_diversity_plot.pivot_table(
        index=['encoder', 'learner'],
        columns='string_diversity_Bin',
        values=score
    ).reset_index()

    rename_map = {'High': 'High-bin', 'Low': 'Low-bin'}
    string_diversity_wide = string_diversity_wide.rename(columns=rename_map)

    # Combine names to create unique pipeline identifiers
    string_diversity_wide['pipeline'] = string_diversity_wide['encoder'] + " + " + string_diversity_wide['learner']

    # --- 1. SETTINGS & STYLES ---
    sns.set_theme(style="whitegrid", context="paper")

    # --- 5. MAIN EXECUTION ---
    fig = plt.figure(figsize=(5, 4)) 

    gs = GridSpec(1, 1)
    gs.update(left=0.15, right=0.99, top=0.99, bottom=0.05)
    ax0 = fig.add_subplot(gs[0])

    # --- 2. PLOTTING ---
    if upper_33:
        # Run the plot on ax0
        plot_regime_custom(string_diversity_wide, ax0, "String Diversity", "string diversity", set_xlim_min=0.4, set_xlim_max=0.9, set_ylim_min=0.4, set_ylim_max=0.9)

        # Note: I commented out the annotations since you had them commented
        # apply_annotations(ax0, card_wide, cardinality_positions)

    # --- 3. LEGEND GENERATION ---
    # Fix: Only get handles from ax0 (ax1 does not exist in this configuration)
    handles, labels = ax0.get_legend_handles_labels()

    # Sort logic
    temp_dict = dict(zip(labels, handles))
    sorted_labels = sorted(temp_dict.keys())
    sorted_handles = [temp_dict[k] for k in sorted_labels]

    # Place Legend
    # Changed to 'center left' at (1.02, 0.5) to anchor it right next to the plot
    fig.legend(
        sorted_handles, 
        sorted_labels,
        loc='center left',      # Anchor the left center of the legend...
        bbox_to_anchor=(1.02, 0.5), # ...to the right edge of the figure
        ncol=2, 
        fontsize=10,            # Reduced slightly to fit better
        frameon=False,
        columnspacing=0.05,
        handletextpad=0.1
    )

        
    today_date = time.strftime("%Y-%m-%d")
    fmt = 'pdf'
    PIC_NAME = f'performance_per_n_gram_1by2_percentile_{percentile}_{task}_{today_date}.{fmt}'
    out_path = os.path.join(
        path_configs["base_path"],
        "results_pics",
        TODAYS_FOLDER,
        PIC_NAME,
    )
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()