'''
COMPARATIVE PARETO PLOTS - split into 2 
using big LLMs list, baselines and E2E models
legend should be included
'''




import os
import time

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    get_encoder_color,
    get_learner_color_simple,
    get_learner_marker,
    get_pareto_front,
    learner_colors,
    learner_shapes,
    results,
    score_list,
    selected_encoders,
)

dtype = 'num-str'
progressive_transparency = False

for metric in score_list:
    
    Y_METRIC = metric

    # 1. Update Data Aggregation for the current metric
    if 'encoder_learner' in results.columns:
        agg_cols = [Y_METRIC, 'inference_time_per_1k', 'run_time_per_1k']
        group_cols = ['encoder_learner', 'encoder', 'learner']
        df_agg = results[(results['method'].str.contains(f'{dtype}_')) & (results['encoder'].isin(selected_encoders)) & (results['method'] != 'num-str_tabpfn_tabpfn_default')].groupby(group_cols)[agg_cols].median().reset_index()
    

    HIGHER_SCORE_IS_BETTER = True

    # We need explicit dictionaries for every unique value in the dataframe
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()

    # A. Marker Palette (Always based on Learner)
    # Maps "XGBoost" -> 's', "XGBoost-tuned" -> 's'
    learner_markers_dict = {L: get_learner_marker(L) for L in unique_learners}

    # B. Color Palettes
    # Palette for Right Plot (Hue = Learner)
    learner_palette_dict = {L: get_learner_color_simple(L) for L in unique_learners}

    # Palette for Left Plot (Hue = Encoder)
    encoder_palette_dict = {E: get_encoder_color(E) for E in unique_encoders}

    # 2. Re-initialize Plotting Parameters
    sns.set_style("white")
    ROW_METRICS = ['inference_time_per_1k', 'run_time_per_1k']
    COL_FACTORS = ['encoder', 'learner']
    COL_TITLES = ['Encoder', 'Learner'] 
    ROW_LABELS = ['Inference Time per 1K samples (s)', 'Total Run Time per 1K samples (s)']

    # --- PALETTE PREPARATION ---
    unique_learners = df_agg['learner'].unique()
    unique_encoders = df_agg['encoder'].unique()

    learner_pal = {L: get_learner_color_simple(L) for L in unique_learners}
    encoder_pal = {E: get_encoder_color(E) for E in unique_encoders}
    learner_markers = {L: get_learner_marker(L) for L in unique_learners}

    # 3. INDENTED PLOTTING LOOP
    for row_idx, x_metric in enumerate(ROW_METRICS):
        fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
        pareto_df = get_pareto_front(df_agg, x_metric, Y_METRIC, True)

        # save pareto_df to latex for later analysis
        today_date = time.strftime("%Y-%m-%d")
        filename = f"pareto_frontier_{Y_METRIC}_vs_{x_metric}_{today_date}.tex"
        save_path = os.path.join(path_configs["base_path"], "results_tables", TODAYS_FOLDER, filename)
        with open(save_path, 'w') as f:
            f.write(pareto_df.to_latex(index=False))

        # =======================================================
        # PRINT PARETO FRONTIER DETAILS
        # =======================================================
        print(f"\n" + "="*60)
        print(f"PARETO FRONTIER: {Y_METRIC} (Max) vs {x_metric} (Min)")
        print("="*60)
        
        # Select and rename columns for cleaner output
        display_cols = ['encoder', 'learner', x_metric, Y_METRIC]
        
        # Sort by the X metric (Time) to show the progression from Fast->Slow
        frontier_view = pareto_df[display_cols].sort_values(by=x_metric)
        
        # Print formatted table
        print(f"{'Encoder':<25} | {'Learner':<20} | {'Time (s)':<10} | {'Score':<8}")
        print("-" * 75)
        
        for _, row in frontier_view.iterrows():
            e = str(row['encoder'])[:25] # Truncate long names for display
            l = str(row['learner'])[:20]
            t = row[x_metric]
            s = row[Y_METRIC]
            print(f"{e:<25} | {l:<20} | {t:<10.4f} | {s:<8.4f}")
        print("="*60 + "\n")
        # =======================================================

        y_bottom = df_agg[Y_METRIC].min()  # Or hardcode e.g., 0.0
        # Right X: Use a value larger than your max X to ensure it hits the edge
        x_right_edge = df_agg[x_metric].max() * 5.0 

        # 3. Create extension points
        # Point A: (First X, Bottom Y) -> Creates vertical line from axis up to first point
        start_point = pd.DataFrame({
            x_metric: [pareto_df[x_metric].iloc[0]], 
            Y_METRIC: [y_bottom]
        })
        
        # Point B: (Right Edge X, Last Y) -> Creates horizontal line to the right
        end_point = pd.DataFrame({
            x_metric: [x_right_edge], 
            Y_METRIC: [pareto_df[Y_METRIC].iloc[-1]]
        })

        pareto_extended = pd.concat([start_point, pareto_df, end_point], ignore_index=True)

        for col_idx, factor in enumerate(COL_FACTORS):
            ax = axes[col_idx]
            
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3, zorder=0)

            # --- SETUP CONTEXT ---
            if factor == 'encoder':
                current_palette = encoder_pal
                hue_col = 'encoder'
                style_col = 'learner'
            else:
                current_palette = learner_pal
                hue_col = 'learner'
                style_col = 'learner'

            if progressive_transparency == False:
                # --- SPLIT DATA ---
                mask_tuned = df_agg['learner'].str.contains('tuned')
                df_default = df_agg[~mask_tuned]
                df_tuned = df_agg[mask_tuned]

                # --- PLOT 1: DEFAULTS (Full Fill) ---
                sns.lineplot(
                    data=df_default, x=x_metric, y=Y_METRIC,
                    hue=hue_col, style=style_col,
                    palette=current_palette, markers=learner_markers,
                    dashes=False, estimator=None, lw=0, markersize=9,
                    ax=ax, legend=False, # Disable auto-legend
                    **{'fillstyle': 'full', 'markeredgewidth': 1.0, 'markeredgecolor': 'white', 'markeredgecolor': 'black'}
                )
                
                # --- PLOT 2: TUNED (Half Fill with Black Contour) ---
                sns.lineplot(
                    data=df_tuned, x=x_metric, y=Y_METRIC,
                    hue=hue_col, style=style_col,
                    palette=current_palette, markers=learner_markers,
                    dashes=False, estimator=None, lw=0, markersize=9,
                    ax=ax, legend=False, # Disable auto-legend
                    # Key fix: Black edge to show the contour of the half-filled shape
                    **{'fillstyle': 'left', 'markerfacecoloralt': 'white', 
                    'markeredgecolor': 'black', 'markeredgewidth': 1.0} 
                )
                
                # --- PARETO LINE ---
                
                ax.step(
                    pareto_extended[x_metric], 
                    pareto_extended[Y_METRIC], 
                    where='post',
                    linestyle='--', 
                    color='black', 
                    linewidth=1.2, 
                    zorder=0
                )
                
                # --- AXIS FORMATTING ---
                ax.set_box_aspect(1) 
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                # Main Title
                if factor == 'encoder':
                    ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                                loc='left', y=1.68, x=-0.25) 
                    subtitle_text = "(shape = learner, color = encoder)"
                    ax.text(0.2, 1.72,  subtitle_text, transform=ax.transAxes, 
                            fontsize=10, style='italic', color='#333333')
                else:
                    ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                                loc='left', y=1.68, x=-0.15) 
                    subtitle_text = "(shape = learner, color = learner)" 
                    ax.text(0.25, 1.72,  subtitle_text, transform=ax.transAxes, 
                            fontsize=10, style='italic', color='#333333')

                ax.set_xlabel('')
                ax.set_ylim(bottom=y_bottom)

                if col_idx == 0:
                    ax.set_ylabel(f'Avg {Y_METRIC_LABELS[Y_METRIC]} ($R^2$ & AUC)', fontsize=10)
                else:
                    ax.set_ylabel('')
                
                
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Force Left and Bottom spines to stay at the "axes" 0-coordinate
                # This ensures they meet perfectly at the corner
                ax.spines['left'].set_position(('axes', 0))
                ax.spines['bottom'].set_position(('axes', 0))
                
                # Ensure they extend fully
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                # 1. Map Encoder Names to Learner Keys
                e2e_map = {
                    'CatBoost': 'CatBoost',
                    'ContextTab': 'ContextTab',
                    'TabSTAR': 'TabSTAR',
                    'TabPFN-2.5': 'TabPFN',
                    # 'Tarte': 'Tarte'
                }

                # 2. Create a 'markers' dictionary for the Encoders
                # Start with default 'D' (Diamond) for everyone
                encoder_markers = {enc: 'D' for enc in unique_encoders}

                # Overwrite the E2E models with their Learner shapes
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_markers:
                        encoder_markers[enc_name] = learner_shapes[learner_name]

                # IMPORTANT: Ensure your encoder_pal (colors) also matches the learner colors for these 3
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_pal:
                        encoder_pal[enc_name] = learner_colors[learner_name]

                # --- MANUAL LEGEND GENERATION ---
                if factor == 'encoder':
                    std_encoders = [e for e in unique_encoders if e not in e2e_map]
                    e2e_encoders = [e for e in unique_encoders if e in e2e_map]
                    
                    # 2. Sort both lists independently
                    std_encoders.sort()
                    e2e_encoders.sort()
                    
                    # 3. Combine: Standard first, E2E last
                    sorted_encoder_list = std_encoders + e2e_encoders

                    enc_handles = []
                    for enc in sorted_encoder_list:
                        current_color = encoder_pal[enc]
                        # current_marker = 'D'  # Default Diamond
                        if enc in e2e_map:
                            learner_key = e2e_map[enc]
                            current_color = learner_colors[learner_key] # Force learner color
                            current_marker = learner_shapes[learner_key] # Force shape color
                            h = mlines.Line2D([], [], color=current_color, marker=current_marker, 
                                        linestyle='', markersize=8, label=enc)
                        else:
                            h = mpatches.Patch(color=current_color, label=enc)
                        enc_handles.append(h)
                                        
                    ax.legend(
                        handles=enc_handles,
                        loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.9  
                    )
                
                else:
                    # RIGHT LEGEND: Shape/Color by Learner, Grouped (Base, Tuned)
                    # 1. Define specific sorting order
                    base_order = ['Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 
                                'TabSTAR', 'ContextTab', 'CatBoost', 
                                # 'Tarte'
                                ]
                    
                    sorted_learners = []
                    # Add families in order: Base then Tuned
                    for base in base_order:
                        if base in unique_learners: 
                            sorted_learners.append(base)
                        if f"{base}-tuned" in unique_learners:
                            sorted_learners.append(f"{base}-tuned")
                    
                    # Catch leftovers
                    for l in unique_learners:
                        if l not in sorted_learners:
                            sorted_learners.append(l)

                    # 2. Build Handles
                    lrn_handles = []
                    for lrn in sorted_learners:
                        is_tuned = 'tuned' in lrn
                        marker = learner_markers[lrn]
                        color = learner_pal[lrn]
                        
                        if is_tuned:
                            # Half-filled with black edge
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='left', markerfacecoloralt='white',
                                            markeredgecolor='black', markeredgewidth=1.0)
                        else:
                            # Full filled
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='full', markeredgecolor='black',markeredgewidth=1.0)
                        lrn_handles.append(h)

                    ax.legend(
                        handles=lrn_handles,
                        loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.4
                    )
            else:
                # --- PREPARE LAYERS (Progressive Transparency) ---
                base_priority = ['Ridge', 'XGBoost', 'ExtraTrees', 'TabPFN', 
                                'TabSTAR', 'ContextTab', 'CatBoost'
                                # , 'Tarte'
                                ]
                
                # KEY CHANGE 1: STRICT SORTING
                # Primary Sort: Base Family order (CatBoost bottom, Tarte top)
                # Secondary Sort: 'tuned' status (Default=0 (Bottom), Tuned=1 (Top))
                z_learners = sorted(list(unique_learners), 
                                    key=lambda x: (
                                        base_priority.index(x.split('-')[0]) if x.split('-')[0] in base_priority else 999,
                                        1 if 'tuned' in x else 0 
                                    ))
                
                total_layers = len(z_learners)

                # 2. Iterate and Plot Layer by Layer
                for i, lrn_name in enumerate(z_learners):
                    
                    # A. Calculate Progressive Alpha (Opacity)
                    # Bottom Layer (i=0) -> 1.0 (Opaque)
                    # Top Layer (i=Max)  -> 0.4 (Transparent)
                    if total_layers > 1:
                        curr_alpha = 1.0 - (0.6 * (i / (total_layers - 1)))
                    else:
                        curr_alpha = 1.0

                    subset = df_agg[df_agg['learner'] == lrn_name]
                    if subset.empty:
                        continue

                    is_tuned = 'tuned' in lrn_name
                    
                    if is_tuned:
                        # KEY CHANGE 2: Transparent Empty Half
                        # Changed markerfacecoloralt from 'white' to 'none'
                        style_kwargs = {
                            'fillstyle': 'left', 
                            'markerfacecoloralt': 'none', # See-through!
                            'markeredgecolor': 'black', 
                            'markeredgewidth': 1.0
                        }
                    else:
                        # Default: Full-filled
                        style_kwargs = {
                            'fillstyle': 'full', 
                            'markeredgecolor': 'black', 
                            'markeredgewidth': 1.0
                        }

                    sns.lineplot(
                        data=subset, x=x_metric, y=Y_METRIC,
                        hue=hue_col, style=style_col,
                        palette=current_palette, markers=learner_markers,
                        dashes=False, estimator=None, lw=0, markersize=9,
                        ax=ax, legend=False,
                        alpha=curr_alpha, # Apply calculated transparency
                        **style_kwargs
                    )
                
                # --- PARETO LINE ---
                ax.step(
                    pareto_df[x_metric], pareto_df[Y_METRIC], where='post',
                    linestyle='--', color='black', linewidth=1.2, zorder=0
                )
                
                # --- AXIS FORMATTING ---
                ax.set_box_aspect(1) 
                ax.set_xscale('log')
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                ax.set_title(COL_TITLES[col_idx], fontsize=14, fontweight='bold', 
                             loc='left', y=1.68) 
            
                # 2. Subtitle (Push slightly below title but above legend, e.g., y=1.35)
                subtitle_text = "(shape = learner, color = encoder)" if factor == 'encoder' else "(shape = learner, color = learner)" 
                ax.text(0.4, 1.72,  subtitle_text, transform=ax.transAxes, 
                        fontsize=10, style='italic', color='#333333')

                ax.set_xlabel('')
                if col_idx == 0:
                    ax.set_ylabel(Y_METRIC_LABELS[Y_METRIC], fontsize=10)
                else:
                    ax.set_ylabel('')

                # sns.despine(ax=ax, trim=False, offset=0) 
                # WITH THIS MANUAL SPINE CONFIGURATION:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Force Left and Bottom spines to stay at the "axes" 0-coordinate
                # This ensures they meet perfectly at the corner
                ax.spines['left'].set_position(('axes', 0))
                ax.spines['bottom'].set_position(('axes', 0))
                
                # Ensure they extend fully
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                e2e_map = {
                    'CatBoost': 'CatBoost',
                    'ContextTab': 'ContextTab',
                    'TabSTAR': 'TabSTAR',
                    'TabPFN-2.5': 'TabPFN',
                    # 'Tarte': 'Tarte'
                }

                # 2. Create a 'markers' dictionary for the Encoders
                # Start with default 'D' (Diamond) for everyone
                encoder_markers = {enc: 'D' for enc in unique_encoders}

                # Overwrite the E2E models with their Learner shapes
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_markers:
                        encoder_markers[enc_name] = learner_shapes[learner_name]

                # IMPORTANT: Ensure your encoder_pal (colors) also matches the learner colors for these 3
                for enc_name, learner_name in e2e_map.items():
                    if enc_name in encoder_pal:
                        encoder_pal[enc_name] = learner_colors[learner_name]

                # --- LEGEND GENERATION ---
                if factor == 'encoder':
                    std_encoders = [e for e in unique_encoders if e not in e2e_map]
                    e2e_encoders = [e for e in unique_encoders if e in e2e_map]
                    
                    # 2. Sort both lists independently
                    std_encoders.sort()
                    e2e_encoders.sort()
                    
                    # 3. Combine: Standard first, E2E last
                    sorted_encoder_list = std_encoders + e2e_encoders

                    enc_handles = []
                    for enc in sorted_encoder_list:
                        current_color = encoder_pal[enc]
                        # current_marker = 'D'  # Default Diamond
                        if enc in e2e_map:
                            learner_key = e2e_map[enc]
                            current_color = learner_colors[learner_key] # Force learner color
                            current_marker = learner_shapes[learner_key] # Force shape color
                            h = mlines.Line2D([], [], color=current_color, marker=current_marker, 
                                        linestyle='', markersize=8, label=enc)
                        else:
                            h = mpatches.Patch(color=current_color, label=enc)
                        enc_handles.append(h)
                    
                    ax.legend(
                        handles=enc_handles,
                        loc='lower center', bbox_to_anchor=(0.5, 1.05),
                        ncol=2, fontsize=8, frameon=False, 
                        title=None, columnspacing=0.8
                    )
                else:
                    # Legend Logic (Same as before)
                    sorted_learners_leg = []
                    for base in base_priority:
                        if base in unique_learners: sorted_learners_leg.append(base)
                        if f"{base}-tuned" in unique_learners: sorted_learners_leg.append(f"{base}-tuned")
                    for l in unique_learners:
                        if l not in sorted_learners_leg: sorted_learners_leg.append(l)

                    lrn_handles = []
                    for lrn in sorted_learners_leg:
                        is_tuned = 'tuned' in lrn
                        marker = learner_markers[lrn]
                        color = learner_pal[lrn]
                        # Keep legend opaque (alpha=1) for clarity
                        if is_tuned:
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='left', markerfacecoloralt='white', # Keep white for legend visibility
                                            markeredgecolor='black', markeredgewidth=1.0)
                        else:
                            h = mlines.Line2D([], [], color=color, marker=marker, linestyle='',
                                            markersize=8, label=lrn,
                                            fillstyle='full', markeredgecolor='black', markeredgewidth=1.0)
                        lrn_handles.append(h)

                    ax.legend(
                        handles=lrn_handles,
                        loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=2, fontsize=10, frameon=False, 
                        title=None, columnspacing=0.4
                    )

        fig.text(0.5, 0.12, f'{ROW_LABELS[row_idx]} (Log Scale)', ha='center', fontsize=10)
        plt.subplots_adjust(bottom=0.13, top=0.75, wspace=0.50, left=0.05, right=0.99)


        today_date = time.strftime("%Y-%m-%d")
        format = 'pdf'
        PIC_NAME = f'comparative_pareto_optimality_plot_1Ksample_scale_progr_transparency_{progressive_transparency}_{Y_METRIC}_{x_metric}_{today_date}.{format}'
        plt.savefig(os.path.join(path_configs["base_path"], "results_pics", TODAYS_FOLDER, PIC_NAME), bbox_inches='tight')
        plt.show()