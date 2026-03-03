'''
Average performance when using combined (Num+Str) features.
Encoders, e2e models and selected LLMs

This code can create also figure_E4, figure_E5, figure_E12 in the Appendix.
'''

import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    Y_METRIC_LABELS,
    get_learner_color_simple,
    get_learner_hatch,
    results,
    score_list,
    selected_encoders,
    selected_encoders_top3,
)



dtype = 'Num+Str'
select_llm = 'selected' #'all_llm', 'top3', 'selected'
score = score_list[0]  # ['score', 'score_norm', 'score_norm_clip', 'score_norm_max1', 'score_centred']

for task in ['all_task', 'classification', 'regression']:
    if (score == 'score') & (task == 'all_task'):
        set_xlim_min = 0.56
        set_xlim_max = 0.78
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score_norm') & (task == 'all_task'):
        set_xlim_min = 0.6
        set_xlim_max = 1.0
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score') & (task == 'regression'):
        set_xlim_min = 0.45
        set_xlim_max = 0.72
        bbox_to_anchor=(1.53, 0.0)
    elif (score == 'score_norm') & (task == 'regression'):
        set_xlim_min = 0.75
        set_xlim_max = 1.0
        bbox_to_anchor=(1.33, 0.0)
    elif (score == 'score') & (task == 'classification'):
        set_xlim_min = 0.75
        set_xlim_max = 0.9
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score_norm') & (task == 'classification'):
        set_xlim_min = 0.6
        set_xlim_max = 0.95
        bbox_to_anchor=(1.28, 0.0)
    elif (score == 'score_clip') & (task == 'all_task'):
        set_xlim_min = 0.55
        set_xlim_max = 0.75
        bbox_to_anchor=(1.53, 0.0)
    elif (score == 'score_clip') & (task == 'regression'):
        set_xlim_min = 0.5
        set_xlim_max = 0.7
        bbox_to_anchor=(1.53, 0.0)
    elif (score == 'score_clip') & (task == 'classification'):
        set_xlim_min = 0.6
        set_xlim_max = 0.9
        bbox_to_anchor=(1.53, 0.0)


    if select_llm == 'all_llm':
        df = results.copy()
    elif select_llm == 'top3':
        df = results[results['encoder'].isin(selected_encoders_top3)].copy()
    else:
        df = results[results['encoder'].isin(selected_encoders)].copy()

    
    if task == 'all_task':
        set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($R^2$ & AUC)"
        pass
    elif task == 'regression':
        df = df[df['task'] == 'regression'].copy()
        set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($R^2$)"
    else: 
        set_x_label = f"Avg {Y_METRIC_LABELS[score]} ($AUC$)"
        df = df[df['task'] != 'regression'].copy()

    plot_data = df[(df['dtype'] == dtype)].copy()

    plot_data = plot_data[(plot_data['method'] != 'num-str_tabpfn_tabpfn_default')]

    unique_learners = sorted(df['learner'].unique())

    color_map = {l: get_learner_color_simple(l) for l in unique_learners}
    hatch_map = {l: get_learner_hatch(l) for l in unique_learners}

    learner_sort_order = [
        'TabPFN-2.5', 'XGBoost-tuned', 'XGBoost',  
        'ExtraTrees-tuned', 'ExtraTrees', 'Ridge', 'CatBoost', 'CatBoost-tuned', 
        'ContextTab', 'TabSTAR', 'Tarte'
    ]
    sort_map = {name: i for i, name in enumerate(learner_sort_order)}
    ordered_learners = unique_learners

    #order by max value of tabular learner
    encoder_order = (
        plot_data.groupby(['encoder','learner'], as_index=False)[score].mean()
        .sort_values(by=['encoder',score], ascending=[True,False])
        .groupby('encoder').head(1)[['encoder',score]]
        .sort_values(by=score, ascending=False)
        .set_index(['encoder']).index
    )


    #separate e2e models from the rest
    e2e_keywords = ['CatBoost', 'ContextTab', 'TabSTAR', 'TabPFN', 'Tarte', 'E2E']

    def is_e2e(name):
        return any(k in str(name) for k in e2e_keywords)

    # 3. Filter into two groups, preserving original order
    e2e_models = [enc for enc in encoder_order if is_e2e(enc)]
    other_models = [enc for enc in encoder_order if not is_e2e(enc)]

    # 4. Concatenate: E2E first (Top), then Others
    new_encoder_order = pd.Index(e2e_models + other_models, name='encoder')

    print("--- New Order (E2E on Top) ---")
    print(new_encoder_order)

    # encoder_order = new_encoder_order

    plt.rcParams['font.sans-serif'] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams['font.family'] = "sans-serif" 
    sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
    sns.set_context("paper")

    fixed_bar_height = 3.5   # Standard height
    inter_group_gap = 6    
    intra_group_sep = 0.01

    # shrink_learners = ['ContextTab', 'TabSTAR', 'TabPFN-2.5']

    fig, ax = plt.subplots(figsize=(3, 6))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    current_y = 0
    yticks_locs = []
    prev_separator = 2.0

    # ============================================
    # 3. DYNAMIC ITERATIVE PLOTTING
    # ============================================
    for group_idx, encoder in enumerate(encoder_order):
        
        enc_df = plot_data[plot_data['encoder'] == encoder]

        present_learners = [l for l in ordered_learners if not enc_df[enc_df['learner'] == l].empty]
        
        print(f"Encoder: {encoder} | Present Learners: {present_learners}")
        
        present_learners.sort(key=lambda x: sort_map.get(x, 999))
            
        num_learners = len(present_learners)
        group_top = current_y
        
        for i, learner in enumerate(present_learners):
            learner_data = enc_df[enc_df['learner'] == learner][score]
            mean_score = learner_data.mean()
            
            bar_y = group_top - (i * (fixed_bar_height + intra_group_sep))
            
            visual_height = fixed_bar_height

            c = color_map[learner]
            h = hatch_map[learner]

            # Bars at zorder 3
            ax.barh(bar_y, mean_score, 
                    height=visual_height, 
                    color=c, 
                    edgecolor='white', 
                    hatch=h,
                    linewidth=0.5, 
                    zorder=3
                    )
        
        top_bar_y = group_top
        bottom_bar_y = group_top - ((num_learners - 1) * (fixed_bar_height + intra_group_sep))
        group_center_y = (top_bar_y + bottom_bar_y) / 2
        yticks_locs.append(group_center_y)

        # Define the separator line (midpoint of the gap)
        separator_y = bottom_bar_y - (fixed_bar_height / 2) - (inter_group_gap / 2)
        
        # --- NEW: ALTERNATING BANDS ---
        # We draw a band from the previous separator down to the current separator
        # This wraps the entire block of bars + half the gap above and below
        if group_idx % 2 == 1:
            ax.axhspan(separator_y, prev_separator, color='#e0e0e0', zorder=0, lw=0)

        # Update trackers
        prev_separator = separator_y
        current_y = bottom_bar_y - inter_group_gap - fixed_bar_height

    e2e_ordered_names = [
        'ContextTab', 
        'TabSTAR', 
        'CatBoost', 
        'CatBoost-tuned'
    ]
    modular_ordered_names = [
        'TabPFN-2.5', 
        'XGBoost', 
        'XGBoost-tuned', 
        'ExtraTrees', 
        'ExtraTrees-tuned', 
        'Ridge'
    ]
    e2e_list = [l for l in e2e_ordered_names if l in unique_learners]
    modular_list = [l for l in modular_ordered_names if l in unique_learners]

    ax.set_yticks(yticks_locs)
    encoder_order_renamed = [name if name != 'TabPFN-2.5' else 'E2E TabPFN-2.5' for name in encoder_order]
    ax.set_yticklabels(encoder_order_renamed, fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel(set_x_label, fontsize=18, ha='right', x=1.0)
    ax.set_ylabel("Encoder (Num+Str)", fontsize=18, labelpad=-2, y=0.45)
    ax.set_xlim(set_xlim_min, set_xlim_max)
    ax.set_ylim(prev_separator + (inter_group_gap/4), 5.0) 
    ax.xaxis.grid(True, linestyle='-', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    def create_patch(l):
        return mpatches.Patch(
            facecolor=color_map[l], edgecolor='white', hatch=hatch_map[l], label=l, linewidth=0.5
        )

    handles_e2e = [create_patch(l) for l in e2e_list]
    handles_mod = [create_patch(l) for l in modular_list]

    header_e2e = mpatches.Patch(visible=False, label=r"$\bf{E2E\ Models}$")
    header_mod = mpatches.Patch(visible=False, label=r"$\bf{Modular\ Learners}$")
    header_base = mpatches.Patch(visible=False, label=r"$\bf{Baselines}$")

    # Combine final list
    final_handles = [header_e2e] + handles_e2e + [header_mod] + handles_mod

    legend = ax.legend(
        handles=final_handles,
        fontsize=7.5,
        loc='lower right',
        bbox_to_anchor=bbox_to_anchor, 
        # bbox_to_anchor=(1.2, 0.0), 
        frameon=True,
        framealpha=1, 
        facecolor='white',     # Background color of the box
        edgecolor='black',
        fancybox=False,        # Set to False for sharp corners, True for rounded
        borderpad=0.6,
        handletextpad=0.4,
        labelspacing=0.4
    )

    # Align text to the left so headers look correct
    for text in legend.get_texts():
        text.set_ha('left')

    sns.despine(left=False, bottom=False) # Keep right spine visible

    ax.tick_params(axis='y', zorder=10)

    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'avg_performance_{dtype}_{score}_{select_llm}_{task}_{today_date}.{format}'
    plt.savefig(os.path.join(path_configs["base_path"], "results_pics", TODAYS_FOLDER, PIC_NAME), bbox_inches='tight')
    plt.show()