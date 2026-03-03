'''
Critical Difference Diagram - selected LLMs, baselines and E2E models

This code can create also figure_E9 and figure_E10 in the Appendix.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import dataset_summary_wide, score_list, clean_method_name, get_learner_color_simple, get_encoder_color
from strable.plots.plot_setup import critical_difference_diagram
import scikit_posthocs as sp

dtype = 'num-str'
score = score_list[0]  
level = 'data_name'  # 'fold_index' or 'data_name'
for task in ['all_task', 'classification', 'regression']:

    # task = 'regression'

    if task == 'classification':
        df_score = results[results['encoder'].isin(selected_encoders)].copy()
        df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)
        df_score = df_score[df_score['task'] != 'regression'].copy()
    elif task == 'regression':
        df_score = results[results['encoder'].isin(selected_encoders)].copy()
        df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)
        df_score = df_score[df_score['task'] == 'regression'].copy()
    else:
        df_score = results[results['encoder'].isin(selected_encoders)].copy()
        df_score = df_score[df_score.method.str.contains(dtype)].reset_index(drop=True)

    # drop TabPFN-2.5 encoder
    df_score = df_score[(df_score['method'] != 'num-str_tabpfn_tabpfn_default')]

    df_score['method'] = df_score['method'].str.replace(f'{dtype}_', '', regex=False)


    # Apply the cleaning logic
    df_score['method'] = df_score['method'].apply(clean_method_name)


    # Ranks and test results
    if level == 'fold_index':
        df_score_final = _generate_marker(df_score)

        avg_rank = (
            df_score_final.groupby(["marker"], group_keys=True)  # marker
            [score].rank(pct=False, ascending=False)
            .groupby(df_score_final.method)
            .mean()
        )
        avg_rank = -1 * avg_rank

        test_results_CF = sp.posthoc_conover_friedman(
            df_score_final,
            melted=True,
            block_col="marker",
            block_id_col="marker",
            group_col="method",
            y_col=score,
        )
    else:
        df_agg = df_score.groupby(["data_name", "method"], as_index=False)[score].mean()
        df_agg["rank"] = df_agg.groupby("data_name")[score].rank(ascending=False)
        avg_rank = df_agg.groupby(['method'])['rank'].mean()

        avg_rank = -1 * avg_rank

        df_agg = df_score.groupby(["data_name", "method"], as_index=False)[score].mean()

        # 2. Pivot to check for missing values (Incomplete Blocks)
        df_pivot = df_agg.pivot(index="data_name", columns="method", values=score)

        # 3. Check if any method is missing for any dataset
        if df_pivot.isnull().values.any():
            print(f"Warning: Dropping {df_pivot.isnull().any(axis=1).sum()} datasets with missing method scores.")
            # Friedman test REQUIRES complete blocks. We must drop datasets where ANY method failed.
            df_pivot = df_pivot.dropna(axis=0)

        df_clean = df_pivot.reset_index().melt(id_vars="data_name", var_name="method", value_name=score)

        # 5. Run the Post-hoc Test
        test_results_CF = sp.posthoc_conover_friedman(
            df_clean,
            melted=True,
            block_col="data_name",
            block_id_col="data_name",
            group_col="method",
            y_col=score
        )

    test_results = test_results_CF.copy()

    test_results = test_results.replace(0, 1e-100)  # Required for visualization


    # Lines
    # line_style = {model: "-" for model in models}
    models = df_score.method.unique()
    line_style = {model: "-" for model in models}
    for model in models:
        if "TargetEncoder" in model:
            line_style[model] = "--"
        if "LLM" in model:
            line_style[model] = "-."

    print(f"Total models: {len(models)}")
    palette_by_learner = {}
    palette_by_encoder = {}

    for model in models:
        # Assuming format is "Encoder - Learner" based on your previous split code
        parts = model.split(' - ')
        encoder_part = parts[0]
        learner_part = parts[-1]
        
        # 1. Learner Palette
        palette_by_learner[model] = get_learner_color_simple(learner_part)
        
        # 2. Encoder Palette
        palette_by_encoder[model] = get_encoder_color(encoder_part)

    name_map = {}
    for model, rank_val in avg_rank.items():
        # rank_val is negative, so we take abs()
        name_map[model] = f"{model} ({abs(rank_val):.1f})"

    # Rename Index/Columns in Data
    avg_rank_plot = avg_rank.rename(index=name_map)
    test_results_plot = test_results.rename(index=name_map, columns=name_map)

    # Update Palettes & Styles to match new names
    palette_by_learner_plot = {name_map[k]: v for k, v in palette_by_learner.items() if k in name_map}
    palette_by_encoder_plot = {name_map[k]: v for k, v in palette_by_encoder.items() if k in name_map}
    line_style_plot = {name_map[k]: v for k, v in line_style.items() if k in name_map}

    # ==========================================
    # COLORED BY LEARNER
    # ==========================================
    sns.set_theme(style="white", font_scale=1)

    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 5))

    critical_difference_diagram(
        ranks=avg_rank_plot,
        sig_matrix=test_results_plot,
        label_fmt_left="{label}",
        label_fmt_right=" {label}",
        label_props={"fontsize": 10},
        crossbar_props={"color": "black", "linewidth": 1},
        marker_props={"marker": ""},
        elbow_props={"linewidth": 1.5},
        text_h_margin=1.2,
        
        color_palette=palette_by_learner_plot,  # <--- APPLIED HERE
        line_style=line_style_plot,
        
        bold_control=True,
        v_space=4,
        ax=ax1,
    )
    n_models = len(models)

    # Create ticks every 5 steps, but ensuring 1 and Max are included
    major_ticks = list(range(0, n_models-4, 5)) 
    if n_models not in major_ticks:
        major_ticks.append(n_models) 

    # Filter 0 (ranks start at 1) and sort
    major_ticks = sorted([t for t in major_ticks if t > 0])

    # Convert to negative space (since CD diagram uses negative ranks)
    plot_ticks = [-t for t in major_ticks]

    # Apply ticks
    ax1.set_xticks(plot_ticks)
    ax1.set_xticklabels(major_ticks, fontsize=12)

    # Force limits to show full range (Left=Max Rank, Right=Rank 1)
    # In negative space: Left = -N, Right = 0
    ax1.set_xlim(-(n_models - 4), 0)


    today_date = time.strftime("%Y-%m-%d")
    format = 'pdf'
    PIC_NAME = f'critical_difference_diagram_selectedLLMs_friedman_colorbylearner_{level}_{score}_{task}_{today_date}.{format}'
    plt.savefig(path_configs["base_path"] + '/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
    plt.show()