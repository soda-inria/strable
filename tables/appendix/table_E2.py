'''
GLM
'''

import os
import time

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    bin_feature_33_66,
    results,
    selected_encoders,
)

# --------------------------------------------------------------
# 1. HELPER: Custom LaTeX Table Generator
# --------------------------------------------------------------
def save_custom_latex(model, algo_name, save_path):
    """
    Manually writes a LaTeX table ensuring:
    - Clean row names (removed "Upper 33%" from rows)
    - Detailed Caption explaining the High/Low comparison
    - Bold p-values < 0.05 with stars (Intercept excluded)
    """
    summary = model.summary()
    results_as_html = summary.tables[1].as_html()
    df_results = pd.read_html(results_as_html, header=0, index_col=0)[0]
    
    # Rename Index: Cleaner names, explanation moved to caption
    name_map = {
        'Intercept': 'Baseline Score (Low Bins)',
        'C(Card_Bin)[T.High]': 'High Cardinality',
        'C(Str_Bin)[T.High]': 'High String Length',
        'C(n_col_Bin)[T.High]': 'High Num Columns',
        'C(n_row_Bin)[T.High]': 'High Num Rows',
        'C(string_diversity_Bin)[T.High]': 'High String Diversity'
    }
    
    df_results.index = [name_map.get(idx, idx) for idx in df_results.index]
    
    # Start LaTeX String
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\begin{tabular}{lcccccc}\n\\toprule\n"
    latex_str += " & \\textbf{coef} & \\textbf{std err} & \\textbf{z} & \\textbf{P$> |$z$|$} & \\textbf{[0.025} & \\textbf{0.975]} \\\\\n\\midrule\n"
    
    for idx, row in df_results.iterrows():
        # Format metrics
        coef = f"{row['coef']:.4f}"
        stderr = f"{row['std err']:.4f}"
        z_val = f"{row['z']:.3f}"
        ci_lower = f"{row['[0.025']:.3f}"
        ci_upper = f"{row['0.975]']:.3f}"
        
        # P-value formatting
        p_val_raw = row['P>|z|']
        
        # Never bold the Intercept/Baseline
        if "Baseline" in idx:
            p_val_str = f"{p_val_raw:.3f}" 
        elif p_val_raw < 0.05:
            p_val_str = f"\\textbf{{{p_val_raw:.3f}*}}" 
        else:
            p_val_str = f"{p_val_raw:.3f}"
            
        row_name_clean = idx.replace("_", "\\_").replace("%", "\\%")
        
        latex_str += f"{row_name_clean} & {coef} & {stderr} & {z_val} & {p_val_str} & {ci_lower} & {ci_upper} \\\\\n"
        
    latex_str += "\\bottomrule\n\\end{tabular}\n"
    
    # --- UPDATED CAPTION ---
    algo_clean = algo_name.replace("_", "\\_").replace("+", " + ")
    latex_str += f"\\caption{{GLM Analysis for: {algo_clean} (High=Upper 33 percentile vs Low=Lower 33 percentile)}}\n"
    latex_str += "\\label{tab:glm_" + algo_name.replace(" ", "_").replace("+", "") + "}\n"
    latex_str += "\\end{table}"
    
    with open(save_path, 'w') as f:
        f.write(latex_str)

# --------------------------------------------------------------
# 2. Prepare Data & Strict Filtering
# --------------------------------------------------------------
# Filter for Num+Str and target configs
df_analysis = results[
    (results['dtype'] == 'Num+Str') & 
    (results['encoder'].isin(selected_encoders))
].copy()

# 1. Apply Binning
cols_to_bin = {
    'Card_Bin': 'avg_cardinality',
    'Str_Bin': 'avg_string_length_per_cell',
    'n_col_Bin': 'num_columns',
    'n_row_Bin': 'num_rows',
    'string_diversity_Bin': 'string_diversity'
}

for bin_col, feat_col in cols_to_bin.items():
    df_analysis[bin_col] = bin_feature_33_66(df_analysis, feat_col)

# 2. STRICT FILTERING: High vs Low ONLY
print(f"Original Row Count: {len(df_analysis)}")

for bin_col in cols_to_bin.keys():
    df_analysis[bin_col] = df_analysis[bin_col].astype(str)
    df_analysis = df_analysis[df_analysis[bin_col].isin(['Low', 'High'])]
    df_analysis[bin_col] = pd.Categorical(
        df_analysis[bin_col], 
        categories=['Low', 'High'], 
        ordered=True
    )

print(f"Filtered Row Count (Intersection): {len(df_analysis)}")

# --------------------------------------------------------------
# 3. Loop: Multivariate GLM per Encoder-Learner
# --------------------------------------------------------------
print("-" * 110)
print(f"{'Algorithm':<40} | {'Feature':<35} | {'Coeff':<8} | {'P-val':<8}")
print("-" * 110)

target_configs = df_analysis['encoder_learner'].drop_duplicates().to_list()

for algo in target_configs:
    df_algo = df_analysis[df_analysis['encoder_learner'] == algo].copy()
    
    if len(df_algo) < 10: 
        print(f"Skipping {algo} (Insufficient data: {len(df_algo)} rows)")
        continue

    formula = (
        "score ~ C(Card_Bin) + C(Str_Bin) + C(n_col_Bin) + C(n_row_Bin) + C(string_diversity_Bin)"
    )
    
    try:
        model = smf.glm(
            formula=formula, 
            data=df_algo, 
            family=sm.families.Gamma(link=sm.families.links.Log())
        ).fit()
        
        # --- Console Output ---
        p_values = model.pvalues
        params = model.params
        for feat in params.index:
            if feat == "Intercept": continue
            if p_values[feat] < 0.05:
                clean_feat = feat.split("[T.")[-1].replace("]", "")
                print(f"{algo:<40} | {clean_feat:<35} | {params[feat]:.4f}   | {p_values[feat]:.4f} *")
                
        # --- Save ---
        today_date = time.strftime("%Y-%m-%d")
        clean_name = algo.replace(" + ", "_").replace(" ", "")
        filename = f"glm_multivariate_{clean_name}_{today_date}.tex"
        
        save_dir = os.path.join(
            path_configs["base_path"],
            "results_tables",
            TODAYS_FOLDER,
        )
        
        save_path = os.path.join(save_dir, filename)
        save_custom_latex(model, algo, save_path)
            
    except Exception as e:
        print(f"Error fitting model for {algo}: {e}")

print("-" * 110)
print("Processing complete. Tables generated with updated captions and row names.")
