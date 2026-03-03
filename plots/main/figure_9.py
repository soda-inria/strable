'''
KENDALL-TAU Correlation vs. Number of Datasets ($N$): How many datasets are sufficient for 
the benchmark to converge to the oracle ranking?
BOOTSTRAPPED CURVE FITTING (EXTRAPOLATION)
'''

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
from scipy.stats import kendalltau

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import (
    TODAYS_FOLDER,
    calculate_rankings,
    model_ref_1,
    results,
)

results_copy = results[(results['dtype']=='Num+Str') & (results['method'] != 'num-str_tabpfn_tabpfn_default')].copy()

pivot_df = results_copy.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# PAY ATTENTION TO DROP NANS IN THE CORRECT DIMENSION
pivot_df = pivot_df.dropna(axis=1)  

# pivot_df = pivot_df.dropna(axis=0)  

df = pivot_df.copy()

# take only those index that contain Num+Str in any column name
df = df[[col for col in df.columns if 'Num+Str' in col]]

n_datasets = len(df)
print(f"Total valid datasets for analysis: {n_datasets}")

# split pivot_df into 2 subsamples for kendall-tau (of equal size)
mid_point = n_datasets // 2
# sample randomly for df_subsample1, compute size, and sample the same size from rest for df_subsample2
indices = df.index.tolist()

# Parameters
# We start at N=10 and go up to the full dataset count in steps of 5
sample_sizes = range(10, mid_point + 1, 1) 
n_iterations = 2000 
stability_scores = []

# Step B: Bootstrapping Loop
for n in sample_sizes:
    print(f"Running simulations for N={n} datasets...")
    for i in range(n_iterations):
        print(f"  Iteration {i+1}/{n_iterations}...")
        random.shuffle(indices)
        df_subsample1 = df.loc[indices[:mid_point], :]
        df_subsample2 = df.loc[indices[mid_point:mid_point*2], :]

        # print("sizes of subsamples:", df_subsample1.shape, df_subsample2.shape)

        # check that sampled datasets in each subsample are different
        assert len(set(df_subsample1.index).intersection(set(df_subsample2.index))) == 0, "Subsamples overlap!"

        # 1. Subsample N datasets (randomly select N rows)
        subset1 = df_subsample1.sample(n=n, replace=False)
        subset2 = df_subsample2.sample(n=n, replace=False)
        
        # 2. Compute rankings on this subset
        subset_rankings_1 = calculate_rankings(subset1)
        subset_rankings_2 = calculate_rankings(subset2)
        
        # 3. Compare subset ranking vs. true ranking
        # We use kendall-tau correlation to see if the ordering is preserved
        corr, _ = kendalltau(subset_rankings_1, subset_rankings_2)
        
        stability_scores.append({
            'N_datasets': n,
            'Kendalltau_Correlation': corr
        })

df_stability = pd.DataFrame(stability_scores)

# Setup
n_bootstraps = 2000
target_y = 0.95
# Calculate the disagreement percentage based on the Kendall Tau formula: (1 - tau) / 2
disagreement_pct = ((1 - target_y) / 2) * 100
max_plot_x = 3000
x_range_smooth = np.linspace(25, max_plot_x, 300)

# Storage for results
results_kendalltau_extrap = {
    'ref1': {'popt': [], 'curves': [], 'preds': []}
}

print(f"Starting {n_bootstraps} bootstrap iterations...")

for k in range(n_bootstraps):
    boot_sample = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
    df_agg = boot_sample.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
    X, Y = df_agg['N_datasets'], df_agg['Kendalltau_Correlation']
    
    # --- FIT MODEL REF 1 ---
    try:
        # Fixed asymptote at 1.0
        p_r1, _ = curve_fit(model_ref_1, X, Y, p0=[0.5, 0.05], bounds=([0, 0], [10, np.inf]))
        
        # 1. STORE THE PARAMETERS HERE
        results_kendalltau_extrap['ref1']['popt'].append(p_r1)
        
        # Finding N requires numerical solver since N is in sqrt and exp
        
        func = lambda n: model_ref_1(n, *p_r1) - target_y
        req_N = fsolve(func, x0=50)[0]
        if 0 < req_N < 10000:
            results_kendalltau_extrap['ref1']['preds'].append(req_N)
            results_kendalltau_extrap['ref1']['curves'].append(model_ref_1(x_range_smooth, *p_r1))
    except: pass

# Convert the list of arrays into a 2D NumPy array (rows=bootstraps, cols=[a, b])
all_popt = np.array(results_kendalltau_extrap['ref1']['popt'])

# compute the line before 25
x_range_dotted = np.linspace(3, 25, 50)

# 2. Re-generate curves for this specific range using your saved bootstrap parameters
#    (We use all_popt which you created earlier: all_popt = np.array(...))
curves_dotted_list = []
for p in all_popt:
    curves_dotted_list.append(model_ref_1(x_range_dotted, *p))

# 3. Calculate the median curve (consistent with your green line logic)
med_line_dotted = np.median(np.array(curves_dotted_list), axis=0)

# Calculate the optimal (mean) parameters
mean_params = np.mean(all_popt, axis=0)
print(f"Optimal parameter a: {mean_params[0]:.4f}")
print(f"Optimal parameter b: {mean_params[1]:.4f}")

# Calculate 95% Confidence Intervals
ci_lower = np.percentile(all_popt, 2.5, axis=0)
ci_upper = np.percentile(all_popt, 97.5, axis=0)

print(f"95% CI for a: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
print(f"95% CI for b: [{ci_lower[1]:.4f}, {ci_upper[1]:.4f}]")

tau_for_strable_benchmark = 1 - (mean_params[0]/np.sqrt(n_datasets)) * np.exp(-mean_params[1]*n_datasets)
disagreement_pct_strable = ((1 - tau_for_strable_benchmark) / 2) * 100

# --- VISUALIZATION ---
plt.rcParams.update({'font.size': 8}) # Standard academic base size
fig, ax = plt.subplots(figsize=(5, 4))

# 1. Plot Observed Data
df_real_agg = df_stability.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].agg(['median', 'sem'])
ax.errorbar(df_real_agg['N_datasets'], df_real_agg['median'], yerr=df_real_agg['sem'], 
             fmt='o', color='blue', markersize=3, elinewidth=0.8, 
             label='Observed (Median ± SE)', zorder=10)

# 2. Extract and Plot Bootstrap CI Band
if len(results_kendalltau_extrap['ref1']['curves']) > 0:
    curves = np.array(results_kendalltau_extrap['ref1']['curves'])
    med_line = np.median(curves, axis=0)
    # Calculate 95% Confidence Interval
    low_ci = np.percentile(curves, 2.5, axis=0)
    high_ci = np.percentile(curves, 97.5, axis=0)
    
    ax.plot(x_range_dotted, med_line_dotted, color='green', 
            linewidth=1.5, linestyle=':') # Dotted style
    
    ax.plot(x_range_smooth, med_line, color='green', linewidth=1.5, 
            label=r'$1 - \frac{a}{\sqrt{N}} * e^{-bN}$')
    
    x_range_smooth_oracle = np.linspace(3, max_plot_x, 300)
    oracle_curve = 1 - (mean_params[0] / (2 * np.sqrt(x_range_smooth_oracle))) * np.exp(-mean_params[1] * x_range_smooth_oracle)
    
    ax.plot(x_range_smooth_oracle, oracle_curve, color="#A900D3", linestyle='--', linewidth=2, 
            label=r'Oracle Correlation: $1 - \frac{a}{2\sqrt{N}} e^{-bN}$')

# 3. Reference Lines & STRABLE Metrics
ax.axvline(x=n_datasets, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=tau_for_strable_benchmark, color='black', linestyle=':', linewidth=1)

# Shortened annotations for 6x3 readability
ax.text(n_datasets - 35, 0.83, f"STRABLE\nsize: {n_datasets}", 
        color='red', fontweight='bold', fontsize=10)

# Calculate and display disagreement succinctly
annot_text = (f"$\\tau={tau_for_strable_benchmark:.1f}$\n"
              f"Disagreement:{disagreement_pct_strable:.1f}%")
ax.annotate(annot_text, xy=(n_datasets, tau_for_strable_benchmark), 
            xytext=(111, 0.865),
            fontsize=9
            )

ax.text(10, 0.925, "asymptotic agreement to oracle\n(theoretical correction)", 
        color='#A900D3', 
        fontsize=9, 
        fontweight='bold', 
        rotation=10, 
        ha='left', va='bottom')

ax.text(45, 0.88, "asymptotic agreement of\ntwo independent benchmarks", 
        color='green', 
        fontsize=9, 
        fontweight='bold', 
        rotation=10, 
        ha='left', va='bottom')

# --- FINAL POLISH ---
ax.set_xlabel('Number of Datasets (N)', fontsize=12)
ax.set_ylabel('Kendall $\\tau$ correlation\nbetween two benchmarks', fontsize=12)
ax.legend(loc='lower right', fontsize=11, frameon=True)
ax.grid(True, alpha=0.2)
ax.set_ylim(0.69, 1.0)
ax.set_xlim(-5, 170)

today_date = time.strftime("%Y-%m-%d")
fmt = 'pdf'
PIC_NAME = f'benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_greenfunction_{today_date}.{fmt}'
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches='tight')
plt.show()