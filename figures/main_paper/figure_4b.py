"""Figure 4(b) — Convergence of STRABLE benchmark rankings to the oracle."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
from scipy.stats import kendalltau

from figures._main import (
    calculate_rankings,
    load_results,
    model_ref_1,
    save_figure,
)


# ---------------------------------------------------------------------------
# 1. Bootstrapping helpers
# ---------------------------------------------------------------------------

def _build_pivot(results):
    """Score matrix indexed by ``data_name`` × ``method_polished``, restricted
    to Num+Str pipelines that have no NaN columns (a NaN in any column would
    poison the per-dataset ranking)."""
    df = results[
        (results['dtype'] == 'Num+Str')
        & (results['method'] != 'num-str_tabpfn_tabpfn_default')
    ].copy()
    pivot = df.pivot_table(
        index='data_name', columns='method_polished',
        values='score', aggfunc='mean',
    )
    pivot = pivot.dropna(axis=1)
    pivot = pivot[[c for c in pivot.columns if 'Num+Str' in c]]
    return pivot


def _stability_scores(df, sample_sizes, n_iterations, seed=0):
    """For each ``N`` in ``sample_sizes``, repeat: shuffle the dataset list,
    split into two halves, draw a random subset of ``N`` from each half,
    compute Kendall-τ between their rankings.

    Returns a long-format DataFrame with columns ``N_datasets`` and
    ``Kendalltau_Correlation``.
    """
    rng = random.Random(seed)
    n_datasets = len(df)
    mid_point = n_datasets // 2
    indices = df.index.tolist()

    scores = []
    for n in sample_sizes:
        print(f"Running simulations for N={n} datasets...")
        for _ in range(n_iterations):
            rng.shuffle(indices)
            half_a = df.loc[indices[:mid_point], :]
            half_b = df.loc[indices[mid_point:mid_point * 2], :]

            # Sanity check that the two halves are disjoint.
            assert not (set(half_a.index) & set(half_b.index)), "Subsamples overlap!"

            subset_a = half_a.sample(n=n, replace=False, random_state=rng.randint(0, 1 << 31))
            subset_b = half_b.sample(n=n, replace=False, random_state=rng.randint(0, 1 << 31))

            corr, _ = kendalltau(
                calculate_rankings(subset_a),
                calculate_rankings(subset_b),
            )
            scores.append({'N_datasets': n, 'Kendalltau_Correlation': corr})
    return pd.DataFrame(scores)


def _bootstrap_fit(df_stability, n_bootstraps, target_y, x_range):
    """Bootstrap-fit ``model_ref_1`` to the (N, mean Kendall-τ) curve.
    Returns ``(all_popt, curves_at_x_range)`` where ``all_popt`` is shape
    ``(n_successful_fits, 2)`` (parameters ``a`` and ``b``)."""
    all_popt = []
    curves = []
    for _ in range(n_bootstraps):
        boot = df_stability.groupby('N_datasets').sample(frac=1.0, replace=True)
        agg = boot.groupby('N_datasets', as_index=False)['Kendalltau_Correlation'].mean()
        try:
            p_r1, _ = curve_fit(
                model_ref_1, agg['N_datasets'], agg['Kendalltau_Correlation'],
                p0=[0.5, 0.05], bounds=([0, 0], [10, np.inf]),
            )
            req_n = fsolve(lambda n: model_ref_1(n, *p_r1) - target_y, x0=50)[0]
            if 0 < req_n < 10000:
                all_popt.append(p_r1)
                curves.append(model_ref_1(x_range, *p_r1))
        except Exception:
            continue
    return np.array(all_popt), np.array(curves)


# ---------------------------------------------------------------------------
# 2. Main
# ---------------------------------------------------------------------------

# Font sizes ported as-is from the original block.
FS_TICK    = 9
FS_AXIS    = 14
FS_LEGEND  = 10
FS_ANNOT   = 11
FS_STRABLE = 12


def plot_kendalltau_convergence(n_iterations=2000, n_bootstraps=2000, target_y=0.95):
    results = load_results()
    df = _build_pivot(results)
    n_datasets = len(df)
    print(f"Total valid datasets for analysis: {n_datasets}")

    sample_sizes = range(10, n_datasets // 2 + 1)
    df_stability = _stability_scores(df, sample_sizes, n_iterations)

    max_plot_x = 3000
    x_range_smooth = np.linspace(25, max_plot_x, 300)

    print(f"Starting {n_bootstraps} bootstrap iterations...")
    all_popt, curves = _bootstrap_fit(
        df_stability, n_bootstraps, target_y, x_range_smooth,
    )

    # Generate the dotted segment in [3, 25] (the band where we don't have
    # observed Kendall-τ but want to show the fit's behaviour).
    x_range_dotted = np.linspace(3, 25, 50)
    curves_dotted = np.array([model_ref_1(x_range_dotted, *p) for p in all_popt])
    med_line_dotted = np.median(curves_dotted, axis=0)

    mean_params = np.mean(all_popt, axis=0)
    ci_lower = np.percentile(all_popt, 2.5, axis=0)
    ci_upper = np.percentile(all_popt, 97.5, axis=0)
    print(f"Optimal parameter a: {mean_params[0]:.4f}")
    print(f"Optimal parameter b: {mean_params[1]:.4f}")
    print(f"95% CI for a: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
    print(f"95% CI for b: [{ci_lower[1]:.4f}, {ci_upper[1]:.4f}]")

    tau_strable = 1 - (mean_params[0] / np.sqrt(n_datasets)) * np.exp(-mean_params[1] * n_datasets)
    disagreement_pct_strable = ((1 - tau_strable) / 2) * 100

    # --- Figure ---
    plt.rcParams.update({'font.size': 8})
    # ``figsize=(5, 5)`` ported from salts (paper figure is roughly square).
    fig, ax = plt.subplots(figsize=(5, 5))

    df_real_agg = (
        df_stability
        .groupby('N_datasets', as_index=False)['Kendalltau_Correlation']
        .agg(['median', 'sem'])
    )
    ax.errorbar(
        df_real_agg['N_datasets'], df_real_agg['median'], yerr=df_real_agg['sem'],
        fmt='o', color='blue', markersize=3, elinewidth=0.8,
        label='Observed (Median ± SE)', zorder=10,
    )

    if curves.size:
        med_line = np.median(curves, axis=0)
        ax.plot(x_range_dotted, med_line_dotted,
                color='green', linewidth=1.5, linestyle=':')
        ax.plot(x_range_smooth, med_line,
                color='green', linewidth=1.5,
                label=r'$1 - \frac{a}{\sqrt{N}} * e^{-bN}$')

        # Oracle correction (theoretical halving of the bias).
        x_oracle = np.linspace(3, max_plot_x, 300)
        oracle_curve = (
            1 - (mean_params[0] / (2 * np.sqrt(x_oracle)))
            * np.exp(-mean_params[1] * x_oracle)
        )
        ax.plot(
            x_oracle, oracle_curve,
            color="#A900D3", linestyle='--', linewidth=2,
            label=r'Oracle Correlation: $1 - \frac{a}{2\sqrt{N}} e^{-bN}$',
        )

    ax.axvline(x=n_datasets, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=tau_strable, color='black', linestyle=':', linewidth=1)

    ax.text(
        n_datasets - 37, 0.83,
        f"STRABLE\nsize: {n_datasets}",
        color='red', fontweight='bold', fontsize=FS_STRABLE,
    )
    ax.annotate(
        f"$\\tau={tau_strable:.1f}$\nDisagreement:\n{disagreement_pct_strable:.1f}%",
        xy=(n_datasets, tau_strable),
        xytext=(111, 0.85),
        fontsize=FS_ANNOT,
    )
    ax.text(
        10, 0.92,
        "asymptotic agreement to oracle\n(theoretical correction)",
        color='#A900D3',
        fontsize=FS_ANNOT, fontweight='bold',
        rotation=10, ha='left', va='bottom',
    )
    ax.text(
        45, 0.88,
        "asymptotic agreement of\ntwo independent benchmarks",
        color='green',
        fontsize=FS_ANNOT, fontweight='bold',
        rotation=10, ha='left', va='bottom',
    )

    ax.set_xlabel('Number of Datasets (N)', fontsize=FS_AXIS)
    ax.set_ylabel(
        'Kendall $\\tau$ correlation\nbetween two benchmarks',
        fontsize=FS_AXIS,
    )
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='lower right', fontsize=FS_LEGEND, frameon=True)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.69, 1.0)
    ax.set_xlim(-5, 170)

    save_figure(
        fig,
        "benchmark_stability_kendalltau_extrapolated_bootstrap_exponential_saturation_greenfunction",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_kendalltau_convergence()
