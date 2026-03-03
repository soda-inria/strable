"""
Figure 7 – Conditioning number per language-model embedding vs TabPFN-2.5 performance.

Relies on the precomputed `singular_values_compiled_with_tfidf.parquet`
created by `figure_E14.py`.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import TODAYS_FOLDER, results


# --- 1. NAME MAPPING & SELECTION ---
name_mapping = {
    "llm-all-MiniLM-L6-v2": "LM All-MiniLM-L6-v2",
    "llm-fasttext": "LM FastText",
    "llm-e5-small-v2": "LM E5-small-v2",
    "llm-llama-3.1-8b": "LM LLaMA-3.1-8B",
    "llm-qwen3-8b": "LM Qwen-3-8B",
    "llm-jasper-token-comp-0.6b": "LM Jasper-0.6B",
    "StringEncoder": "Tf-Idf",
}

selected_models = list(name_mapping.values())


# --- 2. LOAD PRECOMPUTED SINGULAR VALUES ---
svd_path = os.path.join(
    path_configs["base_path"],
    "results",
    "singular_values_compiled_with_tfidf.parquet",
)
svd_df = pd.read_parquet(svd_path)
svd_df["display_name"] = svd_df["model"].map(name_mapping)


# --- 3. TABPFN PERFORMANCE (Y-AXIS) ---
perf_df = results[
    (results["dtype"] == "Num+Str")
    & (results["encoder"].isin(selected_models))
    & (results["learner"] == "TabPFN-2.5")
]

model_performance = perf_df.groupby("encoder")["score"].mean()


# --- 4. CONDITION NUMBERS (X-AXIS) ---
def get_log_cond(group: pd.DataFrame) -> float:
    s_max = group.loc[group["rank"] == 1, "singular_value"].values[0]
    s_min = group.loc[group["rank"] == 30, "singular_value"].values[0]
    return np.log(s_max / s_min) if s_min > 0 else np.nan


cond_per_dataset = svd_df.groupby(["display_name", "dataset_id"]).apply(get_log_cond)
model_cond_numbers = cond_per_dataset.groupby("display_name").mean().to_dict()


# --- 5. SCATTER PLOT ---
plt.figure(figsize=(5, 4))

plot_x: list[float] = []
plot_y: list[float] = []
labels: list[str] = []

for model in selected_models:
    if model in model_cond_numbers and model in model_performance:
        plot_x.append(model_cond_numbers[model])
        plot_y.append(model_performance[model])
        labels.append(model)

plt.scatter(plot_x, plot_y, s=150, c="royalblue", alpha=0.7, edgecolors="black")

for i, txt in enumerate(labels):
    if txt == "LM All-MiniLM-L6-v2":
        xy = (plot_x[i] - 0.03, plot_y[i] - 0.02)
    elif txt == "LM E5-small-v2":
        xy = (plot_x[i] - 0.04, plot_y[i])
    elif txt == "Tf-Idf":
        xy = (plot_x[i], plot_y[i] - 0.01)
    elif txt == "LM FastText":
        xy = (plot_x[i], plot_y[i] - 0.015)
    elif txt == "LM LLaMA-3.1-8B":
        xy = (plot_x[i] - 0.5, plot_y[i])
    elif txt == "LM Qwen-3-8B":
        xy = (plot_x[i] - 0.3, plot_y[i])
    else:
        xy = (plot_x[i], plot_y[i])

    plt.annotate(
        txt,
        xy,
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
    )

plt.xlabel(
    r"Conditioning Number $= \log\left(\frac{SV_1}{SV_{30}}\right)$",
    fontsize=14,
)
plt.ylabel("Average TabPFN-2.5 Score", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)

if len(plot_x) > 1:
    r, _ = pearsonr(plot_x, plot_y)
    z = np.polyfit(plot_x, plot_y, 1)
    p = np.poly1d(z)
    plt.plot(
        plot_x,
        p(plot_x),
        "r--",
        alpha=0.5,
        linewidth=2,
        label=f"Linear Regression Fit\n($Pearson={r:.2f}$)",
    )
    plt.legend(loc=(0.35, 0.50), fontsize=12)

plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
fmt = "pdf"
PIC_NAME = f"conditioning_number_tabpfn_selected_llms_1to30_{today_date}.{fmt}"
out_path = os.path.join(
    path_configs["base_path"],
    "results_pics",
    TODAYS_FOLDER,
    PIC_NAME,
)
plt.savefig(out_path, bbox_inches="tight")
plt.show()