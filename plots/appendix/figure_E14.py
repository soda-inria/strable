"""
Appendix Figure E.14 – Singular values per language-model embedding.

This script:
- Computes singular values (top 30) for selected LLM embeddings across all wide datasets
- Computes singular values for the StringEncoder (Tf-Idf) baseline
- Writes `singular_values_compiled_with_tfidf.parquet`
- Plots the normalized average singular values per model
  and saves `singular_values_1to30_avg_across_datasets_per_model_<date>.pdf`.
"""

import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import TODAYS_FOLDER, get_encoder_color


# --- 1. DATASET LIST (wide datasets) ---
base_path = Path(path_configs["base_path"])
data_processed_path = base_path / "data" / "data_processed"

data_list_wide: list[str] = []
for dataset_dir in data_processed_path.iterdir():
    if not dataset_dir.is_dir():
        continue
    metadata_path = dataset_dir / "config.json"
    if not metadata_path.exists():
        continue
    with metadata_path.open("r") as f:
        # load as dict via json to avoid dtype issues
        import json as _json

        meta = _json.load(f)
    if meta.get("task_type") == "wide":
        data_list_wide.append(dataset_dir.name)


def prepare_tabvec(X_raw: pd.DataFrame):
    """Generate embeddings using skrub's StringEncoder + TableVectorizer."""
    from skrub import StringEncoder, TableVectorizer, SquashingScaler

    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_cleaned = cleaner.fit_transform(X_raw)

    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(high_cardinality=text_encoder, numeric=num_transformer)

    return encoder.fit_transform(X_cleaned)


# --- 2. CONFIGURATION ---
selected_models_raw = [
    "llm-all-MiniLM-L6-v2",
    "llm-fasttext",
    "llm-e5-small-v2",
    "llm-llama-3.1-8b",
    "llm-qwen3-8b",
    "llm-jasper-token-comp-0.6b",
]

num_singular_values = 30
results_svd: dict[str, list[np.ndarray]] = {model: [] for model in selected_models_raw}


# --- 3. SVD FOR LLM EMBEDDINGS ---
print("Computing singular values across datasets for LLM embeddings...")
emb_base = base_path / "data" / "llm_embeding"

for df_name in data_list_wide:
    for model_name in selected_models_raw:
        print(f"Processing model: {model_name} on dataset: {df_name}...")
        try:
            file_path = emb_base / model_name / f"{model_name}|{df_name}.parquet"
            df = pd.read_parquet(file_path)

            emb_cols = [c for c in df.columns if re.match(r"^X\d+$", c)]
            emb_cols.sort(key=lambda x: int(x[1:]))

            embs = df[emb_cols].values
            embs = embs - np.mean(embs, axis=0)

            _, S, _ = svd(embs, full_matrices=False)

            s_top = np.zeros(num_singular_values)
            k = min(len(S), num_singular_values)
            s_top[:k] = S[:k]

            results_svd[model_name].append(s_top)
        except Exception as e:
            print(f"Skipping {model_name} on {df_name} due to error: {e}")
            continue


svd_records: list[dict] = []
for model_name, datasets_sv in results_svd.items():
    for dataset_idx, sv_array in enumerate(datasets_sv):
        for rank, value in enumerate(sv_array, start=1):
            svd_records.append(
                {
                    "model": model_name,
                    "dataset_id": dataset_idx,
                    "rank": rank,
                    "singular_value": value,
                }
            )

svd_df = pd.DataFrame(svd_records)

results_dir = base_path / "results"
results_dir.mkdir(parents=True, exist_ok=True)

compiled_path = results_dir / "singular_values_compiled.parquet"
svd_df.to_parquet(compiled_path, index=False)
print(f"Singular values saved to {compiled_path}")


# --- 4. SVD FOR STRINGENCODER (Tf-Idf) ---
print("Computing singular values for StringEncoder (Tf-Idf)...")
tfidf_svd_results: list[dict] = []

for dataset_idx, df_name in enumerate(data_list_wide):
    try:
        processed_data_path = (
            base_path / "data" / "data_processed" / df_name / "data.parquet"
        )
        processed_json_path = (
            base_path / "data" / "data_processed" / df_name / "config.json"
        )

        import json

        with processed_json_path.open("r") as f:
            processed_json = json.load(f)

        df_raw = pd.read_parquet(processed_data_path)

        target_name = processed_json.get("target_name")
        if target_name in df_raw.columns:
            df_raw = df_raw.drop(columns=[target_name])

        embs_tfidf = prepare_tabvec(df_raw)
        if hasattr(embs_tfidf, "toarray"):
            embs_tfidf = embs_tfidf.toarray()

        embs_tfidf = embs_tfidf - np.mean(embs_tfidf, axis=0)

        _, S, _ = svd(embs_tfidf, full_matrices=False)

        s_top = np.zeros(num_singular_values)
        k = min(len(S), num_singular_values)
        s_top[:k] = S[:k]

        for rank, value in enumerate(s_top, start=1):
            tfidf_svd_results.append(
                {
                    "model": "StringEncoder",
                    "dataset_id": dataset_idx,
                    "rank": rank,
                    "singular_value": value,
                }
            )
    except Exception as e:
        print(f"Skipping Tf-Idf on {df_name} due to error: {e}")
        continue


svd_df_existing = pd.read_parquet(compiled_path)
svd_df_tfidf = pd.DataFrame(tfidf_svd_results)
svd_df_final = pd.concat([svd_df_existing, svd_df_tfidf], ignore_index=True)

compiled_with_tfidf_path = results_dir / "singular_values_compiled_with_tfidf.parquet"
svd_df_final.to_parquet(compiled_with_tfidf_path, index=False)
print(f"Final singular values (including Tf-Idf) saved to {compiled_with_tfidf_path}")


# --- 5. PLOT SINGULAR VALUES (APPENDIX FIGURE E.14) ---
name_mapping = {
    "llm-all-MiniLM-L6-v2": "LM All-MiniLM-L6-v2",
    "llm-fasttext": "LM FastText",
    "llm-e5-small-v2": "LM E5-small-v2",
    "llm-llama-3.1-8b": "LM LLaMA-3.1-8B",
    "llm-qwen3-8b": "LM Qwen-3-8B",
    "llm-jasper-token-comp-0.6b": "LM Jasper-0.6B",
    "StringEncoder": "Tf-Idf",
}

svd_df_final["display_name"] = svd_df_final["model"].map(name_mapping)

avg_svd_per_model_rank = (
    svd_df_final.groupby(["display_name", "rank"], as_index=False)["singular_value"]
    .mean()
)
avg_svd_per_model_rank_pivot = avg_svd_per_model_rank.pivot_table(
    index="rank", columns="display_name", values="singular_value"
)

avg_svd_per_model_rank_pivot = (
    avg_svd_per_model_rank_pivot / avg_svd_per_model_rank_pivot.max()
)

plt.figure(figsize=(5, 4))

for model_name in avg_svd_per_model_rank_pivot.columns:
    series = avg_svd_per_model_rank_pivot[model_name]

    if model_name == "Tf-Idf":
        line_color = "black"
        linestyle = "--"
        marker = "x"
        alpha = 1.0
    else:
        line_color = get_encoder_color(model_name)
        linestyle = "-"
        marker = "o"
        alpha = 0.8

    plt.plot(
        series.index,
        series.values,
        marker=marker,
        label=model_name,
        color=line_color,
        linestyle=linestyle,
        linewidth=2,
        markersize=5,
        alpha=alpha,
    )

plt.xlabel("Singular Value Number (Rank)", fontsize=14)
plt.ylabel(r"Normalized Singular Value $(SV_i / SV_{\max})$", fontsize=14)

max_rank = int(avg_svd_per_model_rank_pivot.index.max())
plt.xticks(range(0, max_rank + 2, 2), fontsize=10)

plt.legend(fontsize=8, bbox_to_anchor=(0.55, 1), loc="upper left")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
fmt = "pdf"
PIC_NAME = f"singular_values_1to30_avg_across_datasets_per_model_{today_date}.{fmt}"
out_path = os.path.join(
    path_configs["base_path"], "results_pics", TODAYS_FOLDER, PIC_NAME
)
plt.savefig(out_path, bbox_inches="tight")
plt.show()

"""
Appendix Figure E.14 – Singular values per language-model embedding.

This script:
- Computes singular values (top 30) for selected LLM embeddings across all wide datasets
- Computes singular values for the StringEncoder (Tf-Idf) baseline
- Writes `singular_values_compiled_with_tfidf.parquet`
- Plots the normalized average singular values per model
  and saves `singular_values_1to30_avg_across_datasets_per_model_<date>.pdf`.
"""

import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd

from strable.configs.path_configs import path_configs
from strable.plots.plot_setup import TODAYS_FOLDER, get_encoder_color


# --- 1. DATASET LIST (wide datasets) ---
base_path = Path(path_configs["base_path"])
data_processed_path = base_path / "data" / "data_processed"

data_list_wide: list[str] = []
for dataset_dir in data_processed_path.iterdir():
    if not dataset_dir.is_dir():
        continue
    metadata_path = dataset_dir / "config.json"
    if not metadata_path.exists():
        continue
    with metadata_path.open("r") as f:
        meta = pd.read_json(f, typ="series")
    if meta.get("task_type") == "wide":
        data_list_wide.append(dataset_dir.name)


def prepare_tabvec(X_raw: pd.DataFrame):
    """Generate embeddings using skrub's StringEncoder + TableVectorizer."""
    from skrub import StringEncoder, TableVectorizer, SquashingScaler

    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_cleaned = cleaner.fit_transform(X_raw)

    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(high_cardinality=text_encoder, numeric=num_transformer)

    return encoder.fit_transform(X_cleaned)


# --- 2. CONFIGURATION ---
selected_models_raw = [
    "llm-all-MiniLM-L6-v2",
    "llm-fasttext",
    "llm-e5-small-v2",
    "llm-llama-3.1-8b",
    "llm-qwen3-8b",
    "llm-jasper-token-comp-0.6b",
]

num_singular_values = 30
results_svd: dict[str, list[np.ndarray]] = {model: [] for model in selected_models_raw}


# --- 3. SVD FOR LLM EMBEDDINGS ---
print("Computing singular values across datasets for LLM embeddings...")
emb_base = base_path / "data" / "llm_embeding"

for df_name in data_list_wide:
    for model_name in selected_models_raw:
        print(f"Processing model: {model_name} on dataset: {df_name}...")
        try:
            file_path = (
                emb_base
                / model_name
                / f"{model_name}|{df_name}.parquet"
            )
            df = pd.read_parquet(file_path)

            emb_cols = [c for c in df.columns if re.match(r"^X\d+$", c)]
            emb_cols.sort(key=lambda x: int(x[1:]))

            embs = df[emb_cols].values
            embs = embs - np.mean(embs, axis=0)

            _, S, _ = svd(embs, full_matrices=False)

            s_top = np.zeros(num_singular_values)
            k = min(len(S), num_singular_values)
            s_top[:k] = S[:k]

            results_svd[model_name].append(s_top)
        except Exception as e:
            print(f"Skipping {model_name} on {df_name} due to error: {e}")
            continue


svd_records: list[dict] = []
for model_name, datasets_sv in results_svd.items():
    for dataset_idx, sv_array in enumerate(datasets_sv):
        for rank, value in enumerate(sv_array, start=1):
            svd_records.append(
                {
                    "model": model_name,
                    "dataset_id": dataset_idx,
                    "rank": rank,
                    "singular_value": value,
                }
            )

svd_df = pd.DataFrame(svd_records)

results_dir = base_path / "results"
results_dir.mkdir(parents=True, exist_ok=True)

compiled_path = results_dir / "singular_values_compiled.parquet"
svd_df.to_parquet(compiled_path, index=False)
print(f"Singular values saved to {compiled_path}")


# --- 4. SVD FOR STRINGENCODER (Tf-Idf) ---
print("Computing singular values for StringEncoder (Tf-Idf)...")
tfidf_svd_results: list[dict] = []

for dataset_idx, df_name in enumerate(data_list_wide):
    try:
        processed_data_path = (
            base_path / "data" / "data_processed" / df_name / "data.parquet"
        )
        processed_json_path = (
            base_path / "data" / "data_processed" / df_name / "config.json"
        )

        import json

        with processed_json_path.open("r") as f:
            processed_json = json.load(f)

        df_raw = pd.read_parquet(processed_data_path)

        target_name = processed_json.get("target_name")
        if target_name in df_raw.columns:
            df_raw = df_raw.drop(columns=[target_name])

        embs_tfidf = prepare_tabvec(df_raw)
        if hasattr(embs_tfidf, "toarray"):
            embs_tfidf = embs_tfidf.toarray()

        embs_tfidf = embs_tfidf - np.mean(embs_tfidf, axis=0)

        _, S, _ = svd(embs_tfidf, full_matrices=False)

        s_top = np.zeros(num_singular_values)
        k = min(len(S), num_singular_values)
        s_top[:k] = S[:k]

        for rank, value in enumerate(s_top, start=1):
            tfidf_svd_results.append(
                {
                    "model": "StringEncoder",
                    "dataset_id": dataset_idx,
                    "rank": rank,
                    "singular_value": value,
                }
            )
    except Exception as e:
        print(f"Skipping Tf-Idf on {df_name} due to error: {e}")
        continue


svd_df_existing = pd.read_parquet(compiled_path)
svd_df_tfidf = pd.DataFrame(tfidf_svd_results)
svd_df_final = pd.concat([svd_df_existing, svd_df_tfidf], ignore_index=True)

compiled_with_tfidf_path = results_dir / "singular_values_compiled_with_tfidf.parquet"
svd_df_final.to_parquet(compiled_with_tfidf_path, index=False)
print(f"Final singular values (including Tf-Idf) saved to {compiled_with_tfidf_path}")


# --- 5. PLOT SINGULAR VALUES (APPENDIX FIGURE E.14) ---
name_mapping = {
    "llm-all-MiniLM-L6-v2": "LM All-MiniLM-L6-v2",
    "llm-fasttext": "LM FastText",
    "llm-e5-small-v2": "LM E5-small-v2",
    "llm-llama-3.1-8b": "LM LLaMA-3.1-8B",
    "llm-qwen3-8b": "LM Qwen-3-8B",
    "llm-jasper-token-comp-0.6b": "LM Jasper-0.6B",
    "StringEncoder": "Tf-Idf",
}

svd_df_final["display_name"] = svd_df_final["model"].map(name_mapping)

avg_svd_per_model_rank = (
    svd_df_final.groupby(["display_name", "rank"], as_index=False)["singular_value"]
    .mean()
)
avg_svd_per_model_rank_pivot = avg_svd_per_model_rank.pivot_table(
    index="rank", columns="display_name", values="singular_value"
)

avg_svd_per_model_rank_pivot = (
    avg_svd_per_model_rank_pivot / avg_svd_per_model_rank_pivot.max()
)

plt.figure(figsize=(5, 4))

for model_name in avg_svd_per_model_rank_pivot.columns:
    series = avg_svd_per_model_rank_pivot[model_name]

    if model_name == "Tf-Idf":
        line_color = "black"
        linestyle = "--"
        marker = "x"
        alpha = 1.0
    else:
        line_color = get_encoder_color(model_name)
        linestyle = "-"
        marker = "o"
        alpha = 0.8

    plt.plot(
        series.index,
        series.values,
        marker=marker,
        label=model_name,
        color=line_color,
        linestyle=linestyle,
        linewidth=2,
        markersize=5,
        alpha=alpha,
    )

plt.xlabel("Singular Value Number (Rank)", fontsize=14)
plt.ylabel(r"Normalized Singular Value $(SV_i / SV_{\max})$", fontsize=14)

max_rank = int(avg_svd_per_model_rank_pivot.index.max())
plt.xticks(range(0, max_rank + 2, 2), fontsize=10)

plt.legend(fontsize=8, bbox_to_anchor=(0.55, 1), loc="upper left")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()

today_date = time.strftime("%Y-%m-%d")
fmt = "pdf"
PIC_NAME = f"singular_values_1to30_avg_across_datasets_per_model_{today_date}.{fmt}"
out_path = os.path.join(
    path_configs["base_path"], "results_pics", TODAYS_FOLDER, PIC_NAME
)
plt.savefig(out_path, bbox_inches="tight")
plt.show()

