"""Centralized path configuration.

All paths are derived from a single ``STRABLE_ROOT`` directory so that the
repository works without any modification on a user's machine.

Resolution order for ``STRABLE_ROOT``:

1. ``$STRABLE_ROOT`` environment variable, if set.
2. The repository root, inferred from this file's location
   (``configs/path_configs.py`` -> repo root is ``..``).

The HuggingFace cache follows the standard ``HF_HOME`` env var (or its
default of ``~/.cache/huggingface``); it is intentionally outside the
repo.
"""

import os
from pathlib import Path

# Repo root = parent of the configs/ directory containing this file.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

REPO_ROOT = Path(os.environ.get("STRABLE_ROOT", str(_DEFAULT_ROOT)))

path_configs = dict()

# --- Top-level roots ----------------------------------------------------
path_configs["base_path"] = str(REPO_ROOT)

# --- Data ---------------------------------------------------------------
path_configs["path_data_processed"] = str(REPO_ROOT / "data" / "data_processed")
path_configs["path_data_raw"] = str(REPO_ROOT / "data" / "data_raw")
path_configs["llm_embeddings"] = str(REPO_ROOT / "data" / "llm_embeding")

# Sibling benchmarks for cross-corpus comparisons (see scripts/
# natural_language_test_VSE_vs_STRABLE_vs_CARTE_vs_TTB.py).
path_configs["vse_datasets"] = str(REPO_ROOT / "data" / "VSE_datasets")
path_configs["carte_datasets"] = str(REPO_ROOT / "data" / "CARTE_datasets")
path_configs["ttb_datasets"] = str(REPO_ROOT / "data" / "TTB_datasets")

# --- Results / outputs --------------------------------------------------
path_configs["results"] = str(REPO_ROOT / "results")
path_configs["compiled_results"] = str(REPO_ROOT / "results" / "compiled_results")
path_configs["results_pics"] = str(REPO_ROOT / "results_pics")
path_configs["results_tables"] = str(REPO_ROOT / "results_tables")

# --- Caches & summaries --------------------------------------------------
path_configs["folder_cache_emb"] = str(REPO_ROOT / "__cache__")
path_configs["dataset_summary_wide"] = str(REPO_ROOT / "dataset_summary_wide.parquet")
path_configs["openml_features"] = str(REPO_ROOT / "openml_features.csv")

# --- NLTK data ----------------------------------------------------------
path_configs["nltk_data"] = str(REPO_ROOT / "nltk_data")

# --- Models -------------------------------------------------------------
path_configs["models"] = str(REPO_ROOT / "data" / "models")
path_configs["fasttext_path"] = str(
    REPO_ROOT / "data" / "language_models" / "cc.en.300.bin"
)

# HuggingFace cache — deliberately outside the repo; falls back to HF default.
path_configs["huggingface_cache_folder"] = os.environ.get(
    "HF_HOME",
    str(Path.home() / ".cache" / "huggingface"),
)
