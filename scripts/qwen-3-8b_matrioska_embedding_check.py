"""Per-dimension variance concentration of cached LLM embeddings.

For every (model, dataset) pair, compute the per-dimension variance, then the
Gini coefficient over those variances. Decoder-only models concentrate
variance into a few dimensions (high Gini); encoder-only models spread it
more evenly (low Gini). Produces:

* a per-model summary of median / mean Gini across the wide-dataset corpus
  -- the paper's ``tab:gini_appendix``
* pairwise Wilcoxon signed-rank tests across the six models with
  Holm-corrected p-values -- the paper's ``tab:gini_pairwise``

Supports the §4.1 claim that decoder embeddings benefit from standard
scaling or direct slicing because their variance is dominated by a few
high-variance dimensions.
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from configs.path_configs import path_configs
from scripts.datasets_metadata_recap import wide_datasets


# (display_name, embedding-cache prefix) — the six models the paper compares.
# The embedding directory for each is path_configs['llm_embeddings']/<prefix>/.
MODELS = [
    ("MiniLM-L6-v2", "llm-all-MiniLM-L6-v2"),
    ("E5-base-v2",   "llm-e5-base-v2"),
    ("BGE-large",    "llm-bge-large"),
    ("LLaMA-3.1-8B", "llm-llama-3.1-8b"),
    ("Qwen3-8B",     "llm-qwen3-8b"),
    ("OPT-6.7B",     "llm-opt-6.7b"),
]


def gini(x):
    """Gini coefficient of a 1D array of non-negative values.
    Returns 0.0 for an all-zero array.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * x.sum()) / (n * x.sum())


def variance_gini(dataset_name, embedding_dir, model_prefix):
    """Gini coefficient of the per-dimension variance of one model's
    embeddings on one dataset. Returns ``None`` if the parquet is missing."""
    file_path = embedding_dir / f"{model_prefix}|{dataset_name}.parquet"
    if not file_path.exists():
        return None
    df = pd.read_parquet(file_path)
    embeddings = df.select_dtypes(include=[np.number]).values
    var_per_dim = np.var(embeddings, axis=0)
    return gini(var_per_dim)


def compute_ginis_for_all_models(datasets, models, base_embeddings_dir):
    """Return ``{display_name: [gini_per_dataset, ...]}`` for every model.
    Missing parquets are stored as ``None`` to keep the list aligned with
    ``datasets`` and to preserve paired structure across models.
    """
    base = Path(base_embeddings_dir)
    out = {}
    for display_name, prefix in models:
        embedding_dir = base / prefix
        out[display_name] = Parallel(n_jobs=-1)(
            delayed(variance_gini)(ds, embedding_dir, prefix) for ds in datasets
        )
    return out


def summarize_ginis(gini_results):
    """Per-model median / mean Gini across datasets — the paper's
    ``tab:gini_appendix`` content."""
    rows = []
    for model, values in gini_results.items():
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        rows.append({
            "Model": model,
            "Median Gini": float(np.median(valid)) if valid.size else float("nan"),
            "Mean Gini":   float(np.mean(valid))   if valid.size else float("nan"),
            "N datasets":  int(valid.size),
        })
    return pd.DataFrame(rows)


def _clean_pair(a, b):
    """Drop dataset positions where either side is missing/NaN."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def pairwise_wilcoxon_gini(gini_results):
    """Pairwise Wilcoxon signed-rank tests on per-dataset Gini values, with
    Holm-corrected p-values across the C(k,2) unique pairs — the paper's
    ``tab:gini_pairwise`` content. Returns a tidy DataFrame of pair, raw p,
    Holm-corrected p, and median paired difference (row - col)."""
    models = list(gini_results.keys())
    n = len(models)

    pair_rows = []
    raw_ps = []
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = models[i], models[j]
            a, b = _clean_pair(gini_results[m1], gini_results[m2])
            diff = a - b
            if a.size == 0 or np.all(diff == 0):
                p = 1.0
            else:
                _, p = wilcoxon(a, b)
            pair_rows.append({
                "Model A": m1,
                "Model B": m2,
                "Median diff (A - B)": float(np.median(diff)) if diff.size else float("nan"),
                "p_raw": p,
            })
            raw_ps.append(p)

    if raw_ps:
        _, holm_corrected, _, _ = multipletests(raw_ps, method="holm")
        for row, p_adj in zip(pair_rows, holm_corrected):
            row["p_holm"] = float(p_adj)

    return pd.DataFrame(pair_rows)


if __name__ == "__main__":
    base_embeddings_dir = Path(path_configs["llm_embeddings"])

    print(f"Computing Gini coefficients for {len(MODELS)} models "
          f"across {len(wide_datasets)} datasets...")
    gini_results = compute_ginis_for_all_models(
        wide_datasets, MODELS, base_embeddings_dir,
    )

    summary = summarize_ginis(gini_results)
    print("\n=== Per-model Gini summary (tab:gini_appendix) ===")
    print(summary.round(4).to_string(index=False))

    pairwise = pairwise_wilcoxon_gini(gini_results)
    print("\n=== Pairwise Wilcoxon, Holm-corrected (tab:gini_pairwise) ===")
    fmt = pairwise.copy()
    fmt["Median diff (A - B)"] = fmt["Median diff (A - B)"].map(lambda x: f"{x:+.4f}")
    fmt["p_raw"]  = fmt["p_raw"].map(lambda x: f"{x:.2e}")
    fmt["p_holm"] = fmt["p_holm"].map(lambda x: f"{x:.2e}")
    print(fmt.to_string(index=False))
