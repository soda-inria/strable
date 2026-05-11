"""Figure E.3 — Average off-diagonal cosine similarity of raw string embeddings, aggregated
across the 108 STRABLE datasets × 5 seeds."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import json
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from configs.path_configs import path_configs
from figures._main import save_figure
from scripts.datasets_metadata_recap import wide_datasets


SAMPLE_SIZE = 50
N_SEEDS     = 5
MIN_ROWS    = SAMPLE_SIZE
ORDER       = ['LLaMA-3.1-8B', 'Qwen3-8B', 'Tf-Idf', 'MiniLM-L6-v2']
LLM_MODELS  = {
    'MiniLM-L6-v2': 'llm-all-MiniLM-L6-v2',
    'LLaMA-3.1-8B': 'llm-llama-3.1-8b',
    'Qwen3-8B':     'llm-qwen3-8b',
}


def _clean_embs(embs):
    """Drop any rows with non-finite values."""
    embs = np.asarray(embs, dtype=np.float64)
    return embs[np.isfinite(embs).all(axis=1)]


def _avg_offdiag_cosine(embs, sample_size, rng):
    """Mean off-diagonal cosine similarity in a random ``sample_size`` rows."""
    n = embs.shape[0]
    if n < 2:
        return np.nan
    k = min(sample_size, n)
    idx = rng.choice(n, size=k, replace=False)
    sim = cosine_similarity(embs[idx])
    mask = ~np.eye(k, dtype=bool)
    return sim[mask].mean()


def _compute_tabvec(df_name):
    """Skrub StringEncoder + SquashingScaler embedding for one dataset."""
    from skrub import StringEncoder, TableVectorizer, SquashingScaler

    data_path = f"{path_configs['path_data_processed']}/{df_name}/data.parquet"
    cfg_path  = f"{path_configs['path_data_processed']}/{df_name}/config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    X = pd.read_parquet(data_path).drop(columns=[cfg['target_name']], errors='ignore')

    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X = cleaner.fit_transform(X)
    encoder = TableVectorizer(
        high_cardinality=StringEncoder(random_state=1234),
        numeric=SquashingScaler(),
    )
    out = encoder.fit_transform(X)
    if hasattr(out, "toarray"):
        return out.toarray()
    return out.values if hasattr(out, 'values') else out


def _load_llm(df_name, model_dir):
    """Load the cached LLM embedding parquet and return only the X1..XN columns."""
    path = f"{path_configs['llm_embeddings']}/{model_dir}/{model_dir}|{df_name}.parquet"
    df = pd.read_parquet(path)
    emb_cols = sorted(
        [c for c in df.columns if re.match(r'^X\d+$', c)],
        key=lambda x: int(x[1:]),
    )
    return df[emb_cols].values


def _gather_cosine_data():
    """Iterate every (dataset, encoder, seed) and collect the cosine similarity.
    Returns a long-format DataFrame keyed by ``(dataset, encoder, seed)``.
    """
    rows = []
    encoder_names = list(LLM_MODELS.keys()) + ['Tf-Idf']
    for df_name in tqdm(wide_datasets, desc='Datasets', position=0):
        embeddings = {}
        for name, model_dir in LLM_MODELS.items():
            try:
                embeddings[name] = _load_llm(df_name, model_dir)
            except Exception as e:
                tqdm.write(f"[{df_name}] skip {name}: {e}")
        try:
            embeddings['Tf-Idf'] = _compute_tabvec(df_name)
        except Exception as e:
            tqdm.write(f"[{df_name}] skip Tf-Idf: {e}")

        for name in encoder_names:
            if name not in embeddings:
                continue
            embs = _clean_embs(embeddings[name])
            if embs.shape[0] < MIN_ROWS:
                tqdm.write(f"[{df_name}] {name}: only {embs.shape[0]} clean rows, skipping")
                continue
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                rows.append({
                    'dataset': df_name,
                    'encoder': name,
                    'seed':    seed,
                    'n_clean': embs.shape[0],
                    'avg_cos': _avg_offdiag_cosine(embs, SAMPLE_SIZE, rng),
                })
    return pd.DataFrame(rows).dropna(subset=['avg_cos'])


def plot_cosine_similarity_before_pca():
    today = time.strftime("%Y-%m-%d")
    cache_path = f"{path_configs['base_path']}/avg_cosine_sim_aggregated_before_PCA_{today}.csv"

    df_results = _gather_cosine_data()
    df_results.to_csv(cache_path, index=False)
    print(f"✅ Wrote {cache_path}")

    summary = (
        df_results
        .groupby('encoder')['avg_cos']
        .agg(['mean', 'std', 'count'])
        .reindex(ORDER)
    )
    print(summary)

    fig, ax = plt.subplots(figsize=(5, 3))
    means = summary.loc[ORDER, 'mean'].values
    stds  = summary.loc[ORDER, 'std'].values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ORDER)))

    bars = ax.bar(
        ORDER, means, yerr=stds, capsize=5,
        color=colors, alpha=0.85,
        edgecolor='black', linewidth=0.5,
    )
    for b, m in zip(bars, means):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f'{m:.3f}',
            ha='center', fontsize=12, fontweight='bold',
        )

    ax.set_ylabel('Avg Off-diagonal Cosine\nSimilarity', fontsize=11)
    ax.set_ylim(0.2, 0.65)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()

    save_figure(fig, "avg_cosine_sim_aggregated_before_PCA")
    plt.close(fig)


if __name__ == "__main__":
    plot_cosine_similarity_before_pca()
