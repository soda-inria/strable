"""The STRABLE leaderboard's published results, one row per pipeline, table and outer fold.

``data/leaderboard/results_per_fold.parquet`` ships with the package: the leaderboard's
``site/results_per_fold.parquet`` at the version and commit of gblayer/strable-leaderboard that
``data/leaderboard/manifest.json`` gives. A pinned strable version therefore pins the tables, the folds and
the field of published results together. ``latest=True`` downloads today's leaderboard instead, from
its copy on the Hugging Face Space inria-soda/strable-leaderboard (cached by huggingface_hub).

Columns: ``id`` (the pipeline: ``<encoder>+<learner>``, or the learner alone for an end-to-end model,
with ``:tuned`` for a tuned learner, e.g. ``tfidf+tabpfn-3.5``, ``catboost:tuned``), ``encoder``,
``learner``, ``variant`` (default or tuned), ``table``, ``task`` (regression, binary or
multiclass), ``fold`` (0, 1 or 2), ``score`` (R² for regression, ROC-AUC for classification: higher
is better, as ``strable.evaluate`` computes it), ``total_s`` (the fold's run time, in seconds),
``n_rows``, ``time_per_1k_s`` (run time per 1,000 rows, in seconds), ``overlap`` (True when the
model's pre-training data may include STRABLE tables, marked ∩ on the leaderboard) and ``api``
(True for a model run on its provider's servers). A pipeline has 324 rows (108 tables x 3 folds), or
fewer when its model supports one task only (96 for a classification-only model).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "leaderboard"
HF_SPACE = "inria-soda/strable-leaderboard"


def load_published_results(latest: bool = False) -> pd.DataFrame:
    """The leaderboard's per-fold results, as a DataFrame with one row per pipeline, table and fold.

    By default, the results shipped with this version of the package; ``latest=True`` downloads
    today's leaderboard. ``df.attrs`` gives the leaderboard version (``leaderboard_version``) and its
    build date (``built``).
    """
    if latest:
        from huggingface_hub import hf_hub_download

        rows = hf_hub_download(HF_SPACE, "results_per_fold.parquet", repo_type="space")
        info = json.loads(Path(hf_hub_download(HF_SPACE, "results.json", repo_type="space")).read_text())
        attrs = {"leaderboard_version": info["version"], "built": info["built"]}
    else:
        rows = RESULTS_DIR / "results_per_fold.parquet"
        info = json.loads((RESULTS_DIR / "manifest.json").read_text())
        attrs = {"leaderboard_version": info["leaderboard_version"], "built": info["built"]}
    df = pd.read_parquet(rows)
    df.attrs.update(attrs)
    return df
