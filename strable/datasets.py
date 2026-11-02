"""The 108 STRABLE tables and their cross-validation folds.

The tables come from the Hugging Face dataset inria-soda/STRABLE-benchmark at a pinned revision and
are cached by huggingface_hub. The folds ship with the package: ``data/folds/<table>.parquet`` gives,
for each row of the table, the outer fold (0, 1 or 2) that tests it. They are the STRABLE
leaderboard's folds (``splits/`` of gblayer/strable-leaderboard), computed as in the paper:
RepeatedKFold for regression, RepeatedStratifiedKFold for classification, 3 splits, 1 repeat,
seed 1234, on the row order of ``data.parquet``. ``data/folds/manifest.json`` is the leaderboard's
manifest: the dataset revision, the seed, the scikit-learn version, and per table the task, the row
count and the SHA-256 of ``data.parquet``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

HF_DATASET = "inria-soda/STRABLE-benchmark"
HF_REVISION = "335c0a77595d4a9ddd86f46dc9c3b414f63a09a8"  # 2026-06-11, the leaderboard's revision
FOLDS_DIR = Path(__file__).resolve().parent / "data" / "folds"
N_FOLDS = 3

# config.json task names -> task names used here and on the leaderboard
TASKS = {"regression": "regression", "b-classification": "binary", "m-classification": "multiclass"}
TASK_GROUPS = {"regression": ("regression",), "binary": ("binary",), "multiclass": ("multiclass",),
               "classification": ("binary", "multiclass")}


@cache
def _manifest() -> dict:
    return json.loads((FOLDS_DIR / "manifest.json").read_text())["tables"]


def _check_name(name: str) -> None:
    if name not in _manifest():
        raise ValueError(f"unknown table {name!r}; strable.list_tables() gives the 108 names")


def list_tables(task: str | None = None) -> list[str]:
    """Names of the 108 tables, sorted.

    ``task`` keeps the tables of one task: "regression", "binary", "multiclass", or
    "classification" for both binary and multiclass.
    """
    if task is not None and task not in TASK_GROUPS:
        raise ValueError(f"task must be one of {sorted(TASK_GROUPS)}, got {task!r}")
    return sorted(name for name, info in _manifest().items() if task is None or info["task"] in TASK_GROUPS[task])


def load_table(name: str, data_dir: str | Path | None = None) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Return ``(X, y, task)`` for one table.

    ``X`` holds every column but the target, rows in the order of ``data.parquet``, which the folds
    rely on. ``task`` is "regression", "binary" or "multiclass". Classification labels are encoded
    as 0..K-1 in the sorted order of the original labels, as in the paper.

    The two files of the table are downloaded from Hugging Face at the pinned revision and cached by
    huggingface_hub (``HF_HOME``). With ``data_dir``, they are read from
    ``<data_dir>/<name>/data.parquet`` and ``config.json`` instead, for example a local copy of
    the dataset.
    """
    _check_name(name)
    if data_dir is not None:
        folder = Path(data_dir) / name
        table_path, config_path = folder / "data.parquet", folder / "config.json"
    else:
        kwargs = dict(repo_id=HF_DATASET, repo_type="dataset", revision=HF_REVISION)
        table_path = hf_hub_download(filename=f"{name}/data.parquet", **kwargs)
        config_path = hf_hub_download(filename=f"{name}/config.json", **kwargs)
    config = json.loads(Path(config_path).read_text())
    data = pd.read_parquet(table_path)
    target, task = config["target_name"], TASKS[config["task"]]
    y = data[target].to_numpy()
    if task != "regression":
        y = np.unique(y, return_inverse=True)[1]
    return data.drop(columns=target), y, task


def get_folds(name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The benchmark's 3 outer folds of one table, as ``(train_idx, test_idx)`` pairs.

    Positions are row numbers of the table that ``load_table`` returns, in increasing order. Every
    row is a test row in exactly one fold. These are the folds of the paper and of the leaderboard.
    """
    _check_name(name)
    outer = pd.read_parquet(FOLDS_DIR / f"{name}.parquet")["outer_fold"].to_numpy()
    return [(np.flatnonzero(outer != k), np.flatnonzero(outer == k)) for k in range(N_FOLDS)]
