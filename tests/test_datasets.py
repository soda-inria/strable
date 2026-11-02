"""The table list, the folds shipped with the package, and loading a table."""

import json

import numpy as np
import pandas as pd
import pytest

import strable
from strable.datasets import FOLDS_DIR


def test_108_tables_split_by_task():
    tables = strable.list_tables()
    assert len(tables) == 108 and tables == sorted(tables)
    binary, multiclass = strable.list_tables("binary"), strable.list_tables("multiclass")
    assert sorted(binary + multiclass) == strable.list_tables("classification")
    assert len(strable.list_tables("regression")) + len(binary) + len(multiclass) == 108


def test_unknown_task_or_table_raises():
    with pytest.raises(ValueError, match="task"):
        strable.list_tables("ranking")
    with pytest.raises(ValueError, match="unknown table"):
        strable.get_folds("not-a-table")


def test_folds_partition_the_rows_of_every_table():
    manifest = json.loads((FOLDS_DIR / "manifest.json").read_text())["tables"]
    for name in strable.list_tables():
        folds, info = strable.get_folds(name), manifest[name]
        assert [len(test) for _, test in folds] == info["test_rows_per_outer_fold"]
        assert np.array_equal(np.sort(np.concatenate([test for _, test in folds])), np.arange(info["n_rows"]))
        for train, test in folds:
            assert len(train) + len(test) == info["n_rows"] and not np.intersect1d(train, test).size


def test_load_table_from_a_folder_encodes_labels(tmp_path):
    folder = tmp_path / "drug-ndc"
    folder.mkdir()
    pd.DataFrame({"a": ["x", "y", "z", "x"], "label": ["b", "a", "c", "a"]}).to_parquet(folder / "data.parquet")
    (folder / "config.json").write_text(json.dumps({"target_name": "label", "task": "m-classification"}))
    X, y, task = strable.load_table("drug-ndc", data_dir=tmp_path)
    assert list(X.columns) == ["a"] and task == "multiclass" and y.tolist() == [1, 0, 2, 0]


@pytest.mark.network
def test_load_table_from_hugging_face():
    X, y, task = strable.load_table("drug-ndc")
    assert task == "multiclass" and X.shape == (583, 21) and sorted(set(y)) == [0, 1, 2, 3]
    assert sum(len(test) for _, test in strable.get_folds("drug-ndc")) == len(X)
