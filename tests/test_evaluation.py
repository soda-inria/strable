"""The fold scores and the evaluation loop, on synthetic tables named after real ones."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline

import strable
from strable.evaluation import _default_tables, _softmax, checked_score, fold_score

REGRESSION, BINARY = "china-overseas-finance-inventory", "animalandveterinary-event"  # 550 and 1,345 rows


def _write_table(data_dir, name, n_rows, task):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n_rows)
    y = 3 * x + rng.normal(scale=0.1, size=n_rows) if task == "regression" else (x > 0).astype(int)
    folder = data_dir / name
    folder.mkdir()
    pd.DataFrame({"city": rng.choice(["Paris", "Lyon", "Nice"], n_rows), "x": x, "target": y}).to_parquet(
        folder / "data.parquet")
    task_name = {"regression": "regression", "binary": "b-classification"}[task]
    (folder / "config.json").write_text(json.dumps({"target_name": "target", "task": task_name}))


def _pipeline(estimator):
    return make_pipeline(strable.encoders.tfidf(), SimpleImputer(keep_empty_features=True), estimator)


def test_softmax_matches_scipy():
    special = pytest.importorskip("scipy.special")
    x = np.random.default_rng(0).normal(size=(20, 4))
    np.testing.assert_array_equal(_softmax(x), special.softmax(x, axis=1))


def test_fold_scores():
    assert fold_score("regression", [1.0, 2.0, 3.0], pred=[1.0, 2.0, 3.0]) == 1.0
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert fold_score("binary", [0, 1, 1], proba=proba, classes=np.array([0, 1])) == 1.0
    proba = np.eye(3)[[0, 1, 2, 0]]
    assert fold_score("multiclass", [0, 1, 2, 0], proba=proba, classes=np.array([0, 1, 2])) == 1.0


def test_checked_score_rejects_bad_predictions():
    with pytest.raises(ValueError, match="shape"):
        checked_score("regression", [0.0, 1.0], [0.0, 1.0], pred=[0.0])
    with pytest.raises(ValueError, match="sum to 1"):
        checked_score("binary", [0, 1], [0, 1], proba=[[0.5, 0.6], [0.5, 0.5]], classes=[0, 1])


def test_default_tables_follow_the_estimator():
    assert _default_tables(RidgeCV()) == strable.list_tables("regression")
    assert _default_tables(_pipeline(LogisticRegression())) == strable.list_tables("classification")
    with pytest.raises(ValueError, match="tables"):
        _default_tables(SimpleImputer())


def test_evaluate_regression_and_binary(tmp_path):
    _write_table(tmp_path, REGRESSION, 550, "regression")
    _write_table(tmp_path, BINARY, 1345, "binary")
    reg = strable.evaluate(_pipeline(RidgeCV()), tables=REGRESSION, data_dir=tmp_path)
    clf = strable.evaluate(_pipeline(LogisticRegression()), tables=[BINARY], data_dir=tmp_path)
    assert list(reg.columns) == ["table", "task", "fold", "score", "n_train", "n_test", "time_s"]
    assert reg["fold"].tolist() == [0, 1, 2] and ((reg["n_train"] + reg["n_test"]) == 550).all()
    assert (reg["score"] > 0.99).all() and (clf["score"] > 0.99).all() and (clf["task"] == "binary").all()


def test_evaluate_rejects_a_table_with_other_rows(tmp_path):
    _write_table(tmp_path, REGRESSION, 549, "regression")
    with pytest.raises(ValueError, match="rows"):
        strable.evaluate(_pipeline(RidgeCV()), tables=REGRESSION, data_dir=tmp_path)
