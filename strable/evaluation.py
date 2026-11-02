"""Score a scikit-learn pipeline on STRABLE tables, with the leaderboard's protocol and metrics.

Per table and outer fold: fit a fresh clone of the pipeline on the training rows, predict the test
rows and score them. R² for regression. ROC-AUC for classification, from ``predict_proba``: the
positive-class column for binary tasks; for multiclass tasks, the one-vs-rest macro average of the
softmax of the probabilities, as in the paper and on the leaderboard.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import r2_score, roc_auc_score

from .datasets import get_folds, list_tables, load_table


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax, computed as scipy.special.softmax(x, axis=1)."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fold_score(task: str, y_true, pred=None, proba=None, classes=None) -> float:
    """R² of ``pred`` for regression; ROC-AUC of ``proba`` for classification.

    ``proba`` has one column per class, in the order of ``classes`` (the sorted training labels).
    Binary: the positive-class column. Multiclass: one-vs-rest macro AUC of the row-wise softmax of
    ``proba`` (``reshape_pred_output`` in the paper's src/utils_evaluation.py).
    """
    if task == "regression":
        return float(r2_score(y_true, pred))
    proba = np.asarray(proba, dtype=float)
    if task == "binary":
        return float(roc_auc_score(np.asarray(y_true) == classes[1], proba[:, 1]))
    return float(roc_auc_score(y_true, _softmax(proba), multi_class="ovr", average="macro", labels=classes))


def checked_score(task: str, y_train, y_test, pred=None, proba=None, classes=None) -> float:
    """``fold_score`` after the leaderboard's checks of the predictions; raises ValueError."""
    n = len(y_test)
    if task == "regression":
        pred = np.asarray(pred, dtype=float)
        if pred.shape != (n,):
            raise ValueError(f"predict returned shape {pred.shape}, expected ({n},)")
        if not np.isfinite(pred).all():
            raise ValueError("predict returned NaN or infinite values")
        return fold_score(task, y_test, pred=pred)
    classes, proba = np.asarray(classes), np.asarray(proba, dtype=float)
    if classes.tolist() != sorted(np.unique(y_train).tolist()):
        raise ValueError("classes_ must be the sorted training labels")
    if proba.shape != (n, len(classes)):
        raise ValueError(f"predict_proba returned shape {proba.shape}, expected ({n}, {len(classes)})")
    if not (np.isfinite(proba).all() and (proba >= 0).all()):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(proba.sum(axis=1), 1, atol=1e-4):
        raise ValueError("probabilities must sum to 1 per row")
    return fold_score(task, y_test, proba=proba, classes=classes)


def _default_tables(pipeline) -> list[str]:
    if is_regressor(pipeline):
        return list_tables("regression")
    if is_classifier(pipeline):
        return list_tables("classification")
    raise ValueError("pass tables=...: the pipeline is neither a scikit-learn regressor nor a classifier")


def evaluate(pipeline, tables: str | list[str] | None = None, data_dir=None) -> pd.DataFrame:
    """Scores of ``pipeline`` on every outer fold of ``tables``, one row per fold.

    ``pipeline`` is a scikit-learn estimator that takes the raw table (a DataFrame with strings,
    numbers and dates), for example ``make_pipeline(strable.encoders.tfidf(), RidgeCV())``. A
    classifier needs ``predict_proba`` and ``classes_``. ``tables`` is one table name or a list of
    names; by default, every regression table for a regressor and every classification table for a
    classifier. ``data_dir`` is passed to ``load_table``.

    Columns: ``table``, ``task``, ``fold``, ``score`` (R² for regression, ROC-AUC for
    classification; higher is better), ``n_train``, ``n_test``, and ``time_s``, the wall-clock
    seconds to fit and predict.
    """
    if tables is None:
        tables = _default_tables(pipeline)
    elif isinstance(tables, str):
        tables = [tables]
    rows = []
    for name in tables:
        X, y, task = load_table(name, data_dir=data_dir)
        folds = get_folds(name)
        if len(X) != len(folds[0][0]) + len(folds[0][1]):
            raise ValueError(f"{name}: {len(X)} rows, but its folds cover {len(folds[0][0]) + len(folds[0][1])}")
        for fold, (train_idx, test_idx) in enumerate(folds):
            model, X_test = clone(pipeline), X.iloc[test_idx]
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Found unknown categories")
                model.fit(X.iloc[train_idx], y[train_idx])
                if task == "regression":
                    out = dict(pred=model.predict(X_test))
                else:
                    out = dict(proba=model.predict_proba(X_test), classes=model.classes_)
            seconds = time.perf_counter() - t0
            rows.append({"table": name, "task": task, "fold": fold,
                         "score": checked_score(task, y[train_idx], y[test_idx], **out),
                         "n_train": len(train_idx), "n_test": len(test_idx), "time_s": seconds})
    return pd.DataFrame(rows)
