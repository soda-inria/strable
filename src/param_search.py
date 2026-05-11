"""Hyperparameter search"""

import time
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import ParameterGrid, ParameterSampler

from configs.exp_configs import estim_configs
from configs.model_parameters import param_distributions_total
from src.utils_evaluation import (
    assign_estimator,
    calculate_output,
    check_pred_output,
    reshape_pred_output,
    return_score,
)


def run_param_search(
    X_train,
    y_train,
    task,
    estim_method,
    tune_indicator,
    cv,
    n_iter,
    n_jobs,
    device,
    cat_features=None,
):
    """Run hyperparameter search for one (estimator, dataset, fold).

    Returns
    -------
    cv_results : pandas.DataFrame or None
        Per-parameter-set summary (one row per sampled hyperparameter set,
        with mean/std test score and timing). ``None`` when no search runs.
    best_params : dict
        The best hyperparameter set, or ``{}`` if no tuning was performed.
    best_estimator : estimator
        Either an already-fitted ``Voting{Regressor,Classifier}`` (when the
        learner uses ``fit_with_val=True`` and tuning ran), or a fresh
        unfitted estimator with ``best_params`` baked in. The caller should
        pass it to ``run_inference``.
    """
    method_cfg = estim_configs.get(estim_method, {"search_method": "no-search", "fit_with_val": False})
    search_method = method_cfg["search_method"]
    fit_with_val = method_cfg["fit_with_val"]

    # Always start with a fresh estimator carrying defaults — used both for
    # no-search and as the template that gets cloned per fold.
    estimator = assign_estimator(
        estim_method,
        task,
        device,
        best_params={},
        cat_features=cat_features,
    )

    # TARTE batch-size override for large training sets (only the finetune
    # variant needs it; harmless for default Tarte).
    if "TARTEFinetune" in estimator.__class__.__name__ and len(X_train) > 500:
        estimator.batch_size = 256

    # Skip the search either when the model has no search space, or when the
    # user explicitly asked for the default (untuned) variant.
    if search_method == "no-search" or tune_indicator == "default":
        return None, {}, estimator

    # TARTE's grid-search uses per-epoch internal validation rather than CV.
    if estim_method == "tarte":
        return _tarte_grid_search(estimator, X_train, y_train)

    # Build the parameter list.
    param_distributions = param_distributions_total[estim_method]
    if search_method == "random-search":
        # Sample n_iter - 1 random points and explicitly include the empty
        # dict so the model's defaults are always evaluated alongside the
        # search.
        param_dict = list(
            ParameterSampler(
                param_distributions,
                n_iter=max(n_iter - 1, 1),
                random_state=1234,
            )
        )
        param_dict += [{}]
    elif search_method == "grid-search":
        param_dict = list(ParameterGrid(param_distributions))
    else:
        raise ValueError(f"Unknown search_method: {search_method!r}")

    # Pre-compute the CV splits once.
    split_index_list = list(enumerate(cv.split(X_train, y_train)))

    # Cartesian product (param_set x cv_fold) to feed joblib in parallel.
    run_args_list = list(product(enumerate(param_dict), split_index_list))

    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score_fold)(
            clone(estimator),
            task,
            X_train,
            y_train,
            fit_with_val,
            run_args,
        )
        for run_args in run_args_list
    )

    # Re-shape into a (n_params, n_folds) grid keyed by (param_idx, cv_idx).
    n_params = len(param_dict)
    n_folds = len(split_index_list)
    score_grid = np.zeros((n_params, n_folds))
    fit_time_grid = np.zeros((n_params, n_folds))
    score_time_grid = np.zeros((n_params, n_folds))
    # Per-(param, fold) trained estimator, kept only when fit_with_val=True.
    estimator_grid: list[list] = [[None] * n_folds for _ in range(n_params)]
    for (param_idx, cv_idx), score, fit_time, score_time, fitted in fold_results:
        score_grid[param_idx, cv_idx] = score
        fit_time_grid[param_idx, cv_idx] = fit_time
        score_time_grid[param_idx, cv_idx] = score_time
        if fit_with_val:
            estimator_grid[param_idx][cv_idx] = fitted

    # Aggregate per-parameter statistics.
    cv_results = _build_cv_results(
        param_dict, score_grid, fit_time_grid, score_time_grid
    )

    # Pick the best parameter set by mean test score.
    best_param_idx = int(np.argmax(cv_results["mean_test_score"].to_numpy()))
    best_params = deepcopy(param_dict[best_param_idx])

    # Build the best estimator. Two cases:
    #
    #   - fit_with_val=True: use the eight already-fitted models from the
    #     winning row, wrapped in FrozenEstimators inside a Voting estimator.
    #     The Voting's `fit` is a no-op (each leaf is frozen) so downstream
    #     `run_inference` can still call .fit(X, y) for symmetry.
    #
    #   - fit_with_val=False: just instantiate a new estimator with
    #     `best_params`. The caller fits it on the full training set.
    if fit_with_val:
        leaves = [
            (f"fold{cv_idx}", FrozenEstimator(estimator_grid[best_param_idx][cv_idx]))
            for cv_idx in range(n_folds)
        ]
        if task == "regression":
            best_estimator = VotingRegressor(estimators=leaves)
        else:
            best_estimator = VotingClassifier(estimators=leaves, voting="soft")
    else:
        best_estimator = assign_estimator(
            estim_method,
            task,
            device,
            best_params=best_params,
            cat_features=cat_features,
        )

    return cv_results, best_params, best_estimator


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _fit_and_score_fold(estimator, task, X_train, y_train, fit_with_val, run_args):
    """Fit ``estimator`` on one (param-set, fold) pair, return val score."""
    (param_idx, params), (cv_idx, split_index) = run_args
    estimator_ = deepcopy(estimator)
    if "CatBoost" in estimator_.__class__.__name__:
        # CatBoost stashes constructor params in `_init_params`; updating
        # `__dict__` directly does not propagate to the underlying booster.
        estimator_.__dict__["_init_params"].update(params)
    else:
        estimator_.__dict__.update(params)

    X_train_, X_valid = X_train[split_index[0]], X_train[split_index[1]]
    y_train_, y_valid = y_train[split_index[0]], y_train[split_index[1]]
    eval_set = [(X_valid, y_valid)]

    start = time.perf_counter()
    if fit_with_val:
        if "XGB" in estimator_.__class__.__name__:
            estimator_.fit(X_train_, y_train_, eval_set=eval_set, verbose=False)
        else:
            estimator_.fit(X_train_, y_train_, eval_set=eval_set)
    else:
        estimator_.fit(X_train_, y_train_)
    fit_time = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    y_prob, y_pred = calculate_output(X_valid, estimator_, task)
    if "classification" in task:
        y_prob = reshape_pred_output(y_prob)
    if task == "regression":
        y_pred = check_pred_output(y_train, y_pred)
    score = return_score(y_valid, y_prob, y_pred, task)
    score_time = round(time.perf_counter() - start, 4)

    return (
        (param_idx, cv_idx),
        score[0],
        fit_time,
        score_time,
        estimator_ if fit_with_val else None,
    )


def _build_cv_results(param_dict, score_grid, fit_time_grid, score_time_grid):
    """Assemble a DataFrame mirroring sklearn ``cv_results_``."""
    n_folds = score_grid.shape[1]
    df = pd.DataFrame(
        score_grid, columns=[f"split{i}_test_score" for i in range(n_folds)]
    )
    df["mean_test_score"] = df.mean(axis=1)
    df["std_test_score"] = df.std(axis=1)
    df["mean_fit_time"] = fit_time_grid.mean(axis=1)
    df["std_fit_time"] = fit_time_grid.std(axis=1)
    df["mean_score_time"] = score_time_grid.mean(axis=1)
    df["std_score_time"] = score_time_grid.std(axis=1)
    df["rank_test_score"] = df["mean_test_score"].rank(ascending=False).astype(int)

    df_params = pd.DataFrame(param_dict).add_prefix("param_")
    df_params["params"] = [str(p) for p in param_dict]
    return pd.concat([df_params, df], axis=1)


def _tarte_grid_search(estimator, X_train, y_train):
    """Grid search for TARTE — uses the model's per-epoch valid_loss_."""
    param_distributions = param_distributions_total["tarte"]
    param_list = list(ParameterGrid(param_distributions))

    if len(X_train) > 1000:
        estimator.set_params(batch_size=256)

    rows = Parallel(n_jobs=1)(
        delayed(_run_single_tarte_fit)(estimator, X_train, y_train, params)
        for params in param_list
    )
    cv_results = pd.concat(rows, axis=0).reset_index(drop=True)
    cv_results["rank_test_score"] = (
        cv_results["mean_test_score"].rank(ascending=True).astype(int)
    )
    best_idx = cv_results["rank_test_score"].argmin()
    best_params = cv_results.loc[best_idx, "params"]
    if str(best_params) == "nan":
        best_params = {}

    # TARTE re-fits fresh on the full training set in run_inference; we just
    # hand back a fresh estimator carrying the chosen params.
    fresh = deepcopy(estimator)
    if best_params:
        fresh.__dict__.update(best_params)
    return cv_results, best_params, fresh


def _run_single_tarte_fit(estimator, X_train, y_train, params):
    """Fit one TARTE configuration and harvest its internal validation curve."""
    start = time.perf_counter()
    estimator_ = deepcopy(estimator)
    estimator_.__dict__.update(params)
    estimator_.fit(X_train, y_train)
    duration = round(time.perf_counter() - start, 4)

    valid_loss = np.array(estimator_.valid_loss_)
    row = {
        "params": params,
        "mean_test_score": float(np.mean(valid_loss)),
        "std_test_score": float(np.std(valid_loss)),
        "mean_fit_time": duration,
    }
    for i, loss in enumerate(valid_loss):
        row[f"split{i}_test_score"] = float(loss)
    return pd.DataFrame([row])
