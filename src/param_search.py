"""Functions used for hyperparameter serach."""

import time
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler, ParameterGrid
from copy import deepcopy
from joblib import Parallel, delayed
from itertools import product
from configs.model_parameters import param_distributions_total
from src.utils_evaluation import (
    assign_estimator,
    calculate_output,
    reshape_pred_output,
    check_pred_output,
    return_score,
    shorten_param,
)


def run_param_search(
    X_train,
    y_train,
    task,
    estim_method,
    cv,
    n_iter,
    n_jobs,
    scoring,
    device,
    cat_features=None,
):
    """Function to run search for evaluation."""

    # Basic settings
    # ridge,realmlp,tabm have its owns search mechanism, so included in no_search logic
    param_method = estim_method.split("-")[-1]
    no_search_estimators = [
        "ridge",
        "tabpfn",
        "realtabpfn",
        "realmlp",
        "tabm",
        "default",
        "tabstar",
        "contexttab",
    ]
    gbdt_estimators_with_val = ["xgb", "catboost"]

    # No search for certain estimators
    if param_method in no_search_estimators:
        return None, {}, None

    # Parameter distribution
    param_distributions = param_distributions_total[param_method]

    # Run hyperparmeter search
    estimator = assign_estimator(
        estim_method,
        task,
        device,
        best_params={},
        cat_features=cat_features,
    )

    if param_method in gbdt_estimators_with_val:
        cv_results, best_params, best_split_idx = run_gbdt_param_search(
            estimator,
            X_train,
            y_train,
            "random",
            param_distributions,
            n_iter,
            cv,
            n_jobs,
        )
        return cv_results, best_params, best_split_idx
    elif param_method == "tarte":
        cv_results, best_params = run_tarte_param_search(
            estimator,
            X_train,
            y_train,
            param_distributions,
            n_jobs,
        )
        return cv_results, best_params, None
    else:
        cv_results, best_params = run_sklearn_param_search(
            estimator,
            X_train,
            y_train,
            param_distributions,
            n_iter,
            cv,
            n_jobs,
            scoring,
        )
        return cv_results, best_params, None

def run_tarte_param_search(
    estimator,
    X_train,
    y_train,
    param_distributions,
    n_jobs,
):
    """Grid Search function for TARTE"""

    # Set paramater list
    param_distributions_ = param_distributions.copy()
    param_list = list(ParameterGrid(param_distributions_))

    if len(X_train) > 1000:
        # Tarte specific setting for large datasets
        print(f"Setting TARTE batch_size to 256 in gridsearch for large dataset: {len(X_train)}")
        estimator.set_params(batch_size=256)

    # Run Gridsearch
    gridsearch_result = Parallel(n_jobs=1)(
        delayed(_run_single_tarte_fit)(estimator, X_train, y_train, params)
        for params in param_list
    )
    cv_results = pd.concat(gridsearch_result, axis=0).reset_index(drop=True)
    
    # Add rank
    cv_results["rank_test_score"] = (
        cv_results["mean_test_score"].rank(ascending=True).astype(int)
    )
    
    # Best params
    best_params = cv_results.loc[cv_results["rank_test_score"].argmin(), "params"]
    if str(best_params) == "nan":
        best_params = {}

    # TARTE does not use external CV splits, so we return None for best_split_idx
    return cv_results, best_params

def _run_single_tarte_fit(estimator, X_train, y_train, params):
    """Helper to run a single TARTE fit and extract internal validation stats."""
    
    import copy
    from time import perf_counter
    
    # Measure time
    start_time = perf_counter()

    # Run estimator
    estimator_ = copy.deepcopy(estimator)
    estimator_.__dict__.update(params)
    estimator_.fit(X_train, y_train)
    
    # Measure time
    duration = round(perf_counter() - start_time, 4)
    
    # Statistics
    vl = np.array(estimator_.valid_loss_)
    
    # Obtain results
    result_run = {
        "params": params,
        "mean_test_score": np.mean(vl),
        "std_test_score": np.std(vl),
        "mean_fit_time": duration,
    }

    # Add individual split scores for completeness
    for i, loss in enumerate(vl):
        result_run[f"split{i}_test_score"] = loss
        
    return pd.DataFrame([result_run])

def _run_estimator(estimator, X_train, y_train, run_args):
    """Function to run fit/predict on the given train/validation set."""

    # Set preliminaries
    param_idx = run_args[0][0]
    params = run_args[0][1]
    cv_idx = run_args[1][0]
    split_index = run_args[1][1]
    if estimator._estimator_type == "regressor":
        task = "regression"
    else:
        if len(np.unique(y_train)) == 2:
            task = "b-classification"
        else:
            task = "m-classification"
    est_name = estimator.__class__.__name__

    # Set the estimator
    estimator_ = deepcopy(estimator)
    if "CatBoost" in est_name:
        estimator_.__dict__["_init_params"].update(params)
    else:
        estimator_.__dict__.update(params)

    # Set the train and validation set
    X_train_, X_valid = X_train[split_index[0]], X_train[split_index[1]]
    y_train_, y_valid = y_train[split_index[0]], y_train[split_index[1]]
    eval_set = [(X_valid, y_valid)]

    # Measure fit_time
    start_time = time.perf_counter()

    if "LGBM" in est_name:
        estimator_.fit(X_train_, y_train_, eval_set=eval_set)
    else:
        estimator_.fit(X_train_, y_train_, eval_set=eval_set, verbose=False)

    end_time = time.perf_counter()
    fit_time = round(end_time - start_time, 4)

    # Measure score_time
    start_time = time.perf_counter()

    y_prob, y_pred = calculate_output(X_valid, estimator_, task)

    # Reshape prediction
    if "classification" in task:
        y_prob = reshape_pred_output(y_prob)

    # Check the output
    if task == "regression":
        y_pred = check_pred_output(y_train, y_pred)

    # obtain scores
    score = return_score(y_valid, y_prob, y_pred, task)

    end_time = time.perf_counter()
    score_time = round(end_time - start_time, 4)

    return (param_idx, cv_idx), score[0], fit_time, score_time

def run_gbdt_param_search(
    estimator,
    X_train,
    y_train,
    search_method,
    param_distributions,
    n_iter,
    cv,
    n_jobs,
):
    """Grid/Random Search function for GBDT (XGB/CatBoost) models."""

    # Set parameters depending on the search method
    if search_method == "random":
        param_dict = list(
            enumerate(
                ParameterSampler(
                    param_distributions,
                    n_iter=n_iter - 1,
                    random_state=1234,
                )
            )
        )
        param_dict += [(n_iter - 1, {})]
    elif search_method == "grid":
        param_dict = list(enumerate(ParameterGrid(param_distributions)))

    # Set splits
    cv_splits = list(enumerate(cv.split(X_train, y_train)))

    # Iterate all the cases to run
    run_args_list = list(product(param_dict, cv_splits))

    search_result = Parallel(n_jobs=n_jobs)(
        delayed(_run_estimator)(estimator, X_train, y_train, run_args)
        for run_args in run_args_list
    )

    # Format into DataFrame (as in sklearn search)
    test_score_result = np.zeros(shape=(len(param_dict), len(cv_splits)))
    fit_time_result = np.zeros(shape=(len(param_dict), len(cv_splits)))
    score_time_result = np.zeros(shape=(len(param_dict), len(cv_splits)))
    for x in search_result:
        test_score_result[x[0]] = x[1]
        fit_time_result[x[0]] = x[2]
        score_time_result[x[0]] = x[3]

    test_score_result = pd.DataFrame(test_score_result)
    split_test_columns = [
        f"split{x}_test_score" for x in range(test_score_result.shape[1])
    ]
    test_score_result.columns = split_test_columns
    test_score_result["mean_test_score"] = test_score_result[split_test_columns].mean(
        axis=1
    )
    test_score_result["std_test_score"] = test_score_result[split_test_columns].std(
        axis=1
    )
    test_score_result["rank_test_score"] = (
        test_score_result["mean_test_score"].rank(ascending=False).astype(int)
    )
    test_score_result["mean_fit_time"] = fit_time_result.mean(axis=1)
    test_score_result["std_fit_time"] = fit_time_result.std(axis=1)
    test_score_result["mean_score_time"] = score_time_result.mean(axis=1)
    test_score_result["std_score_time"] = score_time_result.std(axis=1)

    df_params = pd.DataFrame([params for (_, params) in param_dict])
    df_params.columns = [f"param_{col}" for col in df_params.columns]
    df_params["params"] = pd.DataFrame(param_dict)[1]

    cv_results = pd.concat([df_params, test_score_result], axis=1)
    best_params = cv_results.loc[cv_results["rank_test_score"].argmin(), "params"]
    if str(best_params) == "nan":
        best_params = {}

    best_split_idx = cv_results.loc[
        cv_results["rank_test_score"].argmin(), split_test_columns
    ].idxmax()
    # best_split_idx = (
    #     cv_results[cv_results["rank_test_score"] == 1][split_test_columns]
    #     .max()
    #     .idxmax()
    # )
    best_split_idx = int(re.findall(r"\d+", best_split_idx)[0])

    return cv_results, best_params, best_split_idx


def run_sklearn_param_search(
    estimator,
    X_train,
    y_train,
    param_distributions,
    n_iter,
    cv,
    n_jobs,
    scoring,
):
    """Function to run the scikit-learn parameter search."""

    # First run with the default parameters.
    default_search = RandomizedSearchCV(
        estimator,
        param_distributions=[{}],
        n_iter=1,  # Excluding default
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=n_jobs,
        random_state=1234,
    )
    default_search.fit(X_train, y_train)
    cv_default = pd.DataFrame(default_search.cv_results_)

    # Run without the default
    hyperparameter_search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=n_iter - 1,  # Excluding default
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=n_jobs,
        random_state=1234,
    )
    hyperparameter_search.fit(X_train, y_train)
    cv_results = pd.DataFrame(hyperparameter_search.cv_results_)

    # Format the cv results with the default added
    cv_results = pd.concat([cv_results, cv_default])
    cv_results = cv_results.rename(shorten_param, axis=1)
    cv_results.reset_index(drop=True, inplace=True)
    rank = (
        cv_results["mean_test_score"]
        .rank(method="min", ascending=False)
        .astype(int)
        .copy()
    )
    cv_results["rank_test_score"] = rank
    params_ = cv_results["params"]
    best_params = params_[cv_results["rank_test_score"] == 1].iloc[0]
    if str(best_params) == "nan":
        best_params = {}

    return cv_results, best_params
