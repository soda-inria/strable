"""Functions used for hyperparameter serach."""

import time
import pandas as pd
import numpy as np

from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.base import clone

from configs.model_parameters import param_distributions_total
from configs.exp_configs import estim_configs
from src.utils_evaluation import (
    assign_estimator,
    calculate_output,
    reshape_pred_output,
    check_pred_output,
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
    """Function to run search for evaluation."""

    # Assign estimator
    estimator = assign_estimator(
        estim_method,
        task,
        device,
        best_params={},
        cat_features=cat_features,
    )
    if ("TARTEFinetune" in estimator.__class__.__name__) & (len(X_train) > 500):
        estimator.batch_size = 256

    # No search for certain estimators and default
    if (estim_configs[estim_method]["search_method"] == "no-search") | (
        tune_indicator == "default"
    ):
        return None, {}, estimator

    # Sample parameters
    if estim_configs[estim_method]["search_method"] == "random-search":
        param_distributions = param_distributions_total[estim_method]
        param_dict = list(
            ParameterSampler(
                param_distributions,
                n_iter=n_iter - 1,
                random_state=1234,
            )
        )
        param_dict += [{}]
    elif estim_configs[estim_method]["search_method"] == "grid-search":
        param_distributions = param_distributions_total[estim_method]
        param_dict = list(ParameterGrid(param_distributions))

    # Set split index list for train/val
    split_index_list = list(enumerate(cv.split(X_train, y_train)))

    # Preliminary settings
    df_score = []
    best_score = -9e15
    best_voting_model = None
    best_params = None

    # Run search
    for params in param_dict:
        # Run CV results
        search_result_cv = Parallel(n_jobs=n_jobs)(
            delayed(_run_estimator)(
                clone(estimator),
                task,
                X_train,
                y_train,
                estim_configs[estim_method]["fit_with_val"],
                cv_idx,
                split_index,
                params,
            )
            for cv_idx, split_index in split_index_list
        )
        # Format result (as in sklearn search)
        df_score_cv = {f"split{x[0]}_test_score": x[1] for x in search_result_cv}
        test_score_result = [x[1] for x in search_result_cv]
        fit_time_result = [x[2] for x in search_result_cv]
        score_time_result = [x[3] for x in search_result_cv]
        df_score_cv["mean_test_score"] = np.mean(test_score_result)
        df_score_cv["std_test_score"] = np.std(test_score_result)
        df_score_cv["mean_fit_time"] = np.mean(fit_time_result)
        df_score_cv["std_fit_time"] = np.std(fit_time_result)
        df_score_cv["mean_score_time"] = np.mean(score_time_result)
        df_score_cv["std_score_time"] = np.std(score_time_result)
        df_score += [df_score_cv]

        # Flag for update in best
        if df_score_cv["mean_test_score"] > best_score:
            best_params = deepcopy(params)
            best_score = deepcopy(df_score_cv["mean_test_score"])
            if estim_configs[estim_method]["fit_with_val"]:
                model_list = [
                    (f"model{x[0]}", FrozenEstimator(x[4])) for x in search_result_cv
                ]
                if task == "regression":
                    vote_estimator = VotingRegressor(estimators=model_list)
                else:
                    vote_estimator = VotingClassifier(
                        estimators=model_list, voting="soft"
                    )
                best_voting_model = deepcopy(vote_estimator)

    # Format the results
    df_score = pd.DataFrame(df_score)
    df_score["rank_test_score"] = (
        df_score["mean_test_score"].rank(ascending=False).astype(int)
    )
    df_params = pd.DataFrame(param_dict)
    df_params = df_params.add_prefix("param_")
    df_params["params"] = pd.DataFrame([str(param) for param in param_dict])
    cv_results = pd.concat([df_params, df_score], axis=1)

    # Set the estimator for models without validaion in fit
    if not estim_configs[estim_method]["fit_with_val"]:
        best_voting_model = assign_estimator(
            estim_method,
            task,
            device,
            best_params=best_params,
            cat_features=cat_features,
        )

    return cv_results, best_params, best_voting_model


def _run_estimator(
    estimator,
    task,
    X_train,
    y_train,
    fit_with_val,
    cv_idx,
    split_index,
    params,
):
    """Function to run fit/predict on the given train/validation set."""

    # Set the train and validation set
    if "TARTEFinetune" in estimator.__class__.__name__:
        X_train_ = [X_train[i] for i in split_index[0]]
        X_valid = [X_train[i] for i in split_index[1]]
    else:
        X_train_, X_valid = X_train[split_index[0]], X_train[split_index[1]]
    y_train_, y_valid = y_train[split_index[0]], y_train[split_index[1]]
    y_train_, y_valid = y_train[split_index[0]], y_train[split_index[1]]
    eval_set = [(X_valid, y_valid)]

    if "CatBoost" in estimator.__class__.__name__:
        estimator.__dict__["_init_params"].update(params)
    else:
        estimator.__dict__.update(params)

    # Measure fit_time
    start_time = time.perf_counter()

    # Fit model
    if fit_with_val:
        if "XGB" in estimator.__class__.__name__:
            estimator.fit(X_train_, y_train_, eval_set=eval_set, verbose=False)
        else:
            estimator.fit(X_train_, y_train_, eval_set=eval_set)
    else:
        estimator.fit(X_train_, y_train_)

    end_time = time.perf_counter()
    fit_time = round(end_time - start_time, 4)

    # Measure score_time
    start_time = time.perf_counter()

    y_prob, y_pred = calculate_output(X_valid, estimator, task)

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

    if fit_with_val:
        return cv_idx, score[0], fit_time, score_time, estimator
    else:
        return cv_idx, score[0], fit_time, score_time, None
