"""Script for evaluating models."""

import os
import time
import joblib
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from src.utils_evaluation import (
    load_data,
    set_score_criterion,
    calculate_output,
    reshape_pred_output,
    check_pred_output,
    return_score,
)
from src.encoding import embed_table
from src.param_search import run_param_search
from src.inference import run_inference


def get_width(X):
    """
    Robustly finds the number of columns/features across DataFrames, 
    Arrays, Tensors, or nested List-of-Tuples.
    """
    if hasattr(X, "shape"):
        return X.shape[1] if len(X.shape) > 1 else 1
    
    if isinstance(X, (list, tuple)):
        if not X: 
            return 0
        first = X[0]
        if isinstance(first, (list, tuple)):
            for item in first:
                if hasattr(item, "shape"):
                    return item.shape[1] if len(item.shape) > 1 else 1
            return len(first)
        return 1
    return 0


def run_model(
    data_name,
    method,
    n_split,
    fold_index,
    device="cuda",
    check_result_flag=True,
    override_cache=False,
    normalization=False, 
    no_pca=False, 
):
    """Run model for specific experiment setting."""
    
    # 1. Consolidated Result Pathing
    # Everything goes into a single master 'results' folder
    base_results_dir = "./results"
    
    # Modify the method name string to reflect the PCA/Norm settings for clean tracking
    setting_modifier = ""
    if no_pca:
        setting_modifier = "_no-pca"
    elif normalization:
        setting_modifier = "_pca-norm"
    else:
        setting_modifier = "_pca-no-norm"

    result_save_base_path = f"{base_results_dir}/benchmark/{data_name}/{method}{setting_modifier}"
    marker = f"{data_name}|{method}{setting_modifier}|{n_split}-cv|idx-{fold_index}"
    
    print(f"\n{marker} | STARTING")
    
    # Create directories
    os.makedirs(f"{result_save_base_path}/score", exist_ok=True)
    os.makedirs(f"{result_save_base_path}/log", exist_ok=True)
    
    results_model_path = f"{result_save_base_path}/score/{marker}.csv"
    log_path = f"{result_save_base_path}/log/{marker}_log.csv"

    if check_result_flag and os.path.exists(results_model_path):
        print("✅ The result already exists. Skipping.")
        return None

    # 2. Preliminaries
    print("Loading data and setting criterion...")
    _, data_config = load_data(data_name)
    task = data_config["task"]
    scoring, result_criterion = set_score_criterion(task)
    
    embed_method = ("_").join(method.split("_")[:-1])
    estim_method = method.split("_")[-1]

    # 3. Encode (with cache)
    print("Encoding table...")
    non_cache_embed = ["tabpfn", "catboost", 'tarte', 'contexttab', 'tabstar']
    if embed_method.split("_")[-1] in non_cache_embed:
        X_train, X_test, y_train, y_test, duration_embed, cat_features = embed_table(
            data_name, n_split, fold_index, embed_method,
        )
    else:
        cache_marker = f"{data_name}/{embed_method}/{n_split}-cv|idx-{fold_index}"
        mem = joblib.Memory(f"./__cache__/{cache_marker}", verbose=0)
        if override_cache:
            mem.clear(warn=False)
        X_train, X_test, y_train, y_test, duration_embed, cat_features = mem.cache(
            embed_table
        )(
            data_name, n_split, fold_index, embed_method, normalization, no_pca
        )

    if get_width(X_train) == 0:
        print("❌ Num only features did not return any encoding.")
        return None

    # 4. Estimator Pre-processing
    print("Pre-processing for estimator...")
    start_time = time.perf_counter()
    if estim_method == "ridge":
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    elif estim_method in ["tarte", "tabstar"]: 
        pass
    else:
        X_train, X_test = np.array(X_train), np.array(X_test)
    
    duration_embed += round(time.perf_counter() - start_time, 4)

    # 5. Cross-validation settings
    if task == "regression":
        cv = RepeatedKFold(n_splits=8, n_repeats=1, random_state=1234)
    else:
        cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=1, random_state=1234)
    n_iter, n_jobs = 100, len(os.sched_getaffinity(0))

    # 6. Hyperparameter search
    print("Running hyperparameter search...")
    start_time = time.perf_counter()
    cv_results, best_params, best_split_idx = run_param_search(
        X_train, y_train, task, estim_method, cv, n_iter, n_jobs, scoring, device, cat_features,
    )
    duration_param_search = round(time.perf_counter() - start_time, 4)

    # 7. Final fit and predict
    print("Running final inference...")
    start_time = time.perf_counter()
    estimator = run_inference(
        X_train, y_train, task, estim_method, cv, device, best_params, best_split_idx, cat_features,
    )
    y_prob, y_pred = calculate_output(X_test, estimator, task)

    if estim_method == "realmlp":
        shutil.rmtree(estimator.tmp_folder)

    if "classification" in task:
        y_prob = reshape_pred_output(y_prob)

    if task == "regression":
        y_pred = check_pred_output(y_train, y_pred)

    score = return_score(y_test, y_prob, y_pred, task)
    duration_inference = round(time.perf_counter() - start_time, 4)

    # 8. Format & Save Results
    print("Saving results...")
    results_ = dict()
    for i in range(len(result_criterion[:-4])):
        results_[result_criterion[i]] = score[i]
    results_[result_criterion[-4]] = duration_embed
    results_[result_criterion[-3]] = duration_param_search
    results_[result_criterion[-2]] = duration_inference
    results_[result_criterion[-1]] = duration_embed + duration_param_search + duration_inference
    
    results_model = pd.DataFrame([results_], columns=result_criterion)
    results_model["data_name"] = data_name
    results_model["method"] = method
    results_model["n_cv"] = n_split
    results_model["fold_index"] = fold_index
    results_model["task"] = task

    results_model.to_csv(results_model_path, index=False)
    if cv_results is not None:
        cv_results.to_csv(log_path, index=False)

    print(f"✅ {marker} | COMPLETE")

    return None


if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # USER CONFIGURATION (Example based on bash script)
    # ---------------------------------------------------------
    DATASET_NAME = "commitments-in-trust-funds" 
    METHOD_NAME = "num-str_llm-llama-nemotron-embed-1b-v2_tabpfn" 
    N_SPLITS = 3
    DEVICE = "cuda"
    
    # We loop through the folds. 
    # (If your data has 3 splits, the fold indexes are usually 0, 1, 2)
    FOLD_INDEXES = [0, 1, 2] 
    
    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------
    t_start = time.time()
    
    for fold in FOLD_INDEXES:
        run_model(
            data_name=DATASET_NAME,
            method=METHOD_NAME,
            n_split=N_SPLITS,
            fold_index=fold,
            device=DEVICE,
            check_result_flag=True, #check if the result already exists
            override_cache=True, #override the cache
            normalization=False, #no standard scaling
            no_pca=False #30-dimension PCA
        )
    
    print(f"\nTotal pipeline time: {time.time() - t_start:.2f} seconds")