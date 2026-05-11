"""Cardinality-threshold ablation: low-cardinality columns -> One-Hot Encoding.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import shutil

from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from src.utils_evaluation import (
    get_width,
    load_data,
    set_score_criterion,
    calculate_output,
    reshape_pred_output,
    check_pred_output,
    return_score,
)
from src.encoding_thres_OHE import embed_table
from src.param_search import run_param_search
from src.inference import run_inference
from configs.exp_configs import estim_configs

ABLATION = "ct30-ohe"



def run_model(
    data_name,
    method,
    n_split,
    fold_index,
    device,
    check_result_flag,
    override_cache,
    save_dir='benchmark',
    tune_indicator='default',
    normalization=False, #put False for normal run. True only when no_pca is False. if true = apply standard scaler before pca.
    no_pca=False, #put False for normal run. if True it means we are taking the first X dimensions as embeddings instead of doing PCA.
    n_dimensions=30
):
    """Run model for specific experiment setting."""
       
    save_path = f"./results/{ABLATION}/{save_dir}"
    method_marker = f"{method}_{tune_indicator}"
    marker = f"{data_name}|{method_marker}|{n_split}-cv|idx-{fold_index}"
    result_save_base_path = f"{save_path}/{data_name}/{method_marker}"
    
    print(marker + " start")
    
    
    if not os.path.exists(result_save_base_path):
        os.makedirs(result_save_base_path, exist_ok=True)
    if not os.path.exists(result_save_base_path + "/score"):
        os.makedirs(result_save_base_path + "/score", exist_ok=True)
    if not os.path.exists(result_save_base_path + "/log"):
        os.makedirs(result_save_base_path + "/log", exist_ok=True)
    results_model_path = result_save_base_path + f"/score/{marker}.csv"
    log_path = result_save_base_path + f"/log/{marker}_log.csv"

    if check_result_flag and os.path.exists(results_model_path):
        print("The result already exists.")
        return None

    # Set preliminaries
    _, data_config = load_data(data_name)
    task = data_config["task"]
    scoring, result_criterion = set_score_criterion(task)
    embed_method = ("_").join(method.split("_")[:-1])
    estim_method = method.split("_")[-1]

    # Skip tune-mode runs for models that don't have a search space.
    if (
        estim_configs.get(estim_method, {"search_method": "no-search"})["search_method"]
        == "no-search"
        and tune_indicator == "tune"
    ):
        print("The model requires no tuning; skipping.")
        return None

    # Encode (with cache)
    non_cache_embed = ["catboost", "contexttab", "tabstar", "mambular"]
    if embed_method.split("_")[-1] in non_cache_embed:
        X_train, X_test, y_train, y_test, duration_embed, cat_features = embed_table(
            data_name,
            n_split,
            fold_index,
            embed_method,
        )
    else:
        cache_marker = f"{data_name}/{embed_method}/{n_split}-cv|idx-{fold_index}"
        mem = joblib.Memory(f"./__cache__/{cache_marker}", verbose=0)
        if override_cache:
            mem.clear(warn=False)
        X_train, X_test, y_train, y_test, duration_embed, cat_features = mem.cache(
            embed_table
        )(
            data_name,
            n_split,
            fold_index,
            embed_method,
            normalization, 
            no_pca,
            n_dimensions, # This is the number of dimensions to keep for the no-PCA version.
        )

    # check that features are present
    if get_width(X_train) == 0:
        print("Num only features did not return any encoding.")
        return None

    # Control for learners that don't natively handle NaNs
    start_time = time.perf_counter()
    if estim_method in ["ridge", "tabm", "realmlp"]:
        from sklearn.impute import SimpleImputer

        # Find any columns that are STILL strings/objects
        leaked_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        if len(leaked_cols) > 0:
            print("\n🚨 FATAL LEAK! Skrub failed to encode these columns:")
            for col in leaked_cols:
                print(f" - {col} (First 3 values: {X_train[col].dropna().head(3).tolist()})")
            print("🚨 Exiting script to prevent further errors.\n")
            import sys
            sys.exit(1) # Stop the script so we don't get the confusing imputer error

        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    elif estim_method in ["tabstar", "tabicl", "mambular"]: # learners that do not want an array
        pass
    else:
        X_train, X_test = np.array(X_train), np.array(X_test)
    end_time = time.perf_counter()
    duration_embed += round(end_time - start_time, 4)

    # Set cross-validation settings
    if task == "regression":
        cv = RepeatedKFold(n_splits=8, n_repeats=1, random_state=1234)
    else:
        cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=1, random_state=1234)
    n_iter, n_jobs = 100, len(os.sched_getaffinity(0))

    # Hyperparmeter search
    start_time = time.perf_counter()

    cv_results, best_params, best_estimator = run_param_search(
        X_train,
        y_train,
        task,
        estim_method,
        tune_indicator,
        cv,
        n_iter,
        n_jobs,
        device,
        cat_features,
    )

    end_time = time.perf_counter()
    duration_param_search = round(end_time - start_time, 4)

    # Final fit and predict
    start_time = time.perf_counter()

    estimator = run_inference(
        best_estimator,
        estim_configs[estim_method]["fit_with_val"],
        X_train,
        y_train,
        tune_indicator,
        cv,
    )
    
    y_prob, y_pred = calculate_output(X_test, estimator, task)

    # Exception for pytabkit
    if estim_method == "realmlp":
        filename = estimator.tmp_folder
        shutil.rmtree(filename)


    # Reshape prediction
    if "classification" in task:
        y_prob = reshape_pred_output(y_prob)

    # Check the output
    if task == "regression":
        y_pred = check_pred_output(y_train, y_pred)

    # obtain scores
    score = return_score(y_test, y_prob, y_pred, task)

    end_time = time.perf_counter()
    duration_inference = round(end_time - start_time, 4)

    # Format the results
    results_ = dict()
    for i in range(len(result_criterion[:-4])):
        results_[result_criterion[i]] = score[i]
    results_[result_criterion[-4]] = duration_embed
    results_[result_criterion[-3]] = duration_param_search
    results_[result_criterion[-2]] = duration_inference
    results_[result_criterion[-1]] = (
        duration_embed + duration_param_search + duration_inference
    )
    results_model = pd.DataFrame([results_], columns=result_criterion)
    results_model["data_name"] = data_name
    results_model["method"] = method
    results_model["n_cv"] = n_split
    results_model["fold_index"] = fold_index
    results_model["task"] = task

    # Save the results in csv
    results_model.to_csv(results_model_path, index=False)
    if cv_results is not None:
        cv_results.to_csv(log_path, index=False)

    print(marker + " is complete")

    return None

def _get_experiment_args_list(
    data_name,
    n_split,
    method,
    fold_index,
    device,
    check_result_flag,
    override_cache,
    tune_indicator,
    save_dir,           
    normalization,      
    no_pca,             
    n_dimensions,
):
    """Returns the list of arguments to run evaluations."""

    from sklearn.model_selection import ParameterGrid
    from configs.exp_configs import data_list_wide

    data_name_list = data_name
    if data_name == ["all-wide"]:
        data_name_list = data_list_wide
    else:
        if isinstance(data_name_list, list) == False:
            data_name_list = [data_name_list]

    # Setting for train size
    if isinstance(n_split, list) == False:
        n_split = [n_split]
    n_split = [int(x) if float(x) - int(float(x)) == 0 else float(x) for x in n_split]

    # Setting for method
    method_list = method
    if isinstance(method_list, list) == False:
        method_list = [method_list]

    # Setting for random state
    if "all" in fold_index:
        fold_index = np.arange(n_split[0]).tolist()
    else:
        if isinstance(fold_index, list) == False:
            fold_index = [fold_index]
            fold_index = list(map(int, fold_index))
        else:
            fold_index = list(map(int, fold_index))
    if check_result_flag == "True":
        check_result_flag = True
    else:
        check_result_flag = False
    if override_cache == "True":
        override_cache = True
    else:
        override_cache = False

    # Setting for tune indicator (default / tune / all)
    tune_indicator_list = tune_indicator
    if tune_indicator == ["all"]:
        tune_indicator_list = ["default", "tune"]
    else:
        if isinstance(tune_indicator_list, list) is False:
            tune_indicator_list = [tune_indicator_list]

    # List out all the cases and run
    args_dict = dict()
    args_dict["data_name"] = data_name_list
    args_dict["n_split"] = n_split
    args_dict["method"] = method_list
    args_dict["fold_index"] = fold_index
    args_dict["device"] = [device]
    args_dict["check_result_flag"] = [check_result_flag]
    args_dict["override_cache"] = [override_cache]
    args_dict["tune_indicator"] = tune_indicator_list
    args_dict["save_dir"] = [save_dir]
    args_dict["normalization"] = [normalization]
    args_dict["no_pca"] = [no_pca]
    args_dict["n_dimensions"] = [n_dimensions]
    args_list = list(ParameterGrid(args_dict))

    return args_list


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run caching for datasets.")
    parser.add_argument(
        "-dn",
        "--data_name",
        nargs="+",
        type=str,
        help="Name of data.",
    )
    parser.add_argument(
        "-ns",
        "--n_split",
        nargs="+",
        type=str,
        help="Number of splits (n-CV)",
    )
    parser.add_argument(
        "-m",
        "--method",
        nargs="+",
        type=str,
        help="Method to evaluate",
    )
    parser.add_argument(
        "-fi",
        "--fold_index",
        nargs="+",
        type=str,
        help="Fold Index",
    )
    parser.add_argument(
        "-dv",
        "--device",
        type=str,
        help="Device, cpu or cuda",
    )
    parser.add_argument(
        "-cf",
        "--check_result_flag",
        type=str,
        help="Indicate to check for existing result",
    )
    parser.add_argument(
        "-oc",
        "--override_cache",
        type=str,
        help="Indicate to override the existing cache",
    )
    parser.add_argument(
        "-ti",
        "--tune_indicator",
        nargs="+",
        type=str,
        default=["default"],
        help='Tune mode: "default" (no search), "tune", or "all".',
    )
    parser.add_argument(
        "-sd", 
        "--save_dir", 
        type=str, 
        required=True, 
        help="Directory name to save results"
    )
    parser.add_argument(
        "-norm", 
        "--normalization", 
        type=lambda x: str(x).lower() == 'true',
        default=False,
        help="Apply StandardScaler before PCA"
    )
    parser.add_argument(
        "-nopca", 
        "--no_pca", 
        type=lambda x: str(x).lower() == 'true',
        default=False,
        help="Skip PCA completely"
    )
    parser.add_argument(
        "-ndim", 
        "--n_dimensions", 
        type=int, 
        default=30, 
        help="Number of PC or embedding dimensions to keep"
    )

    args = parser.parse_args()

    # List all parameters to run the computation
    args_list = _get_experiment_args_list(
        args.data_name,
        args.n_split,
        args.method,
        args.fold_index,
        args.device,
        args.check_result_flag,
        args.override_cache,
        args.tune_indicator,
        args.save_dir,
        args.normalization,  # <-- Clean boolean!
        args.no_pca,         # <-- Clean boolean!
        args.n_dimensions
    )
    print(f"Running {len(args_list)} experiments sequentially...", flush=True)
    for kwargs in args_list:
        run_model(**kwargs)
    print("done")
