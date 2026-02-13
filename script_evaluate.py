"""Script for evaluting models."""

import os
import time
import joblib
import numpy as np
import pandas as pd
import submitit
import shutil

from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
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
from configs.exp_configs import embed_configs, estim_configs


def run_model(
    data_name,
    dtype_method,
    embed_method,
    estim_method,
    tune_indicator,
    n_split,
    fold_index,
    device,
    check_result_flag,
    override_cache,
):
    """Run model for specific experiment setting."""

    method_marker = ("_").join(
        [dtype_method, embed_method, estim_method, tune_indicator]
    )
    marker = f"{data_name}|{method_marker}|{n_split}-cv|idx-{fold_index}"
    print(marker + " start")

    if (estim_configs[estim_method]["search_method"] == "no-search") and (
        tune_indicator == "tune"
    ):
        print("The model requries no tuning.")
        return None

    # Set paths to save results
    save_path = "./results/benchmark"
    result_save_base_path = f"{save_path}/{data_name}/{method_marker}"
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
    _, result_criterion = set_score_criterion(task)

    # Encode (with cache)
    if embed_method in embed_configs["non_cache_methods"]:
        X_train, X_test, y_train, y_test, duration_embed, cat_features = embed_table(
            data_name,
            n_split,
            fold_index,
            dtype_method,
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
            dtype_method,
            embed_method,
        )

    # if X_train.shape[1] == 0:
    #     print("Num only features did not return any encoding.")
    #     return None

    # Control for ridge
    # TODO: update for different cases
    start_time = time.perf_counter()
    if estim_method == "ridge":
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    elif estim_method == 'tarte':
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

    cv_results, _, best_estimator = run_param_search(
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

    best_estimator = run_inference(
        best_estimator,
        estim_configs[estim_method]["fit_with_val"],
        X_train,
        y_train,
        tune_indicator,
        cv,
    )

    y_prob, y_pred = calculate_output(X_test, best_estimator, task)

    # Exception for pytabkit
    if estim_method == "realmlp":
        filename = best_estimator.tmp_folder
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
    results_model["method"] = method_marker
    results_model["n_cv"] = n_split
    results_model["fold_index"] = fold_index
    results_model["task"] = task

    # Save the results in csv
    results_model.to_csv(results_model_path, index=False)
    if cv_results is not None:
        cv_results.to_csv(log_path, index=False)

    print(marker + " is complete")

    return None


def get_executor_slurm(
    job_name,
    timeout_hour=60,
    n_cpus=10,
    max_parallel_tasks=10,
    partition="parietal,normal",
    exclude="marg[037-038,042-044]",
    device="cpu",
):
    """Return a submitit executor to launch various tasks on a SLURM cluster.

    Parameters
    ----------
    job_name: str
        Name of the tasks that will be run. It will be used to create an output
        directory and display task info in squeue.
    timeout_hour: int
        Maximal number of hours the task will run before being interupted.
    n_cpus: int
        Number of CPUs requested for each task.
    max_parallel_tasks: int
        Maximal number of tasks that will run at once. This can be used to
        limit the total amount of the cluster used by a script.
    partition: str
        Partition of SLURM where the job would be submitted to.
    exclude: str
        Name of nodes to exclude to submitting jobs.
    gpu: str
        "cpu" or "cuda", If set to "cuda", require one GPU per task.
    """

    folder_path = f"./slurm/{job_name}"
    executor = submitit.AutoExecutor(folder_path)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f"{timeout_hour}:00:00",
        array_parallelism=max_parallel_tasks,
        slurm_additional_parameters={
            "ntasks": 1,
            "partition": f"{partition}",
            "exclude": f"{exclude}",
            "cpus-per-task": n_cpus,
            "distribution": "block:block",
        },
    )
    if device == "cuda":
        executor.update_parameters(
            slurm_gres=f"gpu:1",
        )
    return executor


def _get_experiment_args_list(
    data_name,
    dtype_method,
    embed_method,
    estim_method,
    tune_indicator,
    n_split,
    fold_index,
    device,
    check_result_flag,
    override_cache,
):
    """Returns the list of arguments to run evaluations."""

    from sklearn.model_selection import ParameterGrid
    from glob import glob
    from configs.path_configs import path_configs

    data_name_list = data_name
    if data_name == ["all"]:
        data_name_list = glob(f"{path_configs['path_data_processed']}/*")
        data_name_list = [x.split("/")[-1] for x in data_name_list]
        data_name_list.sort()
    else:
        if isinstance(data_name_list, list) == False:
            data_name_list = [data_name_list]

    # Setting for train size
    if isinstance(n_split, list) == False:
        n_split = [n_split]
    n_split = [int(x) if float(x) - int(float(x)) == 0 else float(x) for x in n_split]

    # Setting for methods
    dtype_method_list = dtype_method
    if dtype_method == ["all"]:
        dtype_method_list = ["num-str", "str-only", "num-only"]
    else:
        if isinstance(dtype_method_list, list) == False:
            dtype_method_list = [dtype_method_list]

    embed_method_list = embed_method
    if isinstance(embed_method_list, list) == False:
        embed_method_list = [embed_method_list]

    estim_method_list = estim_method
    if isinstance(estim_method_list, list) == False:
        estim_method_list = [estim_method_list]

    # Setting for methods
    tune_indicator_list = tune_indicator
    if tune_indicator == ["all"]:
        tune_indicator_list = ["default", "tune"]
    else:
        if isinstance(tune_indicator_list, list) == False:
            tune_indicator_list = [tune_indicator_list]

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

    # List out all the cases and run
    args_dict = dict()
    args_dict["data_name"] = data_name_list
    args_dict["n_split"] = n_split
    args_dict["dtype_method"] = dtype_method_list
    args_dict["embed_method"] = embed_method_list
    args_dict["estim_method"] = estim_method_list
    args_dict["tune_indicator"] = tune_indicator_list
    args_dict["fold_index"] = fold_index
    args_dict["device"] = [device]
    args_dict["check_result_flag"] = [check_result_flag]
    args_dict["override_cache"] = [override_cache]
    args_list = list(ParameterGrid(args_dict))

    return args_list


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run caching for datasets.")
    parser.add_argument(
        "-jn",
        "--job_name",
        type=str,
        help="Name of job submitted.",
    )
    parser.add_argument(
        "-t",
        "--timeout_hour",
        type=int,
        default=10,
        help="Number of CPUs per run of run_one.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether or not to run computation on a GPU.",
    )
    parser.add_argument(
        "-w",
        "--n-cpus",
        type=int,
        default=10,
        help="Number of CPUs per run of run_one.",
    )
    parser.add_argument(
        "-mpt",
        "--max_parallel_tasks",
        type=int,
        default=10,
        help="Maximal number of tasks that will run at once.",
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        default="parietal,normal",
        help="Partition of SLURM to submits jobs to.",
    )
    parser.add_argument(
        "-ex",
        "--exclude",
        type=str,
        default="marg[037-038,042-044]",
        help="Nodes to exclude for submitting jobs.",
    )
    parser.add_argument(
        "-dn",
        "--data_name",
        nargs="+",
        type=str,
        help="Name of data.",
    )
    parser.add_argument(
        "-dm",
        "--dtype_method",
        nargs="+",
        type=str,
        help="Dtype method to evaluate",
    )
    parser.add_argument(
        "-emm",
        "--embed_method",
        nargs="+",
        type=str,
        help="Embed method to evaluate",
    )
    parser.add_argument(
        "-esm",
        "--estim_method",
        nargs="+",
        type=str,
        help="Estimator to evaluate",
    )
    parser.add_argument(
        "-ti",
        "--tune_indicator",
        nargs="+",
        type=str,
        help="Indicate to tune",
    )
    parser.add_argument(
        "-ns",
        "--n_split",
        nargs="+",
        type=str,
        help="Number of splits (n-CV)",
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

    args = parser.parse_args()

    # List all parameters to run the computation
    args_list = _get_experiment_args_list(
        args.data_name,
        args.dtype_method,
        args.embed_method,
        args.estim_method,
        args.tune_indicator,
        args.n_split,
        args.fold_index,
        args.device,
        args.check_result_flag,
        args.override_cache,
    )

    # Submit one task per set of parameters
    executor = get_executor_slurm(
        job_name=args.job_name,
        timeout_hour=args.timeout_hour,
        n_cpus=args.n_cpus,
        max_parallel_tasks=args.max_parallel_tasks,
        partition=args.partition,
        exclude=args.exclude,
        device=args.device,
    )

    # Run the computation on SLURM cluster with `submitit`
    print("Submitting jobs...", end="", flush=True)
    with executor.batch():
        tasks = [executor.submit(run_model, **args) for args in args_list]

    t_start = time.time()
    print("done")
