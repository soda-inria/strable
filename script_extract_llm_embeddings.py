"""Script to extract llm embeddings for machine learning tasks."""

import time
import os
import submitit
import numpy as np
import pandas as pd

from glob import glob
from src.utils_evaluation import load_data, col_names_per_type
from configs.path_configs import path_configs
from configs.exp_configs import llm_configs


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
    """Run llm embedding extraction for specific experiment setting."""

    # Preliminaries
    model_name = embed_method
    llm_embed_folder = f'{path_configs['base_path']}/data/llm_embeding/{model_name}'
    if not os.path.exists(llm_embed_folder):
        os.makedirs(llm_embed_folder, exist_ok=True)
    llm_embed_path = f'{llm_embed_folder}/{model_name}|{data_name}.parquet'
    time_folder = f'{path_configs['base_path']}/data/llm_embed_time/{model_name}'
    if not os.path.exists(time_folder):
        os.makedirs(time_folder, exist_ok=True)
    time_path = f'{time_folder}/{model_name}|{data_name}.npy'

    if check_result_flag and os.path.exists(llm_embed_path):
        print("The embeddings already exists.")
        return None

    # Preliminary check
    cache_folder = path_configs["huggingface_cache_folder"]
    model_configs = llm_configs[model_name]
    model_base_path = (
        f'{cache_folder}/models--{model_configs['hf_model_name'].replace("/", "--")}'
    )
    if os.path.exists(model_base_path):
        model_path = glob(f"{model_base_path}/snapshots/*/config.json")[0].split(
            "config.json"
        )[0]
    else:
        model_path = model_configs['hf_model_name']

    # Load LLM model
    if model_configs['hf_model_name'] == 'fasttext':
        import fasttext

        lm_model = fasttext.load_model(path_configs["fasttext_path"])
    else:
        from sentence_transformers import SentenceTransformer

        lm_model = SentenceTransformer(
            model_name_or_path=model_path,
            cache_folder=cache_folder,
            device="cuda",
            token='',
        )
        # Token control for Llama models
        if "llama" in model_name:
            lm_model.tokenizer.pad_token = lm_model.tokenizer.eos_token

        # Set max. sequence length for memory usage.
        if (lm_model.max_seq_length is not None) and (lm_model.max_seq_length > 512):
            lm_model.max_seq_length = 512

    # Set batch-size with exceptions
    batch_size = 8
    exception_model = []
    exception_model += ["llm-llama-3.1-8b"]
    exception_model += ["llm-qwen3-8b"]
    exception_model += ["llm-opt-6.7b"]
    if model_name in exception_model:
        batch_size = 16

    # Load data
    data, data_config = load_data(data_name)
    _, cat_col, dat_col = col_names_per_type(data, data_config["target_name"])

    # Run embedding
    start_time = time.perf_counter()    

    # Extract total and unique words for comparison
    total_words = []
    if data_name == 'kickstarter-projects':
        extract_col_list = cat_col + dat_col 
    else:
        extract_col_list = cat_col
    for col in extract_col_list:
        total_words += data[col].astype(str).tolist()
    total_words = pd.DataFrame(total_words, columns=["name"])
    unique_words = pd.DataFrame(total_words["name"].unique(), columns=["name"])

    # Exception with Fasttext
    if model_configs['hf_model_name'] == 'fasttext':
        llm_embeddings = [lm_model.get_sentence_vector(str(x)) for x in np.array(unique_words['name'])]
        llm_embeddings = np.array(llm_embeddings)
    else:
        llm_embeddings = lm_model.encode(
            np.array(unique_words['name'].astype(str)),
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=False,
        )

    llm_embeddings = pd.DataFrame(llm_embeddings)
    llm_embeddings.columns = [f"X{x}" for x in range(llm_embeddings.shape[1])]
    llm_embeddings = pd.concat([unique_words, llm_embeddings], axis=1)

    end_time = time.perf_counter()
    duration_emb_extraction = round(end_time - start_time, 4)

    # Save the extracted embeddings
    llm_embeddings.to_parquet(llm_embed_path, index=False)
    np.save(time_path, duration_emb_extraction)

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
