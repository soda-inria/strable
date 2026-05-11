"""Pre-compute and cache LLM embeddings for the feature-engineered tables."""

import json
import time
import os
import numpy as np
import pandas as pd

from glob import glob
from src.utils_evaluation import col_names_per_type
from configs.path_configs import path_configs
from configs.exp_configs import llm_configs


def run_model(
    data_name,
    method,
    n_split,
    fold_index,
    device,
    check_result_flag,
    override_cache,
):
    """Run llm embedding extraction for specific experiment setting."""

    # Preliminaries
    model_name = method
    llm_embed_folder = f'{path_configs["base_path"]}/data/llm_embeding_feat_eng/{model_name}'
    if not os.path.exists(llm_embed_folder):
        os.makedirs(llm_embed_folder, exist_ok=True)
    llm_embed_path = f'{llm_embed_folder}/{model_name}|{data_name}.parquet'
    time_folder = f'{path_configs["base_path"]}/data/llm_embed_time_feat_eng/{model_name}'
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

        if model_configs['hf_model_name'] == 'nvidia/llama-nemotron-embed-1b-v2':
            lm_model = SentenceTransformer(
                model_name_or_path=model_path,
                trust_remote_code=True,
                cache_folder=cache_folder,
                device="cuda",
                token=os.environ.get("HF_TOKEN"),
            )
        else:
            lm_model = SentenceTransformer(
                model_name_or_path=model_path,
                cache_folder=cache_folder,
                device="cuda",
                token=os.environ.get("HF_TOKEN"),
            )
        # Token control for Llama models
        if "llama" in model_name:
            lm_model.tokenizer.pad_token = lm_model.tokenizer.eos_token

    # Set batch-size with exceptions
    batch_size = 32
    exception_model = []
    exception_model += ["llm-llama-3.1-8b"]
    exception_model += ["llm-qwen3-8b"]
    exception_model += ["llm-opt-6.7b"]
    if model_name in exception_model:
        batch_size = 16

    # Load data
    data_folder = f"{path_configs['path_data_processed']}_feat_eng"

    # Dataset
    data_path = f"{data_folder}/{data_name}/data.parquet"
    data = pd.read_parquet(data_path)
    data.fillna(value=np.nan, inplace=True)

    # Configs
    config_path = f"{data_folder}/{data_name}/config.json"
    filename = open(config_path)
    data_config = json.load(filename)
    _, cat_col, _ = col_names_per_type(data, data_config["target_name"])

    # Run embedding
    start_time = time.perf_counter()    

    # Extract total and unique words for comparison
    total_words = []
    for col in cat_col:
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


def _get_experiment_args_list(
    data_name,
    n_split,
    method,
    fold_index,
    device,
    check_result_flag,
    override_cache,
):
    """Returns the list of arguments to run evaluations."""

    from sklearn.model_selection import ParameterGrid
    from configs.exp_configs import data_list_wide, llm_configs

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
    if method == ['all']:
        method_list = list(llm_configs.keys())
    else:
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

    # List out all the cases and run
    args_dict = dict()
    args_dict["data_name"] = data_name_list
    args_dict["n_split"] = n_split
    args_dict["method"] = method_list
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
    )
    print(f"Running {len(args_list)} experiments sequentially...", flush=True)
    for kwargs in args_list:
        run_model(**kwargs)
    print("done")
