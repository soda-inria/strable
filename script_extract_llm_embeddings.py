"""Script to extract llm embeddings for machine learning tasks."""

import time
import os
import numpy as np
import pandas as pd
from glob import glob
from src.utils_evaluation import load_data, col_names_per_type
from configs.path_configs import path_configs
from configs.exp_configs import llm_configs


def run_model(
    data_name,
    method,
    device="cuda",
    check_result_flag="True" # check if the embeddings already exist
):
    """Run llm embedding extraction for specific experiment setting."""

    # Preliminaries
    model_name = method
    llm_embed_folder = f'{path_configs["base_path"]}/data/llm_embeding/{model_name}'
    if not os.path.exists(llm_embed_folder):
        os.makedirs(llm_embed_folder, exist_ok=True)
    llm_embed_path = f'{llm_embed_folder}/{model_name}|{data_name}.parquet'

    time_folder = f'{path_configs["base_path"]}/data/llm_embed_time/{model_name}'
    if not os.path.exists(time_folder):
        os.makedirs(time_folder, exist_ok=True)
    time_path = f'{time_folder}/{model_name}|{data_name}.npy'


    if check_result_flag and os.path.exists(llm_embed_path):
        print(f"The embeddings for {model_name} on {data_name} already exist.")
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
        print("Loading FastText model...")
        lm_model = fasttext.load_model(path_configs["fasttext_path"])
    else:
        from sentence_transformers import SentenceTransformer

        hf_token = os.environ.get("HF_TOKEN")
        print(f"Loading SentenceTransformer model: {model_path}...")

        if hf_token is None:
            raise ValueError("HF_TOKEN environment variable is not set")

        if model_configs['hf_model_name'] == 'nvidia/llama-nemotron-embed-1b-v2':
            lm_model = SentenceTransformer(
                model_name_or_path=model_path,
                trust_remote_code=True,
                cache_folder=cache_folder,
                device=device,
                token=hf_token,
            )
        else:
            lm_model = SentenceTransformer(
                model_name_or_path=model_path,
                cache_folder=cache_folder,
                device=device,
                token=hf_token,
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
    print(f"Loading dataset: {data_name}...")
    data, data_config = load_data(data_name)
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
    print(f"Saving embeddings to: {llm_embed_path}")
    llm_embeddings.to_parquet(llm_embed_path, index=False)
    np.save(time_path, duration_emb_extraction)

    print(f"Done! Extraction took {duration_emb_extraction} seconds.")
    return None


if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # USER CONFIGURATION
    # ---------------------------------------------------------
    DATASET_NAME = "clear-corpus" 
    METHOD_NAME = "llm-e5-small-v2" # Must match a key in configs/exp_configs.py
    DEVICE = "cuda" # 'cuda' or 'cpu'
    
    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------
    t_start = time.time()
    
    run_model(
        data_name=DATASET_NAME,
        method=METHOD_NAME,
        device=DEVICE,
        check_result_flag=True
    )
    
    print(f"Total time taken: {time.time() - t_start:.2f} seconds")