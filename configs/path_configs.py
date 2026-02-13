"""Configuartions of paths."""

path_configs = dict()

# Base path
base_path = "/data/parietal/store3/work/mkim/codes/salts"
path_configs["base_path"] = base_path

# Preprocessed data
path_configs["path_data_processed"] = f"{base_path}/data/data_processed"
path_configs["path_data_processed_gioia"] = (
    f"/data/parietal/store4/soda/gblayer/salts/data/data_processed"
)

# caching folder for embeddings
path_configs["folder_cache_emb"] = f"{base_path}/__cache__"

# models
path_configs["models"] = f"{base_path}/data/models"

# LLM cached
path_configs["huggingface_cache_folder"] = (
    "/data/parietal/store3/work/mkim/huggingface/hub"
)
path_configs["fasttext_path"] = (
    "/data/parietal/store3/work/mkim/data/language_models/cc.en.300.bin"
)