"""Configurations of paths."""
import os
from pathlib import Path

path_configs = dict()

# 1. Dynamically find the base path (the root of the 'strable' repository)
# __file__ is this script. .parent is 'configs/'. .parent.parent is 'strable/'
base_path = Path(__file__).resolve().parent.parent
path_configs["base_path"] = str(base_path)

# 2. Build all other paths relative to the base path
# Preprocessed data
path_configs["path_data_processed"] = str(base_path / "data" / "data_processed")

# caching folder for embeddings
path_configs["folder_cache_emb"] = str(base_path / "__cache__")

# models
path_configs["models"] = str(base_path / "data" / "models")

# LLM cached
# This creates a 'hub' folder inside the repo for HuggingFace downloads. 
# Alternatively, you can remove this entirely to let HF use the user's default ~/.cache/huggingface/
path_configs["huggingface_cache_folder"] = str(base_path / "hub")

# Fasttext path
path_configs["fasttext_path"] = str(base_path / "data" / "language_models" / "cc.en.300.bin")

# 3. (Optional but recommended) Automatically create these folders if they don't exist
folders_to_create = [
    path_configs["path_data_processed"],
    path_configs["folder_cache_emb"],
    path_configs["models"],
    path_configs["huggingface_cache_folder"],
    str(base_path / "data" / "language_models") # Ensure the folder for fasttext exists
]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)