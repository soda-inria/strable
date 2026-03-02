import os
from huggingface_hub import snapshot_download

def main():
    # 1. Determine the path to strable/data/data_processed
    # This assumes the script is running from within the strable/data/ folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_processed_dir = os.path.join(current_dir, "data_processed")
    
    # Create the folder if it doesn't already exist
    os.makedirs(data_processed_dir, exist_ok=True)
    print(f"Directory ready: {data_processed_dir}")

    # 2. Download the dataset repository
    repo_id = "inria-soda/STRABLE-benchmark"
    print(f"Downloading all datasets from {repo_id}...")
    
    # snapshot_download mirrors the exact repository folder structure 
    # directly into your specified local directory.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=data_processed_dir,
        # Uncomment the line below if you ONLY want to download .json and .parquet files,
        # ignoring things like README.md or .gitattributes
        # allow_patterns=["*.json", "*.parquet"] 
    )
    
    print(f"\nSuccess! All datasets have been downloaded and saved to: {data_processed_dir}")

if __name__ == "__main__":
    main()