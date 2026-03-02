import fasttext.util
import os
import shutil

# Import the dynamic paths we set up earlier
# (Adjust the import path if this script is not run from the root directory)
from configs.path_configs import path_configs

def main():
    # 1. Get the exact target path directly from your configs
    target_path = path_configs["fasttext_path"]
    target_dir = os.path.dirname(target_path)

    # 2. Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # 3. Check if it ALREADY exists at the final destination before doing anything
    if os.path.exists(target_path):
        print(f"✅ Model already exists at target path: {target_path}")
        return

    print("Model not found in target directory. Downloading (this may take a while)...")
    
    # 4. Download to current directory
    # fasttext returns the name of the downloaded unzipped file (e.g., 'cc.en.300.bin')
    downloaded_file = fasttext.util.download_model('en', if_exists='ignore')

    # 5. Move it to your path_configs location
    shutil.move(downloaded_file, target_path)
    print(f"✅ Model moved to: {target_path}")

    # 6. Clean up the massive .gz file that fasttext leaves behind
    gz_file = f"{downloaded_file}.gz"
    if os.path.exists(gz_file):
        os.remove(gz_file)
        print("🗑️ Cleaned up leftover .gz file.")

if __name__ == "__main__":
    main()