"""
download_dataset.py
──────────────────────────────────────────────────────────────────────────────
Downloads the Fruits-360 dataset from Kaggle using kagglehub and copies it
into the project's data/ folder.

Prerequisites:
  1. pip install kagglehub
  2. You need a Kaggle account. On first run, kagglehub will ask you to
     authenticate — follow the on-screen instructions to paste your API token.
     Alternatively, place your kaggle.json in  C:\\Users\\<you>\\.kaggle\\kaggle.json

Usage:
  python download_dataset.py
──────────────────────────────────────────────────────────────────────────────
"""
import os
import shutil

def download_fruits360():
    """Download the Fruits-360 dataset and organise it into data/fruits-360/."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Installing now...")
        os.system("pip install kagglehub")
        import kagglehub

    from config import DATA_DIR

    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print(f"[OK] Dataset already exists at: {DATA_DIR}")
        # Show a quick summary
        for split in ("Training", "Test"):
            split_path = os.path.join(DATA_DIR, split)
            if os.path.isdir(split_path):
                num_classes = len([
                    d for d in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, d))
                ])
                print(f"    {split}: {num_classes} fruit classes found")
        return DATA_DIR

    print("[>>] Downloading Fruits-360 dataset from Kaggle...")
    print("    (If prompted, paste your Kaggle API token)\n")

    # kagglehub downloads to a cache dir and returns the local path
    cached_path = kagglehub.dataset_download("moltean/fruits")
    print(f"\n[OK] Downloaded to cache: {cached_path}")

    # Copy from cache into our project's data/ folder
    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)

    # The kagglehub cache usually has the files directly or in a subfolder.
    # We need to find the "Training" and "Test" folders.
    source = cached_path

    # Walk the cache to locate the Training folder
    for root, dirs, files in os.walk(cached_path):
        if "Training" in dirs and "Test" in dirs:
            source = root
            break

    print(f"[>>] Copying dataset to: {DATA_DIR}")
    shutil.copytree(source, DATA_DIR, dirs_exist_ok=True)

    # Summary
    for split in ("Training", "Test"):
        split_path = os.path.join(DATA_DIR, split)
        if os.path.isdir(split_path):
            num_classes = len([
                d for d in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, d))
            ])
            print(f"    {split}: {num_classes} fruit classes found")

    print("\n[OK] Dataset is ready!")
    return DATA_DIR


if __name__ == "__main__":
    download_fruits360()
