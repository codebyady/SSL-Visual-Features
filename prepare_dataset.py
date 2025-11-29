import os
import urllib.request
import zipfile
import shutil

from huggingface_hub import HfFileSystem

# Where to put intermediate + final data
ZIP_DIR = "data/hf_zips"
EXTRACT_DIR = "data/hf_extracted"
OUTPUT_DIR = os.path.join("data", "pretrain")

BASE_URL = (
    "https://huggingface.co/datasets/tsbpp/fall2025_deeplearning/resolve/main"
)

# How many images per shard directory under data/pretrain/
IMAGES_PER_DIR = 1000  # you can bump to 5000 if you like


def list_zip_files_from_hub():
    fs = HfFileSystem()
    paths = fs.glob("hf://datasets/tsbpp/fall2025_deeplearning/**")

    zip_names = sorted({os.path.basename(p) for p in paths if p.endswith(".zip")})
    if not zip_names:
        raise RuntimeError("No .zip files found on the Hub for this dataset.")
    print("Discovered ZIP shards on Hub:")
    for name in zip_names:
        print("  -", name)
    return zip_names


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"[skip] {dest_path} already exists")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[download] {url} -> {dest_path}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            print(
                f"\r  {pct:5.1f}% ({downloaded/1e6:6.1f} MB / {total_size/1e6:6.1f} MB)",
                end="",
            )

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print("\n  done.")


def download_all_zips():
    print("=== Discovering ZIP shards on Hugging Face ===")
    zip_files = list_zip_files_from_hub()

    print("\n=== Downloading ZIP shards from Hugging Face ===")
    for fname in zip_files:
        url = f"{BASE_URL}/{fname}"
        dest = os.path.join(ZIP_DIR, fname)
        download_file(url, dest)


def extract_all_zips():
    print("\n=== Extracting all ZIP shards ===")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")]
    if not zip_files:
        raise RuntimeError(f"No .zip files found in {ZIP_DIR}")

    for fname in zip_files:
        zip_path = os.path.join(ZIP_DIR, fname)
        print(f"  Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(EXTRACT_DIR)
    print("Extraction complete.")


def gather_and_rename():
    print("\n=== Gathering images into sharded data/pretrain ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    counter = 0
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            src = os.path.join(root, f)

            # Decide which shard directory this image belongs to
            shard_idx = counter // IMAGES_PER_DIR
            shard_dir = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:03d}")
            os.makedirs(shard_dir, exist_ok=True)

            ext = os.path.splitext(f)[1].lower()  # .png, .jpg
            dst = os.path.join(shard_dir, f"{counter:07d}{ext}")

            shutil.copy2(src, dst)
            counter += 1

            if counter % 10000 == 0:
                print(f"  Copied {counter} images...")

    print(f"\nDone. Total images: {counter}")
    print(f"Pretrain shards in: {OUTPUT_DIR}")


# NEW -----------------------------------------------------------
def cleanup_intermediate_dirs():
    """Delete hf_zips and hf_extracted safely."""
    print("\n=== Cleaning up intermediate directories ===")

    for path in [ZIP_DIR, EXTRACT_DIR]:
        if os.path.exists(path):
            print(f"  Removing {path} ...")
            shutil.rmtree(path)
        else:
            print(f"  {path} does not exist, skipping")

    print("Cleanup complete.")
# ---------------------------------------------------------------


def main():
    download_all_zips()
    extract_all_zips()
    gather_and_rename()
    cleanup_intermediate_dirs()   # NEW: remove temp dirs


if __name__ == "__main__":
    main()