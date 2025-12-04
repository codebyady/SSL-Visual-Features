import os
import urllib.request
import zipfile
import shutil

from huggingface_hub import HfFileSystem

# Where to put intermediate + final data
ZIP_DIR = "data/hf_zips"
# We no longer need EXTRACT_DIR
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


def gather_and_rename_streaming():
    """
    Stream images directly from ZIPs in ZIP_DIR into sharded OUTPUT_DIR
    without extracting everything to disk first.
    """
    print("\n=== Streaming images from ZIPs into sharded data/pretrain ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Optional: figure out how many images we already wrote
    # so we can resume safely.
    counter = 0
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            name, ext = os.path.splitext(f)
            if name.isdigit():
                counter = max(counter, int(name) + 1)

    print(f"Starting from image index: {counter}")

    # Get all .zip files
    if not os.path.exists(ZIP_DIR):
        raise RuntimeError(f"{ZIP_DIR} does not exist; nothing to stream from.")
    zip_files = sorted(f for f in os.listdir(ZIP_DIR) if f.endswith(".zip"))
    if not zip_files:
        raise RuntimeError(f"No .zip files found in {ZIP_DIR}")

    for fname in zip_files:
        zip_path = os.path.join(ZIP_DIR, fname)
        print(f"\n  Processing zip: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            for info in z.infolist():
                # Skip directories
                if info.is_dir():
                    continue

                # Only images
                ext = os.path.splitext(info.filename)[1].lower()
                if ext not in (".png", ".jpg", ".jpeg"):
                    continue

                # Decide shard for this image
                shard_idx = counter // IMAGES_PER_DIR
                shard_dir = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:03d}")
                os.makedirs(shard_dir, exist_ok=True)

                dst_path = os.path.join(shard_dir, f"{counter:07d}{ext}")

                # If resuming and file already exists, skip
                if os.path.exists(dst_path):
                    counter += 1
                    continue

                # Copy file from zip member to destination
                with z.open(info, "r") as src_f, open(dst_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

                counter += 1
                if counter % 1000 == 0:
                    print(f"  Copied {counter} images...", end="\r")

    print(f"\nDone. Total images: {counter}")
    print(f"Pretrain shards in: {OUTPUT_DIR}")


def cleanup_intermediate_dirs():
    """Delete hf_zips safely."""
    print("\n=== Cleaning up intermediate directories ===")

    # We only have ZIP_DIR now
    for path in [ZIP_DIR]:
        if os.path.exists(path):
            print(f"  Removing {path} ...")
            shutil.rmtree(path)
        else:
            print(f"  {path} does not exist, skipping")

    print("Cleanup complete.")


def main():
    download_all_zips()
    gather_and_rename_streaming()
    cleanup_intermediate_dirs()


if __name__ == "__main__":
    main()