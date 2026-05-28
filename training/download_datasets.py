"""
Dataset download script for NetraEdge training.

Downloads and prepares:
1. LFW (Labeled Faces in the Wild) — benchmark dataset
2. IMFDB (Indian Movie Face Database) — Indian demographics
3. CelebA-Spoof — liveness detection training

All data stored in F:\PROJECTS\NetraEdge\training\data\ (NOT C drive)

Usage:
    python download_datasets.py --output-dir F:/PROJECTS/NetraEdge/training/data
"""

import argparse
import os
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import sys


DATASETS = {
    "lfw": {
        "name": "LFW (Labeled Faces in the Wild)",
        "url": "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
        "filename": "lfw-funneled.tgz",
        "type": "tgz",
        "size_mb": 170,
        "description": "13,233 face images of 5,749 people — standard benchmark",
    },
    "imfdb": {
        "name": "IMFDB (Indian Movie Face Database)",
        "url": "https://cdn.iiit.ac.in/datasets/imfdb_raw.zip",
        "filename": "imfdb_raw.zip",
        "type": "zip",
        "size_mb": 500,
        "description": "34,512 faces of 100 Indian actors — diverse demographics",
    },
    "celeba_spoof_real": {
        "name": "CelebA-Spoof (Real faces)",
        "url": None,  # Manual download required
        "filename": None,
        "type": "manual",
        "size_mb": 0,
        "description": "Real face images — download from https://github.com/EndlessSora/FaceAntiSpoofing",
    },
}


def create_directory_structure(base_dir: Path) -> dict[str, Path]:
    """Create the standard directory structure for datasets."""
    dirs = {
        "lfw": base_dir / "lfw",
        "imfdb": base_dir / "imfdb",
        "celeba_spoof": {
            "train": base_dir / "celeba_spoof" / "train",
            "val": base_dir / "celeba_spoof" / "val",
            "test": base_dir / "celeba_spoof" / "test",
        },
        "recognition_train": base_dir / "recognition" / "train",
        "recognition_val": base_dir / "recognition" / "val",
        "liveness_train": base_dir / "liveness" / "train",
        "liveness_val": base_dir / "liveness" / "val",
    }

    for key, path in dirs.items():
        if isinstance(path, dict):
            for subpath in path.values():
                subpath.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    return dirs


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file with progress reporting."""
    print(f"\n  Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urlretrieve(url, str(dest), reporthook=progress)
        print()  # newline after progress
        return True
    except URLError as e:
        print(f"\n  ERROR: {e}")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def extract_tgz(tgz_path: Path, dest_dir: Path) -> bool:
    """Extract a .tgz archive."""
    print(f"  Extracting to {dest_dir}...")
    try:
        with tarfile.open(str(tgz_path), "r:gz") as tar:
            tar.extractall(str(dest_dir))
        print("  Extraction complete.")
        return True
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a .zip archive."""
    print(f"  Extracting to {dest_dir}...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(dest_dir))
        print("  Extraction complete.")
        return True
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return False


def download_lfw(data_dir: Path) -> bool:
    """Download and extract LFW dataset."""
    info = DATASETS["lfw"]
    dest = data_dir / info["filename"]

    if dest.exists():
        print(f"  {info['filename']} already exists, skipping download.")
    else:
        if not info["url"]:
            return False
        if not download_file(info["url"], dest, info["name"]):
            return False

    lfw_dir = data_dir / "lfw"
    if lfw_dir.exists() and any(lfw_dir.iterdir()):
        print("  LFW already extracted, skipping.")
        return True

    return extract_tgz(dest, data_dir)


def download_imfdb(data_dir: Path) -> bool:
    """Download and extract IMFDB dataset."""
    info = DATASETS["imfdb"]
    dest = data_dir / info["filename"]

    if dest.exists():
        print(f"  {info['filename']} already exists, skipping download.")
    else:
        if not info["url"]:
            return False
        if not download_file(info["url"], dest, info["name"]):
            return False

    imdb_dir = data_dir / "imfdb"
    if imdb_dir.exists() and any(imdb_dir.iterdir()):
        print("  IMFDB already extracted, skipping.")
        return True

    return extract_zip(dest, data_dir)


def print_dataset_info() -> None:
    """Print information about available datasets."""
    print("\n" + "=" * 70)
    print("  NetraEdge Training Datasets")
    print("=" * 70)

    for key, info in DATASETS.items():
        print(f"\n  [{key}] {info['name']}")
        print(f"  {info['description']}")
        if info["size_mb"] > 0:
            print(f"  Size: ~{info['size_mb']} MB")
        if info["url"]:
            print(f"  URL: {info['url']}")
        else:
            print("  Download: Manual (see instructions)")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NetraEdge training datasets")
    parser.add_argument(
        "--output-dir",
        default="F:/PROJECTS/NetraEdge/training/data",
        help="Output directory for datasets",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--lfw", action="store_true", help="Download LFW only")
    parser.add_argument("--imfdb", action="store_true", help="Download IMFDB only")
    parser.add_argument("--all", action="store_true", help="Download all auto-downloadable datasets")
    args = parser.parse_args()

    if args.list:
        print_dataset_info()
        return

    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {data_dir}")

    if args.all or args.lfw:
        print("\n--- LFW Dataset ---")
        download_lfw(data_dir)

    if args.all or args.imfdb:
        print("\n--- IMFDB Dataset ---")
        download_imfdb(data_dir)

    if not (args.all or args.lfw or args.imfdb):
        print("\n  No dataset specified. Use --all, --lfw, --imfdb, or --list.")
        print("  Example: python download_datasets.py --all")
        return

    print("\n" + "=" * 70)
    print("  Dataset download complete!")
    print("  Next step: Run training notebooks in Google Colab")
    print("=" * 70)


if __name__ == "__main__":
    main()
