#!/usr/bin/env python3
"""
Bootstrap Kaggle datasets for casepack fixtures.

Downloads datasets from Kaggle, extracts them, and creates deterministic samples
with manifest files for reproducible testing.

Usage:
    python scripts/bootstrap_kaggle_casepacks.py --all
    python scripts/bootstrap_kaggle_casepacks.py --dataset ab_easy
    python scripts/bootstrap_kaggle_casepacks.py --dataset ihdp_medium --force

Credentials:
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or place
    credentials in ~/.kaggle/kaggle.json with chmod 600 permissions.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

# Dataset configurations
DATASETS = {
    "ab_easy": {
        "kaggle_slug": "amirmotefaker/ab-testing-dataset",
        "description": "A/B testing dataset for easy causal inference",
    },
    "ihdp_medium": {
        "kaggle_slug": "konradb/ihdp-data",
        "description": "Infant Health and Development Program dataset",
    },
    "policy_panel_hard": {
        "kaggle_slug": "mexwell/correlates-of-state-policy-dataset",
        "description": "State policy correlates panel dataset",
    },
}

# Sampling configuration
SAMPLE_SEED = 1337
MAX_SAMPLE_ROWS = 2000
MAX_SAMPLE_PERCENT = 0.01  # 1%
MAX_SAMPLE_CAP = 5000

CASEPACKS_ROOT = Path(__file__).parent.parent / "evals" / "casepacks"


def get_kaggle_credentials() -> tuple[str, str]:
    """Get Kaggle credentials from environment or kaggle.json file."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if username and key:
        return username, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                creds = json.load(f)
            return creds.get("username", ""), creds.get("key", "")
        except (json.JSONDecodeError, KeyError):
            pass

    return "", ""


def check_kaggle_credentials() -> bool:
    """Verify Kaggle credentials are available."""
    username, key = get_kaggle_credentials()
    if not username or not key:
        print("ERROR: Kaggle credentials not found.", file=sys.stderr)
        print("\nTo set up credentials, use one of these options:", file=sys.stderr)
        print("\nOption A - Environment variables:", file=sys.stderr)
        print("  export KAGGLE_USERNAME='your_username'", file=sys.stderr)
        print("  export KAGGLE_KEY='your_key'", file=sys.stderr)
        print("\nOption B - Credentials file:", file=sys.stderr)
        print("  1. Create ~/.kaggle/kaggle.json with:", file=sys.stderr)
        print('     {"username":"your_username","key":"your_key"}', file=sys.stderr)
        print("  2. Set permissions: chmod 600 ~/.kaggle/kaggle.json", file=sys.stderr)
        print("\nGet your API key from: https://www.kaggle.com/settings", file=sys.stderr)
        return False
    return True


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(kaggle_slug: str, dest_dir: Path) -> bool:
    """Download and extract a Kaggle dataset using Kaggle API."""
    username, key = get_kaggle_credentials()

    print(f"  Downloading {kaggle_slug}...")
    try:
        # Set credentials for Kaggle API
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

        # Import and use Kaggle API directly
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        api.dataset_download_files(kaggle_slug, path=str(dest_dir), unzip=True)
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download dataset: {e}", file=sys.stderr)
        return False


def get_kaggle_metadata(kaggle_slug: str, dest_dir: Path) -> dict:
    """Fetch or create metadata for a Kaggle dataset."""
    username, key = get_kaggle_credentials()
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    metadata = {
        "kaggle_slug": kaggle_slug,
        "downloaded_utc": datetime.now(UTC).isoformat(),
    }

    # Try to get metadata from Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_metadata(kaggle_slug, path=str(dest_dir))
        meta_file = dest_dir / "dataset-metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                kaggle_meta = json.load(f)
            metadata["title"] = kaggle_meta.get("title", "")
            metadata["subtitle"] = kaggle_meta.get("subtitle", "")
    except Exception:
        pass  # Use minimal metadata if Kaggle metadata fails

    return metadata


def create_sample_csv(src_path: Path, dest_path: Path) -> dict:
    """Create a deterministic sample from a CSV file."""
    import csv

    # Count total rows (excluding header)
    with open(src_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        total_rows = sum(1 for _ in reader)

    # Calculate sample size
    percent_sample = int(total_rows * MAX_SAMPLE_PERCENT)
    sample_size = min(MAX_SAMPLE_ROWS, percent_sample, MAX_SAMPLE_CAP)
    sample_size = max(sample_size, min(total_rows, MAX_SAMPLE_ROWS))  # At least head N

    # Determine sampling method
    if sample_size >= total_rows:
        sampling_method = "full_copy"
        sample_size = total_rows
    elif sample_size == MAX_SAMPLE_ROWS and total_rows > MAX_SAMPLE_ROWS:
        sampling_method = f"head_{MAX_SAMPLE_ROWS}"
    else:
        sampling_method = f"sample_seed_{SAMPLE_SEED}"

    # Create sample
    with open(src_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)

        if sampling_method.startswith("head_"):
            # Take first N rows
            rows = []
            for i, row in enumerate(reader):
                if i >= sample_size:
                    break
                rows.append(row)
        elif sampling_method.startswith("sample_"):
            # Random sample with seed
            import random
            all_rows = list(reader)
            random.seed(SAMPLE_SEED)
            rows = random.sample(all_rows, min(sample_size, len(all_rows)))
        else:
            rows = list(reader)

    # Write sample
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return {
        "filename": dest_path.name,
        "rows": len(rows),
        "sha256": sha256_file(dest_path),
        "size_bytes": dest_path.stat().st_size,
        "sampling_method": sampling_method,
        "source_rows": total_rows,
    }


def process_dataset(name: str, config: dict, force: bool = False, skip_sample: bool = False) -> bool:
    """Process a single dataset: download, extract, sample, create manifest."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    dataset_dir = CASEPACKS_ROOT / name
    raw_dir = dataset_dir / "raw"
    sample_dir = dataset_dir / "sample"
    meta_dir = dataset_dir / "kaggle_meta"
    manifest_path = dataset_dir / "manifest.json"

    kaggle_slug = config["kaggle_slug"]

    # Check if raw exists
    raw_exists = raw_dir.exists() and any(raw_dir.iterdir())

    if raw_exists and not force:
        print("  Raw data exists. Use --force to re-download.")
    else:
        # Download dataset
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        if not download_dataset(kaggle_slug, raw_dir):
            return False

    # Get metadata
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata = get_kaggle_metadata(kaggle_slug, meta_dir)
    with open(meta_dir / "kaggle_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if skip_sample:
        print("  Skipping sample generation (--skip-sample)")
        return True

    # List raw files
    raw_files = []
    for p in raw_dir.rglob("*"):
        if p.is_file():
            raw_files.append(p.relative_to(raw_dir).as_posix())

    print(f"  Found {len(raw_files)} raw file(s)")

    # Create samples
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing samples
    for f in sample_dir.glob("*"):
        if f.name != ".gitkeep":
            f.unlink()

    sample_files = []
    for raw_file in raw_files:
        raw_path = raw_dir / raw_file
        if raw_path.suffix.lower() == ".csv":
            sample_path = sample_dir / raw_path.name
            print(f"  Sampling: {raw_file}")
            sample_info = create_sample_csv(raw_path, sample_path)
            sample_files.append(sample_info)
            print(f"    -> {sample_info['rows']} rows ({sample_info['sampling_method']})")
        elif raw_path.suffix.lower() in (".npz", ".parquet", ".npy"):
            print(f"  Skipping non-CSV: {raw_file} (manual conversion may be needed)")

    # Create manifest
    manifest = {
        "kaggle_slug": kaggle_slug,
        "description": config["description"],
        "created_utc": datetime.now(UTC).isoformat(),
        "raw_files": raw_files,
        "sample_files": sample_files,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Manifest written: {manifest_path.relative_to(CASEPACKS_ROOT.parent.parent)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Kaggle datasets for casepack fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all datasets"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()), help="Process a specific dataset"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if raw exists"
    )
    parser.add_argument(
        "--skip-sample", action="store_true", help="Skip sample generation"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available datasets"
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, config in DATASETS.items():
            print(f"  {name}: {config['kaggle_slug']}")
            print(f"    {config['description']}")
        return 0

    if not args.all and not args.dataset:
        parser.print_help()
        print("\nError: Specify --all or --dataset", file=sys.stderr)
        return 1

    if not check_kaggle_credentials():
        return 1

    # Ensure casepacks root exists
    CASEPACKS_ROOT.mkdir(parents=True, exist_ok=True)

    datasets_to_process = DATASETS if args.all else {args.dataset: DATASETS[args.dataset]}
    success_count = 0
    fail_count = 0

    for name, config in datasets_to_process.items():
        if process_dataset(name, config, force=args.force, skip_sample=args.skip_sample):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Complete: {success_count} succeeded, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

