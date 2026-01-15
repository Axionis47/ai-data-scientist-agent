#!/usr/bin/env python3
"""
Validate casepack manifests and sample file integrity.

This script runs in CI without Kaggle credentials. It verifies:
1. manifest.json exists for each expected casepack
2. All sample files listed in manifest exist
3. SHA256 checksums match

Usage:
    python scripts/validate_casepacks.py
"""

import hashlib
import json
import sys
from pathlib import Path

CASEPACKS_ROOT = Path(__file__).parent.parent / "evals" / "casepacks"

EXPECTED_CASEPACKS = ["ab_easy", "ihdp_medium", "policy_panel_hard"]


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_casepack(name: str) -> list[str]:
    """Validate a single casepack. Returns list of errors."""
    errors = []
    casepack_dir = CASEPACKS_ROOT / name
    manifest_path = casepack_dir / "manifest.json"
    sample_dir = casepack_dir / "sample"

    # Check manifest exists
    if not manifest_path.exists():
        errors.append(f"{name}: manifest.json not found")
        return errors

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"{name}: Invalid JSON in manifest.json: {e}")
        return errors

    # Check required fields
    if "kaggle_slug" not in manifest:
        errors.append(f"{name}: manifest missing 'kaggle_slug'")

    if "sample_files" not in manifest:
        errors.append(f"{name}: manifest missing 'sample_files'")
        return errors

    # Validate sample files
    for sample_info in manifest.get("sample_files", []):
        filename = sample_info.get("filename")
        expected_sha = sample_info.get("sha256")
        expected_size = sample_info.get("size_bytes")

        if not filename:
            errors.append(f"{name}: sample entry missing 'filename'")
            continue

        sample_path = sample_dir / filename

        if not sample_path.exists():
            errors.append(f"{name}: sample file not found: {filename}")
            continue

        # Verify checksum
        if expected_sha:
            actual_sha = sha256_file(sample_path)
            if actual_sha != expected_sha:
                errors.append(
                    f"{name}: SHA256 mismatch for {filename}\n"
                    f"  Expected: {expected_sha}\n"
                    f"  Actual:   {actual_sha}"
                )

        # Verify size
        if expected_size:
            actual_size = sample_path.stat().st_size
            if actual_size != expected_size:
                errors.append(
                    f"{name}: Size mismatch for {filename}\n"
                    f"  Expected: {expected_size} bytes\n"
                    f"  Actual:   {actual_size} bytes"
                )

    return errors


def main():
    print("Validating casepacks...")
    print(f"Root: {CASEPACKS_ROOT}")
    print()

    all_errors = []

    for name in EXPECTED_CASEPACKS:
        casepack_dir = CASEPACKS_ROOT / name
        if not casepack_dir.exists():
            print(f"⚠️  {name}: directory not found (skipping)")
            continue

        errors = validate_casepack(name)
        if errors:
            print(f"❌ {name}: {len(errors)} error(s)")
            all_errors.extend(errors)
        else:
            manifest_path = casepack_dir / "manifest.json"
            if manifest_path.exists():
                print(f"✅ {name}: valid")
            else:
                print(f"⚠️  {name}: no manifest yet (pending bootstrap)")

    print()

    if all_errors:
        print("=" * 60)
        print("ERRORS:")
        for error in all_errors:
            print(f"  • {error}")
        print("=" * 60)
        return 1

    print("All casepacks valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

