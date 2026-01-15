# Dataset Casepacks

This document describes the casepack fixture system for causal inference testing datasets.

## Overview

Casepacks are standardized dataset fixtures used for evaluating and testing the causal inference pipeline. Each casepack contains:

- **raw/** — Full dataset files (git-ignored, downloaded on demand)
- **sample/** — Deterministic samples for CI testing (committed)
- **kaggle_meta/** — Kaggle metadata (committed)
- **manifest.json** — Dataset manifest with checksums (committed)

## Available Casepacks

| Name | Kaggle Slug | Description |
|------|-------------|-------------|
| `ab_easy` | amirmotefaker/ab-testing-dataset | A/B testing dataset for easy causal inference |
| `ihdp_medium` | konradb/ihdp-data | Infant Health and Development Program dataset |
| `policy_panel_hard` | mexwell/correlates-of-state-policy-dataset | State policy correlates panel dataset |

## Kaggle Credentials Setup

The bootstrap script requires Kaggle API credentials. Choose one of these options:

### Option A: Environment Variables (Recommended for CI)

```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_KEY='your_api_key'
```

### Option B: Credentials File (Recommended for Local Development)

1. Create the credentials file:
   ```bash
   mkdir -p ~/.kaggle
   echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
   ```

2. Set correct permissions (required by Kaggle):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Get your API key from: https://www.kaggle.com/settings

**Security Note:** Never commit your `kaggle.json` file. It is already in `.gitignore`.

## Commands

### Bootstrap All Datasets

```bash
python scripts/bootstrap_kaggle_casepacks.py --all
```

### Bootstrap a Specific Dataset

```bash
python scripts/bootstrap_kaggle_casepacks.py --dataset ab_easy
python scripts/bootstrap_kaggle_casepacks.py --dataset ihdp_medium
python scripts/bootstrap_kaggle_casepacks.py --dataset policy_panel_hard
```

### Force Re-download

```bash
python scripts/bootstrap_kaggle_casepacks.py --dataset ab_easy --force
```

### List Available Datasets

```bash
python scripts/bootstrap_kaggle_casepacks.py --list
```

### Skip Sample Generation

```bash
python scripts/bootstrap_kaggle_casepacks.py --all --skip-sample
```

## Sampling Strategy

Samples are created deterministically to ensure reproducibility:

1. **Sample Size**: min(2000 rows, 1% of dataset capped at 5000 rows)
2. **Method**: 
   - If dataset ≤ 2000 rows: full copy
   - If dataset > 2000 rows: take first 2000 rows (head)
   - For random sampling: use seed 1337
3. **Format**: CSV files are sampled and written as CSV

Non-CSV files (e.g., .npz, .parquet) require manual conversion if needed.

## Manifest Format

Each casepack includes a `manifest.json`:

```json
{
  "kaggle_slug": "amirmotefaker/ab-testing-dataset",
  "description": "A/B testing dataset for easy causal inference",
  "created_utc": "2024-01-15T10:30:00+00:00",
  "raw_files": ["ab_data.csv"],
  "sample_files": [
    {
      "filename": "ab_data.csv",
      "rows": 2000,
      "sha256": "abc123...",
      "size_bytes": 45678,
      "sampling_method": "head_2000",
      "source_rows": 294478
    }
  ]
}
```

## What Gets Committed

| Path | Committed | Description |
|------|-----------|-------------|
| `evals/casepacks/*/raw/` | ❌ No | Full datasets (git-ignored) |
| `evals/casepacks/*/*.zip` | ❌ No | Downloaded archives (git-ignored) |
| `evals/casepacks/*/sample/` | ✅ Yes | Deterministic samples |
| `evals/casepacks/*/kaggle_meta/` | ✅ Yes | Kaggle metadata |
| `evals/casepacks/*/manifest.json` | ✅ Yes | Dataset manifest |

## CI Integration

The CI pipeline validates casepacks without requiring Kaggle credentials:

1. Verifies `manifest.json` exists for each casepack
2. Validates SHA256 checksums of sample files match manifest
3. Does not download or access raw data

This ensures committed samples are consistent and uncorrupted.

## Changing Sample Size

To modify sampling parameters, edit `scripts/bootstrap_kaggle_casepacks.py`:

```python
SAMPLE_SEED = 1337        # Random seed for reproducibility
MAX_SAMPLE_ROWS = 2000    # Default sample size
MAX_SAMPLE_PERCENT = 0.01 # 1% of dataset
MAX_SAMPLE_CAP = 5000     # Maximum rows if using percentage
```

After changing, re-run the bootstrap with `--force` and commit updated samples.

