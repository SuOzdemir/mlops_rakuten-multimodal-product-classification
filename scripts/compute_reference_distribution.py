"""
Computes the training set's class distribution and writes it to
src/api/reference_class_distribution.json, baked into the api image so the
serving container has a drift baseline without needing raw training data
mounted.

Keyed by prdtypecode (the raw Rakuten product-type code), not label_id --
label_id is an arbitrary index assigned by whichever label2id.json happened
to be loaded, and the serving predictor loads its own copy from the asset
dir, not this one. prdtypecode is the stable identifier both sides agree on.

Run after `dvc repro prepare_splits` (or whenever train_split.csv changes):
    uv run python scripts/compute_reference_distribution.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "outputs" / "image_modeling"
OUT_PATH = PROJECT_DIR / "src" / "api" / "reference_class_distribution.json"


def main() -> None:
    train_df = pd.read_csv(SPLIT_DIR / "train_split.csv")
    counts = train_df["prdtypecode"].value_counts().sort_index()
    total = int(counts.sum())
    distribution = {str(int(k)): int(v) / total for k, v in counts.items()}

    OUT_PATH.write_text(json.dumps(distribution, indent=2) + "\n")
    print(f"Wrote {len(distribution)}-class reference distribution ({total:,} rows) to {OUT_PATH}")


if __name__ == "__main__":
    main()
