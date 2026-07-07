"""
prepare_splits.py

Builds the train/val split files consumed by both training scripts:
  - models/Model_I12_ConvNeXt_Base_ModerateAug_Full/train.py
  - models/textModeling/Model_T8_CamemBERT_FullFineTune_L128/train.py

Both scripts read from the same `outputs/image_modeling/` directory
(train_split.csv, val_split.csv, label2id.json) since every row has both
an image and text, and image/text models must be evaluated on the same
held-out products.

Input : data/raw/X_train*.csv, Y_train*.csv (Kaggle Rakuten dataset,
        downloaded via scripts/setup_data.sh)
Output: outputs/image_modeling/{train_split,val_split}.csv, label2id.json

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --data-dir data/raw --output-dir outputs/image_modeling
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 0.15


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def pick_existing_file(base_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files were found in {base_dir}: {list(candidates)}")


def resolve_paths(data_dir: Path) -> dict[str, Path]:
    x_train = pick_existing_file(data_dir, ["X_train.csv", "X_train_update.csv"])
    y_train = pick_existing_file(data_dir, ["Y_train.csv", "Y_train_CVw08PX.csv"])

    image_root_candidates = [data_dir / "images", data_dir.parent / "images"]
    image_root = next((p for p in image_root_candidates if p.exists()), None)
    if image_root is None:
        raise FileNotFoundError(f"Could not find an images folder in {image_root_candidates}")

    image_train = next(
        (p for p in [image_root / "image_train", image_root / "image_training"] if p.exists()),
        None,
    )
    if image_train is None:
        raise FileNotFoundError("Could not find image_train or image_training folder.")

    return {"x_train": x_train, "y_train": y_train, "image_train": image_train}


def build_dataset(data_dir: Path) -> pd.DataFrame:
    paths = resolve_paths(data_dir)
    x_train = pd.read_csv(paths["x_train"])
    y_train = pd.read_csv(paths["y_train"])

    merge_key = "Unnamed: 0" if "Unnamed: 0" in x_train.columns else x_train.columns[0]
    df = pd.merge(x_train, y_train, on=merge_key)

    df["image_path_local"] = df.apply(
        lambda row: str(paths["image_train"] / f"image_{row['imageid']}_product_{row['productid']}.jpg"),
        axis=1,
    )
    return df


def build_label2id(df: pd.DataFrame) -> dict[int, int]:
    classes = sorted(df["prdtypecode"].unique())
    return {int(code): idx for idx, code in enumerate(classes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "outputs" / "image_modeling")
    args = parser.parse_args()

    df = build_dataset(args.data_dir)
    label2id = build_label2id(df)
    df["label_id"] = df["prdtypecode"].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["label_id"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["productid", "imageid", "designation", "description", "image_path_local", "prdtypecode", "label_id"]
    train_df[columns].to_csv(args.output_dir / "train_split.csv", index=False)
    val_df[columns].to_csv(args.output_dir / "val_split.csv", index=False)
    with open(args.output_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    print(f"Wrote {len(train_df)} train / {len(val_df)} val rows, {len(label2id)} classes -> {args.output_dir}")


if __name__ == "__main__":
    main()
