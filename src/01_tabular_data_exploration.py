from __future__ import annotations

"""
01_tabular_data_exploration.py

Shared exploratory analysis for the Rakuten multimodal classification project.
Derived from the original rakuten.py notebook/script and cleaned into a reusable,
script-style workflow.

What this script does
---------------------
- loads X_train, Y_train, X_test
- merges train inputs with labels
- checks dataset structure and missing values
- analyzes class distribution
- verifies image filename linkage
- cleans text columns
- computes character and token statistics
- detects duplicated description blocks
- removes repeated blocks from descriptions
- saves summary tables and figures

Usage
-----
python 01_tabular_data_exploration.py
python 01_tabular_data_exploration.py --data-dir /path/to/data/raw
python 01_tabular_data_exploration.py --output-dir /path/to/outputs/eda

Expected data layout
--------------------
Project root may contain either:
- data/raw/X_train.csv, Y_train.csv, X_test.csv, images/
- or data/X_train.csv, Y_train.csv, X_test.csv, images/
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Logging
# ============================================================


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


# ============================================================
# Paths and loading
# ============================================================


def detect_project_root() -> Path:
    """Assume the script lives in project_root/src or project_root."""
    script_path = Path(__file__).resolve()
    if script_path.parent.name.lower() == "src":
        return script_path.parent.parent
    return script_path.parent



def resolve_data_dir(cli_data_dir: Path | None) -> Path:
    if cli_data_dir is not None:
        return cli_data_dir.resolve()

    project_root = detect_project_root()
    candidates = [project_root / "data" / "raw", project_root / "data"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not auto-detect the data directory. Expected either 'data/raw' or 'data' under "
        f"the project root: {project_root}."
    )



def resolve_output_dir(cli_output_dir: Path | None) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir.resolve()
    return detect_project_root() / "outputs" / "tabular_eda"



def pick_existing_file(base_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of these files were found in {base_dir}: {list(candidates)}"
    )



def build_paths(data_dir: Path) -> dict[str, Path]:
    x_train = pick_existing_file(data_dir, ["X_train.csv", "X_train_update.csv"])
    y_train = pick_existing_file(data_dir, ["Y_train.csv", "Y_train_CVw08PX.csv"])
    x_test = pick_existing_file(data_dir, ["X_test.csv", "X_test_update.csv"])

    image_root_candidates = [
        data_dir / "images",
        data_dir.parent / "images",
    ]
    image_root = next((p for p in image_root_candidates if p.exists()), None)
    if image_root is None:
        raise FileNotFoundError(
            f"Could not find an images folder in {image_root_candidates}"
        )

    image_train = next(
        (p for p in [image_root / "image_train", image_root / "image_training"] if p.exists()),
        None,
    )
    image_test = next(
        (p for p in [image_root / "image_test", image_root / "image_testing"] if p.exists()),
        None,
    )

    if image_train is None:
        raise FileNotFoundError("Could not find image_train or image_training folder.")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "images_root": image_root,
        "image_train": image_train,
        "image_test": image_test,
    }



def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    log("Resolving input files...")
    paths = build_paths(data_dir)
    for key, value in paths.items():
        log(f"  {key}: {value}")

    log("Loading CSV files...")
    x_train = pd.read_csv(paths["x_train"])
    y_train = pd.read_csv(paths["y_train"])
    x_test = pd.read_csv(paths["x_test"])

    merge_key = "Unnamed: 0" if "Unnamed: 0" in x_train.columns else x_train.columns[0]
    log(f"Merging X_train and Y_train on '{merge_key}'...")
    df = pd.merge(x_train, y_train, on=merge_key)

    log(
        f"Loaded shapes -> X_train: {x_train.shape}, Y_train: {y_train.shape}, "
        f"X_test: {x_test.shape}, merged train: {df.shape}"
    )
    return df, x_train, y_train, x_test, paths


# ============================================================
# Text utilities
# ============================================================


def clean_txt_colmn(dtfrme: pd.DataFrame, column: str) -> pd.DataFrame:
    new = column + "_clean"
    dtfrme[new] = dtfrme[column].fillna("")
    dtfrme[new] = dtfrme[new].str.replace(r"<.*?>", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.replace(r"&\w+;", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.lower()
    dtfrme[new] = dtfrme[new].str.replace(r"[^\w\s]", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.replace(r"\s+", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.strip()
    return dtfrme



def outlier_ratio(series: pd.Series) -> float:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()
    return float(outliers / len(series)) if len(series) else 0.0



def has_repeated_block(text: str, min_block_len: int = 100) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_block_len * 2:
        return False
    pattern = rf"(.{{{min_block_len},}})\1+"
    return re.search(pattern, text) is not None



def remove_repeated_blocks(text: str, min_block_len: int = 100) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) < min_block_len * 2:
        return text
    pattern = re.compile(rf"(.{{{min_block_len},}}?)\1+")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"\1", text)
    return text



def box_stats(series: pd.Series) -> dict[str, float]:
    q1 = float(series.quantile(0.25))
    q2 = float(series.quantile(0.50))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return {
        "q1": q1,
        "median": q2,
        "q3": q3,
        "iqr": iqr,
        "lower_whisker": q1 - 1.5 * iqr,
        "upper_whisker": q3 + 1.5 * iqr,
    }



def save_figure(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main analysis
# ============================================================


def run_analysis(data_dir: Path, output_dir: Path) -> None:
    log(f"Using data directory: {data_dir}")
    log(f"Using output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df, x_train, y_train, x_test, paths = load_data(data_dir)

    summary: dict[str, object] = {
        "files": {k: str(v) for k, v in paths.items()},
        "shapes": {
            "X_train": list(x_train.shape),
            "Y_train": list(y_train.shape),
            "X_test": list(x_test.shape),
            "merged_train": list(df.shape),
        },
    }

    log("Saving dataframe info and descriptive statistics...")
    with open(output_dir / "x_train_info.txt", "w", encoding="utf-8") as f:
        x_train.info(buf=f)
    with open(output_dir / "y_train_info.txt", "w", encoding="utf-8") as f:
        y_train.info(buf=f)
    with open(output_dir / "x_test_info.txt", "w", encoding="utf-8") as f:
        x_test.info(buf=f)
    with open(output_dir / "merged_train_info.txt", "w", encoding="utf-8") as f:
        df.info(buf=f)

    x_train.describe(include="all").transpose().to_csv(tables_dir / "x_train_describe.csv")
    df.describe(include="all").transpose().to_csv(tables_dir / "merged_train_describe.csv")

    log("Computing class distribution...")
    class_counts = df["prdtypecode"].value_counts().sort_values(ascending=False)
    class_counts.rename_axis("prdtypecode").reset_index(name="count").to_csv(
        tables_dir / "class_distribution.csv", index=False
    )
    summary["n_classes"] = int(df["prdtypecode"].nunique())
    summary["class_distribution"] = {
        "largest_class": int(class_counts.iloc[0]),
        "smallest_class": int(class_counts.iloc[-1]),
        "top_class": str(class_counts.index[0]),
        "bottom_class": str(class_counts.index[-1]),
    }

    plt.figure(figsize=(10, 5))
    class_counts.head(10).plot(kind="bar")
    plt.title("Top 10 most frequent product categories")
    plt.xlabel("prdtypecode")
    plt.ylabel("Count")
    save_figure(figures_dir, "class_distribution_top10.png")

    plt.figure(figsize=(10, 5))
    class_counts.tail(17).plot(kind="bar")
    plt.title("Least frequent product categories")
    plt.xlabel("prdtypecode")
    plt.ylabel("Count")
    save_figure(figures_dir, "class_distribution_tail.png")

    log("Checking missing values...")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing.rename_axis("column").reset_index(name="missing_count").to_csv(
        tables_dir / "missing_values.csv", index=False
    )
    summary["missing_description_pct"] = float(df["description"].isnull().mean() * 100)

    log("Checking image linkage against training image folder...")
    train_images = set(p.name for p in paths["image_train"].iterdir() if p.is_file())
    expected_filenames = df.apply(
        lambda row: f"image_{row['imageid']}_product_{row['productid']}.jpg", axis=1
    )
    image_exists = expected_filenames.isin(train_images)
    linkage_df = pd.DataFrame({"expected_filename": expected_filenames, "exists": image_exists})
    linkage_df.to_csv(tables_dir / "image_linkage_check.csv", index=False)
    summary["image_linkage"] = {
        "train_images_count": len(train_images),
        "dataset_rows": len(df),
        "matches": int(image_exists.sum()),
        "missing": int((~image_exists).sum()),
    }
    log(
        f"Image linkage complete: {summary['image_linkage']['matches']} matches, "
        f"{summary['image_linkage']['missing']} missing."
    )

    log("Computing raw text length statistics...")
    df["designation_length_chars"] = df["designation"].fillna("").str.len()
    df["description_length_chars"] = df["description"].fillna("").str.len()
    df[["designation_length_chars", "description_length_chars"]].describe().transpose().to_csv(
        tables_dir / "raw_text_length_stats_chars.csv"
    )

    plt.figure(figsize=(8, 5))
    df["designation_length_chars"].hist(bins=50)
    plt.title("Distribution of designation length (characters)")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    save_figure(figures_dir, "designation_length_chars_hist.png")

    log("Cleaning text columns...")
    clean_txt_colmn(df, "designation")
    clean_txt_colmn(df, "description")

    log("Computing cleaned character and token statistics...")
    df["designation_clean_length_chars"] = df["designation_clean"].str.len()
    df["description_clean_length_chars"] = df["description_clean"].str.len()
    df[["designation_clean_length_chars", "description_clean_length_chars"]].describe().transpose().to_csv(
        tables_dir / "clean_text_length_stats_chars.csv"
    )

    df["designation_clean_length_tokens"] = df["designation_clean"].str.split().str.len()
    df["description_clean_length_tokens"] = df["description_clean"].str.split().str.len()
    df[["designation_clean_length_tokens", "description_clean_length_tokens"]].describe().transpose().to_csv(
        tables_dir / "clean_text_length_stats_tokens.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df["designation_clean_length_tokens"].hist(bins=50, ax=axes[0])
    axes[0].set_title("Designation token count")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Frequency")
    df["description_clean_length_tokens"].hist(bins=50, ax=axes[1])
    axes[1].set_title("Description token count")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Frequency")
    save_figure(figures_dir, "token_count_histograms.png")

    log("Computing boxplots and outlier ratios...")
    desc_nonempty = df.loc[df["description_clean"].str.strip() != "", "description_clean_length_tokens"]
    ratio_design = outlier_ratio(df["designation_clean_length_tokens"])
    ratio_descr = outlier_ratio(desc_nonempty)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df["designation_clean_length_tokens"].plot.box(ax=axes[0])
    axes[0].set_title("designation_clean token count")
    axes[0].set_ylabel("Number of tokens")
    axes[0].text(
        0.62,
        0.92,
        f"Outliers: {ratio_design:.2%}",
        transform=axes[0].transAxes,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    desc_nonempty.plot.box(ax=axes[1])
    axes[1].set_title("description_clean token count\n(non-empty only)")
    axes[1].set_ylabel("Number of tokens")
    axes[1].text(
        0.62,
        0.92,
        f"Outliers: {ratio_descr:.2%}",
        transform=axes[1].transAxes,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    save_figure(figures_dir, "token_count_boxplots.png")

    summary["token_length_outliers"] = {
        "designation_ratio": ratio_design,
        "description_nonempty_ratio": ratio_descr,
    }

    log("Detecting repeated description blocks. This can take a while...")
    df["descr_dup_block"] = df["description_clean"].apply(has_repeated_block)
    dup_count = int(df["descr_dup_block"].sum())
    summary["description_repeated_blocks"] = {
        "rows_flagged": dup_count,
        "percentage": float(100 * dup_count / len(df)),
    }
    log(f"Repeated-block detection complete: {dup_count} rows flagged.")

    log("Applying description deduplication...")
    df["description_dedup"] = df["description_clean"].apply(remove_repeated_blocks)
    df["description_dedup_length_tokens"] = df["description_dedup"].str.split().str.len()
    df["removed_chars"] = df["description_clean"].str.len() - df["description_dedup"].str.len()
    df["removed_words"] = (
        df["description_clean"].str.split().str.len() - df["description_dedup"].str.split().str.len()
    )

    changed_rows = int((df["removed_chars"] > 0).sum())
    summary["description_deduplication"] = {
        "rows_changed": changed_rows,
        "percentage_changed": float(100 * changed_rows / len(df)),
    }

    key_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    df.loc[df["removed_chars"] > 0, [
        key_col,
        "productid",
        "imageid",
        "description_clean_length_tokens",
        "description_dedup_length_tokens",
        "removed_words",
        "removed_chars",
    ]].sort_values("removed_chars", ascending=False).to_csv(
        tables_dir / "description_dedup_changes.csv", index=False
    )

    desc_clean = df.loc[df["description_clean"].str.strip() != "", "description_clean_length_tokens"]
    desc_dedup = df.loc[df["description_dedup"].str.strip() != "", "description_dedup_length_tokens"]
    summary["description_box_stats"] = {
        "clean": box_stats(desc_clean),
        "dedup": box_stats(desc_dedup),
    }

    log("Saving cleaned text sample and summary files...")
    sample_cols = [
        c for c in [key_col, "designation", "designation_clean", "description", "description_clean", "description_dedup"]
        if c in df.columns
    ]
    df[sample_cols].head(200).to_csv(tables_dir / "cleaned_text_sample.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"Done. Saved tabular exploration outputs to: {output_dir}")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder containing the CSV files and images folder. If omitted, the script tries data/raw then data relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where analysis outputs will be saved. If omitted, defaults to outputs/tabular_eda under the project root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(data_dir=resolve_data_dir(args.data_dir), output_dir=resolve_output_dir(args.output_dir))
