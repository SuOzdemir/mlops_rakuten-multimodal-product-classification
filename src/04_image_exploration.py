from __future__ import annotations

"""
04_image_exploration.py

Image-focused exploratory analysis for the Rakuten multimodal classification project.
This script extracts the image-specific exploration that was mixed into rakuten.py and
turns it into a reusable project script.

What this script does
---------------------
- loads the train metadata
- verifies image linkage from productid + imageid
- counts training and test images
- inspects image width/height/channels for a sample or full set
- saves class sample images for selected categories
- saves random gallery images
- reports missing/corrupt images

Usage
-----
python 04_image_exploration.py
python 04_image_exploration.py --data-dir /path/to/data/raw
python 04_image_exploration.py --inspect-all

Notes
-----
- By default, image size inspection is sampled for speed.
- You can use --inspect-all to scan every training image.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


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
    return detect_project_root() / "outputs" / "image_eda"



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

    image_root_candidates = [data_dir / "images", data_dir.parent / "images"]
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



def load_train_dataframe(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    log("Resolving input files...")
    paths = build_paths(data_dir)
    for key, value in paths.items():
        log(f"  {key}: {value}")

    log("Loading train metadata CSV files...")
    x_train = pd.read_csv(paths["x_train"])
    y_train = pd.read_csv(paths["y_train"])
    merge_key = "Unnamed: 0" if "Unnamed: 0" in x_train.columns else x_train.columns[0]
    log(f"Merging X_train and Y_train on '{merge_key}'...")
    df = pd.merge(x_train, y_train, on=merge_key)
    log(f"Merged train dataframe shape: {df.shape}")
    return df, paths


# ============================================================
# Helpers
# ============================================================


def build_image_filename(imageid: int, productid: int) -> str:
    return f"image_{imageid}_product_{productid}.jpg"



def save_figure(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()



def inspect_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            arr = np.array(img)
            channels = 1 if arr.ndim == 2 else arr.shape[2]
            return {
                "path": str(path),
                "filename": path.name,
                "ok": True,
                "width": int(width),
                "height": int(height),
                "mode": img.mode,
                "channels": int(channels),
                "error": None,
            }
    except (UnidentifiedImageError, OSError) as exc:
        return {
            "path": str(path),
            "filename": path.name,
            "ok": False,
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "error": str(exc),
        }



def make_gallery(image_paths: list[Path], titles: list[str], output_path: Path, cols: int = 4) -> None:
    if not image_paths:
        return
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx, (img_path, title) in enumerate(zip(image_paths, titles)):
        try:
            with Image.open(img_path) as img:
                axes[idx].imshow(img)
                axes[idx].set_title(title, fontsize=9)
                axes[idx].axis("off")
        except (UnidentifiedImageError, OSError):
            axes[idx].text(0.5, 0.5, f"Unreadable\n{img_path.name}", ha="center", va="center")
            axes[idx].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main analysis
# ============================================================


def run_analysis(
    data_dir: Path,
    output_dir: Path,
    inspect_all: bool,
    inspect_sample_size: int,
    random_seed: int,
    sample_categories: list[int],
    images_per_category: int,
) -> None:
    random.seed(random_seed)

    log(f"Using data directory: {data_dir}")
    log(f"Using output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df, paths = load_train_dataframe(data_dir)
    image_train_dir = paths["image_train"]
    image_test_dir = paths["image_test"]

    summary: dict[str, object] = {
        "files": {k: str(v) for k, v in paths.items()},
        "merged_train_shape": list(df.shape),
    }

    log("Counting image files in train and test folders...")
    train_files = sorted([p for p in image_train_dir.iterdir() if p.is_file()])
    test_files = sorted([p for p in image_test_dir.iterdir() if p.is_file()]) if image_test_dir else []
    summary["image_counts"] = {
        "train_folder": len(train_files),
        "test_folder": len(test_files),
    }
    log(f"Found {len(train_files)} training images and {len(test_files)} test images.")

    log("Checking filename linkage between metadata and training image folder...")
    df["expected_filename"] = df.apply(
        lambda row: build_image_filename(row["imageid"], row["productid"]), axis=1
    )
    train_name_set = {p.name for p in train_files}
    df["image_exists"] = df["expected_filename"].isin(train_name_set)
    key_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    df[[key_col, "productid", "imageid", "prdtypecode", "expected_filename", "image_exists"]].to_csv(
        tables_dir / "image_filename_linkage.csv", index=False
    )
    summary["linkage"] = {
        "matches": int(df["image_exists"].sum()),
        "missing": int((~df["image_exists"]).sum()),
    }
    log(
        f"Image linkage complete: {summary['linkage']['matches']} matches, "
        f"{summary['linkage']['missing']} missing."
    )

    log("Saving random training gallery...")
    gallery_paths = random.sample(train_files, min(12, len(train_files)))
    make_gallery(
        gallery_paths,
        [p.name for p in gallery_paths],
        figures_dir / "random_training_gallery.png",
        cols=4,
    )

    log("Saving category example galleries...")
    for code in sample_categories:
        log(f"  Category {code}...")
        subset = df.loc[df["prdtypecode"] == code].head(images_per_category).copy()
        if subset.empty:
            log(f"    No rows found for category {code}; skipping.")
            continue
        paths_for_cat = []
        titles = []
        for _, row in subset.iterrows():
            img_path = image_train_dir / row["expected_filename"]
            if img_path.exists():
                paths_for_cat.append(img_path)
                titles.append(f"{code} | {row['productid']}")
        make_gallery(
            paths_for_cat,
            titles,
            figures_dir / f"category_{code}_examples.png",
            cols=min(4, max(1, len(paths_for_cat))),
        )

    if inspect_all:
        inspect_paths = train_files
        log(f"Inspecting all training images: {len(inspect_paths)} files. This may take a while...")
    else:
        inspect_paths = random.sample(train_files, min(inspect_sample_size, len(train_files)))
        log(f"Inspecting a sample of {len(inspect_paths)} training images...")

    inspected: list[dict[str, object]] = []
    checkpoint = max(250, len(inspect_paths) // 10) if inspect_paths else 250
    for idx, path in enumerate(inspect_paths, start=1):
        inspected.append(inspect_image(path))
        if idx % checkpoint == 0 or idx == len(inspect_paths):
            log(f"  Inspected {idx}/{len(inspect_paths)} images...")

    inspect_df = pd.DataFrame(inspected)
    inspect_df.to_csv(tables_dir / "image_properties.csv", index=False)

    ok_df = inspect_df.loc[inspect_df["ok"]].copy()
    bad_df = inspect_df.loc[~inspect_df["ok"]].copy()
    bad_df.to_csv(tables_dir / "corrupt_or_unreadable_images.csv", index=False)

    log(f"Image inspection complete: {len(ok_df)} readable, {len(bad_df)} unreadable/corrupt.")

    if not ok_df.empty:
        log("Saving image property summaries and plots...")
        ok_df[["width", "height", "channels"]].describe().transpose().to_csv(
            tables_dir / "image_properties_describe.csv"
        )
        mode_counts = ok_df["mode"].value_counts().rename_axis("mode").reset_index(name="count")
        mode_counts.to_csv(tables_dir / "image_mode_counts.csv", index=False)

        summary["image_properties"] = {
            "inspected_images": int(len(inspect_df)),
            "ok_images": int(len(ok_df)),
            "bad_images": int(len(bad_df)),
            "width": {
                "min": int(ok_df["width"].min()),
                "median": int(ok_df["width"].median()),
                "max": int(ok_df["width"].max()),
            },
            "height": {
                "min": int(ok_df["height"].min()),
                "median": int(ok_df["height"].median()),
                "max": int(ok_df["height"].max()),
            },
            "modes": mode_counts.set_index("mode")["count"].to_dict(),
        }

        plt.figure(figsize=(8, 5))
        ok_df["width"].hist(bins=40)
        plt.title("Image width distribution")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Frequency")
        save_figure(figures_dir, "image_width_hist.png")

        plt.figure(figsize=(8, 5))
        ok_df["height"].hist(bins=40)
        plt.title("Image height distribution")
        plt.xlabel("Height (pixels)")
        plt.ylabel("Frequency")
        save_figure(figures_dir, "image_height_hist.png")

        plt.figure(figsize=(8, 6))
        plt.scatter(ok_df["width"], ok_df["height"], alpha=0.4, s=10)
        plt.title("Image width vs height")
        plt.xlabel("Width")
        plt.ylabel("Height")
        save_figure(figures_dir, "image_width_vs_height_scatter.png")

        ok_df["aspect_ratio"] = ok_df["width"] / ok_df["height"]
        ok_df["aspect_ratio"].describe().to_csv(tables_dir / "aspect_ratio_describe.csv")

    log("Saving class frequency table for image branch...")
    class_counts = df["prdtypecode"].value_counts().rename_axis("prdtypecode").reset_index(name="count")
    class_counts.to_csv(tables_dir / "image_class_distribution.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"Done. Saved image exploration outputs to: {output_dir}")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder containing CSV files and the images folder. If omitted, the script tries data/raw then data relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where analysis outputs will be saved. If omitted, defaults to outputs/image_eda under the project root.",
    )
    parser.add_argument(
        "--inspect-all",
        action="store_true",
        help="Inspect every training image instead of a sample.",
    )
    parser.add_argument(
        "--inspect-sample-size",
        type=int,
        default=3000,
        help="How many training images to inspect when not using --inspect-all.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--sample-categories",
        type=int,
        nargs="*",
        default=[2583, 1560, 1180],
        help="Category codes for example image galleries.",
    )
    parser.add_argument(
        "--images-per-category",
        type=int,
        default=8,
        help="How many example images to save per selected category.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        data_dir=resolve_data_dir(args.data_dir),
        output_dir=resolve_output_dir(args.output_dir),
        inspect_all=args.inspect_all,
        inspect_sample_size=args.inspect_sample_size,
        random_seed=args.random_seed,
        sample_categories=args.sample_categories,
        images_per_category=args.images_per_category,
    )
