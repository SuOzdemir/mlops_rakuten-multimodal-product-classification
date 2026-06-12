#!/usr/bin/env bash

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_FOLDER="${PROJECT_ROOT}/data"
DOWNLOAD_FOLDER="${DATA_FOLDER}/_downloads"
RAW_FOLDER="${DATA_FOLDER}/raw"
IMAGES_FOLDER="${RAW_FOLDER}/images"


create_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        echo "📁 Created directory: $dir"
    fi
}

ensure_kaggle_cli() {
    if ! command -v kaggle &> /dev/null; then
        echo "🔍 Kaggle CLI not found. Initiating installation..."

        if command -v uv &> /dev/null; then
            echo "🚀 Installing kaggle via 'uv'..."
            uv pip install kaggle --system
        else
            echo "📦 Installing kaggle via 'pip'..."
            pip install --upgrade --no-cache-dir kaggle
        fi

        if ! command -v kaggle &> /dev/null; then
            echo "❌ Installation failed or PATH not updated. Please install kaggle manually."
            exit 1
        fi
    fi
}

download_kaggle() {
    local slug="$1"      # Kaggle dataset identifier (user/dataset-name)
    local zip_name="$2"   # Expected filename to check for existence

    # Ensure dependencies are met before proceeding
    ensure_kaggle_cli

    # Check for existing download to avoid redundant bandwidth usage (Idempotency)
    if [[ -f "${DOWNLOAD_FOLDER}/${zip_name}" ]]; then
        echo "⏭️  Already downloaded: $zip_name"
    else
        echo "📥 Downloading dataset: $slug..."
        # Download command with error handling
        kaggle datasets download -d "$slug" -p "${DOWNLOAD_FOLDER}" || {
            echo "❌ Download failed!";
            exit 1;
        }
    fi
}

extract_zip() {
    local zip_path="$1"
    echo "📦 Extracting $(basename "$zip_path")..."
    unzip -uo "$zip_path" -d "${DOWNLOAD_FOLDER}" > /dev/null || { echo "❌ Extraction failed!"; exit 1; }
    echo "✅ Extraction complete."
}


sync_file() {
    local filename="$1"
    local target="${RAW_FOLDER}/${filename}"

    if [[ -f "$target" ]]; then
        echo "⏭️  File already in raw: $filename"
    else
        local src
        src=$(find "${DOWNLOAD_FOLDER}" -type f -iname "$filename" | head -n 1)
        if [[ -n "$src" ]]; then
            cp -f "$src" "$target"
            echo "✅ Moved $filename to raw."
        else
            echo "⚠️  Warning: $filename not found!"
        fi
    fi
}


sync_images() {
    local dir_name="$1"
    local target_dir="${IMAGES_FOLDER}/${dir_name}"
    local src_dir

    src_dir=$(find "${DOWNLOAD_FOLDER}" -type d -iname "$dir_name" | head -n 1)

    if [[ -n "$src_dir" ]]; then
        echo "🔄 Syncing $dir_name (rsync)..."
        create_dir "$target_dir"
        # --ignore-existing
        rsync -a --ignore-existing "${src_dir}/" "${target_dir}/"
        echo "✅ $dir_name synchronized."
    else
        echo "⚠️  Warning: Source folder $dir_name not found!"
    fi
}


main() {
    echo "🚀 Starting functional data setup..."

    create_dir "${DOWNLOAD_FOLDER}"
    create_dir "${RAW_FOLDER}"
    create_dir "${IMAGES_FOLDER}"
    create_dir "${DATA_FOLDER}/processed"

    download_kaggle "arturillenseer/rakuten-product-images-ml" "rakuten-product-images-ml.zip"
    download_kaggle "arturillenseer/csv-files" "csv-files.zip"

    for zip in "${DOWNLOAD_FOLDER}"/*.zip; do
        [[ -e "$zip" ]] && extract_zip "$zip"
    done

    for csv in "X_train.csv" "Y_train.csv" "X_test.csv"; do
        sync_file "$csv"
    done

    sync_images "image_train"
    sync_images "image_test"

    echo -e "\n📊 Current Data Structure:"
    find "${DATA_FOLDER}" -maxdepth 2 -not -path '*/.*' | sort

    echo -e "\n✨ All steps completed successfully!"
}

main