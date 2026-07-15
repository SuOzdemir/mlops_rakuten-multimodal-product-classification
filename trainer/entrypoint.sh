#!/bin/bash
# Ephemeral training entrypoint: reproduces prepare_splits (if stale) then
# the requested model's train stage, both through DVC (records the new
# model weight's hash in dvc.lock). --force --single-item: a manual retrain
# should always actually retrain, and --single-item keeps that forcing
# scoped to the train stage, not the whole upstream pipeline.
set -euo pipefail

: "${MODEL:?Set MODEL=image or MODEL=text}"

case "$MODEL" in
  image) STAGE=train_image ;;
  text)  STAGE=train_text ;;
  *) echo "Unknown MODEL '$MODEL' -- expected 'image' or 'text'" >&2; exit 1 ;;
esac

cd /project
dvc repro prepare_splits
dvc repro "$STAGE" --force --single-item
