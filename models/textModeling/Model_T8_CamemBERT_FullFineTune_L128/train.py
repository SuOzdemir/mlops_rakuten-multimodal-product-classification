# ============================================================
# train.py
# Training script for Rakuten text classification.
# All hyperparameters and paths are defined in config.py.
#
# IMPORTANT: config must be imported first – it sets the
# Windows/HuggingFace env fixes before transformers loads.
#
# Model T8 specifics:
#   - Architecture  : CamemBERT (full fine-tuning)
#   - Input         : designation + description (max 128 tokens)
#   - AMP           : Mixed precision via GradScaler + autocast
#   - Feature export: 768d CLS embeddings + logits for fusion
# ============================================================

# config import MUST come before transformers to apply HF fixes
import models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.config as config  # noqa: E402 (intentional first import)

import gc
import json
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CamembertForSequenceClassification

from models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.dataset import RakutenTextDataset
from models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.utils import (
    export_logits_and_features,
    load_full_checkpoint,
    plot_and_save_history,
    run_epoch,
    save_full_checkpoint,
    save_history_json,
    save_run_metadata,
)


def main() -> None:

    # ----------------------------------------------------------
    # 1. Setup: reproducibility, memory, device
    # ----------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run     : {config.RUN_NAME}")
    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"AMP     : {config.USE_AMP and torch.cuda.is_available()}")

    # ----------------------------------------------------------
    # 2. Create output directories
    # ----------------------------------------------------------
    for d in [config.LOCAL_OUTPUT_ROOT, config.LOCAL_MODEL_ROOT, config.LOCAL_FIG_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 2b. MLflow setup
    # ----------------------------------------------------------
    _mlflow_active = False
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        mlflow.set_experiment("rakuten-text-modeling")
        mlflow.start_run(run_name=config.RUN_NAME)
        mlflow.log_params({
            "architecture":            "CamemBERT-base",
            "seed":                    config.SEED,
            "max_length":              config.MAX_LENGTH,
            "batch_size":              config.BATCH_SIZE,
            "max_epochs":              config.MAX_EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "learning_rate":           config.LEARNING_RATE,
            "weight_decay":            config.WEIGHT_DECAY,
            "feature_dim":             config.FEATURE_DIM,
            "use_amp":                 config.USE_AMP,
        })
        _mlflow_active = True
    except Exception as _e:
        if os.environ.get("MLFLOW_REQUIRED", "true").lower() == "true":
            raise RuntimeError(f"MLflow setup failed: {_e}") from _e
        print(f"[MLflow] Tracking disabled by configuration: {_e}")

    # ----------------------------------------------------------
    # 3. Load splits & label mapping
    # ----------------------------------------------------------
    train_df = pd.read_csv(config.SPLIT_DIR / "train_split.csv")
    val_df   = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    str_label2id = {str(k): v for k, v in label2id.items()}
    train_df["label_id"] = train_df["prdtypecode"].astype(str).map(str_label2id)
    val_df["label_id"]   = val_df["prdtypecode"].astype(str).map(str_label2id)

    if config.TRAIN_ROWS_OVERRIDE or config.VAL_ROWS_OVERRIDE:
        # No random_state here on purpose: this sample is meant to change
        # every run (a quick/bounded sanity check), unlike the rest of
        # training which stays seeded for reproducibility.
        if config.TRAIN_ROWS_OVERRIDE:
            train_df = train_df.sample(n=min(config.TRAIN_ROWS_OVERRIDE, len(train_df))).reset_index(drop=True)
        if config.VAL_ROWS_OVERRIDE:
            val_df = val_df.sample(n=min(config.VAL_ROWS_OVERRIDE, len(val_df))).reset_index(drop=True)
        print(f"[TRAIN_ROWS_OVERRIDE] Sampled down to {len(train_df)} train / {len(val_df)} val rows (random each run)")

    num_classes = len(label2id)
    print(f"Classes : {num_classes}")
    print(f"Train   : {len(train_df):,}  |  Val: {len(val_df):,}")

    # ----------------------------------------------------------
    # 4. Tokenizer & Model
    # ----------------------------------------------------------
    print(f"\nLoading tokenizer & model from: {config.LOCAL_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.LOCAL_MODEL_PATH), use_fast=False)
    model = CamembertForSequenceClassification.from_pretrained(
        str(config.LOCAL_MODEL_PATH),
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # LOCAL_MODEL_PATH's head is sized for 27 classes; reinit if num_classes differs
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params  : {trainable:,} trainable / {total:,} total")

    # ----------------------------------------------------------
    # 5. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_loader = DataLoader(
        RakutenTextDataset(train_df, tokenizer, config.MAX_LENGTH),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        RakutenTextDataset(val_df, tokenizer, config.MAX_LENGTH),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # ----------------------------------------------------------
    # 6. Optimizer, Scheduler, AMP Scaler
    # ----------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )
    scaler = GradScaler("cuda") if (config.USE_AMP and torch.cuda.is_available()) else None

    # ----------------------------------------------------------
    # 7. Resume or initialise training state
    # ----------------------------------------------------------
    start_epoch            = 1
    history: list          = []
    best_macro_f1: float   = -float("inf")
    best_epoch: int        = -1
    epochs_no_improve: int = 0

    if config.RESUME_TRAINING:
        ckpt_map = {
            "local_last": config.LAST_CKPT_LOCAL,
            "local_best": config.BEST_CKPT_LOCAL,
        }
        ckpt_path = ckpt_map[config.CHECKPOINT_SOURCE]
        if ckpt_path.exists():
            start_epoch, history, best_macro_f1, best_epoch = load_full_checkpoint(
                ckpt_path, model, optimizer, scheduler, device
            )
            print(f"Resumed from: {ckpt_path}  (starting epoch {start_epoch})")
        else:
            print(f"Checkpoint not found: {ckpt_path}  – starting from scratch.")

    # ----------------------------------------------------------
    # 8. Training loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Training  (CamemBERT – Full Fine-Tuning) …")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, config.MAX_EPOCHS + 1):

        print(f"\n--- Epoch {epoch}/{config.MAX_EPOCHS} starting "
              f"({time.time() - start_time:.0f}s since training start) ---")

        train_loss, train_acc, train_macro_f1, train_weighted_f1 = run_epoch(
            model, train_loader, device, optimizer, scaler
        )
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = run_epoch(
            model, val_loader, device, optimizer=None, scaler=None
        )

        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch":             epoch,
            "train_loss":        float(train_loss),
            "val_loss":          float(val_loss),
            "train_acc":         float(train_acc),
            "val_acc":           float(val_acc),
            "train_macro_f1":    float(train_macro_f1),
            "val_macro_f1":      float(val_macro_f1),
            "train_weighted_f1": float(train_weighted_f1),
            "val_weighted_f1":   float(val_weighted_f1),
            "lr":                float(current_lr),
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:>2}/{config.MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}  val_macro_f1={val_macro_f1:.4f}  "
            f"val_weighted_f1={val_weighted_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if _mlflow_active:
            mlflow.log_metrics({
                "train_loss":        float(train_loss),
                "val_loss":          float(val_loss),
                "train_acc":         float(train_acc),
                "val_acc":           float(val_acc),
                "train_macro_f1":    float(train_macro_f1),
                "val_macro_f1":      float(val_macro_f1),
                "train_weighted_f1": float(train_weighted_f1),
                "val_weighted_f1":   float(val_weighted_f1),
                "lr":                float(current_lr),
            }, step=epoch)

        save_full_checkpoint(
            config.LAST_CKPT_LOCAL, model, optimizer, scheduler,
            epoch, history, best_macro_f1, best_epoch, config.RUN_NAME,
        )
        save_history_json(history, config.HISTORY_JSON_LOCAL)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1  = val_macro_f1
            best_epoch     = epoch
            epochs_no_improve = 0

            save_full_checkpoint(
                config.BEST_CKPT_LOCAL, model, optimizer, scheduler,
                epoch, history, best_macro_f1, best_epoch, config.RUN_NAME,
            )
            torch.save(model.state_dict(), config.BEST_WEIGHTS_LOCAL)
            print("    >>> New best model saved!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print("    >>> Early stopping triggered.")
            break

    # ----------------------------------------------------------
    # 9. Summary & metadata
    # ----------------------------------------------------------
    total_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("Training Finished")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best F1     : {best_macro_f1:.4f}")
    print(f"  Duration    : {total_minutes:.1f} min")
    print("=" * 60)

    save_run_metadata(
        config.METADATA_JSON_LOCAL,
        model_name=config.RUN_NAME,
        best_f1=best_macro_f1,
        best_epoch=best_epoch,
        duration_min=total_minutes,
    )

    if _mlflow_active:
        mlflow.log_metric("best_macro_f1", best_macro_f1)
        mlflow.log_metric("best_epoch", float(best_epoch))
        mlflow.log_metric("duration_min", total_minutes)
        mlflow.log_artifact(str(config.BEST_WEIGHTS_LOCAL))
        mlflow.log_artifact(str(config.METADATA_JSON_LOCAL))

    # ----------------------------------------------------------
    # 10. Training curves
    # ----------------------------------------------------------
    plot_and_save_history(history, config.LOCAL_FIG_ROOT, best_epoch, best_macro_f1)

    if _mlflow_active:
        curves_png = config.LOCAL_FIG_ROOT / "training_curves.png"
        if curves_png.exists():
            mlflow.log_artifact(str(curves_png))
        mlflow.end_run()

    # ----------------------------------------------------------
    # 11. Feature & logit export  (for fusion models)
    # ----------------------------------------------------------
    if config.EXPORT_FEATURES:
        print("\n" + "=" * 60)
        print("Feature Export  (logits + 768d CLS features for fusion) …")
        print("=" * 60)

        # Reload best weights before export
        model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))

        # Ordered loaders (shuffle=False, safer num_workers for export)
        export_kwargs = dict(
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.EXPORT_NUM_WORKERS,
        )
        val_loader_ordered = DataLoader(
            RakutenTextDataset(val_df, tokenizer, config.MAX_LENGTH), **export_kwargs
        )
        train_loader_ordered = DataLoader(
            RakutenTextDataset(train_df, tokenizer, config.MAX_LENGTH), **export_kwargs
        )

        export_logits_and_features(
            model, val_loader_ordered, device,
            config.LOCAL_OUTPUT_ROOT, "val", feature_dim=config.FEATURE_DIM,
        )
        export_logits_and_features(
            model, train_loader_ordered, device,
            config.LOCAL_OUTPUT_ROOT, "train", feature_dim=config.FEATURE_DIM,
        )

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise
