# ============================================================
# train.py
# Generic training script for Rakuten image classification.
# All hyperparameters and paths are defined in config.py.
#
# Model I12 specifics:
#   - Architecture  : ConvNeXt-Base (full fine-tuning)
#   - AMP           : Mixed precision via GradScaler + autocast
#   - Optimizer     : AdamW with differential LR (backbone vs. head)
#   - Augmentation  : TrivialAugmentWide
#   - Feature export: 1024d features + logits saved for fusion models
# ============================================================

import gc
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")  # mlflow/protobuf>=7.34.0 fix

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Base_Weights

import models.Model_I12_ConvNeXt_Base_ModerateAug_Full.config as config
from models.Model_I12_ConvNeXt_Base_ModerateAug_Full.dataset import RakutenImageDataset
from models.Model_I12_ConvNeXt_Base_ModerateAug_Full.utils import (
    export_logits_and_features,
    load_full_checkpoint,
    plot_and_save_history,
    run_epoch,
    save_full_checkpoint,
    save_history_json,
    save_run_metadata,
)


def build_model(num_classes: int) -> nn.Module:
    """Construct ConvNeXt-Base with custom classification head."""
    model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    n_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.LayerNorm(n_features),
        nn.Dropout(config.DROPOUT),
        nn.Linear(n_features, num_classes),
    )
    return model


def main() -> None:

    # ----------------------------------------------------------
    # 1. Setup: determinism, memory, device
    # ----------------------------------------------------------
    torch.backends.cudnn.benchmark    = config.CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN_DETERMINISTIC
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
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("rakuten-image-modeling")
        mlflow.start_run(run_name=config.RUN_NAME)
        mlflow.log_params({
            "architecture":            "ConvNeXt-Base",
            "seed":                    config.SEED,
            "image_size":              config.IMAGE_SIZE,
            "batch_size":              config.BATCH_SIZE,
            "max_epochs":              config.MAX_EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "lr_backbone":             config.LR_BACKBONE,
            "lr_head":                 config.LR_HEAD,
            "weight_decay":            config.WEIGHT_DECAY,
            "label_smoothing":         config.LABEL_SMOOTHING,
            "dropout":                 config.DROPOUT,
            "use_amp":                 config.USE_AMP,
        })
        _mlflow_active = True
    except Exception as _e:
        print(f"[MLflow] Tracking disabled: {_e}")

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

    if config.SMOKE_TEST:
        train_df = train_df.sample(n=min(config.SMOKE_TEST_TRAIN_ROWS, len(train_df)), random_state=config.SEED).reset_index(drop=True)
        val_df = val_df.sample(n=min(config.SMOKE_TEST_VAL_ROWS, len(val_df)), random_state=config.SEED).reset_index(drop=True)
        print(f"[SMOKE_TEST] Sampled down to {len(train_df)} train / {len(val_df)} val rows")

    def make_path(row) -> str:
        return str(config.LOCAL_IMAGE_TRAIN_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg")

    train_df["image_path_local"] = train_df.apply(make_path, axis=1)
    val_df["image_path_local"]   = val_df.apply(make_path, axis=1)

    num_classes = len(label2id)
    print(f"Classes : {num_classes}")
    print(f"Train   : {len(train_df):,}  |  Val: {len(val_df):,}")

    # ----------------------------------------------------------
    # 4. Transforms
    # ----------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ----------------------------------------------------------
    # 5. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_loader = DataLoader(
        RakutenImageDataset(train_df, transform=train_transform, return_idx=True),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        RakutenImageDataset(val_df, transform=val_transform, return_idx=True),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    # ----------------------------------------------------------
    # 6. Model
    # ----------------------------------------------------------
    model = build_model(num_classes).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params  : {trainable:,} trainable / {total:,} total")

    # ----------------------------------------------------------
    # 7. Loss, Optimizer, Scheduler, AMP Scaler
    # ----------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(),   "lr": config.LR_BACKBONE},
            {"params": model.classifier.parameters(), "lr": config.LR_HEAD},
        ],
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
    # 8. Resume or initialise training state
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
    # 9. Training loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Training  (ConvNeXt-Base – Full Fine-Tuning) …")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, config.MAX_EPOCHS + 1):
        gc.collect()
        torch.cuda.empty_cache()

        train_loss, train_acc, train_macro_f1, train_weighted_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = run_epoch(
            model, val_loader, criterion, device, optimizer=None, scaler=None
        )

        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[1]["lr"]   # head LR as reference

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
    # 10. Summary & metadata
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
    # 11. Training curves
    # ----------------------------------------------------------
    plot_and_save_history(history, config.LOCAL_FIG_ROOT, best_epoch, best_macro_f1)

    if _mlflow_active:
        curves_png = config.LOCAL_FIG_ROOT / "training_curves.png"
        if curves_png.exists():
            mlflow.log_artifact(str(curves_png))
        mlflow.end_run()

    # ----------------------------------------------------------
    # 12. Feature & logit export  (for fusion models)
    # ----------------------------------------------------------
    if config.EXPORT_FEATURES:
        print("\n" + "=" * 60)
        print("Feature Export  (logits + 1024d features for fusion) …")
        print("=" * 60)

        # Reload best weights before export
        model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))

        # Ordered val loader (shuffle=False, already correct above)
        export_logits_and_features(
            model, val_loader, device,
            config.LOCAL_OUTPUT_ROOT, "val", feature_dim=config.FEATURE_DIM,
        )

        # Ordered train loader (must NOT shuffle)
        train_loader_ordered = DataLoader(
            RakutenImageDataset(train_df, transform=val_transform, return_idx=True),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
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
    main()
