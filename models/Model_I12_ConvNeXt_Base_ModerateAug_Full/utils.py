# ============================================================
# utils.py
# Shared utilities: checkpointing, training loop, plotting,
# evaluation (accuracy, macro F1, weighted F1, confusion matrix),
# and feature/logit export for fusion models.
# ============================================================

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.amp import autocast


# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------

def save_full_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    history: list,
    best_f1: float,
    best_epoch: int,
    model_name: str = "",
) -> None:
    """Save a full training checkpoint including RNG states for exact resumability."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "model_name": model_name,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def load_full_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    device,
) -> tuple:
    """
    Load a full training checkpoint and restore all states.

    Returns:
        (start_epoch, history, best_macro_f1, best_epoch)
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    return (
        checkpoint["epoch"] + 1,
        checkpoint["history"],
        checkpoint["best_macro_f1"],
        checkpoint["best_epoch"],
    )


def save_history_json(history: list, path: Path) -> None:
    """Persist the training history list as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_run_metadata(
    path: Path,
    model_name: str,
    best_f1: float,
    best_epoch: int,
    duration_min: float,
) -> None:
    """Save a small JSON with key run results for easy cross-experiment comparison."""
    meta = {
        "model_name":    model_name,
        "best_macro_f1": round(best_f1, 6),
        "best_epoch":    best_epoch,
        "duration_min":  round(duration_min, 2),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------
# Training / Validation loop  (supports AMP via scaler)
# ------------------------------------------------------------

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None) -> tuple:
    """
    Run one full epoch (train or validation).

    The loader may return 2-tuples (image, label) or 3-tuples
    (image, label, idx) – both are handled transparently.

    Args:
        optimizer : Pass None for validation (no gradient updates).
        scaler    : GradScaler for mixed-precision training. Pass None to disable.

    Returns:
        (epoch_loss, epoch_accuracy, epoch_macro_f1, epoch_weighted_f1)
    """
    is_train = optimizer is not None
    use_amp  = scaler is not None and torch.cuda.is_available()

    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds: list = []
    all_true:  list = []

    for batch in loader:
        # Support both (img, label) and (img, label, idx)
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_true.extend(labels.detach().cpu().numpy())

    epoch_loss        = total_loss / len(loader.dataset)
    epoch_acc         = accuracy_score(all_true, all_preds)
    epoch_macro_f1    = f1_score(all_true, all_preds, average="macro",    zero_division=0)
    epoch_weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1, epoch_weighted_f1


# ------------------------------------------------------------
# Feature & Logit Export  (required for fusion models)
# ------------------------------------------------------------

@torch.no_grad()
def export_logits_and_features(
    model,
    loader,
    device,
    output_path: Path,
    split_name: str,
    feature_dim: int = 1024,
) -> None:
    """
    Extract and save logits and intermediate features for fusion.

    Exports two .npy files per split:
        <split_name>_logits.npy          – shape (N, num_classes)
        <split_name>_features_<dim>d.npy – shape (N, feature_dim)

    The loader MUST use shuffle=False and the dataset MUST have
    return_idx=True to guarantee correct sample ordering.

    Args:
        model        : Trained ConvNeXt model (eval mode expected).
        loader       : Ordered DataLoader (shuffle=False, return_idx=True).
        device       : torch device.
        output_path  : Directory where .npy files are written.
        split_name   : "train" or "val" – used in filenames.
        feature_dim  : Feature dimensionality before classifier head
                       (1024 for ConvNeXt-Base, 768 for ConvNeXt-Tiny).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Feature extractor: everything before the final Linear layer
    feature_extractor = nn.Sequential(model.features, model.avgpool)

    all_logits:   list = []
    all_features: list = []

    print(f"Exporting logits & {feature_dim}d features for split: {split_name} …")

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            feats  = feature_extractor(images)
            feats  = torch.flatten(feats, 1)   # (B, feature_dim)

        all_logits.append(logits.cpu().numpy())
        all_features.append(feats.cpu().numpy())

    logits_arr   = np.vstack(all_logits)
    features_arr = np.vstack(all_features)

    np.save(output_path / f"{split_name}_logits.npy",                   logits_arr)
    np.save(output_path / f"{split_name}_features_{feature_dim}d.npy",  features_arr)

    print(
        f"  Logits   : {logits_arr.shape}  → {output_path / f'{split_name}_logits.npy'}"
    )
    print(
        f"  Features : {features_arr.shape}  → "
        f"{output_path / f'{split_name}_features_{feature_dim}d.npy'}"
    )


# ------------------------------------------------------------
# Plotting – training curves
# ------------------------------------------------------------

def plot_and_save_history(
    history: list,
    fig_dir: Path,
    best_epoch: int,
    best_f1: float,
) -> None:
    """
    Generate and save Loss / Accuracy / Macro-F1 / Weighted-F1 curves.
    Four subplots in a 2×2 grid, saved as a single PNG.
    """
    if not history:
        print("No history to plot.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Training Curves  (Best Epoch: {best_epoch} | Best Macro-F1: {best_f1:.4f})",
            fontsize=13,
        )

        plots = [
            ("train_loss",        "val_loss",        "Loss",        axes[0, 0]),
            ("train_acc",         "val_acc",         "Accuracy",    axes[0, 1]),
            ("train_macro_f1",    "val_macro_f1",    "Macro F1",    axes[1, 0]),
            ("train_weighted_f1", "val_weighted_f1", "Weighted F1", axes[1, 1]),
        ]

        for train_col, val_col, title, ax in plots:
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col], "-o", label="Train", linewidth=2)
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col],   "-o", label="Val",   linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to: {fig_dir / 'training_curves.png'}")

    except Exception as e:
        print(f"Error saving training curves: {e}")


# ------------------------------------------------------------
# Plotting – confusion matrix
# ------------------------------------------------------------

def plot_confusion_matrix(
    all_true: list,
    all_preds: list,
    target_names: list,
    fig_dir: Path,
    model_name: str,
    best_f1: float,
) -> None:
    """Compute and save a normalised confusion matrix heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm      = confusion_matrix(all_true, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(
        f"Normalised Confusion Matrix – {model_name}  (Macro-F1: {best_f1:.4f})",
        fontsize=14,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {fig_dir / 'confusion_matrix.png'}")


# ------------------------------------------------------------
# Full evaluation report  (used by evaluate.py)
# ------------------------------------------------------------

def evaluate_model(
    model,
    loader,
    device,
    target_names: list,
    fig_dir: Path,
    model_name: str,
) -> dict:
    """
    Run inference on loader, print classification report,
    save confusion matrix, and return a metrics dict.

    Returns:
        {"accuracy": float, "macro_f1": float, "weighted_f1": float}
    """
    model.eval()
    all_preds: list = []
    all_true:  list = []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())

    acc         = accuracy_score(all_true, all_preds)
    macro_f1    = f1_score(all_true, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"Evaluation Results – {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(all_true, all_preds, target_names=target_names, zero_division=0))

    plot_confusion_matrix(all_true, all_preds, target_names, fig_dir, model_name, macro_f1)

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
