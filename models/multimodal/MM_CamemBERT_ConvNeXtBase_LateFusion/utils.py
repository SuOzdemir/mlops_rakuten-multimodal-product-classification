# ============================================================
# utils.py
# Shared utilities for Late Fusion evaluation:
#   - Alpha grid search
#   - Confusion matrix plot
#   - Alpha sweep plot
#   - Classification report
#   - Metadata saving
# ============================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ------------------------------------------------------------
# Fusion
# ------------------------------------------------------------

def fuse_logits(logits_img: np.ndarray, logits_txt: np.ndarray, alpha: float) -> np.ndarray:
    """
    Weighted combination of image and text logits.

        fused = alpha * logits_img + (1 - alpha) * logits_txt

    Args:
        logits_img : Image model logits, shape (N, num_classes).
        logits_txt : Text model logits,  shape (N, num_classes).
        alpha      : Image weight in [0, 1]. alpha=1 → image only.

    Returns:
        Fused logits, shape (N, num_classes).
    """
    return alpha * logits_img + (1.0 - alpha) * logits_txt


def grid_search_alpha(
    logits_img: np.ndarray,
    logits_txt: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 21,
) -> tuple:
    """
    Find the alpha value that maximises macro F1 on the validation set.

    Args:
        n_steps : Number of evenly spaced alpha values in [0, 1].

    Returns:
        (alphas, scores, best_alpha, best_f1)
    """
    alphas = np.linspace(0, 1, n_steps)
    scores = []

    for alpha in alphas:
        fused = fuse_logits(logits_img, logits_txt, alpha)
        preds = np.argmax(fused, axis=1)
        scores.append(f1_score(y_true, preds, average="macro", zero_division=0))

    scores      = np.array(scores)
    best_idx    = int(np.argmax(scores))
    best_alpha  = float(alphas[best_idx])
    best_f1     = float(scores[best_idx])

    return alphas, scores, best_alpha, best_f1


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_fusion(
    logits_img: np.ndarray,
    logits_txt: np.ndarray,
    y_true: np.ndarray,
    best_alpha: float,
    target_names: list,
    fig_dir: Path,
    output_dir: Path,
    model_name: str,
) -> dict:
    """
    Evaluate the fused model at best_alpha, print full report,
    save confusion matrix and classification report.

    Returns:
        {"accuracy": float, "macro_f1": float, "weighted_f1": float}
    """
    fused  = fuse_logits(logits_img, logits_txt, best_alpha)
    y_pred = np.argmax(fused, axis=1)

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"Fusion Evaluation – {model_name}  (alpha={best_alpha:.2f})")
    print(f"{'='*60}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")

    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(f"\nClassification Report:\n{report}")

    # Save report as text
    report_path = output_dir / "fusion_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model     : {model_name}\n")
        f.write(f"Alpha     : {best_alpha:.4f}  (image weight)\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Macro F1  : {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, target_names, fig_dir, model_name, macro_f1, best_alpha
    )

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_alpha_sweep(
    alphas: np.ndarray,
    scores: np.ndarray,
    best_alpha: float,
    best_f1: float,
    fig_dir: Path,
) -> None:
    """Plot and save the macro F1 score across all tested alpha values."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1_image_only = float(scores[-1])   # alpha = 1.0
    f1_text_only  = float(scores[0])    # alpha = 0.0

    plt.figure(figsize=(10, 5))
    plt.plot(alphas, scores, marker="o", color="purple", linestyle="--", linewidth=2)
    plt.axvline(
        best_alpha, color="red", linestyle=":",
        label=f"Best alpha: {best_alpha:.2f}  (F1={best_f1:.4f})",
    )
    plt.title(
        f"Late Fusion – Alpha Sweep\n"
        f"Image only: {f1_image_only:.4f}  |  "
        f"Text only: {f1_text_only:.4f}  |  "
        f"Best fusion: {best_f1:.4f}",
        fontsize=12,
    )
    plt.xlabel("Image model weight (alpha)")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "alpha_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Alpha sweep plot saved to: {fig_dir / 'alpha_sweep.png'}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    fig_dir: Path,
    model_name: str,
    macro_f1: float,
    alpha: float,
) -> None:
    """Compute and save a normalised confusion matrix heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm      = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(
        f"Normalised Confusion Matrix – {model_name}\n"
        f"Alpha={alpha:.2f}  |  Macro-F1={macro_f1:.4f}",
        fontsize=14,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {fig_dir / 'confusion_matrix.png'}")


def plot_model_comparison(
    model_names: list,
    f1_scores: list,
    fig_dir: Path,
    highlight_last: bool = True,
) -> None:
    """
    Bar chart comparing macro F1 across unimodal and fusion models.

    Args:
        model_names   : Display names for x-axis labels.
        f1_scores     : Corresponding macro F1 values.
        highlight_last: If True, the last bar (fusion) is coloured green.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#3498db"] * len(model_names)
    if highlight_last:
        colors[-1] = "#2ecc71"

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, f1_scores, color=colors, width=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            f"{yval:.4f}",
            ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )

    if len(f1_scores) >= 2:
        boost = (f1_scores[-1] - max(f1_scores[:-1])) * 100
        plt.annotate(
            f"Boost: +{boost:.2f}%",
            xy=(len(model_names) - 1, f1_scores[-1]),
            xytext=(len(model_names) - 1.5, min(f1_scores) + 0.9 * (max(f1_scores) - min(f1_scores))),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
            fontsize=10, fontweight="bold",
        )

    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1", fontsize=12)
    plt.title("Performance Comparison: Unimodal vs. Multimodal", fontsize=14, pad=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Model comparison plot saved to: {fig_dir / 'model_comparison_f1.png'}")


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

def save_run_metadata(
    path: Path,
    model_name: str,
    best_alpha: float,
    best_f1: float,
    accuracy: float,
    weighted_f1: float,
    f1_image_only: float,
    f1_text_only: float,
) -> None:
    """Save fusion results as JSON for easy cross-experiment comparison."""
    meta = {
        "model_name":    model_name,
        "best_alpha":    round(best_alpha, 4),
        "best_macro_f1": round(best_f1, 6),
        "accuracy":      round(accuracy, 6),
        "weighted_f1":   round(weighted_f1, 6),
        "f1_image_only": round(f1_image_only, 6),
        "f1_text_only":  round(f1_text_only, 6),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {path}")
