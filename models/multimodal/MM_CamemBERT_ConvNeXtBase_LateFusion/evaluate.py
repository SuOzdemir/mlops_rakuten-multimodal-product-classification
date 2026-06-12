# ============================================================
# evaluate.py
# Late Fusion evaluation for MM_CamemBERT_ConvNeXtBase_LateFusion.
#
# Prerequisite: both source models must have exported their logits:
#   - Model_I12: outputs/I12_.../val_logits.npy
#   - Model_T8:  outputs/T8_.../text_val_logits.npy
#
# Steps:
#   1. Load pre-exported validation logits from I12 and T8
#   2. Grid search over alpha to find optimal image/text weighting
#   3. Evaluate best fusion: accuracy, macro F1, weighted F1
#   4. Save confusion matrix, alpha sweep plot, classification report
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

import models.multimodal.MM_CamemBERT_ConvNeXtBase_LateFusion.config as config
from models.multimodal.MM_CamemBERT_ConvNeXtBase_LateFusion.utils import (
    evaluate_fusion,
    grid_search_alpha,
    plot_alpha_sweep,
    plot_model_comparison,
    save_run_metadata,
)


def main() -> None:

    # ----------------------------------------------------------
    # 1. Create output directories
    # ----------------------------------------------------------
    config.LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config.LOCAL_FIG_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Run : {config.RUN_NAME}")
    print(f"Image logits : {config.VAL_LOGITS_IMAGE}")
    print(f"Text logits  : {config.VAL_LOGITS_TEXT}")

    # ----------------------------------------------------------
    # 2. Load logits
    # ----------------------------------------------------------
    logits_img = np.load(config.VAL_LOGITS_IMAGE)
    logits_txt = np.load(config.VAL_LOGITS_TEXT)

    assert logits_img.shape == logits_txt.shape, (
        f"Shape mismatch: image logits {logits_img.shape} vs. "
        f"text logits {logits_txt.shape}. "
        "Ensure both were exported from the same val_split.csv."
    )
    print(f"Logits shape : {logits_img.shape}  (N={logits_img.shape[0]}, classes={logits_img.shape[1]})")

    # ----------------------------------------------------------
    # 3. Ground-truth labels
    # ----------------------------------------------------------
    val_df = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    val_df["label_id"] = val_df["prdtypecode"].astype(str).map(
        {str(k): v for k, v in label2id.items()}
    )
    y_true = val_df["label_id"].values

    id2label     = {v: str(k) for k, v in label2id.items()}
    target_names = [id2label[i] for i in range(len(id2label))]

    # ----------------------------------------------------------
    # 4. Alpha grid search
    # ----------------------------------------------------------
    print(f"\nRunning alpha grid search ({config.ALPHA_STEPS} steps) …")
    alphas, scores, best_alpha, best_f1 = grid_search_alpha(
        logits_img, logits_txt, y_true, n_steps=config.ALPHA_STEPS
    )

    f1_image_only = float(scores[-1])   # alpha = 1.0
    f1_text_only  = float(scores[0])    # alpha = 0.0

    print(f"\n  Image only (alpha=1.0) : {f1_image_only:.4f}")
    print(f"  Text only  (alpha=0.0) : {f1_text_only:.4f}")
    print(f"  Best fusion (alpha={best_alpha:.2f}) : {best_f1:.4f}")

    # Plot alpha sweep
    plot_alpha_sweep(alphas, scores, best_alpha, best_f1, config.LOCAL_FIG_ROOT)

    # ----------------------------------------------------------
    # 5. Full evaluation at best alpha
    # ----------------------------------------------------------
    metrics = evaluate_fusion(
        logits_img=logits_img,
        logits_txt=logits_txt,
        y_true=y_true,
        best_alpha=best_alpha,
        target_names=target_names,
        fig_dir=config.LOCAL_FIG_ROOT,
        output_dir=config.LOCAL_OUTPUT_ROOT,
        model_name=config.RUN_NAME,
    )

    # ----------------------------------------------------------
    # 6. Model comparison bar chart
    # ----------------------------------------------------------
    plot_model_comparison(
        model_names=[
            "ConvNeXt-Base\n(Image)",
            "CamemBERT\n(Text)",
            "Late Fusion\n(Image + Text)",
        ],
        f1_scores=[f1_image_only, f1_text_only, metrics["macro_f1"]],
        fig_dir=config.LOCAL_FIG_ROOT,
    )

    # ----------------------------------------------------------
    # 7. Save metadata
    # ----------------------------------------------------------
    save_run_metadata(
        path=config.METADATA_JSON_LOCAL,
        model_name=config.RUN_NAME,
        best_alpha=best_alpha,
        best_f1=metrics["macro_f1"],
        accuracy=metrics["accuracy"],
        weighted_f1=metrics["weighted_f1"],
        f1_image_only=f1_image_only,
        f1_text_only=f1_text_only,
    )

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to:")
    print(f"  Figures : {config.LOCAL_FIG_ROOT}")
    print(f"  Results : {config.LOCAL_OUTPUT_ROOT}")
    print(f"{'='*60}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
