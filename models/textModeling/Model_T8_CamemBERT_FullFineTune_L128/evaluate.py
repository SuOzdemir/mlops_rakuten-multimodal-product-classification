# ============================================================
# evaluate.py
# Standalone evaluation script for Model T8.
# Loads the best saved weights, runs inference on the
# validation set, and produces:
#   - Classification report (accuracy, macro F1, weighted F1)
#   - Normalised confusion matrix (PNG)
# ============================================================

# config import MUST come before transformers to apply HF fixes
import models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.config as config  # noqa: E402

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CamembertForSequenceClassification

from models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.dataset import RakutenTextDataset
from models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.utils import evaluate_model


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run      : {config.RUN_NAME}")
    print(f"Device   : {device}")
    print(f"Weights  : {config.BEST_WEIGHTS_LOCAL}")

    # ----------------------------------------------------------
    # 1. Label mapping
    # ----------------------------------------------------------
    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    id2label     = {v: str(k) for k, v in label2id.items()}
    target_names = [id2label[i] for i in range(len(id2label))]

    # ----------------------------------------------------------
    # 2. Validation dataset & loader
    # ----------------------------------------------------------
    val_df = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    str_label2id = {str(k): v for k, v in label2id.items()}
    val_df["label_id"] = val_df["prdtypecode"].astype(str).map(str_label2id)

    tokenizer = AutoTokenizer.from_pretrained(str(config.LOCAL_MODEL_PATH), use_fast=False)

    val_loader = DataLoader(
        RakutenTextDataset(val_df, tokenizer, config.MAX_LENGTH),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.EXPORT_NUM_WORKERS,
        pin_memory=True,
    )

    # ----------------------------------------------------------
    # 3. Load model & best weights
    # ----------------------------------------------------------
    model = CamembertForSequenceClassification.from_pretrained(
        str(config.LOCAL_MODEL_PATH),
        num_labels=len(label2id),
    )
    model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))
    model = model.to(device)

    # ----------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------
    metrics = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        target_names=target_names,
        fig_dir=config.LOCAL_FIG_ROOT,
        model_name=config.RUN_NAME,
    )

    print(f"\nSummary  |  Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}  |  "
          f"Weighted F1: {metrics['weighted_f1']:.4f}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
