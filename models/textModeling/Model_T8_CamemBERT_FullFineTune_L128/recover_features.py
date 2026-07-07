# ============================================================
# recover_features.py
# Re-export logits and 768d CLS features from the best saved
# weights without re-running training.
#
# Use case: feature export was skipped or failed during training,
# or you need to regenerate .npy files for a different split.
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
from models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.utils import export_logits_and_features


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
    label2id    = {int(k): int(v) for k, v in label2id.items()}
    str_label2id = {str(k): v for k, v in label2id.items()}

    # ----------------------------------------------------------
    # 2. Load splits
    # ----------------------------------------------------------
    train_df = pd.read_csv(config.SPLIT_DIR / "train_split.csv")
    val_df   = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    for df in [train_df, val_df]:
        df["label_id"] = df["prdtypecode"].astype(str).map(str_label2id)

    # ----------------------------------------------------------
    # 3. Tokenizer & model
    # ----------------------------------------------------------
    print(f"Loading tokenizer & model from: {config.LOCAL_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.LOCAL_MODEL_PATH), use_fast=False)

    model = CamembertForSequenceClassification.from_pretrained(
        str(config.LOCAL_MODEL_PATH),
        num_labels=len(label2id),
    )
    model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))
    model = model.to(device).eval()

    # ----------------------------------------------------------
    # 4. Ordered DataLoaders  (shuffle=False is mandatory)
    # ----------------------------------------------------------
    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.EXPORT_NUM_WORKERS,   # 0 is safer on Windows
    )
    val_loader   = DataLoader(RakutenTextDataset(val_df,   tokenizer, config.MAX_LENGTH), **loader_kwargs)
    train_loader = DataLoader(RakutenTextDataset(train_df, tokenizer, config.MAX_LENGTH), **loader_kwargs)

    # ----------------------------------------------------------
    # 5. Export
    # ----------------------------------------------------------
    config.LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    export_logits_and_features(
        model, val_loader, device,
        config.LOCAL_OUTPUT_ROOT, "val", feature_dim=config.FEATURE_DIM,
    )
    export_logits_and_features(
        model, train_loader, device,
        config.LOCAL_OUTPUT_ROOT, "train", feature_dim=config.FEATURE_DIM,
    )

    print("\nFeature recovery complete.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
