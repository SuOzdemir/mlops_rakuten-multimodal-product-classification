# ============================================================
# recover_logits.py
# Utility script to re-export logits (and features) from the
# best saved weights without re-running training.
#
# Use case: feature export was skipped or failed during training,
# or you need to regenerate .npy files for a different split.
# ============================================================

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

import models.Model_I12_ConvNeXt_Base_ModerateAug_Full.config as config
from models.Model_I12_ConvNeXt_Base_ModerateAug_Full.dataset import RakutenImageDataset
from models.Model_I12_ConvNeXt_Base_ModerateAug_Full.utils import export_logits_and_features


def build_model(num_classes: int) -> nn.Module:
    """Rebuild ConvNeXt-Base architecture (must match train.py exactly)."""
    model = models.convnext_base(weights=None)
    n_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.LayerNorm(n_features),
        nn.Dropout(config.DROPOUT),
        nn.Linear(n_features, num_classes),
    )
    return model


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
    str_label2id = {str(k): v for k, v in label2id.items()}

    # ----------------------------------------------------------
    # 2. Load splits
    # ----------------------------------------------------------
    train_df = pd.read_csv(config.SPLIT_DIR / "train_split.csv")
    val_df   = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    for df in [train_df, val_df]:
        df["label_id"] = df["prdtypecode"].astype(str).map(str_label2id)
        df["image_path_local"] = df.apply(
            lambda r: str(config.LOCAL_IMAGE_TRAIN_DIR / f"image_{r['imageid']}_product_{r['productid']}.jpg"),
            axis=1,
        )

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        shuffle=False,                  # MUST be False for correct ordering
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader   = DataLoader(RakutenImageDataset(val_df,   val_transform, return_idx=True), **loader_kwargs)
    train_loader = DataLoader(RakutenImageDataset(train_df, val_transform, return_idx=True), **loader_kwargs)

    # ----------------------------------------------------------
    # 3. Load model
    # ----------------------------------------------------------
    model = build_model(len(label2id))
    model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))
    model = model.to(device).eval()

    # ----------------------------------------------------------
    # 4. Export
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

    print("\nLogit & feature recovery complete.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
