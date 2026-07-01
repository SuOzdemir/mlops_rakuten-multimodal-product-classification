import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, CamembertForSequenceClassification


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

ASSET_DIR = DATA_DIR / "rakuten_streamlit_predictor"

CATEGORY_MAPPING_PATH = ASSET_DIR / "prdtypecode_mapping.json"
IMAGE_WEIGHTS_PATH = ASSET_DIR / "image_model" / "best_model_base.pt"
TEXT_MODEL_DIR = ASSET_DIR / "text_model" / "camembert_run4"


# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------
IMAGE_SIZE = 224
TEXT_MAX_LENGTH = 128

DEFAULT_IMAGE_WEIGHT = 0.45
DEFAULT_TEXT_WEIGHT = 0.55

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# Model builder
# ------------------------------------------------------------
def build_convnext_base(num_classes: int) -> nn.Module:
    model = models.convnext_base(weights=None)
    n_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(n_features, num_classes)
    return model


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Asset loading
# ------------------------------------------------------------
def load_assets() -> dict:
    for path in [
        CATEGORY_MAPPING_PATH,
        IMAGE_WEIGHTS_PATH,
        TEXT_MODEL_DIR / "config.json",
        TEXT_MODEL_DIR / "model.safetensors",
        TEXT_MODEL_DIR / "label2id.json",
        TEXT_MODEL_DIR / "tokenizer.json",
        TEXT_MODEL_DIR / "tokenizer_config.json",
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required asset: {path}")

    raw_label2id = load_json(TEXT_MODEL_DIR / "label2id.json")
    label2id = {int(k): int(v) for k, v in raw_label2id.items()}
    id2label = {v: k for k, v in label2id.items()}

    category_mapping_raw = load_json(CATEGORY_MAPPING_PATH)
    category_mapping = {int(k): str(v) for k, v in category_mapping_raw.items()}

    num_classes = len(label2id)

    # Image model
    image_model = build_convnext_base(num_classes)
    image_state = torch.load(IMAGE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    image_model.load_state_dict(image_state)
    image_model.to(DEVICE)
    image_model.eval()

    # Text model — loaded directly from full HuggingFace checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        str(TEXT_MODEL_DIR),
        local_files_only=True,
    )

    text_model = CamembertForSequenceClassification.from_pretrained(
        str(TEXT_MODEL_DIR),
        num_labels=num_classes,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    text_model.to(DEVICE)
    text_model.eval()

    return {
        "device": DEVICE,
        "label2id": label2id,
        "id2label": id2label,
        "category_mapping": category_mapping,
        "image_model": image_model,
        "text_model": text_model,
        "tokenizer": tokenizer,
    }


# ------------------------------------------------------------
# Input preparation
# ------------------------------------------------------------
def _has_text(designation: str | None, description: str | None) -> bool:
    designation = "" if designation is None else str(designation).strip()
    description = "" if description is None else str(description).strip()
    return bool(designation or description)


def _prepare_text(designation: str | None, description: str | None) -> str:
    designation = "" if designation is None else str(designation).strip()
    description = "" if description is None else str(description).strip()
    return f"{designation} {description}".strip()


def _prepare_image(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    return tensor.to(DEVICE)


# ------------------------------------------------------------
# Logit functions
# ------------------------------------------------------------
@torch.no_grad()
def predict_image_logits(image, assets: dict) -> torch.Tensor:
    image_tensor = _prepare_image(image)
    return assets["image_model"](image_tensor)


@torch.no_grad()
def predict_text_logits(designation: str, description: str, assets: dict) -> torch.Tensor:
    text = _prepare_text(designation, description)
    encoded = assets["tokenizer"](
        text,
        add_special_tokens=True,
        max_length=TEXT_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    outputs = assets["text_model"](input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


# ------------------------------------------------------------
# Top-k formatting
# ------------------------------------------------------------
def logits_to_topk(logits: torch.Tensor, assets: dict, k: int = 3) -> list[dict]:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    top_idx = np.argsort(probs)[::-1][:k]

    results = []
    for rank, class_id in enumerate(top_idx, start=1):
        prdtypecode = int(assets["id2label"][int(class_id)])
        category_name = assets["category_mapping"].get(prdtypecode, "Unknown category")
        confidence = float(probs[int(class_id)] * 100.0)
        results.append({
            "Rank": rank,
            "prdtypecode": prdtypecode,
            "Category name": category_name,
            "Confidence": f"{confidence:.1f}%",
            "confidence_float": confidence,
            "class_id": int(class_id),
        })

    return results


# ------------------------------------------------------------
# Public prediction function
# ------------------------------------------------------------
def predict(
    assets: dict,
    designation: str | None = "",
    description: str | None = "",
    image=None,
    image_weight: float = DEFAULT_IMAGE_WEIGHT,
    top_k: int = 3,
) -> dict:
    text_available = _has_text(designation, description)
    image_available = image is not None

    if not text_available and not image_available:
        raise ValueError("Provide at least a title, a description, or an image.")

    image_weight = float(image_weight)
    image_weight = max(0.0, min(1.0, image_weight))
    text_weight = 1.0 - image_weight

    text_used = _prepare_text(designation, description)

    if text_available and image_available:
        image_logits = predict_image_logits(image, assets)
        text_logits = predict_text_logits(designation, description, assets)
        final_logits = image_weight * image_logits + text_weight * text_logits
        mode = "Late fusion"

    elif text_available:
        final_logits = predict_text_logits(designation, description, assets)
        mode = "Text only"
        image_weight = 0.0
        text_weight = 1.0

    else:
        final_logits = predict_image_logits(image, assets)
        mode = "Image only"
        image_weight = 1.0
        text_weight = 0.0

    top3 = logits_to_topk(final_logits, assets, k=top_k)

    return {
        "mode": mode,
        "text_used": text_used,
        "image_weight": image_weight,
        "text_weight": text_weight,
        "top3": top3,
    }
