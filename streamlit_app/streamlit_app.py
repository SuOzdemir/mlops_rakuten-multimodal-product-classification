import json
from pathlib import Path
import json
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, CamembertModel, CLIPVisionModel, CLIPProcessor

st.set_page_config(page_title="Rakuten Fusion Predictor", layout="centered")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path.cwd()
CAMEMBERT_ID = "camembert-base"
CLIP_ID = "openai/clip-vit-base-patch32"
MAX_TEXT_LEN = 128

OUTPUT_DIR = BASE_DIR / "outputs" / "image_modeling" / "multimodal" / "mm_camembert_clip_gated_fusion_staged_unfreeze"
SPLIT_DIR = BASE_DIR / "outputs" / "image_modeling"
CONFIG_DIR = BASE_DIR / "config"

BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best_checkpoint.pt"
LABEL2ID_PATH = SPLIT_DIR / "label2id.json"
CATEGORY_PATH = CONFIG_DIR / "prdtypecode_mapping.json"

with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
    PRDTYPECODE_TO_NAME = {int(k): v for k, v in json.load(f).items()}
USE_MPS = False
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if USE_MPS and torch.backends.mps.is_available()
    else "cpu"
)

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text = CamembertModel.from_pretrained(CAMEMBERT_ID)
        self.vision = CLIPVisionModel.from_pretrained(CLIP_ID)

        self.gate = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(2304, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)
        vision_outputs = self.vision(pixel_values=pixel_values)

        t = text_outputs.last_hidden_state[:, 0]
        v = vision_outputs.pooler_output

        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        fusion = torch.cat([t, v], dim=1)
        g = self.gate(fusion)

        fused = g * v + (1.0 - g) * t
        final = torch.cat([fused, t, v], dim=1)

        logits = self.classifier(final)
        return logits


@st.cache_resource
def load_assets():
    if not LABEL2ID_PATH.exists():
        raise FileNotFoundError(f"label2id.json not found: {LABEL2ID_PATH}")

    if not BEST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"best_checkpoint.pt not found: {BEST_CHECKPOINT_PATH}")

    with open(LABEL2ID_PATH, "r", encoding="utf-8") as f:
        raw_label2id = json.load(f)
        label2id = {int(k): int(v) for k, v in raw_label2id.items()}

    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location="cpu")

    model = FusionModel(num_classes=num_classes)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer, processor, id2label

def prepare_inputs(image, designation, description, tokenizer, processor):
    designation = "" if designation is None else designation
    description = "" if description is None else description
    text = f"{designation} {description}".strip()

    if text == "":
        text = "unknown product"

    txt = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt"
    )

    img = processor(images=image.convert("RGB"), return_tensors="pt")

    return {
        "input_ids": txt["input_ids"].to(DEVICE),
        "attention_mask": txt["attention_mask"].to(DEVICE),
        "pixel_values": img["pixel_values"].to(DEVICE),
        "text_used": text
    }

def format_confidence(p):
    return f"{p * 100:.2f}%"

@torch.no_grad()
def predict(image, designation, description, model, tokenizer, processor, id2label):
    batch = prepare_inputs(image, designation, description, tokenizer, processor)

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"]
    )

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    result = []
    for idx in top3_idx:
        result.append({
            "class_id": int(idx),
            "prdtypecode": int(id2label[int(idx)]),
            "probability": float(probs[int(idx)])
        })

    return {
        "text_used": batch["text_used"],
        "top1": result[0],
        "top3": result
    }

st.title("Rakuten Fusion Predictor")
st.caption("Upload an image and enter product text to get a multimodal prediction.")

model, tokenizer, processor, id2label = load_assets()

if "prediction_output" not in st.session_state:
    st.session_state.prediction_output = None

left_col, middle_col, right_col = st.columns([1, 1, 1], gap="large")

with left_col:
    st.markdown("### Input")
    uploaded_file = st.file_uploader("Product image", type=["jpg", "jpeg", "png"])
    designation = st.text_input("Designation", placeholder="e.g. PlayStation game")
    description = st.text_area("Description", placeholder="Optional product description", height=140)
    predict_clicked = st.button("Predict", use_container_width=True)

with middle_col:
    st.markdown("### Image")
    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True,
        )
    else:
        st.info("Upload an image to preview it here.")

if predict_clicked:
    if image is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Running prediction..."):
            st.session_state.prediction_output = predict(
                image=image,
                designation=designation,
                description=description,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                id2label=id2label
            )

with right_col:
    st.markdown("### Prediction")

    output = st.session_state.prediction_output

    if output is None:
        st.info("Prediction results will appear here.")
    else:
        top1_code = output["top1"]["prdtypecode"]
        top1_category = PRDTYPECODE_TO_NAME.get(top1_code, "Unknown category")

        st.success(f"🎯 {top1_category}")
        st.write(f"**Code:** {top1_code}")
        st.write(f"**Confidence:** {output['top1']['probability'] * 100:.2f}%")
        st.write(f"**Text used:** `{output['text_used']}`")

        st.markdown("### Top predictions")
        for i, item in enumerate(output["top3"], start=1):
            category_name = PRDTYPECODE_TO_NAME.get(item["prdtypecode"], "Unknown category")
            st.write(
                f"**{i}. {category_name}**  \n"
                f"Code: {item['prdtypecode']}  \n"
                f"Confidence: {item['probability'] * 100:.2f}%"
            )