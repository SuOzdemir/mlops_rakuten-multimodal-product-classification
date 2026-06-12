from pathlib import Path
import json
import re
import runpy
from html import escape

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from services.final_fusion_predictor import load_assets, predict

st.set_page_config(
    page_title="Rakuten Multimodal Product Data Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 0.5rem !important; margin-top: 0 !important; }
    [data-testid="stAppViewContainer"] > section:first-child { padding-top: 0 !important; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[kind="header"],
    .st-emotion-cache-zq5wmm,
    .st-emotion-cache-1f391x5 { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"  # shared data folder agreed upon by the team

MM_LATE_DIR = DATA_DIR / "Streamlit" / "MM_CamemBERT_ConvNeXtBase_LateFusion"
MM_INTER_DIR = DATA_DIR / "Streamlit" / "MM_CamemBERT_ConvNeXtBase_IntermediateFusion"

ION_IMAGE_DIR = APP_DIR / "images"
ION_TRAIN_PATH = DATA_DIR / "raw" / "train_clean.csv"

# -------------------------------------------------------------------
# Search paths: local C:\Streamlit first, then the shared Kaggle dataset.
#
# Expected shareable structure:
#   Local:  C:\Streamlit\I12_ConvNeXT\...
#   Kaggle: /kaggle/input/streamlit/Streamlit/I12_ConvNeXT/...
#
# Both "streamlit" and "Streamlit" are supported on Kaggle to avoid
# case-sensitivity problems. APP_DIR is kept as a final fallback so the app
# also works when all files are placed next to app.py.
# -------------------------------------------------------------------
LOCAL_CANDIDATES = [
    DATA_DIR / "Streamlit" / "I12_ConvNeXT",  # primary — team-agreed project data folder
    DATA_DIR / "Streamlit",
    DATA_DIR,
]

KAGGLE_DATASETS = [
    # New dedicated Kaggle dataset: https://www.kaggle.com/datasets/arturillenseer/streamlit
    Path("/kaggle/input/streamlit"),
    # Older combined dataset path kept as fallback.
    Path("/kaggle/input/rakuten-product-images-ml"),
]
KAGGLE_CANDIDATES = []
for KAGGLE_DATASET in KAGGLE_DATASETS:
    KAGGLE_CANDIDATES.extend([
        KAGGLE_DATASET / "streamlit" / "I12_ConvNeXT",
        KAGGLE_DATASET / "streamlit",
        KAGGLE_DATASET / "Streamlit" / "I12_ConvNeXT",
        KAGGLE_DATASET / "Streamlit",
        # If the dataset root itself is the Streamlit folder.
        KAGGLE_DATASET / "I12_ConvNeXT",
        KAGGLE_DATASET,
    ])

# Fallback: also scan every Kaggle input dataset in case Kaggle changes the
# mounted folder name or the app is copied to a different dataset later.
KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.exists():
    try:
        for child in KAGGLE_INPUT.iterdir():
            if child.is_dir():
                KAGGLE_CANDIDATES.extend([
                    child / "streamlit" / "I12_ConvNeXT",
                    child / "streamlit",
                    child / "Streamlit" / "I12_ConvNeXT",
                    child / "Streamlit",
                    child,
                ])
    except Exception:
        pass

APP_FALLBACK_CANDIDATES = [
    APP_DIR,
]

SEARCH_DIRS = []
for base in LOCAL_CANDIDATES + KAGGLE_CANDIDATES + APP_FALLBACK_CANDIDATES:
    SEARCH_DIRS.extend([
        base,
        base / "artifacts",
        base / "artifacts" / "image_model",
        base / "artifacts" / "convnext_model",
        base / "artifacts" / "gradcam",
        base / "gradcam",
        base / "outputs",
        base / "outputs" / "convnext_base_final_v1",
        base / "convnext_base_final_v1",
        base / "I12_ConvNeXT",
    ])

_seen = set()
SEARCH_DIRS = [p for p in SEARCH_DIRS if not (str(p) in _seen or _seen.add(str(p)))]


def find_file(names):
    if isinstance(names, str):
        names = [names]
    for d in SEARCH_DIRS:
        for name in names:
            p = d / name
            if p.exists():
                return p
    return None


def find_gradcam_images():
    patterns = ["*gradcam*.png", "*GradCAM*.png", "*gradcam*.jpg", "*GradCAM*.jpg"]
    files = []
    for d in SEARCH_DIRS:
        if d.exists() and d.is_dir():
            for pattern in patterns:
                files.extend(d.glob(pattern))
                files.extend((d / "overlays").glob(pattern) if (d / "overlays").exists() else [])
    # Deduplicate while preserving filename-sorted presentation order.
    unique = {}
    for f in files:
        unique[str(f.resolve())] = f
    return sorted(unique.values(), key=lambda p: p.name)


FILES = {
    "metadata": find_file(["model_metadata.json", "convnext_model_metadata.json", "model_metadata_convnext_i12.json"]),
    "checkpoint": find_file(["best_model_base.pt", "best_model.pt"]),
    "history": find_file(["history.csv", "training_log.csv"]),
    "classification_report": find_file(["val_classification_report.txt", "classification_report.txt"]),
    "confusion_png": find_file(["val_confusion_matrix.png", "confusion_matrix.png"]),
    "confusion_npy": find_file(["confusion_matrix.npy", "val_confusion_matrix.npy"]),
    "learning_curves_png": find_file(["learning_curves.png", "training_history.png"]),
    "predictions": find_file(["val_predictions.csv", "predictions.csv"]),
    "id2label": find_file(["id2label.json", "index_to_label.json"]),
    "label2id": find_file(["label2id.json", "label_mapping.json"]),
    "logits": find_file(["val_logits_base.npy", "val_logits.npy", "image_val_logits.npy"]),
    "gradcam_predictions": find_file(["gradcam_prediction_table.csv"]),
    "gradcam_selected": find_file(["selected_gradcam_examples_4_groups.csv", "gradcam_index.csv"]),
}

DEFAULT_METADATA = {
    "display_name": "ConvNeXT-Base Image Model with Augmentation, Fully Unfrozen",
    "framework": "PyTorch",
    "architecture": "ConvNeXT-Base",
    "pretrained": True,
    "augmentation": "With augmentation / true",
    "freezing": "Fully unfrozen / fine-tuned",
    "checkpoint_file": "best_model_base.pt",
    "image_size": 224,
    "run_name": "convnext_base_final_v1",
    "max_epochs": 20,
    "batch_size": 16,
}

DEFAULT_ID2LABEL = {
    "0": 10, "1": 40, "2": 50, "3": 60, "4": 1140, "5": 1160, "6": 1180,
    "7": 1280, "8": 1281, "9": 1300, "10": 1301, "11": 1302, "12": 1320,
    "13": 1560, "14": 1920, "15": 1940, "16": 2060, "17": 2220, "18": 2280,
    "19": 2403, "20": 2462, "21": 2522, "22": 2582, "23": 2583, "24": 2585,
    "25": 2705, "26": 2905,
}

CATEGORY_NAMES = {
    "10": "Books",
    "40": "PC and console video games",
    "50": "Video game accessories",
    "60": "Video game consoles",
    "1140": "Geek merchandise and figurines",
    "1160": "Collectible cards",
    "1180": "Collectible board-game figurines",
    "1280": "Toys, plush toys and dolls",
    "1281": "Board and card games",
    "1300": "Toy cars and models",
    "1301": "Baby/child accessories and game furniture",
    "1302": "Outdoor games",
    "1320": "Women’s bags and early-childhood accessories",
    "1560": "Home furniture, decoration and storage",
    "1920": "Household linen",
    "1940": "Food and groceries",
    "2060": "Home lamps and decorative accessories",
    "2220": "Pet accessories",
    "2280": "Magazines",
    "2403": "Books and comics",
    "2462": "Video-game consoles and games",
    "2522": "Stationery and office storage",
    "2582": "Outdoor furniture and accessories",
    "2583": "Swimming-pool accessories",
    "2585": "Tools and gardening accessories",
    "2705": "Comics and books",
    "2905": "Downloadable games",
}

TABLE_HEIGHT = 680
BIG_TABLE_HEIGHT = 760
GRAPH_WIDTH = None
GRADCAM_WIDTH = 1050

# Fixed presentation order for interpretability examples.
# The dashboard shows up to three examples per category in this order.
GRADCAM_GROUP_ORDER = [
    "high_confidence_right_prediction",
    "high_confidence_wrong_prediction",
    "low_confidence_right_prediction",
    "low_confidence_wrong_prediction",
]
GRADCAM_EXAMPLES_PER_GROUP = 3

# One scenic city/church example was visually misleading for the presentation.
# Excluding it here makes Streamlit automatically use the next available example
# from the same interpretability category.
EXCLUDED_GRADCAM_LABEL_PAIRS = {("2060", "2522")}


def fit_table_height(df, max_height=TABLE_HEIGHT, min_height=78, row_height=36):
    """Height for scrollable dataframes without visible empty rows for short tables."""
    try:
        n_rows = len(df)
    except Exception:
        return max_height
    if n_rows <= 0:
        return min_height
    return min(max_height, max(min_height, (n_rows + 1) * row_height + 8))


def _format_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return escape(str(value))


def render_html_table(df, max_width="100%", compact=False):
    """Readable, wrapping HTML table for short report tables. No virtual empty rows."""
    if df is None or df.empty:
        st.info("No table data available.")
        return
    font = "1.02rem" if compact else "1.06rem"
    pad = "0.45rem 0.6rem" if compact else "0.62rem 0.75rem"
    header = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_format_cell(row[c])}</td>" for c in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    html = f'''
    <div class="table-scroll" style="max-width:{max_width}; overflow-x:auto; margin:0.35rem 0 1.0rem 0;">
      <table class="report-table" style="width:100%; border-collapse:collapse;">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    <style>
    .report-table th {{
        background:#f7f8fa; color:#4b5563; font-weight:600; text-align:left;
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal;
    }}
    .report-table td {{
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal; color:#262730;
    }}
    </style>
    '''
    st.markdown(html, unsafe_allow_html=True)

def render_html_table_with_highlights(df, max_width="100%", compact=False):
    font = "1.02rem" if compact else "1.06rem"
    pad = "0.45rem 0.6rem" if compact else "0.62rem 0.75rem"

    html = df.to_html(index=False, escape=False)

    html = f"""
    <div class="table-scroll" style="max-width:{max_width}; overflow-x:auto; margin:0.35rem 0 1.0rem 0;">
        {html}
    </div>
    <style>
    .table-scroll table {{
        width:100%;
        border-collapse:collapse;
    }}
    .table-scroll th {{
        background:#f7f8fa; color:#4b5563; font-weight:600; text-align:left;
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal;
    }}
    .table-scroll td {{
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal; color:#262730;
    }}
    </style>
    """

    st.markdown(html, unsafe_allow_html=True)


def category_display(label):
    if pd.isna(label):
        return "—"
    key = str(int(label)) if isinstance(label, float) and label.is_integer() else str(label)
    name = CATEGORY_NAMES.get(key)
    return f"{key} — {name}" if name else key


def render_grouped_html_table(df, group_col, highlight_cols=None, max_width="100%", compact=False):
    """HTML table where group_col uses rowspan.
    - Per-group best (by first highlight col): green border on all highlight_cols cells.
    - Global best row overall: red border on all highlight_cols cells (overrides green).
    - Last row of each group gets a thicker bottom separator."""
    if df is None or df.empty:
        st.info("No table data available.")
        return
    if highlight_cols is None:
        highlight_cols = ["Accuracy", "Macro F1"]
    highlight_cols = [c for c in highlight_cols if c in df.columns]
    sort_col = highlight_cols[0] if highlight_cols else None

    font = "1.02rem" if compact else "1.06rem"
    pad = "0.45rem 0.6rem" if compact else "0.62rem 0.75rem"
    other_cols = [c for c in df.columns if c != group_col]
    header = f"<th>{escape(str(group_col))}</th>" + "".join(f"<th>{escape(str(c))}</th>" for c in other_cols)

    seen = []
    for v in df[group_col]:
        if v not in seen:
            seen.append(v)

    def _try_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    # Per-group best local index
    best_local = {}
    global_best_val = None
    global_best_key = None  # (group_val, local_i)
    if sort_col:
        for gv in seen:
            grp = df[df[group_col] == gv]
            pairs = [(_try_float(row[sort_col]), i) for i, (_, row) in enumerate(grp.iterrows())]
            numeric = [(v, i) for v, i in pairs if v is not None]
            if numeric:
                best_v, best_i = max(numeric, key=lambda x: x[0])
                best_local[gv] = best_i
                if global_best_val is None or best_v > global_best_val:
                    global_best_val = best_v
                    global_best_key = (gv, best_i)

    last_group = seen[-1]
    body_rows = []
    for group_val in seen:
        group_rows = df[df[group_col] == group_val]
        n = len(group_rows)
        is_last_group = (group_val == last_group)
        for i, (_, row) in enumerate(group_rows.iterrows()):
            is_last_row = (i == n - 1)
            is_group_best = (best_local.get(group_val) == i)
            is_global_best = (global_best_key == (group_val, i))
            sep = "border-bottom:2.5px solid #9ca3af;" if (is_last_row and not is_last_group) else ""
            cells = ""
            if i == 0:
                cells += (
                    f'<td rowspan="{n}" style="font-weight:700; background:#eef2ff; '
                    f'text-align:center; vertical-align:middle; border:1px solid #c7d2fe; '
                    f'padding:{pad}; font-size:{font}; color:#3730a3; {sep}">'
                    f'{escape(str(group_val))}</td>'
                )
            for c in other_cols:
                cell_val = _format_cell(row[c])
                style = sep
                if c in highlight_cols:
                    if is_global_best:
                        style += "font-weight:700; border:2px solid #dc2626; color:#991b1b;"
                    elif is_group_best:
                        style += "font-weight:700; border:2px solid #059669; color:#065f46;"
                cells += f'<td style="{style}">{cell_val}</td>' if style else f"<td>{cell_val}</td>"
            body_rows.append(f"<tr>{cells}</tr>")

    html = f'''
    <div class="table-scroll" style="max-width:{max_width}; overflow-x:auto; margin:0.35rem 0 1.0rem 0;">
      <table class="report-table" style="width:100%; border-collapse:collapse;">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    <style>
    .report-table th {{
        background:#f7f8fa; color:#4b5563; font-weight:600; text-align:left;
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal;
    }}
    .report-table td {{
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal; color:#262730;
    }}
    </style>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_best_model_card(row):
    """Large, readable card for the best image-only model instead of a small one-row dataframe."""

    def val(name):
        v = row.get(name, "—")
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if name == "Macro F1" else f"{v:.3f}"
        return str(v)

    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb; border-radius:0.9rem; padding:1.1rem 1.25rem; background:#fafafa; margin:0.6rem 0 1rem 0;">
          <div style="font-size:1.45rem; font-weight:800; line-height:1.25; color:#262730; margin-bottom:0.25rem;">{val('Model')}</div>
          <div style="font-size:1.05rem; color:#4b5563; margin-bottom:0.9rem;">
            {val('Family')} · {val('Training strategy')} · {val('Augmentation')} augmentation · {val('Image size')} px
          </div>
          <div style="display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:0.75rem;">
            <div><div style="font-size:0.82rem; color:#6b7280;">Accuracy</div><div style="font-size:1.55rem; font-weight:700;">{val('Accuracy')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Macro F1</div><div style="font-size:1.55rem; font-weight:700;">{val('Macro F1')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Weighted F1</div><div style="font-size:1.55rem; font-weight:700;">{val('Weighted F1')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Best epoch</div><div style="font-size:1.55rem; font-weight:700;">{val('Best epoch')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Hardware / time</div><div style="font-size:1.55rem; font-weight:700; line-height:1.15;">{val('Hardware / time')}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Static report text for the image-modeling chapter.
# -------------------------------------------------------------------
IMAGE_MODEL_RESULTS = [
    {"Model": "Model_I1_CNN128_NoAug_FromScratch", "Family": "CNN baseline", "Image size": "128",
     "Training strategy": "From scratch", "Augmentation": "No", "Accuracy": 0.5746, "Macro F1": 0.5065,
     "Weighted F1": 0.5643, "Best epoch": "30", "Hardware / time": "Tesla T4 / 203 min"},
    {"Model": "Model_I2_CNN128_ModerateAug_FromScratch", "Family": "CNN baseline", "Image size": "128",
     "Training strategy": "From scratch", "Augmentation": "Moderate", "Accuracy": 0.5694, "Macro F1": 0.4984,
     "Weighted F1": 0.5578, "Best epoch": "45", "Hardware / time": "Tesla T4 / not stated"},
    {"Model": "Model_I3_CNN256_NoAug_FromScratch", "Family": "CNN baseline", "Image size": "256",
     "Training strategy": "From scratch", "Augmentation": "No", "Accuracy": 0.5767, "Macro F1": 0.5090,
     "Weighted F1": 0.5627, "Best epoch": "33", "Hardware / time": "Tesla T4 / 68.76 min"},
    {"Model": "Model_I5_ResNet50_NoAug_Frozen", "Family": "ResNet", "Image size": "224",
     "Training strategy": "Frozen pretrained backbone", "Augmentation": "No", "Accuracy": 0.5948, "Macro F1": 0.5540,
     "Weighted F1": 0.5888, "Best epoch": "15", "Hardware / time": "123.9 min; 2x Tesla T4 / Kaggle"},
    {"Model": "Model_I6_ResNet50_ModerateAug_Partial", "Family": "ResNet", "Image size": "224",
     "Training strategy": "Partial unfreezing", "Augmentation": "Moderate", "Accuracy": 0.6752, "Macro F1": 0.6410,
     "Weighted F1": 0.6718, "Best epoch": "17", "Hardware / time": "58.44 min; RTX 5070 Ti"},
    {"Model": "Model_I7_ResNet50_ModerateAug_Full", "Family": "ResNet", "Image size": "224",
     "Training strategy": "Full unfreezing", "Augmentation": "Moderate", "Accuracy": 0.6849, "Macro F1": 0.6533,
     "Weighted F1": 0.6842, "Best epoch": "10", "Hardware / time": "111.65 min; RTX 5070 Ti"},
    {"Model": "Model_I8_ResNet50_ModerateAug_FromScratch", "Family": "ResNet", "Image size": "224",
     "Training strategy": "Random initialization", "Augmentation": "Moderate", "Accuracy": 0.5768, "Macro F1": 0.5105,
     "Weighted F1": 0.5601, "Best epoch": "18", "Hardware / time": "329.37 min; Colab Pro Tesla 4 GPU"},
    {"Model": "Model_I8b_ResNet101_NoAug_Frozen", "Family": "ResNet", "Image size": "224",
     "Training strategy": "Frozen pretrained backbone", "Augmentation": "No", "Accuracy": "0.5425-0.5701",
     "Macro F1": "0.4971-0.5212", "Weighted F1": "—", "Best epoch": "16",
     "Hardware / time": "hardware/time incomplete"},
    {"Model": "Model_I9_ConvNeXt_Tiny_ModerateAug_Full", "Family": "ConvNeXt", "Image size": "224",
     "Training strategy": "Full unfreeze", "Augmentation": "Moderate", "Accuracy": 0.7144, "Macro F1": 0.6850,
     "Weighted F1": 0.7112, "Best epoch": "19", "Hardware / time": "71.04 min; RTX 5070 Ti"},
    {"Model": "Model_I10_EfficientNetB0_NoAug_Partial", "Family": "EfficientNet", "Image size": "224",
     "Training strategy": "Partial fine-tuning", "Augmentation": "No", "Accuracy": 0.6173, "Macro F1": 0.5684,
     "Weighted F1": 0.6089, "Best epoch": "-", "Hardware / time": "33.18 min; RTX PRO 6000 Blackwell SE"},
    {"Model": "Model_I11_EfficientNetB0_ModerateAug_Partial", "Family": "EfficientNet", "Image size": "224",
     "Training strategy": "Partial fine-tuning", "Augmentation": "Moderate", "Accuracy": 0.5990, "Macro F1": 0.5489,
     "Weighted F1": 0.5892, "Best epoch": "-", "Hardware / time": "CUDA GPU; duration not fixed"},
    {"Model": "Model_I12_ConvNeXt_Base_ModerateAug_Full", "Family": "ConvNeXt", "Image size": "224",
     "Training strategy": "Full unfreeze", "Augmentation": "Moderate", "Accuracy": 0.7200, "Macro F1": 0.6924,
     "Weighted F1": 0.7200, "Best epoch": "20", "Hardware / time": "160.82 min; RTX 5070 Ti"},
    {"Model": "Model_I13_DINOv2_TrainAug_Frozen", "Family": "DINOv2", "Image size": "224",
     "Training strategy": "Frozen pretrained backbone", "Augmentation": "Training transform from timm",
     "Accuracy": 0.6647, "Macro F1": 0.6199, "Weighted F1": None, "Best epoch": "-",
     "Hardware / time": "~60-62 min/epoch; Apple Silicon MPS"},
]

IMAGE_MODEL_LABELS = {
    # CNN baseline — differ by input size and augmentation
    "Model_I1_CNN128_NoAug_FromScratch": "128 px · no aug",
    "Model_I2_CNN128_ModerateAug_FromScratch": "128 px · aug",
    "Model_I3_CNN256_NoAug_FromScratch": "256 px · no aug",
    # ResNet — differ by training strategy / backbone depth
    "Model_I5_ResNet50_NoAug_Frozen": "Frozen · no aug",
    "Model_I6_ResNet50_ModerateAug_Partial": "Partial unfreeze",
    "Model_I7_ResNet50_ModerateAug_Full": "Full unfreeze",
    "Model_I8_ResNet50_ModerateAug_FromScratch": "From scratch",
    "Model_I8b_ResNet101_NoAug_Frozen": "ResNet101 · frozen",
    # ConvNeXt — differ by model variant
    "Model_I9_ConvNeXt_Tiny_ModerateAug_Full": "Tiny variant",
    "Model_I12_ConvNeXt_Base_ModerateAug_Full": "Base variant",
    # EfficientNet — differ by augmentation
    "Model_I10_EfficientNetB0_NoAug_Partial": "No aug",
    "Model_I11_EfficientNetB0_ModerateAug_Partial": "Moderate aug",
    # DINOv2 — single model
    "Model_I13_DINOv2_TrainAug_Frozen": "Frozen backbone",
}

# -------------------------------------------------------------------

AUGMENTATION_GROUPS = [
    {
        "Group": "No augmentation",
        "Models": "I1, I3, I5, I8b, I10",
        "Resize / crop": "Resize to model input size; no random crop.",
        "Geometric changes": "None.",
        "Color / policy changes": "None.",
        "Tensor + normalization": "ToTensor + ImageNet normalization where pretrained backbones are used.",
        "Comment": "Stable baseline because product images are usually upright and photographed in fairly consistent catalog-like conditions."
    },
    {
        "Group": "Moderate augmentation — ConvNeXt exact implementation",
        "Models": "I9, I12",
        "Resize / crop": "RandomResizedCrop(224, scale=(0.7, 1.0)).",
        "Geometric changes": "RandomHorizontalFlip().",
        "Color / policy changes": "TrivialAugmentWide().",
        "Tensor + normalization": "ToTensor + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).",
        "Comment": "Identical transform confirmed in both the ConvNeXt-Tiny (I9) and ConvNeXt-Base (I12) training scripts."
    },
    {
        "Group": "Moderate augmentation — all other models",
        "Models": "I2, I6, I7, I8, I11",
        "Resize / crop": "Moderate random cropping / resizing reported where applicable; exact settings vary or are not present in the current app artifact set.",
        "Geometric changes": "Moderate flips, rotations, or translations depending on the specific notebook.",
        "Color / policy changes": "Moderate color, contrast, or policy augmentation depending on the specific notebook.",
        "Tensor + normalization": "Model-specific preprocessing and normalization.",
        "Comment": "Grouped together to avoid falsely claiming that every moderate run used the exact ConvNeXt transform."
    },
    {
        "Group": "Training transform from timm",
        "Models": "I13",
        "Resize / crop": "timm/DINOv2 training transform.",
        "Geometric changes": "Defined by the timm configuration used in the original DINOv2 notebook.",
        "Color / policy changes": "Defined by the timm configuration used in the original DINOv2 notebook.",
        "Tensor + normalization": "timm/DINOv2 preprocessing.",
        "Comment": "Reported separately because it is not the same augmentation family as the CNN/ResNet/ConvNeXt experiments."
    },
]

AUGMENTATION_DECISION_ROWS = [
    {"Point": "Why the report says moderate augmentation",
     "Explanation": "A stronger augmentation setup was tested, but it degraded performance substantially. The report notes validation accuracy around 0.21, training accuracy around 0.17, and early stopping after only three epochs."},
    {"Point": "Why stronger augmentation likely hurt",
     "Explanation": "The dataset already contains high visual variety across many product categories. Excessive synthetic variation can make visually distinct product cues less stable and increase class confusion."},
    {"Point": "Why product images need careful augmentation",
     "Explanation": "Many product photos are upright, centered, and photographed under similar catalog-like conditions. Large rotations, translations, or zooms can create unrealistic examples rather than better generalization."},
    {"Point": "Practical conclusion",
     "Explanation": "Moderate augmentation is a compromise: it adds robustness without destroying the original product structure that the image model needs for classification."},
]


# Loaders and helpers
# -------------------------------------------------------------------
@st.cache_data
def load_json(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_text(path):
    if path is None:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_csv(path):
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_npy(path):
    if path is None:
        return None
    return np.load(path)


def parse_report_metrics(report_text):
    metrics = {}
    acc = re.search(r"\n\s*accuracy\s+([0-9.]+)\s+(\d+)", "\n" + report_text)
    if acc:
        metrics["accuracy"] = float(acc.group(1))
        metrics["validation_samples"] = int(acc.group(2))
    macro = re.search(r"\n\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", "\n" + report_text)
    if macro:
        metrics["macro_precision"] = float(macro.group(1))
        metrics["macro_recall"] = float(macro.group(2))
        metrics["macro_f1"] = float(macro.group(3))
    weighted = re.search(r"\n\s*weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", "\n" + report_text)
    if weighted:
        metrics["weighted_precision"] = float(weighted.group(1))
        metrics["weighted_recall"] = float(weighted.group(2))
        metrics["weighted_f1"] = float(weighted.group(3))
    return metrics


def prepare_predictions(df):
    if df.empty:
        return df
    out = df.copy()
    if "true_label" not in out.columns and "prdtypecode" in out.columns:
        out["true_label"] = out["prdtypecode"]
    if "pred_label" not in out.columns and "pred" in out.columns:
        out["pred_label"] = out["pred"]
    if "correct" not in out.columns and {"true_label", "pred_label"}.issubset(out.columns):
        out["correct"] = out["true_label"].astype(str) == out["pred_label"].astype(str)
    if "image_filename" not in out.columns and {"productid", "imageid"}.issubset(out.columns):
        out["image_filename"] = out.apply(lambda r: f"image_{r['imageid']}_product_{r['productid']}.jpg", axis=1)
    return out


def format_path(p):
    return str(p) if p else "Not found"


def find_image_path(row, image_dirs):
    # Prefer the original absolute path when it exists.
    if "image_path_local" in row and isinstance(row.get("image_path_local"), str):
        p = Path(row.get("image_path_local"))
        if p.exists():
            return p
    filename = row.get("image_filename")
    if not filename:
        return None
    for d in image_dirs:
        p = Path(d.strip()) / filename
        if p.exists():
            return p
    return None


def format_value(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def placeholder_page(title):
    st.title("Rakuten Multimodal Product Classification")
    st.header(title)
    st.info("This chapter is intentionally left blank in this presentation version.")


def render_prediction_tool():
    """Render the standalone Prediction page inside this custom navigation app."""
    prediction_page = APP_DIR / "prediction_tool.py"
    if not prediction_page.exists():
        st.error(f"Prediction page not found: {prediction_page}")
        return

    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        runpy.run_path(str(prediction_page), run_name="__streamlit_prediction_page__")
    finally:
        st.set_page_config = original_set_page_config


def summary_table(metadata, metrics, id2label, preds):
    rows = [
        {"Section": "Model", "Field": "Model / chapter", "Value": "5.5 Best model: ConvNeXT-Base"},
        {"Section": "Model", "Field": "Display name", "Value": metadata.get("display_name")},
        {"Section": "Model", "Field": "Architecture", "Value": metadata.get("architecture")},
        {"Section": "Model", "Field": "Framework", "Value": metadata.get("framework")},
        {"Section": "Model", "Field": "Pretrained", "Value": format_value(metadata.get("pretrained"))},
        {"Section": "Model", "Field": "Augmentation", "Value": metadata.get("augmentation")},
        {"Section": "Model", "Field": "Freezing", "Value": metadata.get("freezing")},
        {"Section": "Model", "Field": "Image size", "Value": metadata.get("image_size")},
        {"Section": "Model", "Field": "Checkpoint", "Value": "Available" if FILES["checkpoint"] else "Not found"},
        {"Section": "Training", "Field": "Max epochs", "Value": metadata.get("max_epochs")},
        {"Section": "Training", "Field": "Batch size", "Value": metadata.get("batch_size")},
        {"Section": "Validation", "Field": "Classes", "Value": len(id2label) or 27},
        {"Section": "Validation", "Field": "Validation samples",
         "Value": metrics.get("validation_samples", len(preds) if not preds.empty else None)},
        {"Section": "Validation", "Field": "Accuracy", "Value": metrics.get("accuracy")},
        {"Section": "Validation", "Field": "Macro F1", "Value": metrics.get("macro_f1")},
        {"Section": "Validation", "Field": "Weighted F1", "Value": metrics.get("weighted_f1")},
        {"Section": "Validation", "Field": "Macro precision", "Value": metrics.get("macro_precision")},
        {"Section": "Validation", "Field": "Macro recall", "Value": metrics.get("macro_recall")},
    ]
    df = pd.DataFrame(rows)
    df["Value"] = df["Value"].apply(format_value)
    return df


def parse_classification_report_table(report_text, id2label):
    """Convert sklearn text classification_report into presentation-friendly tables."""
    class_rows = []
    summary_rows = []
    if not report_text:
        return pd.DataFrame(), pd.DataFrame()

    for line in report_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            idx = m.group(1)
            product_class = id2label.get(idx, id2label.get(str(idx), idx)) if id2label else idx
            class_rows.append({
                "Class index": int(idx),
                "Product class": product_class,
                "Precision": float(m.group(2)),
                "Recall": float(m.group(3)),
                "F1-score": float(m.group(4)),
                "Support": int(m.group(5)),
            })
            continue

        m = re.match(r"^\s*accuracy\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            summary_rows.append({
                "Metric": "Accuracy",
                "Precision": "—",
                "Recall": "—",
                "F1-score / score": float(m.group(1)),
                "Support": int(m.group(2)),
            })
            continue

        m = re.match(r"^\s*(macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            summary_rows.append({
                "Metric": m.group(1).title(),
                "Precision": float(m.group(2)),
                "Recall": float(m.group(3)),
                "F1-score / score": float(m.group(4)),
                "Support": int(m.group(5)),
            })

    return pd.DataFrame(class_rows), pd.DataFrame(summary_rows)


def render_metric_cards(metrics, value_font_size="2.0rem"):
    acc = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    weighted_f1 = metrics.get("weighted_f1")
    support = metrics.get("validation_samples")

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, int):
            return f"{v:,}"
        return f"{v:.3f}"

    st.markdown(
        f"""
        <style>
        .metric-grid {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 1.5rem 0;}}
        .metric-card {{border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1rem 1.2rem; background: #fafafa;}}
        .metric-card-label {{font-size: 0.95rem; color: #4b5563; margin-bottom: 0.35rem;}}
        .metric-card-value {{font-size: {value_font_size}; font-weight: 700; color: #262730; line-height: 1.1;}}
        </style>
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-card-label">Accuracy</div><div class="metric-card-value">{fmt(acc)}</div></div>
          <div class="metric-card"><div class="metric-card-label">Macro F1</div><div class="metric-card-value">{fmt(macro_f1)}</div></div>
          <div class="metric-card"><div class="metric-card-label">Weighted F1</div><div class="metric-card-value">{fmt(weighted_f1)}</div></div>
          <div class="metric-card"><div class="metric-card-label">Validation samples</div><div class="metric-card-value">{fmt(support)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def readable_gradcam_group(group):
    text = str(group or "Grad-CAM example")
    confidence = "↓ Low confidence" if "low_confidence" in text else "↑ High confidence" if "high_confidence" in text else "Confidence"
    correctness = "Wrong prediction" if "wrong_prediction" in text else "Right prediction" if "right_prediction" in text else "Prediction"
    color = "#b91c1c" if "wrong_prediction" in text else "#15803d" if "right_prediction" in text else "#262730"
    return confidence, correctness, color


def infer_gradcam_group(confidence, correct):
    try:
        conf = float(confidence)
    except Exception:
        conf = None
    confidence_part = "low_confidence" if conf is not None and conf < 0.5 else "high_confidence"
    correct_part = "right_prediction" if bool(correct) else "wrong_prediction"
    return f"{confidence_part}_{correct_part}"


def gradcam_observation(row):
    group = str(row.get("group", ""))
    true_label = str(row.get("true_label", ""))
    pred_label = str(row.get("pred_label", ""))
    specific = {
        ("1280",
         "2583"): "Attention is diffuse and partly falls on the background rather than the product itself — a hallmark of the low-confidence wrong prediction the model produces for Toys/Plush. The class is one of the hardest in the dataset (val F1 0.35).",
        ("1302",
         "1560"): "The heatmap focuses on an object detail that is visually consistent with home furniture rather than an outdoor game — the model's attention is plausible but misaligned. Outdoor Games (1302) scores only F1 0.43 on the validation set.",
        ("2060",
         "2522"): "Activation concentrates near the image edges and packaging graphics rather than the central product silhouette, suggesting that ambient context or printed text is driving the stationery prediction instead of product shape.",
        ("1280",
         "2522"): "Attention is localised to a small subset of items in the image rather than the overall product group. The framing makes the product resemble stationery — consistent with the 1280 → 2522 error pattern seen in the confusion matrix.",
        ("1560",
         "2522"): "The model attends strongly to the flat front panel of the item. A shelf or door panel reads visually like stationery storage, explaining a confident but wrong prediction. Furniture (1560) → Lamps/Decor (2060) and Stationery (2522) are the top two error targets for this class.",
        ("1320",
         "1320"): "Correct prediction but uncertain (conf 0.149). The heatmap shows scattered attention across the product rather than a sharp focus — Women's Bags (1320, val F1 0.46) is a challenging class because bags, pouches and early-childhood accessories share similar shapes and colours.",
        ("1302",
         "1302"): "Strong, localised attention on the game equipment with very high confidence (0.999). Outdoor Games products have distinctive shapes and colour patterns that ConvNeXt-Base recognises reliably when the image is clear.",
    }
    if (true_label, pred_label) in specific:
        return specific[(true_label, pred_label)]
    if "right_prediction" in group and "high_confidence" in group:
        return "Sensible focus on the visible product region; the heatmap supports the correct high-confidence prediction."
    if "right_prediction" in group and "low_confidence" in group:
        return "Correct but uncertain; the focus is less decisive or the visual evidence is ambiguous."
    if "wrong_prediction" in group and "high_confidence" in group:
        return "High-confidence mistake; inspect whether the model focuses on background, packaging text, or a misleading product detail."
    if "wrong_prediction" in group and "low_confidence" in group:
        return "Low-confidence error; the image likely contains ambiguous cues or diffuse attention."
    return "Inspect whether the highlighted region corresponds to a plausible product cue."


def render_gradcam_header(row):
    confidence_label, correctness_label, color = readable_gradcam_group(row.get("group", ""))
    true_label = category_display(row.get("true_label", "—"))
    pred_label = category_display(row.get("pred_label", "—"))
    conf = row.get("confidence", None)
    conf_text = f"{float(conf):.3f}" if pd.notna(conf) else "—"
    cam_target = category_display(row.get("cam_target", row.get("pred_label", "—")))
    st.markdown(
        f"""
        <div style="margin-top: 1.4rem; margin-bottom: 0.55rem;">
          <div style="font-size: 1.35rem; font-weight: 700; line-height: 1.2;">
            <span style="color: #262730;">{confidence_label}</span>
            <span style="color: #6b7280;"> — </span>
            <span style="color: {color};">{correctness_label}</span>
          </div>
          <div style="font-size: 1.02rem; color: #262730; margin-top: 0.25rem;">
            Correct category: <b>{true_label}</b><br/>
            Predicted category: <b>{pred_label}</b><br/>
            Confidence: <b>{conf_text}</b> &nbsp;|&nbsp; CAM target: <b>{cam_target}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_observation_box(row):
    st.markdown(
        f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 0.7rem; padding: 0.9rem 1rem; background: #fafafa; margin-top: 1.0rem;">
          <div style="font-weight: 700; margin-bottom: 0.35rem;">Observation</div>
          <div style="font-size: 0.98rem; color: #374151; line-height: 1.45;">{gradcam_observation(row)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_gradcam_display_image(path):
    """Load a pre-rendered Grad-CAM panel and remove the old text title embedded above the pictures."""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # The saved Grad-CAM panels contain a result string at the top.
        # Cropping about 8% removes that result line while keeping the actual panels.
        crop_top = min(max(int(h * 0.135), 95), 150)
        return img.crop((0, crop_top, w, h))
    except Exception:
        return str(path)


def parse_gradcam_filename(path, id2label):
    name = path.name
    # Example: 01_gradcam_idx2101_true11_pred11_conf0.999_pred.png
    m = re.search(r"idx(?P<idx>\d+)_true(?P<true_id>\d+)_pred(?P<pred_id>\d+)_conf(?P<conf>[0-9.]+)_", name)
    if not m:
        return {"file": path, "filename": name, "group": "Grad-CAM example"}
    label_lookup = {**DEFAULT_ID2LABEL, **{str(k): v for k, v in (id2label or {}).items()}}
    true_id = m.group("true_id")
    pred_id = m.group("pred_id")
    true_label = label_lookup.get(str(true_id), true_id)
    pred_label = label_lookup.get(str(pred_id), pred_id)
    conf = float(m.group("conf"))
    correct = str(true_label) == str(pred_label)
    return {
        "file": path,
        "filename": name,
        "idx": int(m.group("idx")),
        "true_id": int(true_id),
        "pred_id": int(pred_id),
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": conf,
        "correct": correct,
        "group": infer_gradcam_group(conf, correct),
    }


def prepare_gradcam_table(images, selected_df, id2label):
    rows = [parse_gradcam_filename(p, id2label) for p in images]
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Enrich group from selected_gradcam_examples_4_groups.csv when possible.
    if not selected_df.empty and {"true_label", "pred_label", "confidence", "group"}.issubset(selected_df.columns):
        groups = []
        for _, r in out.iterrows():
            candidates = selected_df.copy()
            candidates = candidates[candidates["true_label"].astype(str) == str(r.get("true_label"))]
            candidates = candidates[candidates["pred_label"].astype(str) == str(r.get("pred_label"))]
            if "confidence" in candidates.columns and not candidates.empty:
                candidates = candidates.assign(
                    _dist=(candidates["confidence"].astype(float) - float(r.get("confidence", 0))).abs())
                candidates = candidates.sort_values("_dist")
                if not candidates.empty and candidates.iloc[0]["_dist"] < 0.002:
                    groups.append(candidates.iloc[0]["group"])
                    continue
            groups.append(r.get("group", infer_gradcam_group(r.get("confidence"), r.get("correct"))))
        out["group"] = groups
    # Ensure every row has a meaningful group even when no selected CSV was available.
    if "group" not in out.columns or out["group"].eq("Grad-CAM example").any():
        out["group"] = out.apply(lambda r: infer_gradcam_group(r.get("confidence"), r.get("correct")), axis=1)
    return out


def make_label_table(id2label):
    if not id2label:
        return pd.DataFrame()
    label_rows = [{"Index": int(k), "Product class": v} for k, v in id2label.items()]
    labels_df = pd.DataFrame(label_rows).sort_values("Index")
    labels_df["Product class"] = labels_df["Product class"].apply(category_display)
    return labels_df


def is_excluded_gradcam_example(row):
    true_label = str(row.get("true_label", ""))
    pred_label = str(row.get("pred_label", ""))
    return (true_label, pred_label) in EXCLUDED_GRADCAM_LABEL_PAIRS


def ordered_gradcam_examples(gradcam_df):
    if gradcam_df.empty:
        return gradcam_df
    out = gradcam_df.copy()
    out = out[~out.apply(is_excluded_gradcam_example, axis=1)]
    out["_group_order"] = out["group"].apply(
        lambda g: GRADCAM_GROUP_ORDER.index(g) if g in GRADCAM_GROUP_ORDER else len(GRADCAM_GROUP_ORDER))
    # Within each group, keep examples deterministic and easy to review.
    if "confidence" in out.columns:
        out = out.sort_values(["_group_order", "confidence", "filename"], ascending=[True, False, True])
    else:
        out = out.sort_values(["_group_order", "filename"], ascending=[True, True])
    return out.drop(columns=["_group_order"], errors="ignore")


# -------------------------------------------------------------------
# Prediction Tool Helpers
# -------------------------------------------------------------------
def normalize_filename(value):
    if pd.isna(value): return ""
    return Path(str(value)).name.strip().lower()


def normalize_column_name(value):
    return str(value).strip().lower()


def safe_cell_to_text(value):
    if pd.isna(value): return ""
    text = str(value).strip()
    if text.lower() in ["nan", "none", "null"]: return ""
    return text


def find_column(row, candidates):
    normalized_cols = {normalize_column_name(col): col for col in row.index}
    for candidate in candidates:
        col = normalized_cols.get(normalize_column_name(candidate))
        if col is not None: return col
    return None


def find_metadata_row(df, image_filename):
    image_filename_norm = normalize_filename(image_filename)
    candidate_columns = ["image_name", "filename", "file_name", "image_filename", "image_path", "path", "image"]
    normalized_df_cols = {normalize_column_name(col): col for col in df.columns}

    for candidate_col in candidate_columns:
        real_col = normalized_df_cols.get(normalize_column_name(candidate_col))
        if real_col is not None:
            matches = df[df[real_col].apply(normalize_filename) == image_filename_norm]
            if not matches.empty: return matches.iloc[0]

    imageid_col = normalized_df_cols.get("imageid") or normalized_df_cols.get("image_id")
    productid_col = normalized_df_cols.get("productid") or normalized_df_cols.get("product_id")

    if imageid_col is not None and productid_col is not None:
        expected_names = df.apply(lambda row: f"image_{row[imageid_col]}_product_{row[productid_col]}.jpg".lower(),
                                  axis=1)
        matches = df[expected_names == image_filename_norm]
        if not matches.empty: return matches.iloc[0]

    for col in df.columns:
        if df[col].dtype == "object":
            matches = df[df[col].apply(normalize_filename) == image_filename_norm]
            if not matches.empty: return matches.iloc[0]
    return None


def get_text_from_row(row):
    if row is None: return "", ""
    designation_candidates = ["designation", "title", "product_title", "product_name", "name"]
    description_candidates = ["description", "description_dedup", "description dedup", "description_clean",
                              "clean_description", "product_description", "desc"]
    designation_col = find_column(row, designation_candidates)
    description_col = find_column(row, description_candidates)
    designation = safe_cell_to_text(row[designation_col]) if designation_col is not None else ""
    description = safe_cell_to_text(row[description_col]) if description_col is not None else ""
    return designation, description


def get_image_id_from_row(row):
    if row is None: return "not found in csv file"
    image_id_candidates = ["imageid", "image_id", "image id"]
    image_id_col = find_column(row, image_id_candidates)
    if image_id_col is not None:
        image_id = safe_cell_to_text(row[image_id_col])
        if image_id: return image_id
    return "not found in csv file"


def format_prediction_label(code):
    category_name = CATEGORY_NAMES.get(str(code), "Unknown product type")
    return f"{code} - {category_name}"


def clear_image_related_state():
    st.session_state.designation_value = ""
    st.session_state.description_value = ""
    st.session_state.matched_image_id = ""
    st.session_state.prediction_output = None


@st.cache_resource
def get_assets():
    return load_assets()


@st.cache_data
def load_ion_train_data():
    try:
        return pd.read_csv(ION_TRAIN_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def get_ion_dataset_examples(random_state=22):
    df = load_ion_train_data()
    if df.empty:
        return df
    return df.sample(min(5, len(df)), random_state=random_state)


metadata = load_json(FILES["metadata"]) or DEFAULT_METADATA
report_text = load_text(FILES["classification_report"])
metrics = parse_report_metrics(report_text)
id2label = {**DEFAULT_ID2LABEL, **{str(k): v for k, v in (load_json(FILES["id2label"]) or {}).items()}}
label2id = load_json(FILES["label2id"]) or {}
preds = prepare_predictions(load_csv(FILES["predictions"]))
history = load_csv(FILES["history"])
gradcam_selected = load_csv(FILES["gradcam_selected"])
gradcam_prediction_table = load_csv(FILES["gradcam_predictions"])
gradcam_images = find_gradcam_images()
gradcam_table = prepare_gradcam_table(gradcam_images, gradcam_selected, id2label)


def _mm_path(base, filename):
    p = base / filename
    return p if p.exists() else None


mm_late_meta = load_json(_mm_path(MM_LATE_DIR, "run_metadata.json")) or {}
mm_late_report = load_text(_mm_path(MM_LATE_DIR, "fusion_classification_report.txt")) or ""
mm_late_cm_png = _mm_path(MM_LATE_DIR, "confusion_matrix.png")
mm_late_preds = prepare_predictions(load_csv(_mm_path(MM_LATE_DIR, "val_predictions.csv")))
mm_inter_meta = load_json(_mm_path(MM_INTER_DIR, "run_metadata.json")) or {}
mm_inter_history = pd.DataFrame(load_json(_mm_path(MM_INTER_DIR, "history.json")) or [])
mm_inter_cm_png = _mm_path(MM_INTER_DIR, "confusion_matrix.png")

PROJ_DIR = APP_DIR.parent
_rn_i6_hist = PROJ_DIR / "outputs" / "I6_ResNet50_ModerateAug_Partial" / "history.json"
_rn_i7_hist = PROJ_DIR / "outputs" / "I7_ResNet50_ModerateAug_Full" / "history.json"
_rn_i5_png_dir = PROJ_DIR / "outputs" / "image_modeling" / "I5_ResNet50_NoAug_Frozen"
resnet_i6_history = pd.DataFrame(load_json(_rn_i6_hist if _rn_i6_hist.exists() else None) or [])
resnet_i7_history = pd.DataFrame(load_json(_rn_i7_hist if _rn_i7_hist.exists() else None) or [])

_cnxt_i9_hist = PROJ_DIR / "outputs" / "I9_ConvNeXt_Tiny_ModerateAug_Full" / "history.csv"
convnext_i9_history = load_csv(_cnxt_i9_hist if _cnxt_i9_hist.exists() else None)
# I12 training history is already loaded via FILES["history"] → `history` variable


# Global readability tweaks for 1080p and 4K screens.
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 17px; }
    .stMarkdown p, .stMarkdown li { font-size: 1.04rem; line-height: 1.48; }
    h1 { line-height: 1.15 !important; }
    h2, h3 { line-height: 1.22 !important; }
    div[data-testid="stDataFrame"] { font-size: 1.02rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Sidebar navigation: button-style navigation with chapter hierarchy.
# -------------------------------------------------------------------
st.sidebar.title("Rakuten Multimodal Product Data Classification")
# st.sidebar.markdown("### Navigation")

NAV_LEVELS = {
    "1. Overview": 0,
    "1.1 Workflow": 1,
    "2. Data Exploration": 0,
    "2.1 Text": 1,
    "2.11 Vocabulary": 2,
    "2.2 Image": 1,
    "3. Preprocessing": 0,
    "3.1 Text Preprocessing": 1,
    "4. Text Modeling": 0,
    "4.1 Overview": 1,
    "4.2 Best model": 1,
    "4.21 CamemBERT vs TF-IDF": 2,
    "5. Image Modeling": 0,
    "5.1 Data Augmentation": 1,
    "5.2 CNN Models": 1,
    "5.3 ResNet Models": 1,
    "5.4 ConvNeXt Models": 1,
    "5.5 Best Image Model": 1,
    "5.5.1 Interpretability": 2,
    "5.6 Image Modeling Conclusion": 1,
    "6. Multimodal": 0,
    "6.1 Simple Fusion": 1,
    "6.2 Intermediate Fusion": 1,
    "6.3 Gated Fusion": 1,
    "6.4 CLIP Models": 1,
    "6.5 Best model — Summary": 1,
    "6.6 Multimodal Conclusion": 1,
    "7. Prediction Tool": 0,
    "8. Project Conclusions": 0,
    "9. Final Benchmark": 0,
}

NAV_DISPLAY = {
    "1. Overview": "Overview",
    "1.1 Workflow": "Workflow",
    "2. Data Exploration": "Data Exploration",
    "2.1 Text": "Text",
    "2.11 Vocabulary": "Vocabulary",
    "2.2 Image": "Image",
    "3. Preprocessing": "Preprocessing",
    "3.1 Text Preprocessing": "Text",
    "4. Text Modeling": "Text Modeling",
    "4.1 Overview": "Overview Text models",
    "4.2 Best model": "Best model",
    "4.21 CamemBERT vs TF-IDF": "vs TF-IDF",
    "5. Image Modeling": "Image Modeling",
    "5.1 Data Augmentation": "Data Augmentation",
    "5.2 CNN Models": "CNN Models",
    "5.3 ResNet Models": "ResNet Models",
    "5.4 ConvNeXt Models": "ConvNeXt Models",
    "5.5 Best Image Model": "Best Image Model",
    "5.5.1 Interpretability": "Error Analysis & Grad-CAM",
    "5.6 Image Modeling Conclusion": "Image Modeling Conclusion",
    "6. Multimodal": "Multimodal",
    "6.1 Simple Fusion": "Simple Fusion",
    "6.2 Intermediate Fusion": "Intermediate Fusion",
    "6.3 Gated Fusion": "Gated Fusion",
    "6.4 CLIP Models": "CLIP Models",
    "6.5 Best model — Summary": "Best model — Summary",
    "6.6 Multimodal Conclusion": "Multimodal Conclusion",
    "7. Prediction Tool": "Prediction Tool",
    "8. Project Conclusions": "Project Conclusions",
    "9. Final Benchmark": "Final Benchmark",
}
NAV_ITEMS = list(NAV_LEVELS.keys())
DEFAULT_PAGE = "1. Overview"


# Query-parameter navigation makes the sidebar look and behave like regular buttons,
# while keeping the selected chapter stable after Streamlit reruns.
def _get_query_page():
    try:
        raw = st.query_params.get("page")
    except Exception:
        raw = None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return raw if raw in NAV_ITEMS else None


page = _get_query_page() or st.session_state.get("page", DEFAULT_PAGE)
if page not in NAV_ITEMS:
    page = DEFAULT_PAGE
st.session_state["page"] = page

from urllib.parse import quote

# Button-style navigation. The HTML is intentionally left-aligned with no
# leading indentation so Streamlit/Markdown does not render it as a code block.
nav_css = """
<style>
section[data-testid="stSidebar"] {
    background: #ffffff !important;
}
    section[data-testid="stSidebarNav"] {
        display: none !important;
    }
section[data-testid="stSidebar"] h1 {
    font-size: 1.35rem !important;
    line-height: 1.2 !important;
    margin-bottom: 0.75rem !important;
}
.sidebar-nav {
    margin-top: 0.2rem;
    margin-bottom: 0.9rem;
   # max-height: calc(100vh - 100px); # 
    max-height: none; 
    overflow-y: auto;
    overflow-x: hidden;
}
.sidebar-nav a,
.sidebar-nav a:link,
.sidebar-nav a:visited,
.sidebar-nav a:hover,
.sidebar-nav a:active {
    text-decoration: none !important;
    color: inherit !important;
}
.sidebar-nav .nav-button,
.sidebar-nav .nav-button * {
    text-decoration: none !important;
}
.nav-button {
    display: block;
    width: 100%;
    box-sizing: border-box;
    border: 1px solid #e5e7eb;
    border-radius: 0.55rem;
    background: #f3f4f6;
    margin-left: 0 !important;
    margin-right: 0 !important;
    line-height: 1.22;
    white-space: normal;
    overflow-wrap: anywhere;
    text-align: left;
    transition: background 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
}
.nav-button:hover {
    background: #e5e7eb;
    border-color: #cbd5e1;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.nav-button.active {
    border-color: #ff4b4b;
    background: #fff1f1;
    color: #111827;
    font-weight: 700;
}
.nav-level-0 {
    color: #111827;
    font-size: 0.95rem;
    font-weight: 700;
    padding: 0.35rem 0.60rem;
    min-height: 1.9rem;
    margin-top: 0.35rem;
    margin-bottom: 0.08rem;
}
.nav-level-1 {
    color: #374151;
    font-size: 0.84rem;
    font-weight: 600;
    padding: 0.28rem 0.60rem;
    min-height: 1.7rem;
    margin-top: 0.06rem;
    margin-bottom: 0.06rem;
}
.nav-level-2 {
    color: #6b7280;
    font-size: 0.76rem;
    font-weight: 500;
    padding: 0.22rem 0.60rem;
    min-height: 1.5rem;
    margin-top: 0.04rem;
    margin-bottom: 0.04rem;
}
.sidebar-nav a:first-of-type .nav-button {
    margin-top: 0 !important;
}
</style>
"""
#@st.cache_data
def _build_nav_html(active_page: str) -> str:
    items = [nav_css, '<div class="sidebar-nav">']
    for item in NAV_ITEMS:
        label = escape(NAV_DISPLAY.get(item, item))
        level = NAV_LEVELS[item]
        active = " active" if item == active_page else ""
        href = f"?page={quote(item)}"
        items.append(
            f'<a href="{href}" target="_self"><div class="nav-button nav-level-{level}{active}">{label}</div></a>')
    items.append("</div>")
    return "\n".join(items)

st.sidebar.markdown(_build_nav_html(page), unsafe_allow_html=True)

# -------------------------------------------------------------------
# Blank report chapters
# -------------------------------------------------------------------
if page not in NAV_ITEMS:
    placeholder_page(page)

elif page == "1. Overview":
    st.header("1. Project overview")

    st.write(
        """
        This project addresses automatic product classification for the Rakuten marketplace.
        Each product must be assigned to one of 27 product categories (`prdtypecode`) using
        both textual and visual information.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Categories", "27")
    col2.metric("Train samples", "84 916")
    col3.metric("Test samples", "13 812")
    col4.metric("Missing descriptions", "~35%")

    st.subheader("Business motivation")

    st.write(
        """
        Product category prediction is a key task for marketplace platforms.
        Reliable classification improves product search, filtering, recommendation quality,
        catalogue consistency, and user experience.
        """
    )

    st.subheader("Input data")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            **Textual information**
            - `designation`: product title
            - `description`: optional product description
             """
        )

    with c2:
        st.markdown(
            """
            **Visual information**
            - Product images stored separately
            - Linked through `imageid` and `productid`
            """
        )

    st.success(
        """
        The objective is to build a robust classification pipeline that predicts the correct
        product category for unseen products using the available title, description, and image.
        """
    )

elif page == "1.1 Workflow":

    st.header("Modeling strategy")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            **1. Text models**
            - Start with TF-IDF baselines
            - Compare classical classifiers
            - Test sentence embeddings
            - Fine-tune CamemBERT
            """
        )

    with m2:
        st.markdown(
            """
            **2. Image models**
            - Start with a simple CNN baseline
            - Move to transfer learning
            - Compare stronger architectures
            - Use GradCAM for visual checks
            """
        )

    with m3:
        st.markdown(
            """
            **3. Multimodal models**
            - Combine text and image information
            - Compare intermediate and late fusion
            - Test CLIP-based representations
            - Focus on robust final prediction
            """
        )

    st.subheader("Evaluation metric")

    st.write(
        """
        The main evaluation metric is **macro F1-score**. This is more appropriate than
        accuracy because the dataset contains 27 product categories and the classes are
        not equally represented.
        """
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            """
            **Why not accuracy only?**
            - Accuracy can be dominated by frequent classes
            - Rare categories may perform poorly without strongly affecting accuracy
            - A high accuracy score can hide weak per-class performance
            """
        )

    with col_m2:
        st.markdown(
            """
            **Why macro F1?**
            - Computes F1-score independently for each class
            - Gives the same importance to each product category
            - Better reflects performance on both frequent and rare classes
            """
        )

    st.success(
        """
        Macro F1 is therefore used as the main metric because the goal is not only to
        classify common products correctly, but to obtain balanced performance across
        all 27 product categories.
        """
    )

elif page == "5. Image Modeling":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5. Image Modeling")
    st.write(
        "Classifying Rakuten product listings by image alone is harder than it looks: the same category "
        "can contain wildly different visual styles, while different categories sometimes share packaging or shape. "
        "To understand what image models can and cannot contribute, we ran a structured set of experiments — "
        "starting from scratch and progressively adding pretrained knowledge, better architectures, and more "
        "careful training strategies."
    )

    st.subheader("Modeling approach")
    red = lambda text: f"<span style='color:#C44E52; font-weight:600;'>{text}</span>"

    approach_rows = pd.DataFrame([
        {"Step": "1", "Area": f"CNN baselines {red('from scratch')}",
        "Variants": "128 px · no aug  |  128 px · aug  |  256 px · no aug",
        "Purpose": f"Establish a {red('lower-bound reference')} trained purely on this dataset, with no pretrained features."},
        {"Step": "2", "Area": f"ResNet {red('transfer learning')}",
        "Variants": "Frozen · no aug  |  Partial unfreeze  |  Full unfreeze  |  From scratch  |  ResNet101 · frozen",
        "Purpose": f"Bring in {red('ImageNet-pretrained')} representations and compare frozen vs. fine-tuned strategies."},
        {"Step": "3", "Area": f"Upgrade to {red('modern vision')} architectures (EfficientNet, ConvNeXt, DINOv2)",
        "Variants": "B0 · no aug  |  B0 · aug  |  ConvNeXt-Tiny  |  ConvNeXt-Base  |  DINOv2 · frozen",
        "Purpose": f"Evaluate {red('higher-capacity backbones')} for richer image representations."},
        {"Step": "4", "Area": "Best image branch selection",
        "Variants": "ConvNeXt-Base · full unfreeze",
        "Purpose": "Pick the strongest image model to carry forward into multimodal fusion."},
    ])
    render_html_table_with_highlights(approach_rows, max_width="1000px")




elif page == "5.2 CNN Models":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.2 CNN Models")
    st.write(
        "All three CNN baselines use the same custom architecture, trained on this dataset alone — "
        "no ImageNet weights, no pretrained representations. The experiments vary input resolution (128 vs. 256 px) "
        "and augmentation to understand what a fully scratch-trained model can learn."
    )

    _col_arch, _col_setup = st.columns(2)
    with _col_arch:
        st.subheader("Architecture")
        arch_df = pd.DataFrame([
            {"Layer": "Conv block 1", "Details": "Conv2D(32) → BN → ReLU → MaxPool"},
            {"Layer": "Conv block 2", "Details": "Conv2D(64) → BN → ReLU → MaxPool"},
            {"Layer": "Conv block 3", "Details": "Conv2D(128) → BN → ReLU → MaxPool"},
            {"Layer": "Conv block 4", "Details": "Conv2D(256) → BN → ReLU → MaxPool"},
            {"Layer": "Pooling", "Details": "AdaptiveAvgPool2d → Flatten"},
            {"Layer": "Head", "Details": "Linear(512) → ReLU → Dropout(0.5) → Linear(27)"},
        ])
        render_html_table(arch_df, max_width="100%")

    with _col_setup:
        st.subheader("Training setup")
        setup_df = pd.DataFrame([
            {"Setting": "Loss", "Value": "Cross-entropy"},
            {"Setting": "Optimizer", "Value": "Adam"},
            {"Setting": "Learning rate", "Value": "1e-3 + ReduceLROnPlateau"},
            {"Setting": "Batch size", "Value": "64"},
            {"Setting": "Early stop", "Value": "Patience 5–10 epochs"},
            {"Setting": "Hardware", "Value": "Tesla T4 (Google Colab)"},
        ])
        render_html_table(setup_df, max_width="100%")

    st.subheader("Results")
    _cnn_img = APP_DIR / "images" / "cnn_results.png"
    if _cnn_img.exists():
        st.image(str(_cnn_img), width=650)

    st.success(
        "**Effect of resolution:** Higher resolution helped only slightly — I3 (256 px) gained < 0.01 macro F1 over I1 (128 px) at the cost of longer training time.\n\n"
        "**Effect of augmentation:** Augmentation did not improve performance — I2 (augmented) scored lower than I1 (no aug); synthetic transforms removed the stable visual cues the model relies on.\n\n"
        "**Limited data is the bottleneck** — with few examples per class, scratch-trained CNNs cannot build competitive visual representations.\n\n"
        "**Best CNN: I3 · macro F1 ≈ 0.509**\n\n"
        "**Frozen ResNet50 already surpassed every CNN: macro F1 ≈ 0.554** — a backbone never fine-tuned on product images outperformed a network trained entirely on Rakuten data, confirming that transfer learning is the right direction."
    )

elif page == "5.1 Data Augmentation":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.1 Data Augmentation")
    st.write(
        "Before training, each image passes through a pipeline that resizes, optionally augments, and normalises the input. "
        "Augmentation creates synthetic variations on the fly — flipping, random cropping, mild brightness shifts — "
        "so the model learns to recognise the *product category*, not memorise a specific photo."
    )

    st.subheader("Augmentation pipeline stages")
    render_html_table(pd.DataFrame([
        {"Stage": "1. Resize / crop",
         "What happens": "Image is scaled to the model's input size (128², 224², or 256²). With augmentation a random region is cropped first, so the model sees different framings each epoch."},
        {"Stage": "2. Geometric transforms",
         "What happens": "Random horizontal flips and, in stronger setups, small rotations or translations. These simulate natural variation in product photography angles."},
        {"Stage": "3. Color transforms",
         "What happens": "Mild adjustments to brightness, contrast, saturation, or hue. ConvNeXt runs use TrivialAugmentWide, which picks one random policy per image."},
        {"Stage": "4. Normalise",
         "What happens": "Pixel values are rescaled to the range expected by the backbone — ImageNet mean/std (0.485/0.229, 0.456/0.224, 0.406/0.225) for pretrained models."},
    ]), max_width="950px")
    st.markdown(
        """
        **Applied in:**
        - I2 — CNN · aug
        - I6 — ResNet50 · partial unfreeze
        - I7 — ResNet50 · full unfreeze
        - I8 — ResNet50 · from scratch
        - I9 — ConvNeXt-Tiny
        - I11 — EfficientNetB0 · aug
        - I12 — ConvNeXt-Base
        - I13 — DINOv2 · timm transform
        """
    )

    st.subheader("Why we chose moderate augmentation")
    st.markdown(
        """
        - **Heavy augmentation failed in practice.** A stronger setup caused validation accuracy to collapse to ~0.21 and training accuracy to ~0.17 — early stopping triggered after just three epochs.
        - **Product images are catalog-style.** Rakuten listings are mostly upright, centered, and photographed under controlled conditions. Large rotations, heavy crops, or aggressive colour distortion produce unrealistic examples the model was never meant to see.
        - **Natural variety already exists.** The dataset spans 27 categories with genuine visual diversity. Excessive synthetic variation adds noise on top of that diversity, making class boundaries less stable rather than more robust.
        - **Moderate augmentation is the right balance.** Random crop, horizontal flip, and mild colour jitter add just enough variation to reduce overfitting — without destroying the shape, packaging, and colour cues the model depends on for classification.
        """
    )
elif page == "5.3 ResNet Models":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.3 ResNet Models")
    st.write(
        "ResNet50 brings residual (skip) connections to a 50-layer deep network: instead of each block "
        "having to learn the full output from scratch, it only has to learn the *residual* — the difference "
        "from the input. This makes very deep networks stable to train. All four experiments here use the "
        "same ResNet50 backbone pretrained on ImageNet; they differ only in how many layers are allowed "
        "to update during training and whether augmentation is applied."
    )

    st.subheader("Architecture")
    st.write(
        "ResNet50 is organised as five stages. The first stage is a 7 × 7 convolution + max-pool. "
        "Stages 2–5 are stacks of *bottleneck blocks* (1 × 1 → 3 × 3 → 1 × 1 convolutions), each with "
        "a shortcut connection that adds the block input directly to its output. "
        "For classification, the final feature map is average-pooled to a 2 048-dim vector and fed into a "
        "linear head."
    )
    arch_df = pd.DataFrame([
        {"Stage": "Conv 1", "Details": "7×7 Conv (64 filters, stride 2) → BatchNorm → ReLU → MaxPool (stride 2)"},
        {"Stage": "Layer 1", "Details": "3 × bottleneck blocks — output 256 channels"},
        {"Stage": "Layer 2", "Details": "4 × bottleneck blocks — output 512 channels"},
        {"Stage": "Layer 3", "Details": "6 × bottleneck blocks — output 1 024 channels"},
        {"Stage": "Layer 4", "Details": "3 × bottleneck blocks — output 2 048 channels"},
        {"Stage": "Head", "Details": "Global AvgPool → Flatten → [Dropout(0.5)] → Linear(2 048 → 27 classes)"},
    ])
    render_html_table(arch_df, max_width="900px")

    st.subheader("Experiments overview")
    st.write(
        "Four strategies were tested, progressing from maximal knowledge preservation (frozen backbone) "
        "to maximal task adaptation (fully from scratch)."
    )
    exp_df = pd.DataFrame([
        {"ID": "I5", "Strategy": "Frozen backbone", "Augmentation": "No", "Unfrozen layers": "Head only", "LR": "1e-4",
         "Batch": "32", "Dropout": "—", "Label smoothing": "No"},
        {"ID": "I6", "Strategy": "Partial unfreezing", "Augmentation": "Moderate",
         "Unfrozen layers": "Last block + head", "LR": "3e-4", "Batch": "128", "Dropout": "0.5",
         "Label smoothing": "0.1"},
        {"ID": "I7", "Strategy": "Full unfreezing", "Augmentation": "Moderate", "Unfrozen layers": "All", "LR": "1e-4",
         "Batch": "128", "Dropout": "0.5", "Label smoothing": "0.1"},
        {"ID": "I8", "Strategy": "Random initialisation", "Augmentation": "Moderate", "Unfrozen layers": "All",
         "LR": "3e-4", "Batch": "16", "Dropout": "0.3", "Label smoothing": "No"},
    ])
    render_html_table(exp_df, max_width="960px")

    st.subheader("Results")
    rn_models = [r for r in IMAGE_MODEL_RESULTS if r["Family"] == "ResNet"]
    rn_labels = [IMAGE_MODEL_LABELS.get(r["Model"], r["Model"]) for r in rn_models]
    rn_ids = ["I5", "I6", "I7", "I8", "I8b"]
    rn_acc = []
    rn_f1 = []
    for r in rn_models:
        try:
            rn_acc.append(float(r["Accuracy"]))
            rn_f1.append(float(r["Macro F1"]))
        except (TypeError, ValueError):
            rn_acc.append(None)
            rn_f1.append(None)

    # drop I8b (incomplete metrics) for the chart
    plot_labels = [l for l, a in zip(rn_labels, rn_acc) if a is not None]
    plot_acc = [a for a in rn_acc if a is not None]
    plot_f1 = [f for f in rn_f1 if f is not None]

    x = range(len(plot_labels))
    fig_rn, ax_rn = plt.subplots(figsize=(5.5, 2.8))
    bars_acc = ax_rn.bar([i - 0.18 for i in x], plot_acc, width=0.32, label="Accuracy", color="#f97316")
    bars_f1 = ax_rn.bar([i + 0.18 for i in x], plot_f1, width=0.32, label="Macro F1", color="#fdba74")
    for bar in bars_acc:
        ax_rn.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5, color="#9a3412")
    for bar in bars_f1:
        ax_rn.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5, color="#4b5563")
    ax_rn.set_xticks(list(x))
    ax_rn.set_xticklabels(plot_labels, fontsize=7)
    ax_rn.set_ylim(0.45, 0.75)
    ax_rn.set_ylabel("Score", fontsize=7)
    ax_rn.legend(fontsize=7)
    ax_rn.spines[["top", "right"]].set_visible(False)
    ax_rn.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_rn.set_axisbelow(True)
    fig_rn.tight_layout()
    st.pyplot(fig_rn, use_container_width=False)
    plt.close(fig_rn)

    st.subheader("Training curves — I6 & I7")
    st.write(
        "Both I6 and I7 were trained on an RTX 5070 Ti. The curves below show how validation macro F1 "
        "evolved across epochs. The train macro F1 (dashed) rises steeply due to label smoothing; the "
        "gap to validation F1 signals overfitting once the scheduler has reduced the learning rate."
    )
    if not resnet_i6_history.empty or not resnet_i7_history.empty:
        fig_curves, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=False)
        for ax_c, hist, title, c_train, c_val in [
            (axes[0], resnet_i6_history, "I6 — Partial unfreeze", "#f97316", "#c2410c"),
            (axes[1], resnet_i7_history, "I7 — Full unfreeze", "#3b82f6", "#1d4ed8"),
        ]:
            if hist.empty:
                ax_c.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_c.transAxes)
                ax_c.set_title(title, fontsize=9)
                continue
            epochs = hist["epoch"]
            ax_c.plot(epochs, hist["val_macro_f1"], color=c_val, lw=2, label="Val macro F1")
            ax_c.plot(epochs, hist["train_macro_f1"], color=c_train, lw=1.5, linestyle="--", label="Train macro F1",
                      alpha=0.7)
            if "val_acc" in hist.columns:
                ax_c2 = ax_c.twinx()
                ax_c2.plot(epochs, hist["val_acc"], color="#6b7280", lw=1, linestyle=":", label="Val acc", alpha=0.6)
                ax_c2.set_ylabel("Val accuracy", fontsize=7, color="#6b7280")
                ax_c2.tick_params(axis="y", labelsize=6, colors="#6b7280")
            best_ep = int(hist.loc[hist["val_macro_f1"].idxmax(), "epoch"])
            best_f1 = hist["val_macro_f1"].max()
            ax_c.axvline(best_ep, color=c_val, lw=1, linestyle=":", alpha=0.8)
            ax_c.text(best_ep + 0.15, hist["val_macro_f1"].min() + 0.005,
                      f"best ep {best_ep}\nF1={best_f1:.3f}", fontsize=6.5, color=c_val)
            ax_c.set_title(title, fontsize=9, fontweight="bold")
            ax_c.set_xlabel("Epoch", fontsize=7)
            ax_c.set_ylabel("Macro F1", fontsize=7)
            ax_c.tick_params(labelsize=6)
            ax_c.legend(fontsize=6.5, loc="lower right")
            ax_c.spines[["top", "right"]].set_visible(False)
            ax_c.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax_c.set_axisbelow(True)
        fig_curves.tight_layout()
        st.pyplot(fig_curves, use_container_width=True)
        plt.close(fig_curves)
    else:
        st.info("Training history files not found.")

    st.subheader("Effect of the unfreezing strategy")
    st.write(
        "The progression from I5 → I6 → I7 illustrates the classic frozen-to-fine-tuned trade-off:"
    )
    render_html_table(pd.DataFrame([
        {"Model": "I5 — Frozen", "Macro F1": "0.554",
         "Key observation": "Strong ImageNet features, zero adaptation. Head converges but backbone is bottlenecked at generic features."},
        {"Model": "I6 — Partial unfreeze", "Macro F1": "0.641",
         "Key observation": "+0.087 over I5. Unfreezing the last residual block lets the backbone adapt its highest-level features to Rakuten's product styles. Higher LR (3e-4) is safe because only one block changes."},
        {"Model": "I7 — Full unfreeze", "Macro F1": "0.653",
         "Key observation": "+0.012 over I6. Full fine-tuning squeezes a small extra gain. The lower LR (1e-4) protects against catastrophic forgetting of early-layer edge detectors that are still useful here."},
    ]), max_width="960px")
    st.write(
        "The diminishing returns (0.087 then 0.012) suggest that lower-level features — edges, textures, "
        "basic shapes — transfer well from ImageNet to product images; the gain comes almost entirely from "
        "allowing the top layers to re-specialise."
    )

    st.subheader("Effect of training from scratch (I8)")
    st.write(
        "I8 replicates the I7 setup except that the ResNet50 weights are randomly initialised — "
        "no ImageNet pretraining. The result (macro F1 0.511) is worse than even the frozen backbone (I5, 0.554). "
        "This mirrors the CNN baseline finding: with ~68 k training images spread across 27 classes "
        "(roughly 2 500 per class), there simply is not enough data to learn competitive visual features "
        "from scratch in a 23.5 M-parameter network. Transfer learning is not optional here — it is essential."
    )
    if _rn_i5_png_dir.exists():
        _i5_acc_png = _rn_i5_png_dir / "I5_Accuracy_graph.png"
        _i5_loss_png = _rn_i5_png_dir / "I5_Loss_graph.png"
        _i5_cm_png = _rn_i5_png_dir / "I5_Confusion_matrix.png"
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            if _i5_acc_png.exists():
                st.image(str(_i5_acc_png), caption="I5 — Accuracy", use_container_width=True)
        with _c2:
            if _i5_loss_png.exists():
                st.image(str(_i5_loss_png), caption="I5 — Loss", use_container_width=True)
        with _c3:
            if _i5_cm_png.exists():
                st.image(str(_i5_cm_png), caption="I5 — Confusion matrix", use_container_width=True)

    st.success(
        """
        Transfer learning is decisive for ResNet50 on this dataset:
        - Frozen backbone (I5) already beats all scratch-trained CNNs, confirming that ImageNet representations transfer well to product images.
        - Partial unfreezing (I6) delivers the biggest single improvement (+0.087 macro F1), allowing the top residual block to adapt to Rakuten-specific visual patterns.
        - Full fine-tuning (I7, macro F1 0.653) yields a modest further gain, setting the ResNet ceiling before the ConvNeXt experiments push it higher.
        - Training from scratch (I8, macro F1 0.511) confirms that ResNet50 without ImageNet weights cannot beat even the frozen backbone — data volume is the limiting factor.
        """
    )

elif page == "5.4 ConvNeXt Models":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.4 ConvNeXt Models")
    st.write(
        "ConvNeXt is a pure convolutional network whose design was systematically modernised by borrowing "
        "ideas from Vision Transformers (Swin Transformer). The result is a network that looks like a CNN "
        "but matches or exceeds transformer performance on image benchmarks — without requiring self-attention "
        "or positional encodings. Two variants were evaluated here: ConvNeXt-Tiny (I9) as a lighter probe "
        "and ConvNeXt-Base (I12) as the full-capacity model."
    )

    _col_arch, _col_setup = st.columns(2)
    with _col_arch:
        st.subheader("Architecture")
        arch_cnxt_df = pd.DataFrame([
            {"Stage": "Stem", "Details": "4×4 Conv (stride 4) → LayerNorm — aggressive downsampling replaces the early MaxPool"},
            {"Stage": "Stage 1–4", "Details": "Stack of ConvNeXt blocks per stage (depths 3-3-9-3 Tiny, 3-3-27-3 Base)"},
            {"Stage": "ConvNeXt block", "Details": "Depthwise 7×7 Conv → LayerNorm → Linear ×4 (expand) → GELU → Linear (contract) + residual"},
            {"Stage": "Downsampling", "Details": "2×2 LayerNorm + strided Conv between stages (not inside blocks)"},
            {"Stage": "Head", "Details": "Global AvgPool → LayerNorm → Linear(27 classes)"},
        ])
        render_html_table(arch_cnxt_df, max_width="100%")

    with _col_setup:
        st.subheader("Training setup")
        setup_cnxt_df = pd.DataFrame([
            {"Setting": "Loss", "Value": "Cross-entropy + label smoothing 0.1"},
            {"Setting": "Optimizer", "Value": "AdamW"},
            {"Setting": "LR — I9 (Tiny)", "Value": "1e-4 flat (backbone + head)"},
            {"Setting": "LR — I12 (Base)", "Value": "Backbone 5e-6 · Head 5e-5 (diff. LR)"},
            {"Setting": "Batch size", "Value": "I9: 32 · I12: 16"},
            {"Setting": "Scheduler", "Value": "ReduceLROnPlateau"},
            {"Setting": "Epochs", "Value": "20 (no early stopping)"},
            {"Setting": "Hardware", "Value": "RTX 5070 Ti"},
        ])
        render_html_table(setup_cnxt_df, max_width="100%")

    st.write(
        "**Why ConvNeXt after ResNet?** ResNet50 peaked at macro F1 0.653 — switching to ConvNeXt's "
        "modernised design (7×7 depthwise conv, inverted bottleneck, GELU, LayerNorm) and stronger "
        "ImageNet-22k pretraining immediately pushed that ceiling to 0.685 with ConvNeXt-Tiny alone."
    )

    st.subheader("Results")
    cnxt_models = [r for r in IMAGE_MODEL_RESULTS if r["Family"] == "ConvNeXt"]
    cnxt_labels = [IMAGE_MODEL_LABELS.get(r["Model"], r["Model"]) for r in cnxt_models]
    cnxt_acc = [float(r["Accuracy"]) for r in cnxt_models]
    cnxt_f1 = [float(r["Macro F1"]) for r in cnxt_models]

    # add ResNet best (I7) as reference bar
    _rn_ref_label = "ResNet50 best (I7)"
    _rn_ref_acc = 0.6849
    _rn_ref_f1 = 0.6533
    plot_labels = [_rn_ref_label] + cnxt_labels
    plot_acc = [_rn_ref_acc] + cnxt_acc
    plot_f1 = [_rn_ref_f1] + cnxt_f1
    colors_acc = ["#94a3b8"] + ["#059669"] * len(cnxt_labels)
    colors_f1 = ["#cbd5e1"] + ["#6ee7b7"] * len(cnxt_labels)

    x = range(len(plot_labels))
    fig_cx, ax_cx = plt.subplots(figsize=(5.5, 2.8))
    bars_acc = ax_cx.bar([i - 0.18 for i in x], plot_acc, width=0.32, label="Accuracy", color=colors_acc)
    bars_f1 = ax_cx.bar([i + 0.18 for i in x], plot_f1, width=0.32, label="Macro F1", color=colors_f1)
    for bar in bars_acc:
        ax_cx.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5, color="#064e3b")
    for bar in bars_f1:
        ax_cx.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5, color="#4b5563")
    ax_cx.set_xticks(list(x))
    ax_cx.set_xticklabels(plot_labels, fontsize=7)
    ax_cx.set_ylim(0.60, 0.77)
    ax_cx.set_ylabel("Score", fontsize=7)
    ax_cx.legend(fontsize=7)
    ax_cx.spines[["top", "right"]].set_visible(False)
    ax_cx.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_cx.set_axisbelow(True)
    fig_cx.tight_layout()
    _cx_col, _ = st.columns([0.65, 0.35])
    with _cx_col:
        st.pyplot(fig_cx, use_container_width=True)
    plt.close(fig_cx)

    st.subheader("Training curves — I9 & I12")
    st.write(
        "Both models were trained on an RTX 5070 Ti with AdamW, label smoothing 0.1 and "
        "ReduceLROnPlateau. I12 uses differential learning rates (backbone 5e-6, head 5e-5) to "
        "protect the rich pretrained features while still allowing full adaptation."
    )


    # normalise column names so both histories use the same keys for plotting
    def _norm_history(df, val_f1_col, train_f1_col, val_acc_col=None):
        out = pd.DataFrame()
        if df is None or df.empty:
            return out
        out["epoch"] = df["epoch"]
        out["val_macro_f1"] = df[val_f1_col]
        out["train_macro_f1"] = df[train_f1_col]
        if val_acc_col and val_acc_col in df.columns:
            out["val_acc"] = df[val_acc_col]
        return out


    _i9_norm = _norm_history(convnext_i9_history, "val_macro_f1", "train_macro_f1", "val_acc")
    _i12_norm = _norm_history(history, "v_f1", "t_f1")

    if not _i9_norm.empty or not _i12_norm.empty:
        fig_cv, axes_cv = plt.subplots(1, 2, figsize=(10, 3.2), sharey=False)
        for ax_c, hist_n, title, c_train, c_val in [
            (axes_cv[0], _i9_norm, "I9 — ConvNeXt-Tiny", "#059669", "#065f46"),
            (axes_cv[1], _i12_norm, "I12 — ConvNeXt-Base", "#7c3aed", "#4c1d95"),
        ]:
            if hist_n.empty:
                ax_c.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_c.transAxes)
                ax_c.set_title(title, fontsize=9)
                continue
            epochs = hist_n["epoch"]
            ax_c.plot(epochs, hist_n["val_macro_f1"], color=c_val, lw=2, label="Val macro F1")
            ax_c.plot(epochs, hist_n["train_macro_f1"], color=c_train, lw=1.5, linestyle="--", alpha=0.7,
                      label="Train macro F1")
            if "val_acc" in hist_n.columns:
                ax_c2 = ax_c.twinx()
                ax_c2.plot(epochs, hist_n["val_acc"], color="#6b7280", lw=1, linestyle=":", alpha=0.6, label="Val acc")
                ax_c2.set_ylabel("Val accuracy", fontsize=7, color="#6b7280")
                ax_c2.tick_params(axis="y", labelsize=6, colors="#6b7280")
            best_idx = hist_n["val_macro_f1"].idxmax()
            best_ep = int(hist_n.loc[best_idx, "epoch"])
            best_f1 = hist_n["val_macro_f1"].max()
            ax_c.axvline(best_ep, color=c_val, lw=1, linestyle=":", alpha=0.8)
            ax_c.text(best_ep + 0.15, hist_n["val_macro_f1"].min() + 0.003,
                      f"best ep {best_ep}\nF1={best_f1:.3f}", fontsize=6.5, color=c_val)
            ax_c.set_title(title, fontsize=9, fontweight="bold")
            ax_c.set_xlabel("Epoch", fontsize=7)
            ax_c.set_ylabel("Macro F1", fontsize=7)
            ax_c.tick_params(labelsize=6)
            ax_c.legend(fontsize=6.5, loc="lower right")
            ax_c.spines[["top", "right"]].set_visible(False)
            ax_c.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax_c.set_axisbelow(True)
        fig_cv.tight_layout()
        st.pyplot(fig_cv, use_container_width=True)
        plt.close(fig_cv)
    else:
        st.info("Training history files not found.")

    st.info(
        "ConvNeXt-Base (I12) is the best-performing image model. "
        "See **5.4 Best Image Model** for a cross-architecture comparison (CNN vs ResNet vs ConvNeXt) "
        "and a detailed analysis of why it wins, including Grad-CAM visualisations in **5.4.1 Error Analysis & Grad-CAM**."
    )

elif page == "5.5 Best Image Model":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.5 Best Image Model — ConvNeXt-Base (I12)")
    st.write(
        "Across all 13 image-only experiments — from scratch-trained CNNs to ResNets to ConvNeXt — "
        "ConvNeXt-Base (I12) achieves the highest macro F1 and accuracy. "
        "This page compares the best representative from each architecture family and explains "
        "the key factors behind ConvNeXt-Base's win."
    )

    # ── Cross-architecture comparison chart ──────────────────────────────────
    st.subheader("Comparison: best CNN · best ResNet · ConvNeXt-Tiny · ConvNeXt-Base")
    st.write(
        "The chart below shows the best model from each architecture family, "
        "illustrating the progressive performance gains as architectures improve."
    )
    _best_models = [
        {"label": "Best CNN\n(I3 — CNN 256 px)", "acc": 0.5767, "f1": 0.5090, "color_acc": "#f87171", "color_f1": "#fca5a5"},
        {"label": "Best ResNet\n(I7 — ResNet50 full)", "acc": 0.6849, "f1": 0.6533, "color_acc": "#fb923c", "color_f1": "#fdba74"},
        {"label": "ConvNeXt-Tiny\n(I9)", "acc": 0.7144, "f1": 0.6850, "color_acc": "#34d399", "color_f1": "#6ee7b7"},
        {"label": "ConvNeXt-Base\n(I12) ★", "acc": 0.7200, "f1": 0.6924, "color_acc": "#059669", "color_f1": "#10b981"},
    ]
    _bm_labels = [m["label"] for m in _best_models]
    _bm_acc = [m["acc"] for m in _best_models]
    _bm_f1 = [m["f1"] for m in _best_models]
    _bm_colors_acc = [m["color_acc"] for m in _best_models]
    _bm_colors_f1 = [m["color_f1"] for m in _best_models]

    _x = range(len(_bm_labels))
    fig_bm, ax_bm = plt.subplots(figsize=(7, 3.2))
    bars_bm_acc = ax_bm.bar([i - 0.18 for i in _x], _bm_acc, width=0.32, label="Accuracy", color=_bm_colors_acc)
    bars_bm_f1 = ax_bm.bar([i + 0.18 for i in _x], _bm_f1, width=0.32, label="Macro F1", color=_bm_colors_f1)
    for bar in bars_bm_acc:
        ax_bm.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="#1f2937")
    for bar in bars_bm_f1:
        ax_bm.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="#374151")
    ax_bm.set_xticks(list(_x))
    ax_bm.set_xticklabels(_bm_labels, fontsize=7.5)
    ax_bm.set_ylim(0.45, 0.78)
    ax_bm.set_ylabel("Score", fontsize=8)
    ax_bm.legend(fontsize=8)
    ax_bm.spines[["top", "right"]].set_visible(False)
    ax_bm.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_bm.set_axisbelow(True)
    fig_bm.tight_layout()
    _buf_bm = io.BytesIO()
    fig_bm.savefig(_buf_bm, format="png", dpi=150, bbox_inches="tight")
    _buf_bm.seek(0)
    st.image(_buf_bm, width=800)
    plt.close(fig_bm)

    render_html_table(pd.DataFrame([
        {"Architecture": "Custom CNN (I3)", "Params": "~2 M", "Strategy": "From scratch",
         "Val Accuracy": "0.577", "Val Macro F1": "0.509",
         "Note": "Baseline — no pretrained weights, limited capacity"},
        {"Architecture": "ResNet50 (I7)", "Params": "~25 M", "Strategy": "Full fine-tune · flat LR 1e-4",
         "Val Accuracy": "0.685", "Val Macro F1": "0.653",
         "Note": "+0.144 F1 over CNN — ImageNet pretraining is decisive"},
        {"Architecture": "ConvNeXt-Tiny (I9)", "Params": "~28 M", "Strategy": "Full unfreeze · flat LR 1e-4",
         "Val Accuracy": "0.714", "Val Macro F1": "0.685",
         "Note": "+0.032 F1 over ResNet — architecture upgrade matters more than tuning"},
        {"Architecture": "ConvNeXt-Base (I12) ★", "Params": "~89 M", "Strategy": "Full unfreeze · diff. LR 5e-6/5e-5",
         "Val Accuracy": "0.720", "Val Macro F1": "0.692",
         "Note": "Best image model — more capacity + differential LR"},
    ]), max_width="100%")

    # ── Why ConvNeXt-Base wins ────────────────────────────────────────────────
    st.subheader("Why ConvNeXt-Base (I12) is the best image model")
    render_html_table(pd.DataFrame([
        {"Factor": "Architecture upgrade (CNN → ConvNeXt)",
         "Detail": "ConvNeXt's 7×7 depthwise conv, inverted bottleneck, GELU, and LayerNorm give it a fundamentally richer feature space than both CNNs and ResNets at the same input resolution. Moving from ResNet to ConvNeXt-Tiny adds +0.032 macro F1 — a larger jump than any tuning change within the ResNet family."},
        {"Factor": "More parameters (Base vs Tiny)",
         "Detail": "ConvNeXt-Base has ~89 M parameters vs ~28 M for ConvNeXt-Tiny. The extra capacity allows the model to represent finer visual distinctions between similar product categories (e.g., board games vs toy cars)."},
        {"Factor": "Differential learning rates",
         "Detail": "I12 uses backbone LR 5e-6 and head LR 5e-5 — a ×10 ratio. The backbone updates slowly to preserve rich pretrained features; the head updates faster to adapt to Rakuten's 27 classes. I9 uses a flat 1e-4 throughout, which risks slightly distorting the backbone."},
        {"Factor": "Continued improvement through epoch 20",
         "Detail": "I12's val macro F1 rises steadily to 0.692 at epoch 20 with no sign of overfitting — suggesting the model still has room to grow and was not overfitting even at the end of training."},
        {"Factor": "Best absolute scores across all image-only models",
         "Detail": "I12 achieves macro F1 0.692 and accuracy 0.720 — the highest reported figures in the entire image-only experiment set. It becomes the image branch for multimodal fusion in chapters 6.1–6.2."},
    ]), max_width="960px")

    st.subheader("Validation metrics — ConvNeXt-Base (I12)")
    render_metric_cards({
        "accuracy": 0.720,
        "macro_f1": 0.692,
        "weighted_f1": 0.718,
        "validation_samples": 16_984,
    })

    st.success(
        """
        ConvNeXt-Base (I12) wins on every axis:
        - **+0.183 macro F1 over the best CNN** (I3 = 0.509 → I12 = 0.692) — pretrained weights and modern architecture are both essential.
        - **+0.039 macro F1 over the best ResNet** (I7 = 0.653 → I12 = 0.692) — the architecture upgrade alone accounts for +0.032 (ConvNeXt-Tiny), differential LR adds the rest.
        - **+0.007 macro F1 over ConvNeXt-Tiny** (I9 = 0.685 → I12 = 0.692) — scale and training recipe refine the final result.
        - Both ConvNeXt variants train for the full 20 epochs without overfitting — confirming that richer pretrained representations generalise well even on ~68 k training images.
        """
    )

    st.info("For a detailed breakdown of classification errors and Grad-CAM visualisations, see **5.4.1 Error Analysis & Grad-CAM**.")

elif page == "5.6 Image Modeling Conclusion":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.6 Image Modeling Conclusion")
    st.write(
        "This page gives a full picture of all image-only experiments in one place. "
        "The table below covers every model from the simplest scratch-trained CNN to the final "
        "ConvNeXt-Base, letting you compare architectures, training strategies, and validation scores at a glance."
    )

    results_df = pd.DataFrame(IMAGE_MODEL_RESULTS)
    display_df = results_df.copy().replace({None: "—"})
    display_df["Variant"] = display_df["Model"].map(IMAGE_MODEL_LABELS).fillna(display_df["Model"])
    grouped_cols = ["Family", "Variant", "Image size", "Training strategy", "Augmentation",
                    "Accuracy", "Macro F1", "Weighted F1", "Best epoch", "Hardware / time"]
    display_df = display_df[[c for c in grouped_cols if c in display_df.columns]]
    st.subheader("Image model comparison")
    render_grouped_html_table(display_df, group_col="Family")

    st.subheader("Key findings")
    st.markdown(
        """
        - **Transfer learning is essential.** A frozen ResNet50 (I5, F1 0.554) already outperforms the best
          scratch-trained CNN (I3, F1 0.509), even without updating the backbone on product images.
        - **Fine-tuning pays off.** Unlocking the top residual block (I6, +0.087) and then the full network
          (I7, +0.012) progressively improved ResNet50 — but gains diminish quickly, suggesting a ceiling.
        - **Architecture upgrade matters more than further fine-tuning.** ConvNeXt-Tiny (I9) immediately adds
          +0.032 macro F1 over the best ResNet — a larger jump than any tuning change within the ResNet family.
        - **Scale and differential learning rates seal the win.** ConvNeXt-Base (I12, F1 0.692) uses a backbone
          LR 10× lower than the head, protecting pretrained features while the head adapts.
        - **Image-only performance has a ceiling.** Even at F1 0.692, visually similar categories
          (toys, board games, hobby figurines) remain confused — resolved by multimodal fusion in Chapter 6.
        """
    )

elif page == "5.5.1 Interpretability":
    st.title("Rakuten Multimodal Product Classification")
    st.header("5.5.1 Error Analysis & Grad-CAM — ConvNeXt-Base")
    st.write(
        "This page analyses where ConvNeXt-Base (I12) succeeds and fails. "
        "The confusion matrix reveals systematic misclassification patterns at the category level; "
        "the four Grad-CAM examples then zoom in on what the model 'sees' in representative cases."
    )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion matrix — validation set")
    st.write(
        "The row-normalised confusion matrix below shows **per-class recall** on the diagonal and the "
        "fraction of true examples misclassified into each other class off-diagonal. "
        "Cell values are absolute counts. Key patterns visible in the data:"
    )
    st.markdown(
        "- **Strongest classes** — Collectible Cards (F1 0.89), Household Linen (0.79), "
        "Swimming-pool accessories (0.75): visually distinct products with little overlap.\n"
        "- **Hardest classes** — Board & Card Games (F1 0.27), Collectible Board-game Figurines (0.32), "
        "Toys/Plush (0.35), Tools & Garden (0.35): visually similar to neighbouring toy and outdoor categories.\n"
        "- **Largest off-diagonal clusters** — Toys/Plush → Toy Cars (182 errors), "
        "Magazines → Books/Comics (141), Books → Books/Comics (129), "
        "Furniture → Lamps/Decor (108), Tools/Garden → Pool Acc. (105). "
        "These all involve categories that share shapes, colours, or retail context."
    )
    _cm_labeled = ION_IMAGE_DIR / "confusion_matrix_i12_labeled.png"
    if _cm_labeled.exists():
        st.image(str(_cm_labeled),
                 caption="Row-normalised confusion matrix with class names and counts — ConvNeXt-Base (I12)",
                 use_container_width=True)
    elif FILES["confusion_png"]:
        st.image(str(FILES["confusion_png"]), caption="Normalised validation confusion matrix — ConvNeXt-Base (I12)",
                 use_container_width=True)
    else:
        st.info("Confusion matrix image not found.")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    st.subheader("Grad-CAM — selected examples")
    st.write(
        "Grad-CAM highlights the image regions that most influenced the model's prediction. "
        "Four representative cases are shown below — one from each interpretability category — "
        "to illustrate how the model reasons about product images in both success and failure modes. "
        "The examples cover two of the most error-prone classes in the confusion matrix "
        "(Toys/Plush and Outdoor Games) alongside Furniture and Women's Bags."
    )

    if gradcam_table.empty:
        st.warning("No Grad-CAM images found.")
    else:
        ordered_gdf = ordered_gradcam_examples(gradcam_table)

        # (group_key, example_index_within_group, section_label)
        selected_examples = [
            ("high_confidence_right_prediction", 0, "Example 1"),
            ("high_confidence_wrong_prediction", 2, "Example 3"),
            ("low_confidence_right_prediction", 0, "Example 1"),
            ("low_confidence_wrong_prediction", 1, "Example 2"),
        ]

        for group_key, example_idx, example_label in selected_examples:
            group_df = ordered_gdf[ordered_gdf["group"] == group_key]
            if group_df.empty or len(group_df) <= example_idx:
                confidence_label, correctness_label, _ = readable_gradcam_group(group_key)
                st.info(f"No {example_label.lower()} found for '{confidence_label} — {correctness_label}'.")
                continue
            row = group_df.iloc[example_idx]
            render_gradcam_header(row)
            image_col, note_col = st.columns([3, 1])
            with image_col:
                st.image(load_gradcam_display_image(row["file"]), width=GRADCAM_WIDTH)
            with note_col:
                render_observation_box(row)

    # ── Conclusion ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Conclusion — ConvNeXt-Base (I12) as best image model")
    render_html_table(pd.DataFrame([
        {"Metric": "Val macro F1", "Value": "0.692", "Note": "Highest across all image-only models (I1–I12)"},
        {"Metric": "Val accuracy", "Value": "0.720", "Note": "Best single-model accuracy in the image series"},
        {"Metric": "Best epoch", "Value": "20 / 20",
         "Note": "No early stopping — model still improving at the end of training"},
        {"Metric": "Training strategy", "Value": "Differential LR",
         "Note": "Backbone 5e-6, head 5e-5 — preserves pretrained features while adapting to 27 Rakuten classes"},
        {"Metric": "Architecture", "Value": "~89 M params",
         "Note": "ConvNeXt-Base with 7×7 depthwise conv, GELU, LayerNorm — richer representations than ResNet at similar cost"},
    ]), max_width="960px")

    st.success(
        """
        **ConvNeXt-Base (I12) is the best-performing image model**, reaching macro F1 **0.692** — a +0.039 gain
        over the best ResNet and +0.007 over ConvNeXt-Tiny.

        The confusion matrix confirms a clear split between easy and hard categories:
        - **High-recall classes** (Collectible Cards F1 0.89, Linen 0.79, Pool Accessories 0.75) are visually
          distinct with consistent product shapes and colours.
        - **Low-recall classes** (Board Games F1 0.27, Board-game Figurines 0.32, Toys/Plush 0.35,
          Tools & Garden 0.35) share overlapping visual features — toy shapes, product colours, retail packaging —
          that make them hard to distinguish from image alone.
        - The largest confusion clusters (Toys → Toy Cars 182, Magazines → Books 141, Books → Books/Comics 129,
          Furniture → Lamps/Decor 108) all involve categories that a text description would trivially resolve.

        This is the core motivation for **Chapter 6**: product titles and descriptions carry the disambiguation
        signal that images alone cannot provide. Combining I12's visual features with CamemBERT's text
        representations pushes macro F1 from 0.692 to above 0.88.
        """
    )

    st.write(
        "**Next →** Chapter 6 explores two fusion strategies — **Simple Fusion (6.1.1)** blends the probability "
        "outputs of the text and image models at inference time, while **Intermediate Fusion (6.1.2)** "
        "concatenates their feature vectors and trains a joint linear head end-to-end. Both strategies are built "
        "on top of I12 as the image branch and CamemBERT as the text branch."
    )




# ===================================================================
# Chapter 6 — Multimodal
# ===================================================================

elif page == "6. Multimodal":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6. Multimodal")
    st.write(
        "The multimodal stage combines the best text branch (CamemBERT, full fine-tune) and the best image "
        "branch (ConvNeXt-Base, moderate augmentation, fully unfrozen) to push beyond the unimodal ceilings. "
        "Two fusion strategies were explored: a simple late fusion that requires no additional training, and "
        "an intermediate fusion that trains a projection head on top of the frozen branches."
    )
    st.subheader("Fusion strategies at a glance")
    strat_rows = [
        {
            "Strategy": "Late Fusion (Simple Fusion)",
            "How it works": "CamemBERT and ConvNeXt-Base softmax outputs are blended with a single mixing weight α tuned on the validation set — no training required.",
            "Best Macro F1": f"{mm_late_meta['best_macro_f1']:.4f}" if mm_late_meta.get("best_macro_f1") else "—",
        },
        {
            "Strategy": "Intermediate Fusion",
            "How it works": "CamemBERT and ConvNeXt-Base feature vectors are concatenated and a joint classifier is trained on top of the frozen branches.",
            "Best Macro F1": f"{mm_inter_meta['best_macro_f1']:.4f}" if mm_inter_meta.get("best_macro_f1") else "—",
        },
        {
            "Strategy": "CLIP Gate Fusion",
            "How it works": "CamemBERT (text) and CLIP Vision (image) embeddings pass through a sigmoid gate that learns per-feature mixing weights end-to-end.",
            "Best Macro F1": f"{mm_inter_meta['best_macro_f1']:.4f}" if mm_inter_meta.get("best_macro_f1") else "—",
        },
    ]
    render_html_table(pd.DataFrame(strat_rows))
    st.subheader("Unimodal baselines for reference")
    baseline_rows = [
        {"Model": "CamemBERT (text only)",
         "Macro F1": f"{mm_late_meta['f1_text_only']:.4f}" if mm_late_meta.get("f1_text_only") else "—"},
        {"Model": "ConvNeXt-Base (image only)",
         "Macro F1": f"{mm_late_meta['f1_image_only']:.4f}" if mm_late_meta.get("f1_image_only") else "—"},
    ]
    render_html_table(pd.DataFrame(baseline_rows), max_width="600px")

elif page == "6.6 Multimodal Conclusion":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6.6 Multimodal Conclusion")
    st.write(
        "Both fusion models share the same frozen branches. The table below places them next to the best "
        "unimodal baselines to show the gain from combining modalities."
    )

    _mm_cmp_img = APP_DIR / "images" / "multimodal_comparison.png"
    if _mm_cmp_img.exists():
        st.image(str(_mm_cmp_img), width=550)

    f1_text = mm_late_meta.get("f1_text_only")
    f1_image = mm_late_meta.get("f1_image_only")
    late_acc = mm_late_meta.get("accuracy")
    late_f1 = mm_late_meta.get("best_macro_f1")
    late_wf1 = mm_late_meta.get("weighted_f1")
    inter_f1 = mm_inter_meta.get("best_macro_f1")


    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"


    comparison_rows = [
        {
            "Model": "CamemBERT (text only)",
            "Approach": "Text branch alone",
            "Accuracy": "0.8807",
            "Macro F1": "0.8616",
            "Weighted F1": "0.8800",
            "Notes": "Best text-only baseline (T8, 3 epochs, max_len=128)",
        },
        {
            "Model": "ConvNeXt-Base (image only)",
            "Approach": "Image branch alone",
            "Accuracy": "0.720",
            "Macro F1": _f(f1_image),
            "Weighted F1": "0.720",
            "Notes": "Best image-only baseline",
        },
        {
            "Model": "Late Fusion",
            "Approach": "Weighted softmax average (α = 0.55)",
            "Accuracy": _f(late_acc, 3),
            "Macro F1": _f(late_f1),
            "Weighted F1": _f(late_wf1, 4),
            "Notes": "No fusion training; α tuned on val",
        },
        {
            "Model": "Intermediate Fusion",
            "Approach": "Trained projection head on frozen branches",
            "Accuracy": "0.905",
            "Macro F1": _f(inter_f1),
            "Weighted F1": "0.905",
            "Notes": f"Best at epoch {mm_inter_meta.get('best_epoch', '—')}",
        },
        {
            "Model": "CLIP Gate Fusion",
            "Approach": "Sigmoid gate — CamemBERT + CLIP Vision, aug + label smoothing",
            "Accuracy": "0.8765",
            "Macro F1": "0.8802",
            "Weighted F1": "—",
            "Notes": "Best at epoch 14 (stage 3 full unfreeze)",
        },
    ]
    render_html_table(pd.DataFrame(comparison_rows))

    st.markdown(
        """
        <div style="background:#e8f5e9; border:1px solid #c8e6c9; border-radius:8px; padding:1rem 1.4rem;">
        <ul style="margin:0; padding-left:1.2rem; color:#1b5e20;">
          <li style="margin-bottom:0.45rem;">Both fusion models substantially outperform the image-only baseline (+21 pp macro F1).</li>
          <li style="margin-bottom:0.45rem;">Late Fusion narrowly beats Intermediate Fusion in macro F1 despite requiring no additional training.</li>
          <li style="margin-bottom:0.45rem;">CamemBERT carries most of the predictive signal; adding images provides a consistent +1–2 pp gain over text alone.</li>
          <li style="margin-bottom:0;">The optimal α of 0.55 (55 % image weight) shows both modalities contribute meaningfully.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "6.1 Simple Fusion":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6.1 Simple Fusion — Late Fusion")
    st.write(
        "Late Fusion is the simplest possible multimodal strategy: the text model and image model each "
        "produce a probability distribution over the 27 product classes independently, and those "
        "distributions are blended by a weighted average. No joint training is needed."
    )

    st.subheader("How it works")
    _col_text_611, _col_img_611 = st.columns([1, 1.6])
    with _col_text_611:
        st.markdown(
            """
            1. **Text branch** — CamemBERT produces softmax probabilities **P_text** (27 classes).
            2. **Image branch** — ConvNeXt-Base produces softmax probabilities **P_image** (27 classes).
            3. **Fusion** — **P_fusion = α · P_image + (1 − α) · P_text**
            4. **Prediction** — argmax(P_fusion)

            The weight α was swept over a grid and the value that maximised macro F1 on the validation set was kept.
            """
        )
    with _col_img_611:
        st.image(str(ION_IMAGE_DIR / "late_fusion_architecture.png"), use_container_width=True)

    st.caption(
        "**What are softmax probabilities (logits)?** — Each model first produces one raw score "
        "(a logit) per class. A softmax function then converts those scores into a proper probability "
        "distribution: all 27 values are positive and sum to exactly 1, so P_text[i] and P_image[i] "
        "each express the model's confidence that the product belongs to class i. "
        "Late Fusion works entirely in this probability space — no retraining is needed."
    )

    alpha = mm_late_meta.get("best_alpha")
    macro_f1 = mm_late_meta.get("best_macro_f1")

    if alpha is not None:
        st.subheader("Optimal mixing weight")
        img_pct = int(round(alpha * 100))
        text_pct = 100 - img_pct
        _col_alpha_txt, _col_alpha_img = st.columns([1, 1.6])
        with _col_alpha_txt:
            st.markdown(
                f"**α = {alpha}** → image gets **{img_pct} %**, text gets **{text_pct} %**"
            )
            st.write(
                "α was found by sweeping values from 0 to 1 in steps of 0.01 and picking the α that "
                "maximised macro F1 on the validation set. The curve opposite shows how performance "
                "changes across the full range: the model rises steeply as image information is added "
                "to the strong text baseline, peaks at α = 0.50–0.55, then drops as the image branch "
                "dominates and the text signal is diluted."
            )
        with _col_alpha_img:
            _sweep_png = ION_IMAGE_DIR / "alpha_sweep_late_fusion.png"
            if _sweep_png.exists():
                st.image(str(_sweep_png), use_container_width=True)

    st.subheader("Validation metrics at optimal α")
    metrics_late = {
        "accuracy": mm_late_meta.get("accuracy"),
        "macro_f1": macro_f1,
        "weighted_f1": mm_late_meta.get("weighted_f1"),
        "validation_samples": 16984,
    }
    render_metric_cards(metrics_late, value_font_size="1.3rem")

    st.subheader("Unimodal vs. fusion comparison")


    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"


    comp_df = pd.DataFrame([
        {"Model": "Text only — CamemBERT", "Macro F1": _f(mm_late_meta.get("f1_text_only"))},
        {"Model": "Image only — ConvNeXt-Base", "Macro F1": _f(mm_late_meta.get("f1_image_only"))},
        {"Model": f"Late Fusion (α = {alpha})", "Macro F1": _f(macro_f1)},
    ])
    render_html_table(comp_df, max_width="600px")

    st.success(
        f"""
        Simple Fusion combines CamemBERT's text understanding with ConvNeXt-Base's visual features
        through a single weighted average — no additional training required.
        - Text-only (CamemBERT) already reaches a strong macro F1 of **{_f(mm_late_meta.get('f1_text_only'))}**;
          image-only (ConvNeXt-Base) reaches **{_f(mm_late_meta.get('f1_image_only'))}**.
        - The optimal blend (α = {alpha}) pushes macro F1 to **{_f(macro_f1)}** — a gain of
          **+{round(macro_f1 - mm_late_meta.get('f1_text_only', 0), 4):.4f}** over text alone and
          **+{round(macro_f1 - mm_late_meta.get('f1_image_only', 0), 4):.4f}** over image alone.
        - The image branch contributes most where product titles are ambiguous but visual packaging
          is distinctive (video game boxes, book covers, figurine illustrations).
        - Late Fusion is robust and reproducible: the only hyperparameter is α, and the sweep
          shows a wide, flat plateau around the optimum — small deviations from 0.50–0.55 have
          virtually no impact on performance.
        """
    )

elif page == "6.2 Intermediate Fusion":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6.2 Intermediate Fusion — Learned Joint Classifier")
    st.write(
        "Intermediate Fusion goes one step further than Late Fusion: instead of blending "
        "independent probability distributions, the feature representations from both modalities "
        "are concatenated and a new classification head is trained jointly on top of them. "
        "This allows the model to learn cross-modal interactions at the representation level."
    )

    st.subheader("How it works")
    _col_text_612, _col_img_612 = st.columns([1, 1.6])
    with _col_text_612:
        st.markdown(
            """
            1. **Text branch** — CamemBERT produces a feature vector **h_text**.
            2. **Image branch** — ConvNeXt-Base produces a feature vector **h_image**.
            3. **Fusion** — **h_fused = concat(h_text, h_image)**
            4. **Classifier** — a linear head jointly trained on **h_fused** predicts the 27 classes.
            5. **Prediction** — argmax(softmax(**W** · **h_fused** + **b**))

            Unlike Late Fusion, the classification head is trained end-to-end so it can weight the
            contribution of each modality per feature dimension rather than applying a single global α.
            """
        )
    with _col_img_612:
        st.image(str(ION_IMAGE_DIR / "intermediate_fusion_architecture.png"), use_container_width=True)

    st.caption(
        "**What are feature vectors?** — Before a neural network makes a class prediction it builds "
        "an internal numerical representation of the input: a high-dimensional vector of floating-point "
        "values (the penultimate layer's activations). These vectors — h_text from CamemBERT, h_image "
        "from ConvNeXt-Base — encode what each model has learned about the text and image respectively, "
        "without yet committing to a specific class. Concatenating them gives the joint classifier access "
        "to both modalities' raw learned signal in a single vector h_fused."
    )

    best_ep = mm_inter_meta.get("best_epoch", "—")
    inter_macro_f1 = mm_inter_meta.get("best_macro_f1")

    inter_acc, inter_wf1 = None, None
    if not mm_inter_history.empty and isinstance(best_ep, int):
        _row = mm_inter_history[mm_inter_history["epoch"] == best_ep]
        if not _row.empty:
            inter_acc = float(_row["val_acc"].iloc[0])
            inter_wf1 = float(_row["val_weighted_f1"].iloc[0])

    st.subheader(f"Validation metrics at best epoch (epoch {best_ep})")
    metrics_inter = {
        "accuracy": inter_acc,
        "macro_f1": inter_macro_f1,
        "weighted_f1": inter_wf1,
        "validation_samples": 16984,
    }
    render_metric_cards(metrics_inter, value_font_size="1.3rem")

    st.subheader("Unimodal vs. fusion comparison")


    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"


    comp_df = pd.DataFrame([
        {"Model": "Text only — CamemBERT", "Macro F1": _f(mm_late_meta.get("f1_text_only"))},
        {"Model": "Image only — ConvNeXt-Base", "Macro F1": _f(mm_late_meta.get("f1_image_only"))},
        {"Model": f"Late Fusion (α = {mm_late_meta.get('best_alpha', '—')})",
         "Macro F1": _f(mm_late_meta.get("best_macro_f1"))},
        {"Model": "Intermediate Fusion", "Macro F1": _f(inter_macro_f1)},
    ])
    render_html_table(comp_df, max_width="600px")

    st.success(
        """
        Intermediate Fusion trains a shared classifier on the concatenated text and image features,
        giving the model the ability to learn cross-modal patterns beyond a fixed mixing weight.
        Despite this added expressiveness, Late Fusion slightly outperforms Intermediate Fusion on
        this dataset — the unimodal probability distributions already capture the class signal well
        enough that a weighted average proves hard to beat. Both strategies clearly surpass any
        single modality, confirming the value of combining text and image information.
        """
    )

elif page == "6.3 Gated Fusion":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6.3 Gated Fusion — Learned Feature Mixing")
    st.write(
        "Gated Fusion combines CamemBERT (text) and CLIP Vision (image) at the feature level. "
        "Instead of blending probability distributions (Late Fusion) or concatenating features with a fixed head "
        "(Intermediate Fusion), a small gating network learns how much to rely on text versus image "
        "per feature dimension for each sample."
    )

    st.subheader("How it works")
    _col_text_613, _col_img_613 = st.columns([1, 1.6])
    with _col_text_613:
        st.markdown(
            """
            1. **Text branch** — CamemBERT → 768-dim CLS embedding **h_text**
            2. **Image branch** — CLIP ViT-B/32 → 768-dim pooled embedding **h_image**
            3. **Gate** — `g = sigmoid(MLP([h_text, h_image]))` — 768-dim vector
            4. **Fusion** — `fused = g · h_image + (1 − g) · h_text`
            5. **Classifier** — `[fused, h_text, h_image]` → 2304-dim → 27 classes

            The gate learns a **per-dimension** mixing weight: for each feature it decides
            whether text or image is more informative, rather than applying a single global weight.
            """
        )
    with _col_img_613:
        _gate_flow_img = APP_DIR / "images" / "gate_fusion_flow.png"
        if _gate_flow_img.exists():
            st.image(str(_gate_flow_img), use_container_width=True)

    st.caption(
        "**What does the gate learn?** — For text-rich categories (e.g. books, software) the gate "
        "weights h_text heavily; for visually distinctive categories (e.g. outdoor games, tools) "
        "it shifts weight towards h_image. This dynamic mixing is learned end-to-end."
    )

    st.subheader("Staged unfreezing strategy")
    st.write("Training proceeds in three stages to avoid catastrophic forgetting of the pretrained backbones.")
    unfreeze_df = pd.DataFrame([
        {"Stage": "Stage 1 — Head only",       "Trainable":   "Gate + classifier",                          "Backbones": "Frozen",            "Purpose": "Warm up fusion layers without disturbing pretrained weights."},
        {"Stage": "Stage 2 — Partial unfreeze", "Trainable":  "Gate + classifier + last 2 encoder blocks",  "Backbones": "Partially unfrozen", "Purpose": "Adapt high-level features to the product domain."},
        {"Stage": "Stage 3 — Full unfreeze",    "Trainable":  "All layers",                                 "Backbones": "Fully unfrozen",     "Purpose": "End-to-end fine-tuning for maximum task adaptation."},
    ])
    render_html_table(unfreeze_df, max_width="960px")

    st.subheader("Validation metrics — best model (aug + softmax gate)")
    metrics_gate = {"accuracy": 0.892, "macro_f1": 0.880, "weighted_f1": None, "validation_samples": 16984}
    render_metric_cards(metrics_gate, value_font_size="1.3rem")

    st.subheader("Unimodal vs. fusion comparison")
    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"
    comp_gate_df = pd.DataFrame([
        {"Model": "Text only — CamemBERT",          "Macro F1": _f(mm_late_meta.get("f1_text_only"))},
        {"Model": "Image only — ConvNeXt-Base",      "Macro F1": _f(mm_late_meta.get("f1_image_only"))},
        {"Model": "Gated Fusion — frozen",           "Macro F1": "0.8640"},
        {"Model": "Gated Fusion — staged unfreeze",  "Macro F1": "0.8640"},
        {"Model": "Gated Fusion — aug + softmax ★",  "Macro F1": "0.8800"},
    ])
    render_html_table(comp_gate_df, max_width="600px")

    st.success(
        "Gated Fusion learns to dynamically weight text and image features per dimension, "
        "giving it more expressive power than a fixed mixing weight. "
        "The best result (macro F1 0.880) came not from unfreezing alone, but from combining "
        "a shared projection space, softmax gating, image augmentation, and label smoothing. "
        "Late Fusion with ConvNeXt still outperforms (0.891), suggesting that ConvNeXt's richer "
        "image features matter more than the fusion mechanism itself."
    )

elif page == "6.4 CLIP Models":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6.4 CLIP Models")
    st.markdown(
        """
CLIP (`openai/clip-vit-base-patch32`) learns visual and textual representations in a shared embedding space.
It was pretrained on large-scale image-text pairs, which makes its image encoder a strong general-purpose
feature extractor for product images.

However, CLIP's built-in text encoder is not ideal for this dataset. It was mainly designed for English prompts
and is limited to 77 tokens, while Rakuten product descriptions are in French and often longer.

For this reason, we kept the CLIP Vision encoder for image understanding and replaced the text branch with
CamemBERT, which is pretrained on French text. This combines strong visual features from CLIP with stronger
French-language understanding from CamemBERT.
        """
    )

    _clip_arch = APP_DIR / "images" / "clip.png"
    if _clip_arch.exists():
        col, _ = st.columns([0.6, 0.4])
        col.image(str(_clip_arch), use_container_width=True)


    clip_chart = APP_DIR / "images" / "clip_model_comparison.png"
    if clip_chart.exists():
        col, _ = st.columns([0.6, 0.4])
        col.image(str(clip_chart), use_container_width=True)

    st.subheader("Key findings")
    st.markdown(
        """
CLIP-based models were multimodal in architecture, but mostly text-dominated in practice.

CLIP alone performed weaker because its text encoder is designed for short, mostly English image-text pairs. Rakuten descriptions are French and often longer, so CLIP's 77-token text limit became a bottleneck.

Replacing CLIP text with CamemBERT improved performance strongly, from about 0.672 to 0.864 macro F1. Since the CLIP vision encoder stayed the same, this shows that the main limitation was the text branch.

Frozen and staged-unfreeze models performed almost the same because CamemBERT already captured most category information. The image branch added only a very small gain, around +0.002 macro F1.

The best CLIP-based model reached 0.880 macro F1, mainly due to better projection, softmax gating, augmentation, and label smoothing — not unfreezing alone.

The best overall result was ConvNeXt + CamemBERT late fusion, reaching 0.891 macro F1. This suggests that ConvNeXt provided more useful product-image features than CLIP Vision.
        """
    )

    st.success(
        "Takeaway: For this dataset, text is very strong. CLIP gating was technically multimodal, but mostly relied on CamemBERT. "
        "The strongest model used CamemBERT for French text and ConvNeXt for product images."
    )

elif page == "6.5 Best model — Summary":
    st.title("Rakuten Multimodal Product Classification")
    st.header("6.5 Best Multimodal Model — Simple Fusion")

    alpha = mm_late_meta.get("best_alpha", "—")
    accuracy = mm_late_meta.get("accuracy")
    macro_f1 = mm_late_meta.get("best_macro_f1")
    wf1 = mm_late_meta.get("weighted_f1")
    f1_text = mm_late_meta.get("f1_text_only")
    f1_image = mm_late_meta.get("f1_image_only")
    inter_f1 = mm_inter_meta.get("best_macro_f1")


    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"


    # ── Why Simple Fusion wins ────────────────────────────────────────────────
    st.subheader("Why Simple Fusion is the best multimodal model")
    st.write(
        "Three fusion strategies were evaluated. The table below compares their best validation macro F1 "
        "alongside the two unimodal baselines."
    )
    render_html_table(pd.DataFrame([
        {"Strategy": "Text only — CamemBERT", "Macro F1": _f(f1_text), "Notes": "Unimodal baseline"},
        {"Strategy": "Image only — ConvNeXt-Base (I12)", "Macro F1": _f(f1_image), "Notes": "Unimodal baseline"},
        {"Strategy": "Intermediate Fusion (trained head)", "Macro F1": _f(inter_f1),
         "Notes": "Feature concat + joint linear head, 6 epochs"},
        {"Strategy": "CLIP Gated Fusion (best run)", "Macro F1": "0.8800",
         "Notes": "CamemBERT + CLIP ViT-B/32, augmentation + label smoothing"},
        {"Strategy": "Simple Fusion — Late Fusion (alpha-sweep)", "Macro F1": _f(macro_f1),
         "Notes": "Weighted average of softmax outputs, no training"},
    ]), max_width="960px")

    render_html_table(pd.DataFrame([
        {"Factor": "No additional training",
         "Detail": "Late Fusion requires only a grid search over alpha on the validation set — no gradient step, "
                   "no risk of overfitting or catastrophic forgetting. Both branches keep their individually "
                   "optimised weights."},
        {"Factor": "Text branch carries most signal",
         "Detail": f"CamemBERT alone scores F1 {_f(f1_text)} — already very strong. The alpha sweep settles on "
                   f"alpha=0.55, meaning image gets 55% and text 45%. Text is almost equally weighted, "
                   "and even a small image contribution resolves the cases text alone gets wrong."},
        {"Factor": "Intermediate Fusion overfit",
         "Detail": f"Training a joint head on top of frozen branches (Intermediate Fusion, F1 {_f(inter_f1)}) "
                   "shows the model peaked at epoch 3 and then stagnated — the head had too few gradient "
                   "steps to add value over Late Fusion's calibrated blend."},
        {"Factor": "CLIP image encoder is weaker than ConvNeXt-Base",
         "Detail": "The best CLIP run reaches F1 0.8800 — below Late Fusion's 0.8994. "
                   "CLIP ViT-B/32 produces general-purpose visual features; ConvNeXt-Base was fine-tuned "
                   "end-to-end on Rakuten product images, giving it a decisive task-specific advantage."},
    ]), max_width="960px")

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("Validation metrics at optimal alpha")
    metrics_best = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": wf1,
        "validation_samples": 16984,
    }
    render_metric_cards(metrics_best, value_font_size="1.3rem")

    st.subheader("Unimodal vs. fusion comparison")
    comp_df = pd.DataFrame([
        {"Model": "Image only — ConvNeXt-Base", "Macro F1": _f(f1_image), "Accuracy": "0.7197"},
        {"Model": "Text only — CamemBERT", "Macro F1": _f(f1_text), "Accuracy": "—"},
        {"Model": f"Simple Fusion (alpha={alpha})", "Macro F1": _f(macro_f1), "Accuracy": _f(accuracy, 4)},
    ])
    render_html_table(comp_df, max_width="640px")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion matrix — validation set")
    st.write(
        "The row-normalised confusion matrix shows per-class recall on the diagonal. "
        "Compared to the image-only model (I12), the diagonal is dramatically brighter across almost "
        "all 27 classes — CamemBERT's text features resolve the visual ambiguities that ConvNeXt-Base "
        "alone could not handle. Key patterns:"
    )
    st.markdown(
        "- **Near-perfect classes** — Tool & Garden (F1 0.955), DL Games (0.994), Baby/Child (0.988), "
        "Toy Cars (0.992), Pool Accessories (0.987): text titles make these unambiguous.\n"
        "- **Most improved over I12** — Game Accessories (+0.503), Pet Accessories (+0.471), "
        "Tools/Garden (+0.604), Toys/Plush (+0.460): categories the image model confused with "
        "visually similar neighbours, but whose product names are highly distinctive.\n"
        "- **Remaining hard classes** — Board & Card Games (F1 0.716), Collectible Board-game Figurines (0.771), "
        "Toys/Plush (0.807): these categories still overlap both visually and textually "
        "(e.g. a 'figurine set' can appear in multiple classes).\n"
        "- **Largest residual errors** — Board Games → Toys/Plush (90), "
        "Books/Comics → Books (55), Comics/Books → Books (43): "
        "even combined, text descriptions for these pairs are genuinely similar."
    )
    _lf_cm = ION_IMAGE_DIR / "confusion_matrix_latefusion_labeled.png"
    if _lf_cm.exists():
        st.image(str(_lf_cm),
                 caption="Row-normalised confusion matrix with class names and counts — Simple Fusion (CamemBERT + ConvNeXt-Base)",
                 width=650)
    elif mm_late_cm_png and mm_late_cm_png.exists():
        st.image(str(mm_late_cm_png), caption="Validation confusion matrix — Simple Fusion", width=650)
    else:
        st.info("Confusion matrix image not found.")

    # ── I12 vs T8 vs Fusion per-class comparison chart ───────────────────────
    st.subheader("Per-class F1: Image-only (I12) vs. Text-only (T8) vs. Simple Fusion")
    st.write(
        "Each group of three bars shows the image-only, text-only, and fusion score for every class. "
        "Text (blue) already dominates most classes; the green fusion bar shows where combining both modalities "
        "squeezes out a meaningful extra gain."
    )
    _cls_names = ["Books", "Video Games", "Game Acc.", "Game Consoles", "Figurines", "Coll. Cards", "Board Figs.",
                  "Toys/Plush", "Board Games", "Toy Cars", "Baby/Child", "Outdoor Games", "Womens Bags",
                  "Furniture", "Linen", "Food", "Lamps/Decor", "Pet Acc.", "Magazines", "Books/Comics",
                  "Console/Games", "Stationery", "Outdoor Furn.", "Pool Acc.", "Tools/Garden", "Comics/Books",
                  "DL Games"]
    _i12_f1s = [0.469, 0.471, 0.396, 0.660, 0.609, 0.887, 0.321, 0.347, 0.274, 0.665, 0.579, 0.429, 0.465,
                0.573, 0.792, 0.625, 0.499, 0.428, 0.662, 0.615, 0.521, 0.647, 0.458, 0.750, 0.351, 0.690, 0.578]
    _t8_f1s = [0.727, 0.815, 0.894, 0.926, 0.835, 0.970, 0.714, 0.804, 0.718, 0.990, 0.991, 0.844, 0.875,
               0.900, 0.933, 0.934, 0.868, 0.907, 0.924, 0.866, 0.848, 0.948, 0.867, 0.987, 0.945, 0.836, 0.991]
    _lf_f1s = [0.793, 0.891, 0.899, 0.935, 0.861, 0.983, 0.771, 0.807, 0.716, 0.992, 0.988, 0.864, 0.895,
               0.893, 0.928, 0.932, 0.869, 0.899, 0.930, 0.883, 0.882, 0.956, 0.872, 0.987, 0.955, 0.911, 0.994]
    _x = range(len(_cls_names))
    fig_cmp, ax_cmp = plt.subplots(figsize=(18, 4.5))
    ax_cmp.bar([i - 0.27 for i in _x], _i12_f1s, width=0.25, label="ConvNeXt-Base I12 (image)", color="#94a3b8")
    ax_cmp.bar([i for i in _x], _t8_f1s, width=0.25, label="CamemBERT T8 (text)", color="#3b82f6")
    ax_cmp.bar([i + 0.27 for i in _x], _lf_f1s, width=0.25, label="Simple Fusion", color="#059669")
    ax_cmp.set_xticks(list(_x))
    ax_cmp.set_xticklabels(_cls_names, rotation=45, ha="right", fontsize=7)
    ax_cmp.set_ylim(0, 1.08)
    ax_cmp.set_ylabel("Macro F1", fontsize=8)
    ax_cmp.legend(fontsize=8)
    ax_cmp.spines[["top", "right"]].set_visible(False)
    ax_cmp.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_cmp.set_axisbelow(True)
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    st.subheader("Where fusion adds the little extra over text-only")
    st.write(
        "CamemBERT already handles most of the classification work. The image branch contributes a "
        "meaningful additional signal in two specific situations:"
    )
    render_html_table(pd.DataFrame([
        {
            "Category group": "Comics / Books family",
            "Text F1": "0.836 / 0.727 / 0.866",
            "Fusion F1": "0.911 / 0.793 / 0.883",
            "Delta": "+0.075 / +0.066 / +0.017",
            "Why image helps": "Comics/Books, Books and Books/Comics share near-identical vocabulary "
                               "(author names, titles, genre terms). The cover design, illustration style "
                               "and layout are visually distinctive — e.g. manga-style art vs. novel cover "
                               "vs. illustrated comics — and the image branch resolves these confusions.",
        },
        {
            "Category group": "Video Games / Console-Games",
            "Text F1": "0.815 / 0.848",
            "Fusion F1": "0.891 / 0.882",
            "Delta": "+0.076 / +0.034",
            "Why image helps": "Product names like 'FIFA 23' or 'Nintendo Switch bundle' can appear in "
                               "multiple classes. Box art, controller shape and platform logo on the "
                               "packaging give the image branch a clean visual signal that text alone misses.",
        },
        {
            "Category group": "Board-game Figurines",
            "Text F1": "0.714",
            "Fusion F1": "0.771",
            "Delta": "+0.057",
            "Why image helps": "Descriptions use shared terms ('figurine', 'miniature', 'set'). "
                               "The physical appearance of painted game pieces vs. decorative figurines "
                               "is visually distinct, giving the image branch a useful tie-breaking signal.",
        },
        {
            "Category group": "Classes where fusion is neutral or slightly worse",
            "Text F1": "e.g. Baby/Child 0.991, Pet Acc. 0.907",
            "Fusion F1": "0.988 / 0.899",
            "Delta": "−0.003 / −0.008",
            "Why image helps": "Text is already near-perfect for these classes. The image branch "
                               "introduces a tiny amount of noise from edge cases where visual appearance "
                               "is ambiguous, marginally reducing the text-only score. The effect is small "
                               "and consistent with alpha=0.55 giving 45% weight to text.",
        },
    ]), max_width="100%")

    # ── Conclusion ────────────────────────────────────────────────────────────
    st.divider()
    st.success(
        f"""
        **Simple Fusion (CamemBERT + ConvNeXt-Base, alpha={alpha}) is the best multimodal model**,
        reaching macro F1 **{_f(macro_f1)}** and accuracy **{_f(accuracy, 4)}** on 16 984 validation samples.

        - Macro F1 rises from **{_f(f1_image)}** (image only) and **{_f(f1_text)}** (text only) to **{_f(macro_f1)}** — a gain of
          +{round(macro_f1 - max(f1_image, f1_text), 4) if macro_f1 and f1_image and f1_text else '—'} over the stronger unimodal branch.
        - The largest improvements are in classes that are visually ambiguous but textually distinctive:
          Tools/Garden (+0.604), Game Accessories (+0.503), DL Games (+0.416), Pet Accessories (+0.471).
        - Remaining confusion is concentrated in three clusters where both text and image evidence overlaps:
          board games / toys / figurines, and the books / comics family.
        - No extra training was needed — Late Fusion is inference-only, making it robust, reproducible,
          and free of hyperparameter sensitivity beyond the single alpha value.
        """
    )


# =========================
# 2. Data Exploration (parent)
# =========================
elif page == "2. Data Exploration":
    st.title("2 Data Exploration")

    st.write(
        """
        Key dataset characteristics that influenced preprocessing, modeling, and evaluation.
        """
    )

    st.subheader("Category distribution")
    col1, col2, col3 = st.columns(3)
    col1.metric("Categories", "27")
    col2.metric("Largest class", ">10 000")
    col3.metric("Smallest classes", "~700–800")
    col_text, col_img = st.columns([1, 2])

    with col_text:
        st.write(
            """
            - The dataset shows a **moderate but clear class imbalance**
            - The largest class, `2583`, contains **10,209 samples** and represents about **12%** of the training set
            - Several other large classes contain around **5,000 samples** each
            - The median class size is **2,671 samples**
            - The smallest classes still contain around **760–870 samples**
            - This means the dataset is imbalanced, but not extremely sparse
            - Accuracy alone could be biased toward the largest categories
            - **Macro F1** is important because it gives equal weight to each class, including smaller categories
            """
        )

    with col_img:
        _img = ION_IMAGE_DIR / "category_balance.png"
        if _img.exists():
            st.image(
                str(_img),
                caption="Category distribution: largest vs smallest classes",
                width="stretch"
            )

# =========================
# 2.1 Text Exploration
# =========================
elif page == "2.1 Text":
    st.header("2.1 Text Exploration")

    col_T10, col_T11 = st.columns([1, 3.5])

    with col_T10:
        st.subheader("Text fields")
        st.write(
            """
            **Designation**
            - Product title
            - Always available
            - Short and consistent
            - Strong category signal
            """
        )

        st.write(
            """
            **Description**
            - ~35% missing
            - Longer and more variable
            - Very long descriptions often contain duplicated content
            - Adds product attributes and context
            """
        )

    with col_T11:
        _img = ION_IMAGE_DIR / "text_length.png"
        if _img.exists():
            st.image(str(_img),
                     caption="Titles are short and consistent; descriptions are longer, variable, and often missing",
                     width="stretch")
# =========================
# 2.11 Vocabulary
# =========================
elif page == "2.11 Vocabulary":

    st.subheader("Data quality")
    col1, col2, col3 = st.columns(3)
    col1.metric("Missing descriptions", "~35%")
    col2.metric("Duplicated text blocks", "~1.5%")
    col3.metric("Numeric tokens", "~8–9%")
    st.write(
        """
        Duplicated description segments were removed to reduce noise while preserving the majority of samples.
        Numeric tokens were kept because product references, sizes, and model numbers may be useful.
        """
    )

    st.subheader("Vocabulary insights")
    col1, col2 = st.columns(2)
    col1.metric("Title vocabulary", "~82k tokens")
    col2.metric("Description vocabulary", "~137k tokens")
    with col1:
        st.write(
            """
            - Titles often contain category-defining product keywords
            """
        )

    with col2:
        st.write(
            """
            - Descriptions mostly add attributes such as size, color, material, and condition
            """
        )
    _img = ION_IMAGE_DIR / "token_comparison.png"
    if _img.exists():
        st.image(
            str(_img),
            caption="Titles contain category-defining keywords, while descriptions add broader and often less specific vocabulary (blue = shared, red = description-only)",
            use_container_width=True
        )

    st.success(
        """
        Titles are the strongest text feature.  \n
        Descriptions provide useful but noisier context.  \n
        """
    )

# =========================
# 2.2 Image Exploration
# =========================
elif page == "2.2 Image":
    st.header("2.2 Image Exploration")

    st.subheader("Dataset overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total images", "98 728")
    col2.metric("Train images", "84 916")
    col3.metric("Test images", "13 812")
    col4.metric("Disk size", "2.44 GB")
    col5.metric("Missing images", "0%")

    st.write("Each product is associated with one image, linked via `imageid` and `productid`.")

    col_T20, col_T21 = st.columns(2)
    with col_T20:
        st.subheader("Image properties")
        st.write(
            """
            - format: JPG (standardized across dataset)
            - dimension : 500 × 500 pixels
            - color depth: 24-bit
            - resolution: 96 dpi
            """
        )

    with col_T21:

        st.subheader("Data quality")
        st.write(
            """
            - images are of good quality
            - all images follow a consistent format and size.            
            """
        )

    st.subheader("Sample products")
    sample = get_ion_dataset_examples()
    if sample.empty:
        st.warning("Training data not found — cannot show sample products.")
    else:
        for _, row in sample.iterrows():
            image_path = ION_IMAGE_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg"
            col_img, col_text = st.columns([1, 2])
            with col_img:
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.caption("Image not available")
            with col_text:
                st.write(f"**Category:** `{row['prdtypecode']}`")
                st.write(f"**Designation:**  \n {row['designation']}")
                desc = row.get("description", "")
                if pd.isna(desc) or str(desc).strip() == "":
                    desc = "Missing"
                else:
                    desc = str(desc)[:500] + "..."
                st.write(f"**Description:**  \n {desc}")
            st.divider()

    st.success(
        """
        Images provide complementary information to text, but their variability and
        makes standalone image classification more challenging.
        """
    )

# =========================
# 3. Preprocessing
# =========================
elif page == "3. Preprocessing":
    st.header("3. Text Preprocessing")

    st.write(
        """
        Text preprocessing prepares raw product text for different modeling approaches.
        The pipeline is designed to clean noise while preserving informative signals.
        """
    )

    st.subheader("Cleaning pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            - lowercase conversion
            - removal of punctuation and special characters
            """
        )
    with col2:
        st.markdown(
            """
            - removal of HTML tags and encoded text
            - whitespace normalization
            """
        )
    with col3:
        st.markdown(
            """
            - removal of short tokens (<2 characters)
            - stopword removal (French, English, German)
            """
        )

    st.subheader("Numeric information")
    col_text, col_plot = st.columns([1.6, 0.9])
    with col_text:
        st.markdown(
            """
            - ~8–9% of tokens are purely numeric
            - numbers may encode useful information (size, quantity, model IDs)
            """
        )
        st.markdown(
            """
            Two configurations are evaluated:
            - keep numeric tokens
            - remove numeric tokens

            This allows measuring their impact on model performance.
            """
        )
    with col_plot:
        _img = ION_IMAGE_DIR / "numeric_token_share.png"
        if _img.exists():
            st.image(str(_img), caption="Share of numeric tokens in titles and descriptions.", width=380)

# =========================
# 3.1  Text Preprocessing
# =========================
elif page == "3.1 Text Preprocessing":
    st.subheader("Description deduplication")
    st.markdown("~1.5% of descriptions contain repeated text blocks that artificially inflate length.")
    st.markdown(
        """
        A preprocessing step removes consecutive duplicated segments while preserving
        the original content.
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- reduces extreme outliers")
    with col2:
        st.markdown("- does not affect the majority of samples")
    with col3:
        st.markdown("- improves overall data quality")

    col_T30, col_T31 = st.columns([1, 2])
    with col_T30:
        st.subheader("Tokenization")
        st.markdown(
            """
            After cleaning, text is tokenized to enable:
            - vocabulary construction
            - frequency-based representations (TF-IDF)
            - input formatting for neural models
            """
        )
    with col_T31:
        _img = ION_IMAGE_DIR / "tokenization_example.png"
        if _img.exists():
            st.image(str(_img),
                     caption="Example transformation from raw product title to tokenized input.",
                     width="stretch")

    st.success("Preprocessing removes noise while preserving informative signals.")

# =========================
# 4. Text Modeling (parent)
# =========================
elif page == "4. Text Modeling":
    st.header("4.1 Text Modeling Overview")

    st.subheader("Key questions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <ul>
                <li><span style="color:#5B8DB8;">which text source matters most?</span></li>
                <li><span style="color:#E69F56;">do bigrams improve performance?</span></li>
            </ul>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <ul>
                <li><span style="color:#6FA36F;">should numeric tokens be kept?</span></li>
                <li><span style="color:#8E7BBE;">which classifier works best?</span></li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    st.subheader("TF-IDF results")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "0.81")
    col2.metric("Macro F1", "0.79")

    with col1:
        st.write(
            """
            <ul>
                <li><span style="color:#5B8DB8;">combining designation + description outperforms either field alone</span></li>
                <li><span style="color:#E69F56;">bigrams add a modest improvement over unigrams</span></li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.write(
            """
            <ul>
                <li><span style="color:#6FA36F;">numeric tokens are slightly helpful</span></li>
                <li><span style="color:#8E7BBE;">6 classifiers checked. LinearSVC slightly outperforms Logistic Regression</span></li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Best TF-IDF configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Classifier", "LinearSVC")
    col2.metric("N-grams", "1–2")
    col3.metric("Numeric tokens", "Kept")
    st.success("Strong, fast, and interpretable baseline.")


# =========================
# 4.1 Text Modeling — Overview
# =========================
elif page == "4.1 Overview":
    st.subheader("Model comparison")
    col_T40, col_T41 = st.columns([1, 2])

    with col_T40:
        st.markdown(
            """
            **TF-IDF**
            - sparse
            - strong keyword matching
            - competitive baseline
            """
        )

        st.markdown(
            """
            **MiniLM**
            - dense
            - loses lexical detail
            - weakest performance
            """
        )

        st.markdown(
            """
            **CamemBERT**
            - context-aware
            - task adaptive
            - best overall performance
            """
        )
    with col_T41:
        _img = ION_IMAGE_DIR / "model_comparison.png"
        if _img.exists():
            st.image(str(_img),
                     caption="CamemBERT slightly outperforms TF-IDF, while MiniLM-based models lag behind.",
                     width="stretch")

    st.success("Semantic compression reduces performance for product classification.")

# =========================
# 4.2 Text Modeling — Best model
# =========================
elif page == "4.2 Best model":
    st.header("4.2 Best Text Model: CamemBERT")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "CamemBERT")
    col2.metric("Input", "Title + description")
    col3.metric("Type", "Transformer")

    col1.metric("Accuracy", "0.8719")
    col2.metric("Macro F1", "0.8557")
    col3.metric("Rank", "Best")
    st.success("Best-performing text model across all evaluated approaches.")

    col_T50, col_T51 = st.columns([1, 3.5])

    with col_T50:
        st.subheader("Key training decisions")
        st.markdown(
            """
            - longer sequences improve performance
            - gains continue up to 4 epochs
            - best configuration: **256 tokens / 4 epochs**
            """
        )

        st.subheader("Training behavior")
        st.markdown(
            """
            - training loss ↓ steadily
            - validation improves up to epoch 4
            - slight overfitting appears after epoch 2
            """
        )

    with col_T51:
        _img = ION_IMAGE_DIR / "camembert_training_curves_split.png"
        if _img.exists():
            st.image(str(_img),
                     caption="Performance stabilizes around epochs 3–4.",
                     width="stretch")

# =========================
# 4.21 CamemBERT vs TF-IDF
# =========================
elif page == "4.21 CamemBERT vs TF-IDF":

    col_T60, col_T61 = st.columns([1, 2.2])
    with col_T60:
        st.subheader("Why it works")
        st.markdown(
            """
            **Strengths**
            - captures context
            - uses full sequences
            - adapts via fine-tuning
            """
        )

        st.markdown(
            """
            **Compared to TF-IDF**
            - not limited to keywords
            - understands phrasing
            """
        )

        st.subheader("Per-class performance")
        col_T60.metric("Categories improved", "17 / 27")
        st.markdown("TF-IDF remains competitive in keyword-driven categories.")

    with col_T61:

        _img = ION_IMAGE_DIR / "per_class_f1_delta_tfidf_camembert_top_changes.png"
        if _img.exists():
            st.image(str(_img), caption="Per-class performance differences.",
                     width="stretch")

    col_T62, col_T63 = st.columns([1, 2.2])

    with col_T62:
        st.subheader("Where models differ")
        st.markdown(
            """
            **TF-IDF**
            - repetitive keywords
            - strong lexical signals
            - term-driven categories
            """
        )

        st.markdown(
            """
            **CamemBERT**
            - diverse language
            - distributed meaning
            - context-dependent
            """
        )

        st.subheader("Limitations")
        st.markdown(
            """
            - overlapping vocabulary across classes
            - text alone not always sufficient
            """
        )

    with col_T63:
        _img = ION_IMAGE_DIR / "bow_class_comparison.png"
        if _img.exists():
            st.image(str(_img),
                     caption="Red: TF-IDF better | Blue: CamemBERT better.",
                     width="stretch")

    st.success(
        """
        CamemBERT improves performance for most categories, especially where wording and context matter. \n
        TF-IDF remains strong when categories are defined by repeated, distinctive keywords.
        """
    )


elif page == "7. Prediction Tool":
    render_prediction_tool()

# -------------------------------------------------------------------
# 8. Conclusions
# -------------------------------------------------------------------

elif page == "8. Project Conclusions":

    st.header("8. Project Conclusions")

    st.write(
        "This project built and compared text, image, and multimodal models for classifying "
        "Rakuten products into 27 categories. The pipeline progressed from simple baselines "
        "to a full multimodal fusion system combining CamemBERT and ConvNeXt-Base."
    )

    _conclusion_img = APP_DIR / "images" / "project_conclusion.png"
    if _conclusion_img.exists():
        _conc_col, _ = st.columns([0.8, 0.2])
        with _conc_col:
            st.image(str(_conclusion_img), use_container_width=True)

    st.subheader("")
    st.markdown(
        """
        <div style="max-width:960px; overflow-x:auto; margin:0.35rem 0 1.0rem 0;">
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr>
              <th style="background:#2e7d32; color:#fff; font-weight:600; text-align:left; border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; width:22%;">Lesson</th>
              <th style="background:#2e7d32; color:#fff; font-weight:600; text-align:left; border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem;">Explanation</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background:#e8f5e9;"><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#1b5e20; font-weight:600;">Text dominates</td><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#262730;">French product titles and descriptions carry the strongest category signal — CamemBERT alone reaches macro F1 ≈ 0.871.</td></tr>
            <tr style="background:#f1f8f1;"><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#1b5e20; font-weight:600;">Images are complementary</td><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#262730;">ConvNeXt-Base adds signal for visually distinctive categories but cannot resolve text-ambiguous ones alone.</td></tr>
            <tr style="background:#e8f5e9;"><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#1b5e20; font-weight:600;">Late fusion is powerful</td><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#262730;">A simple weighted average of the two unimodal outputs achieves the best overall result without any additional training.</td></tr>
            <tr style="background:#f1f8f1;"><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#1b5e20; font-weight:600;">Architecture &gt; tuning</td><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#262730;">Switching backbone (ResNet → ConvNeXt) gave larger gains than any fine-tuning change within the same architecture.</td></tr>
            <tr style="background:#e8f5e9;"><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#1b5e20; font-weight:600;">Macro F1 is essential</td><td style="border:1px solid #c8e6c9; padding:0.62rem 0.75rem; font-size:1.06rem; color:#262730;">Accuracy masks class imbalance; macro F1 reflects true performance across all 27 categories equally.</td></tr>
          </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "9. Final Benchmark":

    st.header("9. Challenge Result")

    st.write(
        """
        As a final validation, we submitted our predictions to the challenge platform.
        Our final model reached the **top 5 on the public leaderboard**.
        """
    )

    st.markdown(
        """
        Public leaderboard:  
        https://challengedata.ens.fr/participants/challenges/35/ranking/public
        """
    )

    col_T70, col_T71, col_T72 = st.columns([1, 2, 1])

    with col_T71 :
        _img = ION_IMAGE_DIR / "leaderboard_top5.png"
        if _img.exists():
            st.image(
                str(_img),
                caption="Public leaderboard result from the Rakuten challenge platform.",
                width="stretch"
            )

    st.info(
        """
        Final note: reaching the top 5 is a strong outcome for the project.
        It confirms that the workflow was effective, while leaving room for further refinement.
        """
    )

