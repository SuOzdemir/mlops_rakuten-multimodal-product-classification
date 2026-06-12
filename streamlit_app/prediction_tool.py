import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from services.final_fusion_predictor import load_assets, predict

st.set_page_config(page_title="Prediction", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0.4rem !important;
        min-height: 70px !important;
    }
    div[data-testid="stFileUploader"] label {
        font-size: 0.9rem !important;
    }
    .small-info {
        font-size: 0.9rem;
        padding: 0.4rem 0.6rem;
        border-radius: 8px;
        border: 1px solid rgba(120,120,120,0.2);
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent
CATEGORY_PATH = BASE_DIR / "config" / "prdtypecode_mapping.json"

with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
    PRDTYPECODE_TO_NAME = {int(k): v for k, v in json.load(f).items()}

st.title("Prediction")
st.caption("Upload a CSV and product image. Metadata is filled automatically if a matching row is found.")


@st.cache_resource
def get_assets():
    return load_assets()


def normalize_filename(value):
    if pd.isna(value):
        return ""
    return Path(str(value)).name.strip().lower()


def normalize_column_name(value):
    return str(value).strip().lower()


def safe_cell_to_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in ["nan", "none", "null"]:
        return ""
    return text


def find_column(row, candidates):
    normalized_cols = {
        normalize_column_name(col): col
        for col in row.index
    }

    for candidate in candidates:
        col = normalized_cols.get(normalize_column_name(candidate))
        if col is not None:
            return col

    return None


def find_metadata_row(df, image_filename):
    image_filename_norm = normalize_filename(image_filename)

    candidate_columns = [
        "image_name",
        "filename",
        "file_name",
        "image_filename",
        "image_path",
        "path",
        "image",
    ]

    normalized_df_cols = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate_col in candidate_columns:
        real_col = normalized_df_cols.get(normalize_column_name(candidate_col))
        if real_col is not None:
            matches = df[df[real_col].apply(normalize_filename) == image_filename_norm]
            if not matches.empty:
                return matches.iloc[0]

    imageid_col = normalized_df_cols.get("imageid") or normalized_df_cols.get("image_id")
    productid_col = normalized_df_cols.get("productid") or normalized_df_cols.get("product_id")

    if imageid_col is not None and productid_col is not None:
        expected_names = df.apply(
            lambda row: f"image_{row[imageid_col]}_product_{row[productid_col]}.jpg".lower(),
            axis=1
        )
        matches = df[expected_names == image_filename_norm]
        if not matches.empty:
            return matches.iloc[0]

    for col in df.columns:
        if df[col].dtype == "object":
            matches = df[df[col].apply(normalize_filename) == image_filename_norm]
            if not matches.empty:
                return matches.iloc[0]

    return None


def get_text_from_row(row):
    if row is None:
        return "", ""

    designation_candidates = [
        "designation",
        "title",
        "product_title",
        "product_name",
        "name",
    ]

    description_candidates = [
        "description",
        "description_dedup",
        "description dedup",
        "description_clean",
        "description clean",
        "clean_description",
        "clean description",
        "product_description",
        "product description",
        "desc",
    ]

    designation_col = find_column(row, designation_candidates)
    description_col = find_column(row, description_candidates)

    designation = safe_cell_to_text(row[designation_col]) if designation_col is not None else ""
    description = safe_cell_to_text(row[description_col]) if description_col is not None else ""

    return designation, description


def get_image_id_from_row(row):
    if row is None:
        return "not found in csv file"

    image_id_candidates = ["imageid", "image_id", "image id"]
    image_id_col = find_column(row, image_id_candidates)

    if image_id_col is not None:
        image_id = safe_cell_to_text(row[image_id_col])
        if image_id:
            return image_id

    return "not found in csv file"


def format_prediction_label(code):
    category_name = PRDTYPECODE_TO_NAME.get(int(code), "Unknown product type")
    return f"{code} - {category_name}"


def clear_image_related_state():
    st.session_state.designation_value = ""
    st.session_state.description_value = ""
    st.session_state.matched_image_id = ""
    st.session_state.prediction_output = None


if "designation_value" not in st.session_state:
    st.session_state.designation_value = ""

if "description_value" not in st.session_state:
    st.session_state.description_value = ""

if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

if "last_csv_name" not in st.session_state:
    st.session_state.last_csv_name = None

if "matched_image_id" not in st.session_state:
    st.session_state.matched_image_id = ""

if "csv_status" not in st.session_state:
    st.session_state.csv_status = "No CSV loaded."

if "prediction_output" not in st.session_state:
    st.session_state.prediction_output = None

left_col, middle_col, right_col = st.columns([1.25, 1.05, 0.95], gap="medium")

df = None
image = None

with left_col:
    st.markdown("### Input")

    csv_file = st.file_uploader(
        "Metadata CSV",
        type=["csv"],
        label_visibility="visible"
    )

    if csv_file is None:
        if st.session_state.last_csv_name is not None:
            st.session_state.csv_status = "CSV deleted."
            st.session_state.last_csv_name = None
        else:
            st.session_state.csv_status = "No CSV loaded."
    else:
        try:
            df = pd.read_csv(csv_file)
            df.columns = [str(col).strip() for col in df.columns]
            st.session_state.csv_status = f"CSV loaded: {len(df)} rows"
            st.session_state.last_csv_name = csv_file.name
        except Exception:
            df = None
            st.session_state.csv_status = "CSV could not be read."
            st.session_state.last_csv_name = csv_file.name

    st.markdown(
        f"<div class='small-info'>{st.session_state.csv_status}</div>",
        unsafe_allow_html=True
    )

    image_file = st.file_uploader(
        "Product image",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if image_file is None:
        if st.session_state.last_image_name is not None:
            st.session_state.last_image_name = None
            clear_image_related_state()
    else:
        current_image_name = image_file.name

        if current_image_name != st.session_state.last_image_name:
            st.session_state.last_image_name = current_image_name
            st.session_state.prediction_output = None

            if df is not None:
                row = find_metadata_row(df, current_image_name)

                if row is not None:
                    designation_found, description_found = get_text_from_row(row)
                    image_id_found = get_image_id_from_row(row)

                    st.session_state.designation_value = designation_found
                    st.session_state.description_value = description_found
                    st.session_state.matched_image_id = image_id_found
                else:
                    st.session_state.matched_image_id = "not found in csv file"
            else:
                st.session_state.matched_image_id = "not found in csv file"

    st.markdown(f"**Image ID:** `{st.session_state.matched_image_id}`")

    designation = st.text_input(
        "Designation",
        key="designation_value",
        placeholder="Product title"
    )

    description = st.text_area(
        "Description",
        key="description_value",
        placeholder="Product description",
        height=130
    )

    predict_clicked = st.button("Predict", use_container_width=True)

with middle_col:
    st.markdown("### Image")

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to preview it here.")

with right_col:
    st.markdown("### Prediction")

    if predict_clicked:
        if image is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Running prediction..."):
                model, tokenizer, processor, id2label = get_assets()

                st.session_state.prediction_output = predict(
                    image=image,
                    designation=designation,
                    description=description,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    id2label=id2label
                )

    output = st.session_state.prediction_output

    if output is None:
        st.info("Prediction results will appear here.")
    else:
        top1 = output["top1"]
        top1_label = format_prediction_label(top1["prdtypecode"])

        st.success("Prediction completed")
        st.write(f"**Predicted product type:** {top1_label}")
        st.write(f"**Confidence:** {top1['probability'] * 100:.2f}%")
       # TO DO st.write(f"**Text used:** `{output['text_used']}`")

        st.markdown("### Top 3")
        for i, item in enumerate(output["top3"], start=1):
            item_label = format_prediction_label(item["prdtypecode"])
            st.write(
                f"**{i}. {item_label}**  \n"
                f"Confidence: {item['probability'] * 100:.2f}%"
            )