import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Rakuten Product Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 2rem !important; }
    section[data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------

def api_login(username: str, password: str) -> str | None:
    try:
        resp = requests.post(
            f"{API_URL}/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
        return None
    except Exception:
        return None


def api_logout(token: str):
    try:
        requests.post(
            f"{API_URL}/logout",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
    except Exception:
        pass


def api_predict(token: str, designation: str, description: str, image: Image.Image | None) -> dict | None:
    headers = {"Authorization": f"Bearer {token}"}
    data = {"designation": designation, "description": description}
    files = {}

    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        files["image"] = ("image.jpg", buf, "image/jpeg")

    try:
        resp = requests.post(
            f"{API_URL}/predict",
            headers=headers,
            data=data,
            files=files if files else None,
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text or f"HTTP {resp.status_code}"
        st.error(f"API error {resp.status_code}: {detail}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is the API server running?")
        return None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# -------------------------------------------------------------------
# CSV helpers
# -------------------------------------------------------------------

def _norm_fname(value) -> str:
    if pd.isna(value):
        return ""
    return Path(str(value)).name.strip().lower()


def _norm_col(value) -> str:
    return str(value).strip().lower()


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    t = str(value).strip()
    return "" if t.lower() in {"nan", "none", "null"} else t


def _find_col(row, candidates):
    norm = {_norm_col(c): c for c in row.index}
    for cand in candidates:
        col = norm.get(_norm_col(cand))
        if col is not None:
            return col
    return None


def find_metadata_row(df, image_filename):
    norm = _norm_fname(image_filename)
    norm_cols = {_norm_col(c): c for c in df.columns}

    for cand in ["image_name", "filename", "file_name", "image_filename", "image_path", "path", "image"]:
        real = norm_cols.get(_norm_col(cand))
        if real is not None:
            matches = df[df[real].apply(_norm_fname) == norm]
            if not matches.empty:
                return matches.iloc[0]

    id_col = norm_cols.get("imageid") or norm_cols.get("image_id")
    pid_col = norm_cols.get("productid") or norm_cols.get("product_id")
    if id_col and pid_col:
        expected = df.apply(lambda r: f"image_{r[id_col]}_product_{r[pid_col]}.jpg".lower(), axis=1)
        matches = df[expected == norm]
        if not matches.empty:
            return matches.iloc[0]

    for col in df.columns:
        if df[col].dtype == "object":
            matches = df[df[col].apply(_norm_fname) == norm]
            if not matches.empty:
                return matches.iloc[0]
    return None


def get_text_from_row(row):
    if row is None:
        return "", ""
    desig_col = _find_col(row, ["designation", "title", "product_title", "product_name", "name"])
    desc_col = _find_col(row, ["description", "description_dedup", "description_clean", "clean_description", "product_description", "desc"])
    return (
        _safe_text(row[desig_col]) if desig_col else "",
        _safe_text(row[desc_col]) if desc_col else "",
    )


def get_image_id_from_row(row):
    if row is None:
        return ""
    col = _find_col(row, ["imageid", "image_id", "image id"])
    if col:
        val = _safe_text(row[col])
        if val:
            return val
    return ""


def clear_image_state():
    st.session_state.designation_value = ""
    st.session_state.description_value = ""
    st.session_state.matched_image_id = ""
    st.session_state.prediction_output = None


def fmt_label(item) -> str:
    return f"{item['prdtypecode']} - {item['Category name']}"


# -------------------------------------------------------------------
# Login screen
# -------------------------------------------------------------------

def render_login():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:2rem;">
                <h1 style="font-size:2rem; font-weight:800;">Rakuten Product Classifier</h1>
                <p style="color:#6b7280;">Sign in to access the prediction tool</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            with st.spinner("Signing in..."):
                token = api_login(username, password)
            if token:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["token"] = token
                st.rerun()
            else:
                st.error("Invalid username or password.")


# -------------------------------------------------------------------
# Prediction screen
# -------------------------------------------------------------------

def render_prediction():
    st.sidebar.title("Rakuten Classifier")
    st.sidebar.markdown(f"Logged in as **{st.session_state.get('username', '')}**")
    if st.sidebar.button("Log out"):
        api_logout(st.session_state.get("token", ""))
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["token"] = ""
        st.rerun()

    st.title("Product Classification")
    st.caption("Upload a product image and enter text to get a multimodal prediction.")

    for key, default in [
        ("designation_value", ""),
        ("description_value", ""),
        ("last_image_name", None),
        ("last_csv_name", None),
        ("matched_image_id", ""),
        ("csv_status", "No CSV loaded."),
        ("prediction_output", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    left_col, middle_col, right_col = st.columns([1.25, 1.05, 0.95], gap="medium")
    df = None
    image = None

    with left_col:
        st.markdown("### Input")

        csv_file = st.file_uploader("Metadata CSV (optional)", type=["csv"])
        if csv_file is None:
            if st.session_state.last_csv_name is not None:
                st.session_state.csv_status = "CSV deleted."
                st.session_state.last_csv_name = None
            else:
                st.session_state.csv_status = "No CSV loaded."
        else:
            try:
                df = pd.read_csv(csv_file)
                df.columns = [str(c).strip() for c in df.columns]
                st.session_state.csv_status = f"CSV loaded: {len(df)} rows"
                st.session_state.last_csv_name = csv_file.name
            except Exception:
                df = None
                st.session_state.csv_status = "CSV could not be read."
                st.session_state.last_csv_name = csv_file.name

        st.caption(st.session_state.csv_status)

        image_file = st.file_uploader("Product image", type=["jpg", "jpeg", "png"])
        if image_file is None:
            if st.session_state.last_image_name is not None:
                st.session_state.last_image_name = None
                clear_image_state()
        else:
            if image_file.name != st.session_state.last_image_name:
                st.session_state.last_image_name = image_file.name
                st.session_state.prediction_output = None
                if df is not None:
                    row = find_metadata_row(df, image_file.name)
                    if row is not None:
                        d, desc = get_text_from_row(row)
                        st.session_state.designation_value = d
                        st.session_state.description_value = desc
                        st.session_state.matched_image_id = get_image_id_from_row(row)
                    else:
                        st.session_state.matched_image_id = "not found in csv"
                else:
                    st.session_state.matched_image_id = ""

        if st.session_state.matched_image_id:
            st.markdown(f"**Image ID:** `{st.session_state.matched_image_id}`")

        designation = st.text_input("Designation", key="designation_value", placeholder="Product title")
        description = st.text_area("Description", key="description_value", placeholder="Product description", height=130)
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
            if image is None and not designation.strip() and not description.strip():
                st.warning("Please upload an image or enter a title/description.")
            else:
                with st.spinner("Running prediction..."):
                    st.session_state.prediction_output = api_predict(
                        token=st.session_state["token"],
                        designation=designation,
                        description=description,
                        image=image,
                    )

        output = st.session_state.prediction_output
        if output is None:
            st.info("Prediction results will appear here.")
        else:
            top1 = output["top3"][0]
            st.success("Prediction completed")
            st.markdown(f"**Mode:** {output['mode']}")
            st.write(f"**Predicted product type:** {fmt_label(top1)}")
            st.write(f"**Confidence:** {top1['Confidence']}")
            st.markdown("### Top 3")
            for item in output["top3"]:
                st.write(f"**{item['Rank']}. {fmt_label(item)}**  \nConfidence: {item['Confidence']}")


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------

if not st.session_state.get("logged_in", False):
    render_login()
else:
    render_prediction()
