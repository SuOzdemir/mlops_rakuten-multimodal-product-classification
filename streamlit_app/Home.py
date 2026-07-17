import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from project_story import (
    render_data_training,
    render_drift_types,
    render_links,
    render_monitoring,
    render_orchestration,
    render_overview,
    render_tracking,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_PUBLIC_URL = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5001")
AIRFLOW_PUBLIC_URL = os.environ.get("AIRFLOW_PUBLIC_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Rakuten Product Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
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

def api_login(username: str, password: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
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


def api_retrain(
    token: str,
    model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    early_stopping_patience: int,
    weight_decay: float,
    use_amp: bool,
    label_smoothing: float | None = None,
    dropout: float | None = None,
) -> dict | None:
    try:
        payload = {
            "model": model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
            "weight_decay": weight_decay,
            "use_amp": use_amp,
        }
        if label_smoothing is not None:
            payload["label_smoothing"] = label_smoothing
        if dropout is not None:
            payload["dropout"] = dropout
        resp = requests.post(
            f"{API_URL}/retrain",
            headers={"Authorization": f"Bearer {token}"},
            data=payload,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Retrain failed ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is the API server running?")
        return None


def api_retrain_status(token: str, job_id: str) -> dict | None:
    try:
        resp = requests.get(
            f"{API_URL}/retrain/{job_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
    except requests.exceptions.ConnectionError:
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
                login_result = api_login(username, password)
            if login_result:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["token"] = login_result["access_token"]
                st.session_state["role"] = login_result.get("role", "viewer")
                st.rerun()
            else:
                st.error("Invalid username or password.")


# -------------------------------------------------------------------
# Prediction screen
# -------------------------------------------------------------------

def render_prediction():
    st.sidebar.title("Rakuten Classifier")
    role = st.session_state.get("role", "viewer")
    st.sidebar.markdown(f"Logged in as **{st.session_state.get('username', '')}** ({role})")

    is_admin = role == "admin"
    story_pages = [
        "Overview",
        "Data & Training",
        "Tracking & Versioning",
        "Orchestration & Deployment",
        "Monitoring",
        "Drift Types",
    ]
    pages = (
        story_pages
        + ["Predict"]
        + (["Retrain (Admin)"] if is_admin else [])
        + ["Plugins & Links"]
    )
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    if st.sidebar.button("Log out"):
        api_logout(st.session_state.get("token", ""))
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["token"] = ""
        st.session_state["role"] = ""
        st.rerun()

    story_renderers = {
        "Overview": render_overview,
        "Data & Training": render_data_training,
        "Tracking & Versioning": render_tracking,
        "Orchestration & Deployment": render_orchestration,
        "Monitoring": render_monitoring,
        "Drift Types": render_drift_types,
        "Plugins & Links": render_links,
    }
    if page in story_renderers:
        story_renderers[page]()
        return

    if page == "Retrain (Admin)":
        render_retrain()
        return

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
# Retrain screen (admin only)
# -------------------------------------------------------------------

STATUS_COLOR = {
    "queued": "🟡",
    "running": "🔵",
    "completed": "🟢",
    "failed": "🔴",
}


def render_retrain():
    st.title("Model Retraining")
    st.caption("Triggers the Airflow DAG `rakuten_model_training` (DVC prep + train) via its REST API.")
    st.info(
        "After the job completes, open MLflow and select the latest image or text run. "
        "Use the Metrics section for interactive epoch charts, or open "
        "Artifacts → training_curves.png for the generated summary image."
    )
    st.link_button("Open MLflow results", MLFLOW_PUBLIC_URL)
    st.link_button("View all retrain jobs in Airflow", AIRFLOW_PUBLIC_URL)

    if "retrain_job_id" not in st.session_state:
        st.session_state.retrain_job_id = None

    model = st.selectbox(
        "Model",
        ["image", "text"],
        format_func=lambda m: {
            "image": "Image — ConvNeXt-Base (I12)",
            "text": "Text — CamemBERT (T8)",
        }[m],
        key="retrain_model",
    )
    defaults = {
        "image": {
            "architecture": "ConvNeXt-Base",
            "input_shape": 224,
            "input_label": "Image size",
            "feature_dim": 1024,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "seed": 42,
            "early_stopping_patience": 6,
            "weight_decay": 0.05,
            "use_amp": True,
            "label_smoothing": 0.1,
            "dropout": 0.5,
        },
        "text": {
            "architecture": "CamemBERT-base",
            "input_shape": 128,
            "input_label": "Max token length",
            "feature_dim": 768,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "seed": 42,
            "early_stopping_patience": 3,
            "weight_decay": 0.01,
            "use_amp": True,
        },
    }[model]

    with st.form("retrain_form"):
        fixed_arch, fixed_input, fixed_feature = st.columns(3)
        with fixed_arch:
            st.text_input(
                "Architecture",
                value=defaults["architecture"],
                disabled=True,
                help="Fixed because the checkpoint and production serving code depend on this architecture.",
            )
        with fixed_input:
            st.number_input(
                defaults["input_label"],
                value=defaults["input_shape"],
                disabled=True,
                help="Fixed to match preprocessing and production serving.",
            )
        with fixed_feature:
            st.number_input(
                "Feature dimension",
                value=defaults["feature_dim"],
                disabled=True,
                help="Fixed by the selected pretrained architecture.",
            )

        st.markdown("#### Core training parameters")
        epoch_col, batch_col, lr_col = st.columns(3)
        with epoch_col:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                help="Maximum training epochs. Early stopping may finish the run sooner.",
            )
        with batch_col:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=32,
                value=defaults["batch_size"],
                step=1,
                key=f"retrain_batch_size_{model}",
                help="Larger batches use more memory. ConvNeXt-Base defaults to 16; CamemBERT defaults to 32.",
            )
        with lr_col:
            learning_rate = st.number_input(
                "Head learning rate" if model == "image" else "Learning rate",
                min_value=1e-7,
                max_value=1e-2,
                value=defaults["learning_rate"],
                step=1e-6,
                format="%.7f",
                key=f"retrain_learning_rate_{model}",
                help=(
                    "For CamemBERT this is the optimizer LR. For ConvNeXt this is "
                    "the classifier-head LR; the backbone uses LR / 10."
                ),
            )
        if model == "image":
            st.caption(
                "Derived parameter: **Backbone learning rate = Head learning rate / 10** "
                f"(default: `{defaults['learning_rate'] / 10:g}`)."
            )

        with st.expander("Advanced hyperparameters", expanded=True):
            seed_col, patience_col = st.columns(2)
            with seed_col:
                seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=defaults["seed"],
                    step=1,
                    key=f"retrain_seed_{model}",
                )
            with patience_col:
                early_stopping_patience = st.number_input(
                    "Early-stopping patience",
                    min_value=1,
                    max_value=50,
                    value=defaults["early_stopping_patience"],
                    step=1,
                    key=f"retrain_patience_{model}",
                )

            decay_col, amp_col = st.columns(2)
            with decay_col:
                weight_decay = st.number_input(
                    "Weight decay",
                    min_value=0.0,
                    max_value=1.0,
                    value=defaults["weight_decay"],
                    step=0.001,
                    format="%.4f",
                    key=f"retrain_weight_decay_{model}",
                )
            with amp_col:
                use_amp = st.checkbox(
                    "Use mixed precision (AMP)",
                    value=defaults["use_amp"],
                    key=f"retrain_use_amp_{model}",
                    help="Used only when CUDA is available; CPU training remains full precision.",
                )

            label_smoothing = None
            dropout = None
            if model == "image":
                smoothing_col, dropout_col = st.columns(2)
                with smoothing_col:
                    label_smoothing = st.number_input(
                        "Label smoothing",
                        min_value=0.0,
                        max_value=0.5,
                        value=defaults["label_smoothing"],
                        step=0.01,
                        format="%.2f",
                        key="retrain_label_smoothing_image",
                    )
                with dropout_col:
                    dropout = st.number_input(
                        "Classifier dropout",
                        min_value=0.0,
                        max_value=0.9,
                        value=defaults["dropout"],
                        step=0.05,
                        format="%.2f",
                        key="retrain_dropout_image",
                    )
        submitted = st.form_submit_button("Start Retrain", use_container_width=True)

    if submitted:
        with st.spinner("Triggering Airflow DAG..."):
            job = api_retrain(
                st.session_state["token"],
                model=model,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                seed=int(seed),
                early_stopping_patience=int(early_stopping_patience),
                weight_decay=float(weight_decay),
                use_amp=bool(use_amp),
                label_smoothing=(
                    float(label_smoothing)
                    if label_smoothing is not None
                    else None
                ),
                dropout=float(dropout) if dropout is not None else None,
            )
        if job:
            st.session_state.retrain_job_id = job["job_id"]
            st.success(f"Job started: `{job['job_id']}`")

    st.divider()

    col_refresh, col_auto = st.columns([1, 3])
    with col_refresh:
        refresh_clicked = st.button("Refresh status")
    with col_auto:
        auto_refresh = st.checkbox("Auto-refresh every 5s", value=False)

    if st.session_state.retrain_job_id:
        st.markdown(f"### Current job: `{st.session_state.retrain_job_id}`")
        job = api_retrain_status(st.session_state["token"], st.session_state.retrain_job_id)
        if job:
            emoji = STATUS_COLOR.get(job["status"], "⚪")
            st.markdown(
                f"**Status:** {emoji} {job['status']}  |  "
                f"**Model:** {job['model']}  |  "
                f"**Epochs:** {job.get('epochs', '-')}  |  "
                f"**Batch:** {job.get('batch_size', '-')}  |  "
                f"**LR:** {job.get('learning_rate', '-')}  |  "
                f"**Started:** {job.get('started_at') or '-'}"
            )
        else:
            st.warning("Could not fetch job status (Airflow unreachable or job not found).")

    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------

if not st.session_state.get("logged_in", False):
    render_login()
else:
    render_prediction()
