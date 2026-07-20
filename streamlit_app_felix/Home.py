"""
streamlit_app_felix/Home.py

Full demo app with all project pages, grouped by presenter.
Pages from streamlit_app/project_story.py are imported via sys.path.

Run:
    streamlit run streamlit_app_felix/Home.py --server.port 8502
"""

import io
import os
import sys
import time

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ── import shared pages from sibling streamlit_app/ directory ─────────────────
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamlit_app"),
)
from project_story import (  # noqa: E402
    render_data_training,
    render_drift_types,
    render_links,
    render_monitoring,
    render_orchestration,
    render_overview,
    render_tracking,
)

# ── constants ─────────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_PUBLIC_URL = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5001")
AIRFLOW_PUBLIC_URL = os.environ.get("AIRFLOW_PUBLIC_URL", "http://localhost:8080")
GRAFANA_PUBLIC_URL = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3000")
PROMETHEUS_PUBLIC_URL = os.environ.get("PROMETHEUS_PUBLIC_URL", "http://localhost:9090")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rakuten MLOps — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"]          { display: none !important; }
    .block-container                         { padding-top: 2rem !important; }
    section[data-testid="stSidebarNav"]     { display: none !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"]        { display: none !important; }
    /* nav button: remove extra spacing between sidebar buttons */
    section[data-testid="stSidebar"] .stButton { margin-bottom: -6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── architecture diagram (inline HTML — no CDN, works offline) ────────────────
_ARCH_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: transparent;
  padding: 6px 2px 12px;
  font-size: 12px;
  color: #1e293b;
}

.arch-title {
  text-align: center;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #64748b;
  text-transform: uppercase;
  margin-bottom: 10px;
}

/* ── grid ── */
.top-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

.lane {
  border: 1.5px solid #e2e8f0;
  border-radius: 10px;
  padding: 10px;
}

.lane-header {
  font-size: 9px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #94a3b8;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid #e2e8f0;
}

/* ── nodes ── */
.node {
  border-radius: 7px;
  padding: 7px 10px;
  color: #fff;
  font-weight: 600;
  font-size: 11.5px;
  text-align: center;
  cursor: default;
  position: relative;
  transition: transform 0.12s, box-shadow 0.12s;
  user-select: none;
}
.node:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 5px 16px rgba(0,0,0,0.22);
  z-index: 20;
}

/* tooltip */
.node .tip {
  display: none;
  position: absolute;
  bottom: calc(100% + 7px);
  left: 50%;
  transform: translateX(-50%);
  background: #1e293b;
  color: #f1f5f9;
  font-size: 10.5px;
  font-weight: 400;
  line-height: 1.5;
  padding: 7px 10px;
  border-radius: 7px;
  white-space: nowrap;
  z-index: 100;
  box-shadow: 0 4px 14px rgba(0,0,0,0.32);
  pointer-events: none;
  text-align: center;
}
.node:hover .tip { display: block; }

/* ── arrow ── */
.arrow {
  text-align: center;
  color: #94a3b8;
  font-size: 11px;
  line-height: 1;
  padding: 1px 0;
}
.arrow-label {
  font-size: 9px;
  color: #94a3b8;
}

/* ── infra bar ── */
.infra {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  border: 1.5px solid #e2e8f0;
  border-radius: 10px;
  padding: 10px;
}
.infra-header {
  grid-column: 1 / -1;
  font-size: 9px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #94a3b8;
  padding-bottom: 6px;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 2px;
}

@media (prefers-color-scheme: dark) {
  body { color: #f1f5f9; }
  .lane, .infra { border-color: #334155; }
  .lane-header, .infra-header { color: #475569; border-bottom-color: #334155; }
  .arch-title { color: #475569; }
  .arrow, .arrow-label { color: #475569; }
}
</style>
</head>
<body>

<div class="arch-title">Rakuten MLOps — Full System Overview</div>

<div class="top-grid">

  <!-- ── Lane 1: Interface ── -->
  <div class="lane">
    <div class="lane-header">User Interface</div>

    <div class="node" style="background:#475569">
      👤 User
      <div class="tip">Browses Rakuten products<br>Submits predictions</div>
    </div>
    <div class="arrow">↕<br><span class="arrow-label">browser</span></div>
    <div class="node" style="background:#dc2626">
      Streamlit
      <div class="tip">Login · Predict · Retrain UI<br>HTTP client to FastAPI only</div>
    </div>
    <div class="arrow">↕<br><span class="arrow-label">HTTP</span></div>
    <div class="node" style="background:#16a34a">
      FastAPI
      <div class="tip">/login /predict /retrain<br>/retrain/{id} /metrics<br>Loads champion from disk</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">POST /retrain</span></div>
    <div class="node" style="background:#7c3aed">
      Airflow
      <div class="tip">Receives retrain trigger<br>Orchestrates 3-task DAG:<br>prepare → train → promote</div>
    </div>
  </div>

  <!-- ── Lane 2: Training Pipeline ── -->
  <div class="lane">
    <div class="lane-header">Training Pipeline</div>

    <div class="node" style="background:#7c3aed">
      Airflow DAG
      <div class="tip">rakuten_model_training<br>prepare_data → launch_trainer<br>→ trigger_promotion</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">DockerOperator</span></div>
    <div class="node" style="background:#4338ca">
      Trainer Container
      <div class="tip">Ephemeral Docker container<br>dvc repro train_X --force<br>Trains image or text model</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">logs metrics</span></div>
    <div class="node" style="background:#0891b2">
      MLflow
      <div class="tip">Records epoch metrics + params<br>Stores model artifacts<br>rakuten-multimodal-classifier</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">DockerOperator</span></div>
    <div class="node" style="background:#4338ca">
      Promotion Container
      <div class="tip">Builds multimodal bundle<br>Champion gate: Macro-F1<br>dvc push checkpoint</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">if gate passes</span></div>
    <div class="node" style="background:#16a34a">
      FastAPI reloads
      <div class="tip">Atomic replace of<br>data/rakuten_streamlit_predictor/<br>New champion serves live</div>
    </div>
  </div>

  <!-- ── Lane 3: Monitoring & CI ── -->
  <div class="lane">
    <div class="lane-header">Monitoring &amp; CI/CD</div>

    <div class="node" style="background:#16a34a">
      FastAPI /metrics
      <div class="tip">Exposes Prometheus metrics:<br>prediction_drift_psi<br>latency · request rate · errors</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">scrape every 5 s</span></div>
    <div class="node" style="background:#d97706">
      Prometheus
      <div class="tip">Time-series storage<br>Query: prediction_drift_psi<br>Alerts evaluated by Grafana</div>
    </div>
    <div class="arrow">↓<br><span class="arrow-label">evaluate rules</span></div>
    <div class="node" style="background:#ea580c">
      Grafana
      <div class="tip">Rakuten API Overview dashboard<br>PSI alert: &gt; 0.60 for 2 min<br>Routes to email / Slack</div>
    </div>

    <div style="margin-top:10px;border-top:1px dashed #e2e8f0;padding-top:10px;">
      <div class="lane-header" style="border:none;margin:0 0 6px">CI / CD</div>
      <div class="node" style="background:#1f2937">
        GitHub Actions
        <div class="tip">On PR: pytest + image builds<br>On main push: publish to GHCR</div>
      </div>
      <div class="arrow">↓<br><span class="arrow-label">publishes images</span></div>
      <div class="node" style="background:#6d28d9">
        GHCR
        <div class="tip">ghcr.io/suozdemir/*:latest<br>api · streamlit · airflow<br>trainer · promotion</div>
      </div>
    </div>
  </div>
</div>

<!-- ── Infrastructure bar ── -->
<div class="infra">
  <div class="infra-header">Shared Infrastructure</div>

  <div class="node" style="background:#92400e">
    MinIO (S3-compatible)
    <div class="tip">Two buckets:<br><b>dvc-data</b> — checkpoints by content hash<br><b>mlflow-artifacts</b> — runs + model bundles</div>
  </div>
  <div class="node" style="background:#1d4ed8">
    PostgreSQL
    <div class="tip">Three databases:<br>MLflow run metadata<br>Airflow pipeline state<br>API user accounts</div>
  </div>
  <div class="node" style="background:#374151">
    GitHub Repository
    <div class="tip">Source code · dvc.yaml · dvc.lock<br>Docker Compose configs<br>Never stores large files or secrets</div>
  </div>
</div>

</body>
</html>
"""

# ── API helpers ───────────────────────────────────────────────────────────────

def api_login(username: str, password: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_logout(token: str) -> None:
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
        st.error(f"Prediction error {resp.status_code}: {detail}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None


def api_retrain(
    token: str, model: str, epochs: int, batch_size: int,
    learning_rate: float, seed: int, early_stopping_patience: int,
    weight_decay: float, use_amp: bool,
) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/retrain",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "early_stopping_patience": early_stopping_patience,
                "weight_decay": weight_decay,
                "use_amp": use_amp,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Retrain failed: {resp.json().get('detail', resp.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None


def api_retrain_status(token: str, job_id: str) -> dict | None:
    try:
        resp = requests.get(
            f"{API_URL}/retrain/{job_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def fetch_live_psi() -> float | None:
    """Read the current prediction_drift_psi gauge from FastAPI /metrics."""
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=4)
        if resp.status_code != 200:
            return None
        for line in resp.text.splitlines():
            if line.startswith("prediction_drift_psi ") and not line.startswith("#"):
                return float(line.split()[1])
        return None
    except Exception:
        return None


# ── pipeline visualization ────────────────────────────────────────────────────

_PIPELINE_STEPS = [
    ("Streamlit",  "User triggers"),
    ("FastAPI",    "Job accepted"),
    ("Airflow",    "DAG running"),
    ("Trainer",    "Training model"),
    ("MLflow",     "Gate & register"),
    ("Deploy",     "Champion live"),
]

_STATUS_DONE = {"queued": 2, "running": 3, "completed": 6, "failed": 3}


def _step_icon(index: int, n_done: int, status: str | None) -> str:
    if status == "completed":
        return "✅"
    if status == "failed" and index == n_done:
        return "🔴"
    if index < n_done:
        return "✅"
    if index == n_done and status in ("queued", "running"):
        return "🔵"
    return "⚪"


def render_pipeline(status: str | None) -> None:
    n_done = _STATUS_DONE.get(status or "", 0)
    cols = st.columns(len(_PIPELINE_STEPS))
    for i, (name, label) in enumerate(_PIPELINE_STEPS):
        icon = _step_icon(i, n_done, status)
        with cols[i]:
            st.markdown(
                f"<div style='text-align:center;padding:0.5rem 0'>"
                f"<div style='font-size:2rem;line-height:1'>{icon}</div>"
                f"<div style='font-weight:700;font-size:0.88rem;margin-top:4px'>{name}</div>"
                f"<div style='color:#6b7280;font-size:0.75rem'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if status == "completed":
        st.success("Pipeline finished — new champion is live.")
    elif status == "failed":
        st.error("Pipeline failed. Open Airflow for task-level details.")
    elif status == "running":
        st.info("Trainer is running — this takes a few minutes. Watch Airflow for task progress.")
    elif status == "queued":
        st.info("Job queued — Airflow is about to pick it up.")
    else:
        st.caption("Trigger a retraining run above to start the pipeline.")


# ── model defaults ────────────────────────────────────────────────────────────

_MODEL_DEFAULTS = {
    "text": {
        "label":        "Text — CamemBERT (T8)",
        "batch_size":   32,
        "learning_rate": 2e-5,
        "seed":         42,
        "patience":     3,
        "weight_decay": 0.01,
        "use_amp":      True,
    },
    "image": {
        "label":        "Image — ConvNeXt-Base (I12)",
        "batch_size":   16,
        "learning_rate": 5e-5,
        "seed":         42,
        "patience":     6,
        "weight_decay": 0.05,
        "use_amp":      True,
    },
}


# ── page: Retrain Story ───────────────────────────────────────────────────────

def render_retrain_story() -> None:
    st.title("User Story: Retraining the Model")

    with st.container(border=True):
        st.markdown("#### The Situation")

        psi = fetch_live_psi()
        c1, c2, c3 = st.columns(3)

        if psi is not None:
            psi_delta = "above threshold (> 0.60)" if psi > 0.60 else "below threshold (≤ 0.60)"
            psi_color = "inverse" if psi > 0.60 else "normal"
            c1.metric("Live Prediction Drift (PSI)", f"{psi:.3f}", delta=psi_delta, delta_color=psi_color)
        else:
            c1.metric("Prediction Drift (PSI)", "—", delta="Make predictions to measure")

        c2.metric("Drift window", "200 predictions", delta="rolling, in-memory")
        c3.metric("Alert threshold", "PSI > 0.60", delta="2 consecutive minutes → Grafana alert")

        st.write(
            "A new wave of French product listings has shifted the category distribution "
            "of incoming requests. The PSI alert in Grafana fired — an admin investigates "
            "and decides to retrain the text model on fresh data."
        )
        st.caption(
            "Tip: go to **Predict**, classify several products from a single category "
            "repeatedly, then come back here to see the PSI update."
        )

    st.divider()

    st.markdown("#### Trigger Retraining")

    is_admin = st.session_state.get("role") == "admin"
    if not is_admin:
        st.warning("Admin access required to trigger retraining. Log in as **admin**.")
        return

    if "retrain_job_id" not in st.session_state:
        st.session_state.retrain_job_id = None
    if "retrain_model_select" not in st.session_state:
        st.session_state.retrain_model_select = "text"

    model = st.selectbox(
        "Model to retrain",
        list(_MODEL_DEFAULTS),
        format_func=lambda m: _MODEL_DEFAULTS[m]["label"],
        key="retrain_model_select",
    )
    d = _MODEL_DEFAULTS[model]

    with st.form("retrain_form"):
        col_epochs, col_batch, col_lr = st.columns(3)
        epochs = col_epochs.number_input(
            "Epochs", min_value=1, max_value=50, value=1, step=1,
            help="MAX_EPOCHS_OVERRIDE is 1 in .env for speed; increase here to override.",
        )
        batch_size = col_batch.number_input(
            "Batch size", min_value=1, max_value=32, value=d["batch_size"], step=1,
            key=f"bs_{model}",
        )
        learning_rate = col_lr.number_input(
            "Learning rate", value=d["learning_rate"], format="%.2e",
            min_value=1e-7, max_value=1e-2, key=f"lr_{model}",
        )

        with st.expander("Advanced hyperparameters", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            seed = c1.number_input("Seed", value=d["seed"], step=1, key=f"seed_{model}")
            patience = c2.number_input(
                "Early-stopping patience", min_value=1, max_value=50,
                value=d["patience"], key=f"pat_{model}",
            )
            weight_decay = c3.number_input(
                "Weight decay", value=d["weight_decay"], format="%.4f",
                min_value=0.0, max_value=1.0, key=f"wd_{model}",
            )
            use_amp = c4.checkbox("Mixed precision (AMP)", value=d["use_amp"], key=f"amp_{model}")

        submitted = st.form_submit_button(
            "▶  Start Retraining", use_container_width=True, type="primary",
        )

    if submitted:
        with st.spinner("Sending request to FastAPI → Airflow..."):
            job = api_retrain(
                token=st.session_state["token"],
                model=model,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                seed=int(seed),
                early_stopping_patience=int(patience),
                weight_decay=float(weight_decay),
                use_amp=bool(use_amp),
            )
        if job:
            st.session_state.retrain_job_id = job["job_id"]
            st.success(f"Job accepted by FastAPI.  Job ID: `{job['job_id']}`")
            st.rerun()

    st.divider()

    st.markdown("#### Live Pipeline")

    col_refresh, col_auto, _ = st.columns([1, 2, 4])
    refresh_clicked = col_refresh.button("Refresh now")  # noqa: F841
    auto_refresh = col_auto.checkbox("Auto-refresh (5 s)", value=True)

    status = None
    job_info = None

    if st.session_state.retrain_job_id:
        job_info = api_retrain_status(st.session_state["token"], st.session_state.retrain_job_id)
        if job_info:
            status = job_info["status"]

    render_pipeline(status)

    if job_info:
        status_emoji = {"queued": "🟡", "running": "🔵", "completed": "🟢", "failed": "🔴"}.get(status, "⚪")
        st.caption(
            f"{status_emoji} **{status}** · "
            f"Job `{st.session_state.retrain_job_id}` · "
            f"Model: **{job_info.get('model', '—')}** · "
            f"Epochs: {job_info.get('epochs', '—')} · "
            f"Started: {job_info.get('started_at') or '—'}"
        )

    st.divider()

    st.markdown("#### Follow the Pipeline")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Airflow**")
            st.write(
                "Watch the three tasks execute in real time:  \n"
                "`prepare_data` → `launch_trainer` → `trigger_promotion`"
            )
            st.link_button("Open Airflow →", AIRFLOW_PUBLIC_URL, use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("**MLflow**")
            st.write(
                "After training: inspect epoch metrics, training curves, "
                "the Model Registry gate, and the promoted champion version."
            )
            st.link_button("Open MLflow →", MLFLOW_PUBLIC_URL, use_container_width=True)

    if auto_refresh and status in ("queued", "running"):
        time.sleep(5)
        st.rerun()


# ── page: Predict ─────────────────────────────────────────────────────────────

def render_predict() -> None:
    st.title("Product Classification")
    st.caption(
        "Upload a product image and enter text — the API fuses both modalities "
        "via late fusion (ConvNeXt-Base + CamemBERT)."
    )

    for key, default in [
        ("designation_value", ""),
        ("description_value", ""),
        ("last_image_name", None),
        ("prediction_output", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    left_col, mid_col, right_col = st.columns([1.1, 1.1, 0.95], gap="medium")
    image = None

    with left_col:
        st.markdown("### Input")
        image_file = st.file_uploader("Product image", type=["jpg", "jpeg", "png"])

        if image_file is None:
            if st.session_state.last_image_name is not None:
                st.session_state.last_image_name = None
                st.session_state.prediction_output = None
        elif image_file.name != st.session_state.last_image_name:
            st.session_state.last_image_name = image_file.name
            st.session_state.prediction_output = None

        designation = st.text_input(
            "Designation", key="designation_value", placeholder="Product title",
        )
        description = st.text_area(
            "Description", key="description_value",
            placeholder="Product description (optional)", height=120,
        )
        predict_clicked = st.button("Predict", use_container_width=True, type="primary")

    with mid_col:
        st.markdown("### Image")
        if image_file is not None:
            image = Image.open(image_file).convert("RGB")
            st.image(image, use_container_width=True)
        else:
            st.info("Upload an image to preview it here.")

    with right_col:
        st.markdown("### Result")
        if predict_clicked:
            if image is None and not designation.strip() and not description.strip():
                st.warning("Please provide at least an image or some text.")
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
            mode_label = {
                "multimodal": "Multimodal (image + text)",
                "text_only":  "Text only (image missing)",
                "image_only": "Image only (text missing)",
            }.get(output.get("mode", ""), output.get("mode", "—"))

            st.success("Prediction complete")
            st.markdown(f"**Mode:** {mode_label}")
            st.divider()
            st.markdown(f"**Category:** {top1['prdtypecode']} — {top1['Category name']}")
            st.markdown(f"**Confidence:** {top1['Confidence']}")
            st.divider()
            st.markdown("**Top 3**")
            for item in output["top3"]:
                st.markdown(
                    f"**{item['Rank']}.** {item['prdtypecode']} — {item['Category name']}  \n"
                    f"Confidence: {item['Confidence']}"
                )


# ── page: System Health ───────────────────────────────────────────────────────

def render_system_health() -> None:
    st.title("System Health & Monitoring")
    st.caption(
        "Prometheus scrapes the API every 5 seconds. "
        "Grafana visualizes the metrics and evaluates alert rules."
    )

    psi = fetch_live_psi()
    col1, col2, col3 = st.columns(3)
    if psi is not None:
        psi_delta = "⚠ above alert threshold" if psi > 0.60 else "within normal range"
        psi_color = "inverse" if psi > 0.60 else "normal"
        col1.metric("Live PSI (prediction drift)", f"{psi:.4f}", delta=psi_delta, delta_color=psi_color)
    else:
        col1.metric("Live PSI", "—", delta="No predictions recorded yet")
    col2.metric("PSI alert threshold", "0.60", delta="for 2 consecutive minutes")
    col3.metric("Scrape interval", "5 s", delta="Prometheus → FastAPI /metrics")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### Grafana — API Overview Dashboard")
            st.write(
                "Request rate · p50/p95 latency · 5xx error rate · PSI drift gauge.  \n"
                "Login: `admin` / `adminadmin` (from `.env`)"
            )
            st.link_button("Open Grafana →", GRAFANA_PUBLIC_URL, use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("#### Prometheus — Raw Metrics")
            st.write(
                "Query `prediction_drift_psi` for live drift.  \n"
                "Query `http_request_duration_seconds` for latency percentiles."
            )
            st.link_button("Open Prometheus →", PROMETHEUS_PUBLIC_URL, use_container_width=True)

    st.divider()
    st.markdown("### What is monitored")
    st.markdown(
        "| Signal | Tool chain | Status |\n"
        "|---|---|---|\n"
        "| Request rate & latency (p50/p95) | FastAPI → Prometheus → Grafana | Metrics live, no alert rule yet |\n"
        "| 5xx error rate | FastAPI → Prometheus → Grafana | Metrics live, no alert rule yet |\n"
        "| **Prediction drift (PSI)** | `drift.py` → Prometheus → Grafana | **Active alert: PSI > 0.60 for 2 min** |\n"
        "| Input / concept drift | Not built | Would require delayed ground-truth labels |\n"
    )

    with st.expander("How to trigger the drift alert in the demo"):
        st.write(
            "1. Go to **Predict** and classify the same product image repeatedly (5+ times).  \n"
            "2. Return to this page — the PSI metric above will update.  \n"
            "3. Open Grafana and navigate to **Rakuten API Overview** to see the drift gauge rise.  \n"
            "4. If PSI stays above 0.60 for two minutes, the Grafana alert fires.  \n"
            "5. Then go to **Retrain Story** and show the full retraining loop."
        )

    st.info(
        "PSI measures how far the live predicted-class distribution has shifted from the "
        "training baseline. A high PSI is a signal to investigate — not automatic proof "
        "that retraining is needed. Concept drift can only be confirmed with ground-truth labels."
    )


# ── page: Architecture ────────────────────────────────────────────────────────

def render_architecture() -> None:
    st.title("System Architecture")
    st.caption(
        "Hover over any component to see its role in the system. "
        "Replace the diagram below with Sumeyra's image by placing "
        "`architecture.png` in `streamlit_app_felix/`."
    )

    arch_image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "architecture.png"
    )
    if os.path.exists(arch_image_path):
        st.image(arch_image_path, caption="System Architecture", use_container_width=True)
        with st.expander("Interactive component map", expanded=False):
            components.html(_ARCH_HTML, height=660, scrolling=False)
    else:
        components.html(_ARCH_HTML, height=660, scrolling=False)

    st.divider()
    st.markdown("### Key design decisions")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Late fusion**")
            st.write(
                "Image logits (ConvNeXt-Base) and text logits (CamemBERT) are combined "
                "after softmax — no shared backbone. Each modality degrades gracefully "
                "when the other is absent."
            )
    with c2:
        with st.container(border=True):
            st.markdown("**Ephemeral containers**")
            st.write(
                "Training and promotion run in short-lived Docker containers launched "
                "by Airflow's DockerOperator. The main stack (API, MLflow, …) stays "
                "up while training happens in isolation."
            )
    with c3:
        with st.container(border=True):
            st.markdown("**Champion gate**")
            st.write(
                "A new model only replaces the serving champion when its component "
                "Macro-F1 exceeds the current champion's — preventing accidental "
                "regressions from low-quality retrain runs."
            )


# ── login screen ──────────────────────────────────────────────────────────────

def render_login() -> None:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:2rem;">
                <h1 style="font-size:2rem;font-weight:800;">Rakuten MLOps</h1>
                <p style="color:#6b7280;">Sign in to access the demo</p>
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
                result = api_login(username, password)
            if result:
                st.session_state.update({
                    "logged_in": True,
                    "username":  username,
                    "token":     result["access_token"],
                    "role":      result.get("role", "viewer"),
                    "page":      "Retrain Story",
                })
                st.rerun()
            else:
                st.error("Invalid username or password.")


# ── navigation helpers ────────────────────────────────────────────────────────

_NAV_SECTIONS = {
    "PROJECT CONTEXT": ["Overview", "Data & Training", "Drift Types"],
    "LIVE DEMO": ["Retrain Story", "Predict", "System Health"],
    "ARCHITECTURE": ["Architecture"],
    "TECHNICAL DEEP DIVE": [
        "Tracking & Versioning",
        "Orchestration & Deployment",
        "Monitoring",
        "Plugins & Links",
    ],
}

_PAGE_RENDER = {
    "Overview":                render_overview,
    "Data & Training":         render_data_training,
    "Drift Types":             render_drift_types,
    "Retrain Story":           render_retrain_story,
    "Predict":                 render_predict,
    "System Health":           render_system_health,
    "Architecture":            render_architecture,
    "Tracking & Versioning":   render_tracking,
    "Orchestration & Deployment": render_orchestration,
    "Monitoring":              render_monitoring,
    "Plugins & Links":         render_links,
}


def _sidebar_nav(current: str) -> str:
    """Render section-labelled button nav; return the active page name."""
    for section, pages in _NAV_SECTIONS.items():
        st.sidebar.caption(section)
        for page in pages:
            is_active = current == page
            if st.sidebar.button(
                page,
                key=f"nav__{page}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state.page = page
                    st.rerun()
        st.sidebar.write("")  # small gap between sections
    return current


# ── main router ───────────────────────────────────────────────────────────────

def render_app() -> None:
    st.sidebar.title("Rakuten MLOps")
    role = st.session_state.get("role", "viewer")
    st.sidebar.markdown(f"**{st.session_state.get('username', '')}** · {role}")
    st.sidebar.divider()

    if "page" not in st.session_state:
        st.session_state.page = "Retrain Story"

    current = _sidebar_nav(st.session_state.page)

    st.sidebar.divider()
    if st.sidebar.button("Log out", use_container_width=True):
        api_logout(st.session_state.get("token", ""))
        st.session_state.update({
            "logged_in": False, "username": "", "token": "", "role": "", "page": "Retrain Story",
        })
        st.rerun()

    render_fn = _PAGE_RENDER.get(current)
    if render_fn:
        render_fn()
    else:
        st.error(f"Unknown page: {current}")


# ── entry point ───────────────────────────────────────────────────────────────

if not st.session_state.get("logged_in", False):
    render_login()
else:
    render_app()