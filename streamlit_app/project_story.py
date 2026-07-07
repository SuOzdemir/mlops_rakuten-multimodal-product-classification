import streamlit as st

# -------------------------------------------------------------------
# 1. Overview
# -------------------------------------------------------------------
def render_overview():
    st.title("Rakuten Product Classifier — Overview")
    st.caption("A walkthrough of what this project does and how it's built.")

    st.markdown("### Task")
    st.write(
        "Classify Rakuten France product listings into 27 categories using both "
        "the product image and its text (designation + description)."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Train samples", "84,916")
    col2.metric("Test samples", "13,812")
    col3.metric("Categories", "27")
    st.caption("~35% of listings are missing a description — text-only or image-only fallback is required.")

    st.markdown("### Models")
    st.markdown(
        "- **Image** — ConvNeXt-Base (`Model_I12_ConvNeXt_Base_ModerateAug_Full`)\n"
        "- **Text** — CamemBERT (`Model_T8_CamemBERT_FullFineTune_L128`)\n"
        "- **Fusion** — weighted late fusion of both models' logits "
        "(`streamlit_app/services/raw_late_fusion_predictor.py`), with automatic "
        "fallback to text-only or image-only when one modality is missing."
    )

    st.markdown("### Tracked metrics")
    st.write("`val_accuracy` and `val_f1` (macro F1) are logged per epoch to MLflow during training.")


# -------------------------------------------------------------------
# 2. Data & Training
# -------------------------------------------------------------------
def render_data_training():
    st.title("Data & Training")

    st.markdown("### Getting the data")
    st.code("./scripts/setup_data.sh", language="bash")
    st.write("Downloads the raw Kaggle dataset into `data/raw/` (not committed to git).")

    st.markdown("### DVC pipeline")
    st.write(
        "The `prepare_splits` DVC stage (`dvc.yaml`) turns the raw CSVs into "
        "reproducible train/val split files:"
    )
    st.markdown(
        "- `outputs/image_modeling/train_split.csv`\n"
        "- `outputs/image_modeling/val_split.csv`\n"
        "- `outputs/image_modeling/label2id.json`"
    )
    st.code("uv run dvc repro prepare_splits", language="bash")

    st.markdown("### Training")
    st.write(
        "Each model has its own `train.py` + `config.py` under `models/`, run as a "
        "separate DVC stage (`train_image`, `train_text`). Both write a best-checkpoint "
        "`.pt` file plus `history.json`/`run_metadata.json`, and support a `SMOKE_TEST=1` "
        "mode for a fast 1-epoch sanity run of the whole pipeline."
    )


# -------------------------------------------------------------------
# 3. Tracking & Versioning
# -------------------------------------------------------------------
def render_tracking():
    st.title("Tracking & Versioning")

    st.markdown("### MLflow")
    st.write(
        "Every training run logs metrics, params, and artifacts to MLflow "
        "(experiment tracking + Model Registry), backed by a shared Postgres "
        "instance also used by Airflow's metadata store — a single database "
        "server instead of two separate SQLite files."
    )

    st.markdown("### DVC")
    st.write(
        "`dvc.yaml`/`dvc.lock` version the raw data, split files, and trained model "
        "weights by content hash, independent of git."
    )
    st.warning(
        "No DVC remote is configured yet — `dvc repro`/`dvc status` work locally via the "
        "DVC cache, but `dvc push`/`pull` (sharing data/models across machines or CI) "
        "isn't wired up. This is a deliberate, known gap, not an oversight."
    )


# -------------------------------------------------------------------
# 4. Orchestration & Deployment
# -------------------------------------------------------------------
def render_orchestration():
    st.title("Orchestration & Deployment")

    st.markdown("### Airflow DAGs")
    st.markdown(
        "- **`rakuten_model_training`** — DVC prep + train, triggered by the API's `/retrain` endpoint\n"
        "- **`rakuten_model_promotion`** — copies the best trained weights into the serving directory read by the API"
    )

    st.markdown("### Serving")
    st.write(
        "FastAPI (`src/api/main.py`) exposes `/login`, `/predict`, and (admin-only) `/retrain`. "
        "Streamlit is a thin client that calls this API over HTTP — it never loads the models directly."
    )

    st.markdown("### Containers & CI")
    st.write(
        "`docker-compose.yaml` runs the serving stack (MLflow, API, Streamlit, Prometheus, Grafana); "
        "the Airflow stack is a separate compose project with its own network. "
        "`.github/workflows/ci.yml` runs the test suite and builds all three application images on every push/PR."
    )


# -------------------------------------------------------------------
# 5. Monitoring
# -------------------------------------------------------------------
def render_monitoring():
    st.title("Monitoring")

    st.markdown("### What's tracked")
    st.write("The API exposes Prometheus metrics at `/metrics`:")
    st.markdown(
        "- Request rate and latency (p50/p95), per endpoint\n"
        "- Error rate (5xx responses)\n"
        "- Prediction confidence distribution (top-1 confidence of every `/predict` call)"
    )
    st.write(
        "Grafana auto-provisions a **Rakuten API Overview** dashboard on startup — "
        "no manual dashboard setup needed."
    )

    st.markdown("### Not built yet")
    st.write(
        "Drift detection (comparing live prediction/input distributions against the training "
        "set to catch model decay, e.g. with Evidently) is still an open gap."
    )


# -------------------------------------------------------------------
# 6. Plugins & Links
# -------------------------------------------------------------------
_LINKS = [
    ("MLflow", "http://localhost:5001", "Experiment tracking + Model Registry"),
    ("Prometheus", "http://localhost:9090", "Raw metrics + target health"),
    ("Grafana", "http://localhost:3000", "API Overview dashboard (admin/admin by default)"),
    ("Airflow", "http://localhost:8080", "Training + promotion DAGs"),
    ("API docs (Swagger)", "http://localhost:8000/docs", "Interactive FastAPI docs"),
]


def render_links():
    st.title("Plugins & Links")
    st.caption("Opens in a new tab — requires the relevant service to be running (`make up` / `make up-all`).")

    for name, url, description in _LINKS:
        col_label, col_button = st.columns([3, 1])
        with col_label:
            st.markdown(f"**{name}**")
            st.caption(description)
        with col_button:
            st.link_button(f"Open {name}", url, use_container_width=True)
        st.divider()
