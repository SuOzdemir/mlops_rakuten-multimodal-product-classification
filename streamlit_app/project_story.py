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
        "`.pt` file plus `history.json`/`run_metadata.json`, and support "
        "`MAX_EPOCHS_OVERRIDE`/`TRAIN_ROWS_OVERRIDE`/`VAL_ROWS_OVERRIDE` env vars for a "
        "bounded, real (non-subsampled-model) sanity run of the whole pipeline."
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
    st.success(
        "DVC remote is wired up to the local MinIO instance (`s3://dvc-data`, same MinIO "
        "MLflow's artifacts use) — `dvc push`/`pull` share data/models across machines "
        "without a real cloud account. Non-secret remote config "
        "(`url`, `endpointurl`) lives in `.dvc/config` (committed); credentials go in "
        "`.dvc/config.local` (gitignored, one-time `dvc remote modify --local` setup per machine)."
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
    st.title("Monitoring & Alerts")

    st.markdown("### Metrics collection")
    st.write(
        "The API exposes Prometheus metrics at `/metrics`, scraped every 5s "
        "(`monitoring/prometheus/prometheus.yml`):"
    )
    st.markdown(
        "- Request rate and latency (p50/p95), per endpoint — auto-instrumented via "
        "`prometheus-fastapi-instrumentator`\n"
        "- Error rate (5xx responses)\n"
        "- `prediction_drift_psi` / `prediction_drift_window_size` — see **Prediction drift** below"
    )
    st.write(
        "Grafana auto-provisions a **Rakuten API Overview** dashboard on startup "
        "(`monitoring/grafana/dashboards/api-overview.json`) — no manual dashboard setup needed."
    )

    st.divider()
    st.markdown("### Detection & alert types")
    st.write(
        "An MLOps stack can alert on several distinct failure modes. "
        "Status of each in this project:"
    )
    st.markdown(
        "| # | Type | Status |\n"
        "|---|------|--------|\n"
        "| 1 | Input drift (feature distribution vs training set) | ❌ Not built |\n"
        "| 2 | Prediction / concept drift | ✅ **Built** |\n"
        "| 3 | Model performance decay (accuracy/F1 vs real ground truth) | ❌ Not built |\n"
        "| 4 | System/service health (latency, error rate) | 🟡 Metrics yes, alert rule no |\n"
        "| 5 | Pipeline/orchestration failures (Airflow DAG fails) | ❌ Not built |\n"
        "| 6 | Data quality (missing values, schema drift) | ❌ Not built |\n"
        "| 7 | Infra/resource (disk, container OOM) | 🟡 Healthchecks only, not in Prometheus |"
    )

    with st.expander("2. Prediction / concept drift — details", expanded=True):
        st.markdown(
            "- **Metric**: Population Stability Index (PSI) between the live rolling window of "
            "predicted `prdtypecode`s and the training set's class distribution "
            "(`<0.1` none, `0.1–0.25` moderate, `>0.25` significant).\n"
            "- **Where stored**: in-memory `deque(maxlen=200)` in `src/api/drift.py` — resets on "
            "API restart, no database needed for a single-process demo. Reference distribution: "
            "`src/api/reference_class_distribution.json`.\n"
            "- **When triggered**: scoring needs ≥30 recorded predictions "
            "(`MIN_PREDICTIONS_FOR_SCORE`, below that the window is too noisy). The alert fires "
            "when PSI stays **> 0.25 for 2 consecutive 1-minute evaluations** "
            "(`monitoring/grafana/provisioning/alerting/rules.yaml`, `for: 2m`) so a single noisy "
            "window doesn't page anyone.\n"
            "- **Where observed**:\n"
            "  - Raw metric: Prometheus gauges `prediction_drift_psi` / "
            "`prediction_drift_window_size` (`/metrics`)\n"
            "  - Dashboard: Grafana → *Rakuten API Overview*\n"
            "  - Alert: Grafana → *Alerting* → *Rakuten Alerts* folder, routed to Slack via "
            "`contactpoints.yaml` (`SLACK_WEBHOOK_URL`, set in a local `.env`)"
        )

    with st.expander("4. System/service health — details"):
        st.markdown(
            "- **Where stored**: `http_requests_total` / `http_request_duration_seconds` "
            "(auto-instrumented), scraped into Prometheus's own time-series storage.\n"
            "- **When triggered**: nothing yet — no Grafana alert rule watches latency or "
            "error-rate, only the drift rule above exists.\n"
            "- **Where observed**: Grafana → *Rakuten API Overview* (request-rate, p50/p95 "
            "latency, and error-rate panels)."
        )

    st.info(
        "Rows 1, 3, 5, 6 have no code yet. If built, each would likely follow the same shape as "
        "drift detection: a small in-process tracker module (like `drift.py`), a Prometheus gauge "
        "wired into `src/api/main.py`, and a new rule in `rules.yaml`."
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
