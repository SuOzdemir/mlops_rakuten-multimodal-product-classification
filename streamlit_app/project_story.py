import os
from html import escape

import streamlit as st


_TOOL_ICONS = {
    "MLflow": ("mlflow",),
    "Git & DVC": ("git", "dvc"),
    "MinIO": ("minio",),
    "PostgreSQL": ("postgresql",),
    "Apache Airflow": ("apacheairflow",),
    "FastAPI": ("fastapi",),
    "Streamlit": ("streamlit",),
    "Docker & Docker Compose": ("docker",),
    "GitHub Actions": ("githubactions",),
    "Prometheus": ("prometheus",),
    "Grafana": ("grafana",),
    "PSI (Population Stability Index)": ("chartdotjs",),
}


def _tool_explanation(name: str, purpose: str, project_usage: str) -> None:
    """Render a compact tool summary with small official-style icons."""
    def inline_code(text: str) -> str:
        parts = text.split("`")
        return "".join(
            f"<code>{escape(part)}</code>" if index % 2 else escape(part)
            for index, part in enumerate(parts)
        )

    icons = "".join(
        f'<img src="https://cdn.simpleicons.org/{escape(slug)}" '
        f'alt="" width="18" height="18" style="vertical-align:middle;margin-right:5px;">'
        for slug in _TOOL_ICONS.get(name, ("tools",))
    )
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
            f'{icons}<strong>{escape(name)}</strong></div>'
            f'<ul style="margin-top:0;margin-bottom:0;padding-left:22px;">'
            f'<li>{inline_code(purpose)}</li>'
            f'<li><strong>In this project:</strong> {inline_code(project_usage)}</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024


@st.cache_data(ttl=30, show_spinner=False)
def _list_minio_objects() -> tuple[list[dict], str | None]:
    """Return model-related objects from the local MinIO buckets."""
    try:
        from minio import Minio

        client = Minio(
            os.environ.get("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", ""),
            secret_key=os.environ.get("MINIO_SECRET_KEY", ""),
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        )
        rows = []
        model_markers = (
            ".pt", ".pth", ".safetensors", "mlmodel", "model.pkl",
            "registered_model", "checkpoint", "best_model",
        )
        for bucket in ("mlflow-artifacts", "dvc-data"):
            if not client.bucket_exists(bucket):
                continue
            for item in client.list_objects(bucket, recursive=True):
                object_name = item.object_name or ""
                if not any(marker in object_name.lower() for marker in model_markers):
                    continue
                rows.append({
                    "Bucket": bucket,
                    "Object": object_name,
                    "Size": _format_bytes(item.size or 0),
                    "Last modified": (
                        item.last_modified.strftime("%Y-%m-%d %H:%M UTC")
                        if item.last_modified else "—"
                    ),
                })
        return rows, None
    except Exception as exc:
        return [], str(exc)


# -------------------------------------------------------------------
# 1. Overview
# -------------------------------------------------------------------
def render_overview():
    st.title("Rakuten Product Classifier — Overview")
    st.caption("An overview of the problem, models, and end-to-end MLOps workflow.")

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

    st.markdown("### MLOps Project Goal")
    st.write(
        "This project goes beyond training a classification model. Its goal is to build a "
        "reproducible and observable machine-learning system that covers the complete model "
        "lifecycle: data preparation, versioned training, experiment tracking, model registration, "
        "deployment, prediction, retraining, and monitoring."
    )
    st.write(
        "A user can request a prediction through Streamlit, while an administrator can trigger "
        "retraining from the same interface. FastAPI provides the service layer, Airflow orchestrates "
        "the retraining and promotion workflow, DVC versions data and checkpoints, and MLflow records "
        "experiments and the deployable model version. Prometheus and Grafana make the running system "
        "observable after deployment."
    )


# -------------------------------------------------------------------
# 2. Data & Training
# -------------------------------------------------------------------
def render_data_training():
    st.title("Data & Training")
    st.caption("How raw data becomes reproducible, versioned model artifacts.")

    st.markdown("### 1. Data acquisition")
    st.write(
        "The project starts by downloading the Rakuten product CSV files and images "
        "from Kaggle. Raw data is stored under `data/raw/` and is excluded from Git "
        "because it is large and should not be versioned as source code."
    )
    st.code("./scripts/setup_data.sh", language="bash")

    st.markdown("### 2. Reproducible preprocessing with DVC")
    st.write(
        "Preprocessing is defined as the `prepare_splits` stage in `dvc.yaml`. "
        "DVC connects the raw dataset, preprocessing script, and generated outputs "
        "as one reproducible pipeline stage. If neither the data nor the script changes, "
        "the stage is not run again unnecessarily."
    )
    st.code("uv run dvc repro prepare_splits", language="bash")
    st.write("The stage produces versioned inputs shared by both models:")
    st.markdown(
        "- `outputs/image_modeling/train_split.csv`\n"
        "- `outputs/image_modeling/val_split.csv`\n"
        "- `outputs/image_modeling/label2id.json`"
    )

    st.markdown("### 3. Versioned model training")
    st.write(
        "Image and text training are separate DVC stages: `train_image` and `train_text`. "
        "Both consume the same versioned split and label mapping, which keeps their "
        "validation results comparable. Each stage produces a best checkpoint plus "
        "`history.json` and `run_metadata.json`. DVC records the resulting artifact hashes "
        "in `dvc.lock`, linking each model output to the data and code that produced it."
    )

    st.markdown("### Why this is MLOps")
    st.info(
        "The important outcome is not only a trained model. The complete path from raw data "
        "to a checkpoint is repeatable, dependency-aware, and versioned. The same DVC stages "
        "can be executed locally, from Airflow, or inside the trainer container without "
        "manually repeating preprocessing steps."
    )


# -------------------------------------------------------------------
# 3. Tracking & Versioning
# -------------------------------------------------------------------
def render_tracking():
    st.title("Tracking & Versioning")

    _tool_explanation(
        "MLflow",
        "Tracks experiment parameters, metrics, and artifacts and manages deployable versions through the Model Registry.",
        "Each image or text retraining run logs epoch metrics and its best checkpoint. The promotion DAG registers the complete multimodal bundle as `rakuten-multimodal-classifier` and deploys the version referenced by the `champion` alias.",
    )
    _tool_explanation(
        "Git & DVC",
        "Git versions source code, while DVC versions large data and model files that are unsuitable for Git.",
        "We track Python, YAML, and Docker files with Git, and track data splits and model checkpoints by content hash with DVC.",
    )
    _tool_explanation(
        "MinIO",
        "An S3-compatible object-storage service for keeping large artifacts in centralized persistent storage.",
        "We use separate buckets on the same MinIO server for MLflow artifacts and the DVC remote, so model and data files survive container replacement.",
    )

    st.markdown("### DVC paths and artifact destinations")
    st.caption(
        "Large files, experiment artifacts, deployable models, and source code have different owners. "
        "The table shows both the project path and the system responsible for storing each artifact."
    )
    st.markdown(
        "| Content | Project path | Destination | What it is used for |\n"
        "|---|---|---|---|\n"
        "| Raw dataset | `data/raw/` | **DVC → `dvc-data`** *(planned; currently downloaded from Kaggle)* | Original CSV and image inputs from which every reproducible pipeline run starts. |\n"
        "| Train/validation splits | `outputs/image_modeling/train_split.csv`, `val_split.csv`, `label2id.json` | **DVC → `dvc-data`** | Versioned, shared inputs used by both image and text training. |\n"
        "| Training checkpoints | `models/Model_I12_ConvNeXt_Base_ModerateAug_Full/best_model_state_dict.pt`, `models/textModeling/Model_T8_CamemBERT_FullFineTune_L128/best_model_text.pt` | **DVC → `dvc-data`** | Reproducible model weights tied to the code, data, and `dvc.lock` version that produced them. |\n"
        "| Experiment plots and metrics | Logged by the training scripts | **MLflow → `mlflow-artifacts`** | Compares runs, parameters, validation metrics, learning curves, and reports. |\n"
        "| Deployable multimodal bundle | MLflow model `rakuten-multimodal-classifier` | **MLflow Model Registry → `champion`** | Identifies the exact image + text + tokenizer + mappings bundle approved for serving. |\n"
        "| Active API model bundle | `data/rakuten_streamlit_predictor/` | **Downloaded from Registry `champion`** | Read-only model files loaded by FastAPI for live predictions. |\n"
        "| Source code and DVC metadata | Python/YAML/Docker files, `dvc.yaml`, `dvc.lock`, `.dvc/config` | **GitHub** | Versions pipeline definitions and hashes, but never stores large data, checkpoints, or secrets. |"
    )
    st.info(
        "DVC does not create readable `raw/`, `splits/`, or `best-model/` folders inside MinIO. "
        "The workspace paths above are mapped to content hashes in the local `.dvc/cache`, and "
        "`dvc push` copies those hash-addressed objects to the `s3://dvc-data` remote. "
        "The production model is selected by MLflow's `champion` alias rather than by manually "
        "copying a file into a `best-model` folder."
    )

    st.markdown("### Where to view each storage layer")
    st.caption(
        "DVC has no separate Registry UI in this project. Its remote is the `dvc-data` bucket in "
        "MinIO; MLflow artifacts use a second bucket, while model versions and aliases are viewed in MLflow."
    )
    dvc_col, artifacts_col, registry_col = st.columns(3)
    with dvc_col:
        with st.container(border=True):
            st.markdown("#### DVC remote")
            st.write("Stores large datasets, splits, and checkpoints by content hash.")
            st.code("MinIO → Object Browser → dvc-data", language="text")
            st.link_button("Open DVC remote in MinIO", "http://localhost:9001", width="stretch")
    with artifacts_col:
        with st.container(border=True):
            st.markdown("#### MLflow artifacts")
            st.write("Stores run plots, metrics files, checkpoints, and complete model bundles.")
            st.code("MinIO → Object Browser → mlflow-artifacts", language="text")
            st.link_button("Open MLflow artifacts in MinIO", "http://localhost:9001", width="stretch")
    with registry_col:
        with st.container(border=True):
            st.markdown("#### MLflow Model Registry")
            st.write("Manages candidate versions, promotion tags, and the production `champion` alias.")
            st.code("MLflow → Models → rakuten-multimodal-classifier", language="text")
            st.link_button("Open MLflow Model Registry", "http://localhost:5001/#/models", width="stretch")
    st.caption(
        "MinIO login uses `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` from the local `.env`. "
        "A Registry model appears only after a promotion container successfully registers its first candidate."
    )

    st.markdown("### Does this survive a restart?")
    st.write(
        "Named Docker volumes (`minio_data`, `postgres_data`, and the monitoring volumes) are stored "
        "on disk, not in memory, so they are not tied to any single container's lifetime. "
        "On macOS/Windows, Docker Desktop keeps every volume inside one virtual disk image "
        "(`Docker.raw`/`Docker.vhdx`) that lives on the host disk; on Linux, volumes sit directly "
        "under `/var/lib/docker/volumes/`. Either way, the data outlives individual containers."
    )
    safe_col, destroy_col = st.columns(2)
    with safe_col:
        st.success(
            "**Data stays**\n\n"
            "- `docker compose restart`\n"
            "- `docker compose down` (no flags)\n"
            "- `docker compose up`\n"
            "- container crash or host reboot"
        )
    with destroy_col:
        st.error(
            "**Data is deleted**\n\n"
            "- `docker compose down -v`\n"
            "- `docker volume rm` / `docker volume prune`\n"
            "- Docker Desktop → *Clean / Purge data* or *Reset to factory defaults*"
        )
    st.caption(
        "So `dvc-data`, `mlflow-artifacts`, and the `champion` alias all survive normal restarts — "
        "they are only lost through one of the explicit volume-removal actions above."
    )

    st.markdown("### Models stored in MinIO")
    st.write(
        "This is a live view of model-related objects stored in MinIO. "
        "`mlflow-artifacts` contains artifacts uploaded by MLflow runs and Model Registry bundles. "
        "`dvc-data` contains content-addressed files pushed by DVC; its object names may therefore "
        "look like hashes instead of human-readable model names."
    )
    if st.button("Refresh MinIO objects", width="content"):
        _list_minio_objects.clear()
    minio_rows, minio_error = _list_minio_objects()
    if minio_error:
        st.warning(
            "MinIO objects could not be loaded. Make sure the MinIO service is running. "
            f"Connection detail: {minio_error}"
        )
    elif minio_rows:
        st.dataframe(minio_rows, width="stretch", hide_index=True)
        st.caption(f"Showing {len(minio_rows)} model-related objects. MinIO Console: http://localhost:9001")
    else:
        st.info(
            "No model-related objects were found yet. Complete a tracked training run for "
            "MLflow artifacts, or run `uv run dvc push` to populate the DVC remote."
        )
    _tool_explanation(
        "PostgreSQL",
        "Stores persistent, queryable application data in relational tables.",
        "One PostgreSQL server hosts separate databases for MLflow run metadata, Airflow pipeline metadata, and API users.",
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

    st.markdown("### Tools used in this stage")
    _tool_explanation(
        "Apache Airflow",
        "Schedules multi-step workflows as DAGs and records the status of every task.",
        "The Retrain button in Streamlit triggers the `rakuten_model_training` DAG through FastAPI. The DAG prepares data, trains the selected model in a trainer container, and starts the promotion DAG after a successful run.",
    )
    _tool_explanation(
        "FastAPI",
        "A Python framework for building typed REST APIs with automatic Swagger documentation.",
        "It provides login/logout, multimodal `/predict`, retraining, and job-status endpoints. The production model is loaded only by the API service.",
    )
    _tool_explanation(
        "Streamlit",
        "Builds interactive data and machine-learning web interfaces directly in Python.",
        "It provides authentication, product image/text inputs, prediction results, and the administrator retraining screen. It communicates with FastAPI over HTTP instead of loading models directly.",
    )
    _tool_explanation(
        "Docker & Docker Compose",
        "Runs applications with their dependencies in isolated containers, while Compose manages multiple services together.",
        "We run the API, Streamlit, MLflow, MinIO, PostgreSQL, and monitoring services in one development stack. Resource-intensive training runs in a temporary trainer container.",
    )
    _tool_explanation(
        "GitHub Actions",
        "A continuous-integration service that automatically tests code and publishes deployable container images.",
        "On pull requests it runs pytest and verifies all five application images. After a successful push to `main`, it publishes immutable commit-tagged images and `latest` tags to GitHub Container Registry (GHCR).",
    )

    st.markdown("### What is stored where?")
    st.caption("Each artifact type has a separate remote store; Docker images and model files are not committed to Git.")
    st.markdown(
        "| Artifact | Destination | Purpose |\n"
        "|---|---|---|\n"
        "| Source code and configuration | **GitHub repository** | Git versions Python, YAML, Dockerfiles and documentation. |\n"
        "| Docker images | **GitHub Container Registry (GHCR)** | CI publishes the API, Streamlit, Airflow, trainer and promotion images under `ghcr.io/suozdemir/`. |\n"
        "| Datasets and model checkpoints | **DVC remote on MinIO/S3** | DVC versions large files by content hash without adding them to Git. |\n"
        "| MLflow run and Registry artifacts | **MinIO** | MLflow stores checkpoints, plots and deployable model bundles in `mlflow-artifacts`. |"
    )
    st.info(
        "In short: `git push` sends code to GitHub, the CI workflow sends container images "
        "to GHCR, `dvc push` sends versioned data/models to the DVC bucket, and MLflow "
        "writes experiment artifacts to its MinIO bucket."
    )

    st.markdown("### Airflow DAGs")
    st.markdown(
        "- **`rakuten_model_training`** — DVC prep + train, triggered by the API's `/retrain` endpoint\n"
        "- **`rakuten_model_promotion`** — launches an ephemeral promotion container, pushes the retrained DVC checkpoint, registers a candidate, applies the component Macro-F1 gate, and deploys only a passing `champion`"
    )

    st.markdown("### DVC tasks inside the Airflow training DAG")
    st.write(
        "Airflow orchestrates DVC; it does not train the model inside the scheduler. "
        "Each task has one clear responsibility and a failed task stops every downstream step."
    )
    st.markdown(
        "| Airflow task | Where it runs | DVC operation | Result |\n"
        "|---|---|---|---|\n"
        "| `prepare_data` | Airflow scheduler (lightweight) | `dvc repro prepare_splits` | Rebuilds shared train/validation CSVs and `label2id.json` only when their dependencies changed. |\n"
        "| `launch_trainer` | Ephemeral trainer container | `dvc repro train_image\\|train_text --force --single-item` | Applies the requested epochs, batch size and learning rate, retrains the selected component, and records the best validation Macro-F1 checkpoint in `dvc.lock`. |\n"
        "| `trigger_promotion` | Airflow operator | Passes `model=image\\|text` and waits for `rakuten_model_promotion` | Keeps the retrain job open until candidate registration, gating, and deployment finish. |\n"
        "| `register_gate_and_deploy` | Ephemeral promotion container | `dvc push dvc.yaml:train_image\\|train_text` | Copies the reproduced checkpoint from the local DVC cache to the `s3://dvc-data` MinIO/S3 remote. |"
    )
    st.code(
        "POST /retrain {model, epochs, batch_size, learning_rate}\n"
        "  ↓\n"
        "prepare_data\n"
        "  dvc repro prepare_splits\n"
        "  ↓\n"
        "launch_trainer (temporary container)\n"
        "  dvc repro train_<model> --force --single-item\n"
        "  ↓\n"
        "trigger_promotion (waits for completion)\n"
        "  ↓\n"
        "register_gate_and_deploy (temporary container)\n"
        "  dvc push dvc.yaml:train_<model>",
        language="text",
    )
    st.caption(
        "`dvc push` archives the checkpoint by content hash; it does not make the model production. "
        "Production approval is handled separately by the MLflow Registry gate below."
    )
    st.info(
        "Retrain defaults are model-aware: image uses batch 16 and head LR 5e-5; text "
        "uses batch 32 and LR 2e-5. For ConvNeXt, the selected LR applies to the "
        "classifier head and the pretrained backbone uses LR / 10. The form also "
        "exposes seed, early-stopping patience, weight decay and AMP; image training "
        "adds label smoothing and classifier dropout. Architecture and input shape "
        "remain read-only to preserve serving compatibility."
    )

    st.markdown("### What model promotion does")
    st.code(
        "Image checkpoint\n"
        "+ Text checkpoint\n"
        "+ CamemBERT base model and tokenizer\n"
        "+ label map\n"
        "+ category map\n"
        "  ↓\n"
        "Complete, loadable multimodal bundle\n"
        "  ↓\n"
        "MLflow Registry candidate version\n"
        "  ↓\n"
        "Champion gate (candidate vs current component Macro-F1)\n"
        "  ├─ rejected → keep candidate for audit; champion and API stay unchanged\n"
        "  └─ accepted\n"
        "       ↓\n"
        "champion alias\n"
        "       ↓\n"
        "Download models:/rakuten-multimodal-classifier@champion\n"
        "       ↓\n"
        "Atomic replacement of data/rakuten_streamlit_predictor/\n"
        "       ↓\n"
        "FastAPI reloads the new serving bundle",
        language="text",
    )
    st.info(
        "A `best_model_*.pt` file visible in MinIO is only the best checkpoint within one "
        "training run. It becomes the production model only after the complete bundle is registered, "
        "passes the champion gate, receives the `champion` alias, and is installed in the API serving directory."
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
        "`.github/workflows/ci.yml` runs the test suite and builds all five application images on every push/PR; "
        "a successful `main` push publishes them to GHCR."
    )


# -------------------------------------------------------------------
# 5. Monitoring
# -------------------------------------------------------------------
def render_monitoring():
    st.title("Monitoring & Alerts")

    st.markdown("### Tools used in this stage")
    _tool_explanation(
        "Prometheus",
        "Collects time-series metrics from applications at regular intervals and stores them for querying.",
        "It scrapes FastAPI's `/metrics` endpoint every five seconds and collects request count, latency, error rate, prediction confidence, and drift metrics.",
    )
    _tool_explanation(
        "Grafana",
        "Visualizes metrics from sources such as Prometheus through dashboards and alert rules.",
        "The `Rakuten API Overview` dashboard displays request rate, p50/p95 latency, and 5xx errors. Grafana also evaluates the PSI drift alert.",
    )
    _tool_explanation(
        "PSI (Population Stability Index)",
        "Measures how far a live distribution has moved from a reference distribution to indicate possible prediction drift.",
        "We compare the category distribution of the latest 200 predictions with the training reference. The current local-demo setting calculates PSI after 5 predictions (use 30+ in production), and Grafana raises an alert if it remains above the configured 0.60 threshold.",
    )

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
        "The table separates the detector, metric storage, alert engine, and trigger policy:"
    )
    st.markdown(
        "| # | Type | Status | Tools and responsibilities | When to alert |\n"
        "|---|---|---|---|---|\n"
        "| 1 | Input drift | ❌ Not built | Python/Evidently detector → Prometheus → Grafana | "
        "After a representative window (for example 200 rows), when feature drift remains above "
        "the agreed threshold for 2 evaluation windows |\n"
        "| 2 | Prediction drift | ✅ **Built** | `drift.py` calculates PSI → Prometheus stores "
        "gauges → Grafana evaluates and notifies | After ≥5 predictions in this demo (30+ in "
        "production), when PSI is **>0.60 for 2 minutes** |\n"
        "| 3 | Concept drift | ❌ Not built | Airflow label-join/evaluation job → Prometheus and "
        "MLflow → Grafana | After ground-truth labels arrive and class/segment errors degrade "
        "significantly across 2 labeled windows |\n"
        "| 4 | Model performance decay | ❌ Not built | Airflow evaluator → MLflow for run "
        "history + Prometheus for live metrics → Grafana | When rolling accuracy/Macro-F1 falls "
        "below the champion baseline or SLA for 2 evaluations |\n"
        "| 5 | System/service health | 🟡 Metrics built; rules missing | FastAPI metrics → "
        "Prometheus → Grafana | Recommended: 5xx rate >5% or p95 latency above the SLA for 5 min |\n"
        "| 6 | Pipeline/orchestration failure | ❌ Alert callback not built | Airflow task state "
        "and retry callbacks → email/Slack | Immediately when retries are exhausted or a critical "
        "DAG/task fails |\n"
        "| 7 | Data quality and schema drift | ❌ Not built | Pandera/Great Expectations in API "
        "or Airflow → Prometheus → Grafana | Schema mismatch immediately; quality violation rate "
        "above the accepted limit in a batch/window |\n"
        "| 8 | Infra/resource | 🟡 Docker healthchecks only | cAdvisor + node-exporter → "
        "Prometheus → Grafana | Recommended: container down for 2 min, memory >90% for 5 min, "
        "or low-disk threshold reached |"
    )
    st.caption(
        "Only row 2 has a live alert threshold in this repository. Thresholds shown for the other "
        "rows are recommended starting policies and must be tuned from production baselines."
    )
    st.info(
        "**Who does what?** The application or a batch detector calculates the signal. "
        "**Prometheus** scrapes and stores the numeric metric. **Grafana Alerting** queries that "
        "metric, applies the time/threshold rule, and routes the notification. For DAG failures, "
        "**Airflow** should alert directly through task failure and retry callbacks."
    )

    st.code(
        "Application or detector → exposes metric → Prometheus stores time series\n"
        "                                      → Grafana evaluates rule → email/Slack\n\n"
        "Airflow task failure → Airflow retry/failure callback → email/Slack",
        language="text",
    )

    with st.expander("2. Prediction drift — details", expanded=True):
        st.markdown(
            "- **Metric**: Population Stability Index (PSI) between the live rolling window of "
            "predicted `prdtypecode`s and the training set's class distribution "
            "(`<0.1` none, `0.1–0.25` moderate, `>0.25` significant).\n"
            "- **Where stored**: in-memory `deque(maxlen=200)` in `src/api/drift.py` — resets on "
            "API restart, no database needed for a single-process demo. Reference distribution: "
            "`src/api/reference_class_distribution.json`.\n"
            "- **When triggered**: the current local-demo setting starts scoring after ≥5 recorded "
            "predictions (`MIN_PREDICTIONS_FOR_SCORE`; use 30+ for a less noisy production signal). The alert fires "
            "when PSI stays **> 0.60 for 2 consecutive 1-minute evaluations** "
            "(`monitoring/grafana/provisioning/alerting/rules.yaml`, `for: 2m`) so a single noisy "
            "window doesn't page anyone.\n"
            "- **Where observed**:\n"
            "  - Raw metric: Prometheus gauges `prediction_drift_psi` / "
            "`prediction_drift_window_size` (`/metrics`)\n"
            "  - Dashboard: Grafana → *Rakuten API Overview*\n"
            "  - Alert: Grafana → *Alerting* → *Rakuten Alerts* folder, routed to email and "
            "optionally Slack via `contactpoints.yaml` (credentials are set in a local `.env`)"
        )

    with st.expander("5. System/service health — details"):
        st.markdown(
            "- **Where stored**: `http_requests_total` / `http_request_duration_seconds` "
            "(auto-instrumented), scraped into Prometheus's own time-series storage.\n"
            "- **When triggered**: nothing yet — no Grafana alert rule watches latency or "
            "error-rate, only the drift rule above exists.\n"
            "- **Where observed**: Grafana → *Rakuten API Overview* (request-rate, p50/p95 "
            "latency, and error-rate panels)."
        )

    st.info(
        "Only prediction drift is currently calculated as an ML drift signal. "
        "Prediction drift is not proof of concept drift: confirming concept or performance drift "
        "requires delayed ground-truth labels. Open the Drift Types page for the distinctions and "
        "the recommended response workflow."
    )


# -------------------------------------------------------------------
# 6. Drift Types
# -------------------------------------------------------------------
def render_drift_types():
    st.title("Drift Types")
    st.caption(
        "A practical guide to distribution changes, model degradation, and the signals "
        "implemented in this project."
    )

    st.markdown("### What is model drift?")
    st.write(
        "Drift means that production data, predictions, or the relationship between inputs and "
        "the correct outcome has changed compared with the model's reference period. A drift "
        "signal is an investigation trigger; it does not automatically prove that the model is wrong."
    )

    st.markdown("### Main drift types")
    st.markdown(
        "| Drift type | What changes? | Rakuten example | How to detect it |\n"
        "|---|---|---|---|\n"
        "| **Data / input drift** (covariate shift) | Input distribution `P(X)` | Product-title "
        "lengths, image brightness, language, or image embeddings change | PSI, KS test, "
        "Jensen-Shannon distance, or embedding-distance monitoring |\n"
        "| **Label drift** (prior-probability shift) | Class distribution `P(Y)` | The real share "
        "of books, toys, and electronics changes | Compare delayed ground-truth label "
        "distributions with the reference period |\n"
        "| **Concept drift** | Relationship `P(Y given X)` | The same product characteristics now map "
        "to a different `prdtypecode` because taxonomy or customer behavior changed | Join "
        "predictions with new labels and monitor class-level accuracy, F1, and error patterns |\n"
        "| ✅ **Prediction drift** | Model-output distribution `P(Ŷ)` | The API suddenly predicts "
        "one `prdtypecode` much more frequently | Compare predicted-class distributions with PSI "
        "or another distribution-distance metric |\n"
        "| **Performance drift** | Quality metrics decline | Accuracy, macro-F1, calibration, or "
        "top-3 accuracy falls below the accepted baseline | Evaluate labeled production samples "
        "over rolling time windows |\n"
        "| **Schema drift** | Data structure or contract | A column is renamed, an image field "
        "changes type, or a required field disappears | Schema validation, contract tests, and "
        "validation-failure metrics |\n"
        "| **Data-quality drift** | Data validity or completeness | More empty descriptions, "
        "corrupt images, duplicates, or out-of-range values | Missing-value, duplicate, range, "
        "and parse-failure monitoring |"
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### Prediction drift ≠ concept drift")
            st.write(
                "Prediction drift can be measured immediately without labels. It may come from "
                "real demand changes, input drift, model behavior, or a pipeline bug. Concept "
                "drift can only be confirmed after ground-truth labels arrive."
            )
    with right:
        with st.container(border=True):
            st.markdown("#### Drift ≠ automatic retraining")
            st.write(
                "A high drift score should first trigger data and service checks. Retraining is "
                "appropriate only when labeled evidence shows degradation and a candidate model "
                "passes the validation and champion gates."
            )

    with st.expander("Current prediction-drift configuration", expanded=True):
        st.markdown(
            "- **Reference:** training class distribution in "
            "`src/api/reference_class_distribution.json`\n"
            "- **Live window:** latest 200 API predictions, held in memory\n"
            "- **Demo minimum:** PSI starts after 5 predictions; use at least 30 in production\n"
            "- **Alert:** `prediction_drift_psi > 0.60` for two minutes\n"
            "- **Metrics:** `prediction_drift_psi` and `prediction_drift_window_size` in "
            "Prometheus; dashboard and alert in Grafana\n"
            "- **Limitation:** the live window resets when the API process restarts"
        )

    st.markdown("### Recommended response workflow")
    st.code(
        "Drift signal\n"
        "  → validate schema, data quality, and service health\n"
        "  → identify affected classes, features, and time window\n"
        "  → collect delayed ground-truth labels\n"
        "  → measure champion performance\n"
        "      → performance is stable: document and continue monitoring\n"
        "      → performance degraded: train a candidate model\n"
        "          → validation + MLflow champion gate\n"
        "              → promote only if the candidate passes",
        language="text",
    )
    st.warning(
        "Do not launch automatic retraining from PSI alone. A PSI alert is an early-warning "
        "signal, not evidence that the current champion has lost accuracy."
    )


# -------------------------------------------------------------------
# 7. Plugins & Links
# -------------------------------------------------------------------
_LINKS = [
    ("MLflow", "mlflow", "http://localhost:5001", "Experiment tracking + Model Registry"),
    ("MinIO Console", "minio", "http://localhost:9001", "MLflow and DVC object storage"),
    ("Adminer", "adminer", "http://localhost:8081", "Browser UI for PostgreSQL databases"),
    ("PostgreSQL", "postgresql", "postgresql://localhost:5432", "Database endpoint; connect with a database client"),
    ("Prometheus", "prometheus", "http://localhost:9090", "Raw metrics + target health"),
    ("Grafana", "grafana", "http://localhost:3000", "API Overview dashboard (credentials come from the local .env)"),
    ("Airflow", "apacheairflow", "http://localhost:8080", "Training + promotion DAGs"),
    ("API docs (Swagger)", "swagger", "http://localhost:8000/docs", "Interactive FastAPI docs"),
]


def render_links():
    st.title("Plugins & Links")
    st.caption("Interfaces for the services used in this project. The relevant service must be running through `make up` or `make up-all`.")
    items = "".join(
        f'<li style="display:flex;align-items:center;gap:8px;margin:7px 0;">'
        f'<img src="https://cdn.simpleicons.org/{escape(icon)}" alt="" '
        f'width="16" height="16" style="flex:0 0 16px;">'
        f'<a href="{escape(url)}" target="_blank" '
        f'style="font-weight:600;text-decoration:none;">{escape(name)}</a>'
        f'<a href="{escape(url)}" target="_blank" '
        f'style="font-family:monospace;font-size:0.78rem;text-decoration:none;">'
        f'{escape(url)}</a>'
        f'<span style="font-size:0.82rem;color:#808495;">— {escape(description)}</span>'
        f'</li>'
        for name, icon, url, description in _LINKS
    )
    st.markdown(
        f'<ul style="list-style:none;padding-left:0;margin-top:8px;">{items}</ul>',
        unsafe_allow_html=True,
    )
