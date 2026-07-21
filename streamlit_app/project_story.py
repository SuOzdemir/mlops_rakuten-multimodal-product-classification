from html import escape
from pathlib import Path

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
    "Evidently": ("evidentlyai",),
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
        "`.github/workflows/ci.yml` runs the test suite and builds all six application images on every push/PR; "
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
        "Evidently",
        "Compares a current production window with reference data and produces drift reports.",
        "An optional parallel container reads privacy-safe prediction features from PostgreSQL, writes HTML/JSON reports to MinIO, and exposes drift metrics to Prometheus. It never replaces or blocks the live PSI path.",
    )
    _tool_explanation(
        "PSI (Population Stability Index)",
        "Measures how far a live distribution has moved from a reference distribution to indicate possible prediction drift.",
        "We compare the category distribution of the latest 200 predictions with the training reference. The current local-demo setting calculates PSI after 5 predictions (use 30+ in production), and Grafana raises an alert if it remains above the configured 0.60 threshold.",
    )
# -------------------------------------------------------------------
# 6. MLOps & Model Drift
# -------------------------------------------------------------------
def render_drift_types():
    st.title("MLOps & Model Drift")
    st.caption(
        "Why a trained model requires lifecycle management, continuous monitoring, and "
        "controlled response to change."
    )

    st.markdown("### Why MLOps is needed")
    st.write(
        "After finding a strong model for a prediction problem and a particular dataset, the "
        "work may appear complete. In a changing world, however, data, requirements, and targets "
        "evolve over time. The model can therefore become gradually and almost invisibly less "
        "appropriate for its production environment."
    )
    st.write("MLOps controls model development and deployment by:")
    st.markdown(
        "- Managing the model's dependency on data and accounting for how that data changes over time\n"
        "- Automating data preparation, model training, validation, promotion, and deployment\n"
        "- Assuring reproducibility through versioned data, code, parameters, and model artifacts"
    )

    st.markdown("### Drift as an operational signal")
    st.write(
        "A key responsibility of MLOps is detecting and measuring changes in model behavior and "
        "prediction quality. Prediction drift can be measured immediately from model outputs, "
        "while performance drift requires ground-truth labels and quality metrics such as "
        "accuracy or F1. A drift signal triggers investigation and controlled MLOps actions; it "
        "does not by itself prove that the model is wrong or require automatic retraining."
    )

    st.markdown("### Four core drift types")
    st.markdown(
        "| Drift type | Definition | Rakuten example |\n"
        "|---|---|---|\n"
        "| **Data drift** | The statistical distribution of input data changes over time. For "
        "multimodal data this can include changes in language, text length, image properties, or "
        "other input characteristics, potentially reducing prediction quality. | A new wave of "
        "listings starts using much longer, keyword-heavy French descriptions than the training "
        "set, shifting the `designation_length` / `description_length` distributions (tracked via "
        "the optional Evidently sidecar). |\n"
        "| **Concept drift** | The relationship between input variables and the correct target "
        "changes; similar products may need to map to different categories because the taxonomy "
        "or real-world behavior changed. | Rakuten reclassifies wireless earbuds from one "
        "`prdtypecode` to another after a taxonomy update -- same text/image, different correct "
        "answer. |\n"
        "| ✅ **Prediction drift** | The statistical distribution of predicted values changes. It "
        "can be measured with PSI, Jensen-Shannon distance, or related distribution metrics. "
        "Prediction drift may accompany lower quality, but it can also reflect a legitimate "
        "population change. | Predicted \"toys\" share jumps from ~10% to ~40% overnight -- could "
        "be a real seasonal shift (holiday season) or a model/pipeline bug. |\n"
        "| **Performance drift** | Labeled production metrics such as accuracy, Macro-F1, or "
        "calibration deteriorate over time, providing direct evidence that prediction quality "
        "changed. | Macro-F1 on newly labeled validation batches drops from 0.87 to 0.70 over a "
        "quarter as new product types appear. |"
    )

# -------------------------------------------------------------------
# 7. Key Notes
# -------------------------------------------------------------------
_ARCHITECTURAL_DECISIONS = [
    (
        "Modular multimodal architecture",
        "ConvNeXt and CamemBERT remain independent components: only the selected image "
        "or text model is retrained, then promotion combines the new checkpoint with the "
        "unchanged component, tokenizer, and label/category maps to rebuild one complete "
        "serving bundle. During inference, their logits are combined before softmax with "
        "configurable weights, with image-only or text-only fallback when one modality is absent.",
    ),
    (
        "Ephemeral containers",
        "For each training or promotion run, Airflow starts a fresh Docker container "
        "with only the dependencies required for that task. The container stops when "
        "the task finishes, while long-running services such as FastAPI, Streamlit, "
        "MLflow, PostgreSQL, and MinIO remain available. This isolates training failures "
        "and heavy workloads from online inference.",
    ),
    (
        "Metric-gated champion promotion",
        "A candidate becomes the serving model only when the retrained component's "
        "Macro-F1 meets the current champion plus the configured minimum gain. If accepted, "
        "promotion moves the MLflow `champion` alias to that version and serving follows the "
        "alias instead of a hard-coded version number. If rejected, the version remains "
        "available for audit while the champion alias and API stay unchanged.",
    ),
    (
        "DVC and MLflow separation",
        "DVC versions datasets, splits, and training checkpoints. MLflow records runs, "
        "parameters, metrics, artifacts, and candidate/champion model versions.",
    ),
    (
        "Atomic deployment and hot reload",
        "Promotion swaps the complete serving directory atomically and then updates the "
        "manifest. FastAPI loads the new bundle on the next prediction without restarting.",
    ),
    (
        "Non-blocking monitoring",
        "A failed `prediction_events` telemetry write is logged but does not fail an "
        "otherwise successful prediction. Evidently runs asynchronously, so a monitoring "
        "outage may create an observation gap without interrupting inference.",
    ),
]

_LINKS = [
    ("MLflow", "mlflow", "http://localhost:5001", "Experiment tracking + Model Registry"),
    ("MinIO Console", "minio", "http://localhost:9001", "MLflow and DVC object storage"),
    ("Adminer", "adminer", "http://localhost:8081", "Browser UI for PostgreSQL databases"),
    ("PostgreSQL", "postgresql", "postgresql://localhost:5432", "Database endpoint; connect with a database client"),
    ("Prometheus", "prometheus", "http://localhost:9090", "Raw metrics + target health"),
    ("Grafana", "grafana", "http://localhost:3000", "API Overview dashboard (credentials come from the local .env)"),
    ("Evidently", "evidentlyai", "http://localhost:8002/reports/latest.html", "Optional parallel drift report (`make evidently-up`)"),
    ("Airflow", "apacheairflow", "http://localhost:8080", "Training + promotion DAGs"),
    ("API docs (Swagger)", "swagger", "http://localhost:8000/docs", "Interactive FastAPI docs"),
]


def render_key_notes():
    st.title("Key Notes")
    st.caption("The architectural choices that shape model training, deployment, and serving.")

    st.markdown("### Architectural decisions")
    for title, description in _ARCHITECTURAL_DECISIONS:
        st.markdown(f"- **{title}** — {description}")

    st.markdown("### Artifact and package terminology")
    st.markdown(
        "| Concept | Contains | Purpose |\n"
        "|---|---|---|\n"
        "| **Checkpoint** | Trained weights for one image or text component | Preserve the output of a component training run |\n"
        "| **Complete serving bundle** | Both checkpoints, CamemBERT and tokenizer, label map, and category map | Give FastAPI every model asset required for inference |\n"
        "| **MLflow model package** | Complete serving bundle, inference-code snapshot, and Python dependencies | Register and load a portable, executable model version |\n"
        "| **`dvc-data`** | Content-addressed DVC objects: datasets, splits, and checkpoints | Store reproducible, versioned pipeline data in the MinIO DVC remote |"
    )

    st.divider()
    st.markdown("### Service links")
    st.caption(
        "The relevant service must be running through `make up`, `make up-all`, "
        "or its optional profile command."
    )
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
