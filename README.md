# Rakuten Multimodal Product Classification

Multimodal (text + image) product category classification for the Rakuten France e-commerce dataset. Products are assigned to one of **27 categories** (`prdtypecode`) using product title, description, and image.

---

## Objectives & Key Metrics

**Objective:** classify Rakuten France products into 27 categories from title, description, and image, and serve the resulting model behind an authenticated REST API + web UI — with experiment tracking and a repeatable promotion path from a trained checkpoint to production.

| Metric | Type | Result |
|--------|------|--------|
| Val Macro F1 — CamemBERT (text) | Model quality | **0.871** |
| Val Macro F1 — ConvNeXt-Base (image) | Model quality | **0.692** |
| Val Macro F1 — Late fusion (α=0.45) | Model quality | **~0.893** |
| Public leaderboard rank | Business | Top 5 ([challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/35/ranking/public)) |
| API test coverage | Engineering | 45 unit tests, gated in CI on every push/PR |
| Serving latency / uptime | Ops | Prometheus + Grafana (`Rakuten API Overview` dashboard) |
| Prediction drift | Ops | PSI-based tracker, alerted via Grafana → Slack when it crosses 0.25 for 2 minutes |

See [docs/methodology_phase1.md](docs/methodology_phase1.md) for modeling detail and [docs/methodology_phase2.md](docs/methodology_phase2.md) for the MLOps pipeline.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         docker-compose (main stack)                        │
│                                                                            │
│  ┌────────────┐   ┌─────────────┐        ┌────────┐      ┌─────────────┐ │
│  │ Streamlit  │──▶│  FastAPI    │───────▶ │ MLflow │────▶ │   MinIO     │ │
│  │  :8501     │   │  :8000      │         │ :5001  │      │ :9000/9001  │ │
│  └────────────┘   └──────┬──────┘        └───┬────┘      └─────────────┘ │
│                          │                    │        (artifacts + DVC   │
│                          ▼                    ▼            remote data)  │
│                    ┌───────────┐        ┌───────────┐                    │
│                    │ Prometheus│──────▶ │  Grafana  │──▶ Slack (alerts)  │
│                    │  :9090    │        │  :3000    │                    │
│                    └───────────┘        └───────────┘                    │
│                          ▲                    │                          │
│                          │              ┌───────────┐                    │
│                          └──────────────│ Postgres  │ (mlflow/airflow/   │
│                                         │  :5432    │  api auth DBs)     │
│                                         └───────────┘                    │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                   airflow_dst compose (separate network)                   │
│                                                                            │
│   Airflow  ──▶  launch_trainer (DockerOperator, via docker-socket-proxy)  │
│   :8080         ──▶ ephemeral `trainer` container (dvc repro train_*)     │
│                 ──▶ on success, triggers rakuten_model_promotion DAG      │
└────────────────────────────────────────────────────────────────────────────┘
```

### Request flow

**Login**
```
User → POST /login (username + password)
     → API validates credentials
     → returns Bearer token
     → Streamlit stores token in session
```

**Prediction**
```
User → image + text → clicks Predict
     → POST /predict (Bearer token + form data)
     → API validates token
     → raw_late_fusion_predictor:
           ConvNeXt-Base  (image logits)
         + CamemBERT      (text logits)
         → weighted late fusion
         → top 3 categories
     → JSON → Streamlit renders result
```

**Logout**
```
User → clicks Log out
     → POST /logout → API removes token
     → redirected to login screen
```

---

## MLflow

MLflow is used for **experiment tracking** and **model promotion** across the training pipeline.

### What is tracked

During training (`train.py`), each run logs:

| Type | Examples |
|------|---------|
| Parameters | `lr`, `epochs`, `batch_size`, `architecture` |
| Metrics | `val_f1`, `val_accuracy`, `val_loss` per epoch |
| Artifacts | confusion matrix, learning curves, classification report |

```python
# Example from train.py
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("val_f1", 0.87, step=epoch)
    mlflow.log_artifact("confusion_matrix.png")
```

### UI

MLflow UI is accessible at `http://localhost:5001` after `docker compose up`.  
All runs across models (ConvNeXt, CamemBERT, fusion experiments) can be compared side by side.

MLflow's backend store is a shared Postgres instance (`postgres` service in `docker-compose.yaml`) —
the same Postgres also backs Airflow's metadata DB (reached via `host.docker.internal` from the
separate `airflow_dst` compose stack), so there's one database server instead of two SQLite files.

### Model Registry and deployment (Airflow DAG)

After training, `rakuten_model_promotion` assembles a complete, loadable
multimodal serving bundle and logs it as an MLflow `pyfunc` model named
`rakuten-multimodal-classifier`. The new version receives the `champion`
alias. The DAG then downloads `models:/rakuten-multimodal-classifier@champion`
back from MLflow and atomically replaces `data/rakuten_streamlit_predictor/`.

The registered model and the API therefore use the same image weights,
fine-tuned text weights, tokenizer, label map and category map. A
`deployment_manifest.json` in the serving directory records the deployed
MLflow version and run ID.

### Retrain pipeline (Airflow DAG + ephemeral trainer + DVC + API)

`POST /retrain` (admin only) doesn't run training itself — it triggers the Airflow DAG `rakuten_model_training` via Airflow's REST API. That DAG doesn't run training as a subprocess of its own scheduler either: it launches a fresh, throwaway **`trainer`** container per run (via `DockerOperator`), does the job, and is removed. Rationale, and why this replaced an earlier design that ran training as a subprocess inside the always-on Airflow scheduler:

```
POST /retrain {model: image|text}   (admin)
   → API calls Airflow REST API: POST /api/v1/dags/rakuten_model_training/dagRuns
        ↓
   DAG: prepare_data     (dvc repro prepare_splits, in the scheduler -- cheap, no ML deps)
        ↓
   DAG: launch_trainer   (DockerOperator, via docker-socket-proxy, launches a fresh
                          `trainer` container: dvc repro train_image|train_text --force
                          --single-item, model from dag_run conf; container is removed
                          on success)
        ↓
   DAG: trigger_promotion (on success, triggers rakuten_model_promotion and waits for it)

GET /retrain            → list all dag runs (any authenticated user)
GET /retrain/{job_id}   → poll a specific dag run's status
```

Why an ephemeral container instead of a subprocess: a full-size ConvNeXt-Base/CamemBERT
training run needs several GB of RAM, and running it as a subprocess of the Airflow
scheduler meant it competed for memory with every other always-on service in that
container — it got OOM-killed under any real load. The `trainer` container gets its own
`mem_limit`/`shm_size` (`docker-compose.yaml`), is launched fresh per run, and is
discarded after.

A few things worth knowing:
- MLflow's artifact store is MinIO (S3-compatible), not a bind-mounted directory, so the
  `trainer` container never needs storage credentials — only `MLFLOW_TRACKING_URI`.
- Airflow talks to `docker-socket-proxy`, not the raw host Docker socket, so
  `DockerOperator` can launch/stop containers but can't touch volumes or anything else
  on the host.
- DVC's lock is repo-wide, not per-stage, so two retrains — even of different models —
  collide on `.dvc/tmp/rwlock` if run concurrently. `src/api/retrain.py` enforces one
  retrain at a time for this reason.
- `mem_limit`/`shm_size` cap the trainer's own usage, but don't add headroom beyond
  Docker Desktop's overall VM memory ceiling — that still needs raising separately if
  the always-on services plus training together exceed it.
- `scripts/diagnose_disk.sh` is a read-only helper for figuring out what's actually
  eating disk (Docker build cache, named volumes, or the repo's own checkpoints).

Also still true from the original design:
- **`scripts/prepare_splits.py`** builds `outputs/image_modeling/{train_split,val_split}.csv` + `label2id.json` from the raw Kaggle CSVs (`data/raw/X_train.csv`, `Y_train.csv`) — both I12 and T8 read from the same split so image/text models are evaluated on identical held-out products.
- **`dvc.yaml`** wraps that script as a pipeline stage (`prepare_splits`) — `dvc repro` only re-runs it if the raw data or script changed.
- Requires the raw Kaggle dataset in `data/raw/` (`./scripts/setup_data.sh`) — without it, `dvc repro` fails at the `prepare_data` task, which is expected until the dataset is downloaded on the machine running Airflow.

---

## Monitoring & Drift Detection

The API exposes Prometheus metrics at `/metrics` (`prometheus-fastapi-instrumentator`): request rate, latency (p50/p95), and error rate per endpoint. `monitoring/prometheus/prometheus.yml` scrapes it every 5s, and Grafana auto-provisions a **Rakuten API Overview** dashboard on startup.

`src/api/drift.py` also tracks prediction drift — whether the live mix of predicted categories is drifting away from the training set's class distribution:
- Metric: Population Stability Index (PSI) — `<0.1` none, `0.1–0.25` moderate, `>0.25` significant
- Compared against `src/api/reference_class_distribution.json` (built by `scripts/compute_reference_distribution.py`)
- Scored over the last 200 predictions (in-memory, resets on API restart), needs at least 30 before it scores at all
- Exposed as Prometheus gauges `prediction_drift_psi` / `prediction_drift_window_size`, and alerted on in Grafana (`monitoring/grafana/provisioning/alerting/`) — fires when PSI stays above 0.25 for 2 minutes, routed to Slack via `SLACK_WEBHOOK_URL` (see `.env.example`)

Not covered yet: input-feature drift, live model-performance decay against real ground truth, Airflow pipeline-failure alerts, data-quality checks, and infra-resource alerts. See the Streamlit app's Monitoring page for the full breakdown of what's built vs. open.

---

## Models

### Text — CamemBERT (T8)
- Architecture: `camembert-base` fine-tuned for sequence classification
- Input: `designation + description` (max 128 tokens)
- Val macro F1: **0.871**

### Image — ConvNeXt-Base (I12)
- Architecture: ConvNeXt-Base, fully fine-tuned
- Input: 224×224, TrivialAugmentWide
- Val macro F1: **0.692**

### Fusion — Late Fusion
- Weighted average of text and image logits
- Default weights: text `0.55`, image `0.45`
- Supports: text only / image only / text + image

---

## Project Structure

```
.
├── docker-compose.yaml              # API + Streamlit + MLflow + MinIO + Postgres + Prometheus + Grafana
├── pyproject.toml                   # Python dependencies (uv) — dev/training env
├── Makefile                         # make up / airflow-up / test / health / restart-all shortcuts
├── .github/workflows/ci.yml         # pytest + Docker build checks on push/PR
├── dvc.yaml                         # DVC pipeline: raw CSVs -> train/val splits
├── .dvc/                            # DVC metadata (config: MinIO remote; config.local: per-machine creds, gitignored)
├── .env.example                     # Template for the root .env (SLACK_WEBHOOK_URL, gitignored)
│
├── models/
│   ├── Model_I12_ConvNeXt_Base_*/   # Image model training
│   └── textModeling/
│       └── Model_T8_CamemBERT_*/    # Text model training
│
├── trainer/                          # Ephemeral training container (launched by the training DAG)
│   ├── Dockerfile                    # CPU-only torch + mlflow/dvc deps
│   ├── entrypoint.sh                 # dvc repro prepare_splits && dvc repro train_$MODEL --force
│   └── requirements.txt
│
├── scripts/
│   ├── setup_data.sh                 # Downloads raw Kaggle dataset -> data/raw/
│   ├── prepare_splits.py             # DVC stage: builds train/val split CSVs + label2id.json
│   ├── compute_reference_distribution.py  # Builds src/api/reference_class_distribution.json (drift baseline)
│   ├── diagnose_disk.sh              # Read-only: what's consuming disk (Docker vs. repo vs. volumes)
│   ├── git-helper.sh                 # Git shortcuts for contributors without a GUI git client
│   ├── run_api.sh / run_streamlit.sh
│
├── src/
│   └── api/
│       ├── main.py                  # FastAPI: /login /logout /predict /retrain /metrics
│       ├── retrain.py               # Calls Airflow REST API to trigger training DAG
│       ├── drift.py                 # Prediction-drift PSI tracker
│       ├── reference_class_distribution.json  # Training-set class distribution (drift baseline)
│       ├── db.py                    # Postgres-backed user store
│       ├── Dockerfile
│       └── requirements.txt
│
├── streamlit_app/
│   ├── Home.py                      # Login + Prediction + Retrain (admin) UI
│   ├── project_story.py             # Walkthrough pages: Overview/Data/Tracking/Orchestration/Monitoring/Links
│   ├── Dockerfile
│   ├── requirements.txt
│   └── services/
│       └── raw_late_fusion_predictor.py  # ConvNeXt + CamemBERT fusion
│
├── monitoring/
│   ├── prometheus/prometheus.yml     # Scrapes api /metrics (docker + manual-up targets)
│   └── grafana/
│       ├── dashboards/                # Auto-provisioned "Rakuten API Overview" dashboard
│       └── provisioning/alerting/     # Prediction-drift alert rule, Slack contact point, routing policy
│
├── airflow_dst/
│   ├── docker-compose.yaml          # Airflow stack (own network, built image, docker-socket-proxy)
│   ├── Dockerfile                   # apache/airflow:2.10.0 + mlflow client + dvc (no torch — training moved to trainer/)
│   ├── requirements.txt
│   └── dags/
│       ├── rakuten_model_promotion.py    # Model promotion DAG
│       └── rakuten_model_training.py     # DVC prep + launch_trainer + trigger_promotion DAG
│
├── data/                            # Not in git
│   ├── raw/                         # Rakuten dataset (CSV + images)
│   └── rakuten_streamlit_predictor/ # Serving weights (populated by DAG)
│
└── outputs/                         # Training artifacts
```

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Model weights in `data/rakuten_streamlit_predictor/` (run Airflow DAG after training)

### Run

```bash
docker compose up --build
# or: make up
```

The `make up`, `make up-all`, `make airflow-up`, `make serve`, and
`make manual-up` commands first run `make docker-up`. On macOS this starts
Docker Desktop automatically when it is closed and waits for the Docker
daemon to become ready.

Airflow (separate stack, own network — used for model promotion after training):

```bash
cd airflow_dst && docker compose up --build
# or from repo root: make airflow-up
```

| Service | URL |
|---------|-----|
| Streamlit app | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |
| Airflow UI | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/adminadmin) |

Check all serving-stack services are healthy: `make health`

### Credentials

Users are stored in the shared Postgres instance (`api` database, `DATABASE_URL` in `docker-compose.yaml` — same Postgres server as MLflow/Airflow, one database server instead of separate SQLite files per service), created and seeded automatically on first API startup. Passwords are salted + hashed (PBKDF2-HMAC-SHA256, 100k iterations), never stored in plain text.

| Username | Password | Role |
|----------|----------|------|
| `admin` | `adminadmin` | `admin` |
| `user` | `user` | `viewer` |

The `role` field is stored per-user but not yet enforced on any endpoint — every authenticated user currently has the same access (no admin-only routes exist yet).

### DVC remote (MinIO)

`dvc push`/`dvc pull` share DVC-tracked data/models via the local MinIO instance (`s3://dvc-data`, same MinIO MLflow's artifacts use — started by `make up`). The remote's URL/endpoint (`.dvc/config`) is committed, but credentials are per-machine and gitignored (`.dvc/config.local`) — run this once after cloning:

```bash
dvc remote modify --local minio access_key_id admin
dvc remote modify --local minio secret_access_key adminadmin
```

### Run tests

```bash
# Install dev dependencies
uv sync
# or: make install

# Run API unit tests (no Docker, no model weights needed)
python -m pytest tests/ -v
# or: make test
```

### CI

Every push/PR to `main` runs via GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)):
- `test` — `uv sync` + `pytest tests/` (45 tests, mocked model assets — no GPU/weights needed)
- `docker-build` — builds the `api`, `streamlit`, and `airflow` images to catch Dockerfile breakage before merge (does not push)

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Dependency management | uv |
| Deep learning | PyTorch, Transformers (HuggingFace) |
| Image models | ConvNeXt-Base (torchvision) |
| Text models | CamemBERT (camembert-base) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Experiment tracking | MLflow (Postgres backend store, MinIO artifact store) |
| Object storage | MinIO (S3-compatible — MLflow artifacts + DVC remote) |
| Database | Postgres (shared: MLflow, Airflow metadata, API auth) |
| Orchestration | Apache Airflow (DockerOperator-launched ephemeral `trainer` container) |
| Containerisation | Docker, Docker Compose |
| CI | GitHub Actions |
| Monitoring | Prometheus + Grafana, PSI-based prediction-drift alerting → Slack |

---

## Roadmap

Known gaps, not yet built:

| Item | What it would add |
|------|--------------------|
| Input-feature drift | Compare live input (text/image) distribution against the training set, not just predicted classes (see [Monitoring & Drift Detection](#monitoring--drift-detection)) |
| Live model-performance decay | Score accuracy/F1 against real ground truth once labels become available, instead of only offline eval at training time |
| Pipeline-failure alerting | Airflow DAG failures (training/promotion) don't currently notify anyone — no `on_failure_callback`/retries configured |
| Data-quality checks | No runtime validation of missing values or unexpected `prdtypecode`/schema in `prepare_splits.py` |
| Infra-resource alerts | Docker healthchecks exist, but disk usage / container OOM aren't scraped by Prometheus (no node-exporter/cadvisor) |

---

## Dataset

**Rakuten France Multimodal Product Data Classification**  
- 84 916 training samples, 13 812 test samples  
- 27 product categories  
- ~35% missing descriptions  
- Raw data and images are stored locally under `data/raw/` (not committed to git)

## Manuel start

.venv/bin/uvicorn src.api.main:app --reload --port 8000
.venv/bin/streamlit run streamlit_app/Home.py
