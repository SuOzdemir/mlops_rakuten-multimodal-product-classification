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
| API test coverage | Engineering | 22 unit tests, gated in CI on every push/PR |
| Serving latency / uptime / drift | Ops | Not yet instrumented — no Prometheus/Grafana or drift monitoring in place |

See [docs/methodology_phase1.md](docs/methodology_phase1.md) for modeling detail and [docs/methodology_phase2.md](docs/methodology_phase2.md) for the MLOps pipeline.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      docker-compose                          │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  Streamlit  │───▶│  FastAPI     │    │     MLflow      │ │
│  │   :8501     │    │   :8000      │───▶│     :5000       │ │
│  └─────────────┘    └──────────────┘    └─────────────────┘ │
└──────────────────────────────────────────────────────────────┘
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

### Model promotion (Airflow DAG)

After training is complete, the Airflow DAG `rakuten_model_promotion` copies the best model weights to the serving directory consumed by the FastAPI service:

```
Training outputs                   Serving directory
──────────────────────────────     ──────────────────────────────────────
models/I12_ConvNeXt_Base/      →   data/rakuten_streamlit_predictor/
  best_model_state_dict.pt           image_model/best_model_base.pt
models/T8_CamemBERT/           →     text_model/best_model_text.pt
  best_model_text.pt                 text_model/camembert_base/
outputs/image_modeling/        →   label2id.json
  label2id.json                    prdtypecode_mapping.json
```

The DAG runs manually (triggered after training) and validates the serving directory before finishing.

### Retrain pipeline (Airflow DAG + DVC + API)

`POST /retrain` (admin only) doesn't run training itself — it triggers the Airflow DAG `rakuten_model_training` via Airflow's REST API, so the long-running job lives in Airflow (retries, logs, concurrency all handled there) instead of inside the FastAPI process.

```
POST /retrain {model: image|text}   (admin)
   → API calls Airflow REST API: POST /api/v1/dags/rakuten_model_training/dagRuns
        ↓
   DAG: prepare_data (dvc repro prepare_splits)
        ↓
   DAG: train_model  (python -m models.<...>.train, model from dag_run conf)

GET /retrain            → list all dag runs (any authenticated user)
GET /retrain/{job_id}   → poll a specific dag run's status
```

- **`scripts/prepare_splits.py`** builds `outputs/image_modeling/{train_split,val_split}.csv` + `label2id.json` from the raw Kaggle CSVs (`data/raw/X_train.csv`, `Y_train.csv`) — both I12 and T8 read from the same split so image/text models are evaluated on identical held-out products.
- **`dvc.yaml`** wraps that script as a pipeline stage (`prepare_splits`) — `dvc repro` only re-runs it if the raw data or script changed.
- The Airflow container reaches the API's host via `AIRFLOW_API_URL` (default `http://host.docker.internal:8080`, since `airflow_dst` is a separate compose stack on its own network); Basic Auth uses the `admin`/`admin` user created by `airflow-init`.
- Requires the raw Kaggle dataset in `data/raw/` (`./scripts/setup_data.sh`) — without it, `dvc repro` fails at the `prepare_data` task, which is expected until the dataset is downloaded on the machine running Airflow.

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
├── docker-compose.yaml              # MLflow + API + Streamlit + Prometheus + Grafana
├── pyproject.toml                   # Python dependencies (uv) — dev/training env
├── Makefile                         # make up / airflow-up / test / health shortcuts
├── .github/workflows/ci.yml         # pytest + Docker build checks on push/PR
├── dvc.yaml                         # DVC pipeline: raw CSVs -> train/val splits
├── .dvc/                            # DVC metadata (dvc init)
│
├── models/
│   ├── Model_I12_ConvNeXt_Base_*/   # Image model training
│   └── textModeling/
│       └── Model_T8_CamemBERT_*/    # Text model training
│
├── scripts/
│   ├── setup_data.sh                # Downloads raw Kaggle dataset -> data/raw/
│   ├── prepare_splits.py            # DVC stage: builds train/val split CSVs + label2id.json
│   ├── run_api.sh / run_streamlit.sh
│
├── src/
│   └── api/
│       ├── main.py                  # FastAPI: /login /logout /predict /retrain
│       ├── retrain.py               # Calls Airflow REST API to trigger training DAG
│       ├── db.py                    # SQLite user store
│       ├── Dockerfile
│       └── requirements.txt
│
├── streamlit_app/
│   ├── Home.py                      # Login + Prediction UI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── services/
│       └── raw_late_fusion_predictor.py  # ConvNeXt + CamemBERT fusion
│
├── monitoring/
│   ├── prometheus/prometheus.yml     # Scrapes api /metrics (docker + manual-up targets)
│   └── grafana/                      # Auto-provisioned datasource + "Rakuten API Overview" dashboard
│
├── airflow_dst/
│   ├── docker-compose.yaml          # Airflow stack (own network, built image)
│   ├── Dockerfile                   # apache/airflow:2.10.0 + mlflow client + dvc + torch(cpu)
│   ├── requirements.txt
│   └── dags/
│       ├── rakuten_model_promotion.py    # Model promotion DAG
│       └── rakuten_model_training.py     # DVC prep + train DAG (triggered by /retrain)
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
| Grafana | http://localhost:3000 (admin/admin) |

Check all serving-stack services are healthy: `make health`

### Credentials

Users are stored in a SQLite DB (`config/users.db`, gitignored), created and seeded automatically on first API startup. Passwords are salted + hashed (PBKDF2-HMAC-SHA256), never stored in plain text.

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin` | `admin` |
| `user` | `user` | `viewer` |

The `role` field is stored per-user but not yet enforced on any endpoint — every authenticated user currently has the same access (no admin-only routes exist yet).

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
- `test` — `uv sync` + `pytest tests/` (22 tests, mocked model assets — no GPU/weights needed)
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
| Experiment tracking | MLflow |
| Orchestration | Apache Airflow |
| Containerisation | Docker, Docker Compose |
| CI | GitHub Actions |
| Monitoring | Not yet implemented — see [Roadmap](#roadmap) |

---

## Roadmap

Known gaps, not yet built:

| Item | What it would add |
|------|--------------------|
| Drift detection (e.g. Evidently) | Compare live prediction/input distribution against the training set to catch model decay |
| MLflow ↔ Airflow (promotion) | Promotion DAG picks the best run from MLflow (Model Registry) instead of a fixed local file path — separate from the retrain DAG's `/retrain` trigger, which is already wired |
| DVC remote | `dvc.yaml` works locally (cache-based reproducibility) but has no configured remote yet — needed to `dvc push`/`pull` data across machines/CI |

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
