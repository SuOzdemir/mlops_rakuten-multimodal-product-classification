# Methodology — Phase 2: MLOps

## Objective

Take the best models from Phase 1 (CamemBERT + ConvNeXt-Base late fusion) and build a production-ready serving pipeline with experiment tracking, automated model promotion, a REST API, and a web interface.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        docker-compose.yaml                      │
│                                                                 │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │  Streamlit   │───▶│   FastAPI     │    │     MLflow      │  │
│  │  :8501       │    │   :8000       │    │     :5001       │  │
│  └──────────────┘    └───────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

FastAPI no longer depends on MLflow at request time (see §6) — the arrow
above is dropped intentionally; MLflow is only used during training.

Training pipeline (separate, runs locally or on GPU machine):
  train.py → MLflow tracking → Airflow DAG → serving directory

Local dev without Docker (hot-reload on every save):
  ./scripts/run_api.sh        # uvicorn --reload, :8000
  ./scripts/run_streamlit.sh  # streamlit run, :8501
```

---

## 1. Experiment Tracking — MLflow

Both training scripts (`train.py` for I12 and T8) log to MLflow at `http://localhost:5000`.

### What is logged

| Type | Examples |
|------|---------|
| Parameters | `lr_backbone`, `lr_head`, `batch_size`, `max_epochs`, `dropout`, `label_smoothing`, `image_size` |
| Metrics | `train_loss`, `val_loss`, `val_accuracy`, `val_f1` — logged per epoch |
| Artifacts | confusion matrix PNG, learning curves PNG, classification report TXT, model metadata JSON |

### Setup in train.py

```python
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("rakuten-text-modeling")   # or "rakuten-image-modeling"

with mlflow.start_run(run_name=config.RUN_NAME):
    mlflow.log_params({...})
    # per epoch:
    mlflow.log_metrics({"val_f1": ..., "val_loss": ...}, step=epoch)
    mlflow.log_artifact("confusion_matrix.png")
```

### MLflow UI

Access at `http://localhost:5000` after `docker compose up`.

Use cases:
- Compare all runs across model families (CNN, ResNet, ConvNeXt, CamemBERT)
- Filter by macro F1 to identify the best checkpoint
- Download artifacts for inspection
- Track which hyperparameter changes had the most impact

---

## 2. Model Promotion — Airflow DAG

After training, model weights must be copied from the training output directory to the serving directory used by the FastAPI service. This is automated by the Airflow DAG `rakuten_model_promotion`.

### DAG: `rakuten_model_promotion`

- **Schedule:** None (triggered manually after training completes)
- **Location:** `airflow_dst/dags/rakuten_model_promotion.py`

### Task graph

```
check_artifacts
      │
      ├──▶ promote_image_model  ──┐
      ├──▶ promote_text_model   ──┤──▶ validate_serving_dir
      ├──▶ promote_tokenizer    ──┤
      └──▶ promote_label_maps   ──┘
```

### What each task does

| Task | Source | Destination |
|------|--------|-------------|
| `check_artifacts` | Verifies all required training outputs exist | — |
| `promote_image_model` | `models/I12_ConvNeXt_Base_*/best_model_state_dict.pt` | `data/rakuten_streamlit_predictor/image_model/best_model_base.pt` |
| `promote_text_model` | `models/T8_CamemBERT_*/best_model_text.pt` | `data/rakuten_streamlit_predictor/text_model/best_model_text.pt` |
| `promote_tokenizer` | Local `camembert-base` directory | `data/rakuten_streamlit_predictor/text_model/camembert_base/` |
| `promote_label_maps` | `outputs/image_modeling/label2id.json` | `data/rakuten_streamlit_predictor/label2id.json` |
| `validate_serving_dir` | Checks all required serving files exist and are non-empty | — |

### Trigger

```bash
# From Airflow UI or CLI:
airflow dags trigger rakuten_model_promotion
```

### Run Airflow

```bash
cd airflow_dst
docker compose up
# UI at http://localhost:8080
```

---

## 3. Model Retraining — DVC + Airflow DAG + REST API

Unlike promotion (manual, post-training), retraining is triggered from the FastAPI service and runs entirely inside Airflow — the API never runs training itself, it only calls Airflow's REST API and polls status.

### Flow

```
POST /retrain {model: image|text}     (admin only, src/api/main.py)
   → src/api/retrain.py: requests.post to Airflow REST API
        POST http://<airflow>/api/v1/dags/rakuten_model_training/dagRuns
        body: {"dag_run_id": "retrain_<model>_<uuid>", "conf": {"model": "<model>"}}
        ↓
   DAG rakuten_model_training (airflow_dst/dags/rakuten_model_training.py):
        prepare_data:  dvc repro prepare_splits
        train_model:   dvc repro train_image | train_text   (stage picked from conf.model)

GET /retrain            -> lists all dag runs for rakuten_model_training
GET /retrain/{job_id}   -> polls one dag run (job_id == dag_run_id)
```

### DVC — `dvc.yaml`

```yaml
stages:
  prepare_splits:
    cmd: python scripts/prepare_splits.py
    deps: [scripts/prepare_splits.py, data/raw]
    outs: [outputs/image_modeling/train_split.csv, outputs/image_modeling/val_split.csv, outputs/image_modeling/label2id.json]

  train_image:
    cmd: python -m models.Model_I12_ConvNeXt_Base_ModerateAug_Full.train
    deps: [outputs/image_modeling/train_split.csv, outputs/image_modeling/val_split.csv, outputs/image_modeling/label2id.json, models/Model_I12_ConvNeXt_Base_ModerateAug_Full/{train.py,config.py,dataset.py,utils.py}]
    outs: [models/Model_I12_ConvNeXt_Base_ModerateAug_Full/best_model_state_dict.pt]

  train_text:
    cmd: python -m models.textModeling.Model_T8_CamemBERT_FullFineTune_L128.train
    deps: [outputs/image_modeling/train_split.csv, outputs/image_modeling/val_split.csv, outputs/image_modeling/label2id.json, models/textModeling/Model_T8_CamemBERT_FullFineTune_L128/{train.py,config.py,dataset.py,utils.py}]
    outs: [models/textModeling/Model_T8_CamemBERT_FullFineTune_L128/best_model_text.pt]
```

`scripts/prepare_splits.py` merges `X_train.csv` + `Y_train.csv` (from `scripts/setup_data.sh`), builds `image_path_local` (`image_<imageid>_product_<productid>.jpg`) and `label_id` (encoded `prdtypecode`), and writes a single stratified train/val split — shared by both I12 and T8 so image and text models are evaluated on the same held-out products.

`train_image`/`train_text` depend on the split files (and each model's own source files) and output the trained weight (`.pt`, gitignored — DVC tracks its content hash independently of git, exactly the case DVC is for). `dvc repro <stage>` recomputes dep hashes first: if nothing changed since the last successful run, it skips the stage entirely; if the split data or script changed, it reruns and records the new output hash in `dvc.lock` — that hash is what `dvc diff`/`dvc status`/`dvc push` operate on. Before this, DVC had no visibility into the model weights at all.

No DVC remote is configured yet — `dvc repro`/`dvc status` work locally via the DVC cache; `dvc push`/`pull` (for sharing data/model blobs across machines) is future work.

### Airflow REST API access

`airflow_dst/docker-compose.yaml` sets `AIRFLOW__API__AUTH_BACKENDS: airflow.api.auth.backend.basic_auth` so the stable REST API accepts Basic Auth against the `admin`/`admin` user created by `airflow-init`. The `api` service reaches it via `AIRFLOW_API_URL` (default `http://host.docker.internal:8080` — `airflow_dst` is a separate compose stack, own network, same cross-network pattern already used for `MLFLOW_TRACKING_URI` in the promotion DAG).

### Status mapping

`src/api/retrain.py` maps Airflow's DAG-run `state` to a simpler API-facing `status`:

| Airflow state | API status |
|---|---|
| `queued` | `queued` |
| `running` | `running` |
| `success` | `completed` |
| `failed` | `failed` |

A `RuntimeError` (-> HTTP 409) is raised client-side if a run for the same model is already `queued`/`running`, checked via `GET /retrain` before triggering. If Airflow itself is unreachable, `requests` raises a connection error that the API turns into HTTP 503.

### Known limitation

Both `models/*/train.py` need the raw Kaggle dataset in `data/raw/` (`./scripts/setup_data.sh`) to get past `prepare_data`. Without it, `dvc repro` fails there — this is a data-availability gap, not a code bug (see README Known Gaps).

---

## 4. Prediction Service — FastAPI

**Location:** `src/api/main.py`  
**Port:** 8000  
**Docs:** `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | None | Health check |
| `POST` | `/login` | None | Validate credentials, return Bearer token |
| `POST` | `/logout` | Bearer | Invalidate token |
| `POST` | `/predict` | Bearer | Multimodal prediction |

### Authentication flow

```
POST /login
  body: username=admin&password=adminadmin
  response: {"access_token": "<token>", "token_type": "bearer"}

POST /predict
  header: Authorization: Bearer <token>
  body: designation=...&description=...&image=<file>
  response: {"mode": "Late fusion", "top3": [...]}
```

Tokens are stored in memory (`dict[token → {"username", "role"}]`). On restart, all sessions are cleared.

### User store — SQLite (`src/api/db.py`)

Users used to live in a plaintext `config/credentials.json`. They are now stored in a SQLite DB (`config/users.db`, gitignored):

- `init_db()` creates the `users` table on first API startup and seeds `admin`/`admin` (role `admin`) and `user`/`user` (role `viewer`) if the table is empty.
- Passwords are never stored in plain text — each user has a random 16-byte salt, and the password is hashed with PBKDF2-HMAC-SHA256 (100k iterations) before being written.
- `get_user(username)` / `verify_password(password, user)` are the two functions `/login` calls.
- The `role` column exists but is **not yet enforced** on any endpoint — every authenticated user has identical access today.

### Prediction modes

The predictor automatically selects the mode based on available inputs:

| Input | Mode | Model used |
|-------|------|-----------|
| text + image | Late fusion | CamemBERT + ConvNeXt-Base |
| text only | Text only | CamemBERT |
| image only | Image only | ConvNeXt-Base |

### Late fusion formula

```
final_logits = image_weight × image_logits + text_weight × text_logits
top3 = softmax(final_logits).argsort()[::-1][:3]
```

Default weights: `image_weight=0.45`, `text_weight=0.55` (tuned in Phase 1).

### Predictor module

`streamlit_app/services/raw_late_fusion_predictor.py`

Shared between the API and (optionally) the Streamlit app. Loads models once at startup via `load_assets()` and exposes a single `predict()` function.

Assets loaded at startup:
- ConvNeXt-Base weights: `data/rakuten_streamlit_predictor/image_model/best_model_base.pt`
- CamemBERT weights: `data/rakuten_streamlit_predictor/text_model/best_model_text.pt`
- CamemBERT tokenizer: `data/rakuten_streamlit_predictor/text_model/camembert_base/` (offline, no HuggingFace download)
- Label maps: `data/rakuten_streamlit_predictor/label2id.json`
- Category names: `data/rakuten_streamlit_predictor/prdtypecode_mapping.json`

---

## 5. Web Interface — Streamlit

**Location:** `streamlit_app/Home.py`  
**Port:** 8501  
**API dependency:** `API_URL` environment variable (default: `http://localhost:8000`)

### Screens

**Login screen**
- Calls `POST /login` via HTTP
- Stores Bearer token in `st.session_state`

**Prediction screen**
- Optional CSV upload for automatic text pre-fill (matches image filename to CSV row)
- Image upload (JPG/PNG)
- Manual designation + description fields
- Calls `POST /predict` with Bearer token
- Displays mode, top 1 result, and top 3 table

**Logout**
- Calls `POST /logout` to invalidate server-side token
- Clears session state

### Streamlit does NOT load models

The Streamlit container has no PyTorch or Transformers dependency. All inference happens in the API container. This keeps the Streamlit image small and the frontend stateless.

---

## 6. Containerisation

### docker-compose services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mlflow` | `ghcr.io/mlflow/mlflow` | 5000 | Experiment tracking UI + backend store |
| `api` | Built from `src/api/Dockerfile` | 8000 | Model inference + auth |
| `streamlit` | Built from `streamlit_app/Dockerfile` | 8501 | Web interface |

### Dependency chain

```
api ← streamlit
```

`api` no longer waits on `mlflow` (removed — the API never calls MLflow at request time, so the `depends_on` was a false dependency). `mlflow` can be started/stopped independently: `docker compose up mlflow` or `docker compose up api streamlit` (a.k.a. `make serve`).

### Data volume

```yaml
volumes:
  - ./data/rakuten_streamlit_predictor:/project/data/rakuten_streamlit_predictor:ro   # model weights, read-only
  - ./config:/project/config                                                          # users.db, read-write
```

The Streamlit container has no volume mount — it communicates entirely through the API.

### Per-service requirements.txt

Each buildable image installs only what it needs (not the full `pyproject.toml` dev/training environment):

| Service | Dependency file |
|---------|------------------|
| `src/api/Dockerfile` | `src/api/requirements.txt` (+ CPU-only torch/torchvision installed inline) |
| `streamlit_app/Dockerfile` | `streamlit_app/requirements.txt` |
| `airflow_dst/Dockerfile` | `airflow_dst/requirements.txt` (adds `mlflow` client on top of `apache/airflow:2.10.0`) |

Airflow builds its own image with the MLflow client. The promotion DAG registers a complete pyfunc bundle, assigns the `champion` alias, downloads that Registry version, and atomically installs its assets in the serving directory.

---

## 7. End-to-End Pipeline

```
1. TRAIN
   python models/Model_I12_.../train.py     # logs to MLflow
   python models/textModeling/Model_T8_.../train.py  # logs to MLflow

2. INSPECT
   MLflow UI → http://localhost:5001
   Compare runs, confirm best checkpoints

3. PROMOTE
   Airflow DAG: rakuten_model_promotion
   Registers the complete bundle and deploys models:/rakuten-multimodal-classifier@champion

4. SERVE
   docker compose up --build      # or: make up (incl. mlflow) / make serve (api+streamlit only)
   API loads weights at startup (lifespan event)
   API also creates+seeds config/users.db on first startup (see §4)

5. USE
   http://localhost:8501  →  Login  →  Predict
```

CI (`.github/workflows/ci.yml`) runs `pytest` + a Docker build check for `api`/`streamlit`/`airflow` on every push/PR to `main`.

---

## 8. Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `MLFLOW_TRACKING_URI` | trainer | `http://mlflow:5001` | Required experiment-tracking endpoint |
| `MLFLOW_TRACKING_URI` | airflow DAG | `http://host.docker.internal:5001` | Registry registration and champion deployment endpoint |
| `USERS_DB_PATH` | api | `config/users.db` | SQLite user store location |
| `AIRFLOW_API_URL` | api | `http://host.docker.internal:8080` | Airflow REST API base URL, used by `/retrain` |
| `AIRFLOW_API_USER` / `AIRFLOW_API_PASSWORD` | api | `admin` / `adminadmin` | Basic Auth credentials for the Airflow REST API |
| `API_URL` | streamlit | `http://localhost:8000` | FastAPI server URL |
| `RAKUTEN_PROJECT_DIR` | airflow DAG | `/project` | Project root in container |
| `CAMEMBERT_BASE_DIR` | airflow DAG | `streamlit_app/models/camembert_run4` | Path to local CamemBERT base files |
