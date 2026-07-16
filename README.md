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
| API test coverage | Engineering | 55 unit tests, gated in CI on every push/PR |
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

After training, `rakuten_model_promotion` launches a temporary promotion
container rather than loading models in the Airflow scheduler. It pushes the
retrained checkpoint to the DVC remote, assembles a complete multimodal serving
bundle and logs it as an MLflow `pyfunc` candidate named
`rakuten-multimodal-classifier`. The candidate receives the `champion` alias only
when its retrained component's validation Macro-F1 passes the configured gate.
The container then downloads `models:/rakuten-multimodal-classifier@champion`
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
├── promotion/                        # Ephemeral DVC push + MLflow candidate gate/deployment
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── promote.py
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

## Step-by-step setup

### 1. Prerequisites

Install the following tools before cloning the project:

- Git
- Docker Desktop (or Docker Engine + Compose v2)
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for the local Python environment
- Kaggle credentials only if the raw dataset will be downloaded

The API can start without model weights, but `/predict` returns `503` until a complete
serving bundle exists under `data/rakuten_streamlit_predictor/`.

### 2. Clone and configure

```bash
git clone git@github.com:SuOzdemir/mlops_rakuten-multimodal-product-classification.git
cd mlops_rakuten-multimodal-product-classification
cp .env.example .env
make install
```

Edit `.env` only on the machine that runs the stack. It is gitignored and must never be
committed. SMTP and Slack values are optional; the stack starts without them.

### 3. Download the dataset (optional for API unit tests)

Put the Kaggle token at `~/.kaggle/kaggle.json`, then run:

```bash
make setup-data
make prepare-splits
```

`setup-data` downloads the raw CSV and image files into `data/raw/`.
`prepare-splits` executes the DVC `prepare_splits` stage and creates the shared image/text
train-validation split under `outputs/image_modeling/`.

### 4. Start the serving stack

```bash
make up
```

This starts Postgres, Adminer, MinIO, MLflow, API, Streamlit, Prometheus and Grafana.
Compose waits for dependency health checks, so API and Streamlit do not race Postgres.
On macOS, the Makefile starts Docker Desktop first if necessary and waits up to two minutes.

To include the separate Airflow stack:

```bash
make up-all
```

Airflow is intentionally a second Compose project. It reaches the main Postgres and MLflow
services through `host.docker.internal` and launches training containers through the scoped
Docker socket proxy.

### 5. Verify the services

```bash
make health
docker compose ps
cd airflow_dst && docker compose ps
```

| Service | URL / credentials |
|---------|-------------------|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| OpenAPI docs | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Airflow | http://localhost:8080 (`AIRFLOW_API_USER` / `AIRFLOW_API_PASSWORD`) |
| Adminer | http://localhost:8081 |
| MinIO console | http://localhost:9001 (`MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`) |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (`GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`) |

Application users are seeded by the API on first startup. Passwords are stored as salted
PBKDF2-HMAC-SHA256 hashes in the `api` database.

| Username | Password | Role |
|----------|----------|------|
| `admin` | Value of `API_ADMIN_PASSWORD` | `admin` |
| `user` | Value of `API_VIEWER_PASSWORD` | `viewer` |

The admin role may trigger `/retrain`; viewer accounts may inspect retrain jobs but cannot
start one. These credentials are local-development defaults and must be replaced before a
public deployment.

## Database initialization

The root Compose stack runs one PostgreSQL 15 server and separates service data into three
databases:

| Database | Owner | Used by |
|----------|-------|---------|
| `mlflow` | `mlflow` | MLflow runs and Model Registry metadata |
| `airflow` | `airflow` | Airflow metadata, DAG runs and task state |
| `api` | `api` | API users and authentication state |

On the **first** Postgres start, Docker mounts
`scripts/postgres_init/create_databases.sql` into `/docker-entrypoint-initdb.d/`. The official
Postgres entrypoint executes it and creates all three users and databases. The API creates
its own tables and seed users afterward; `airflow-init` runs migrations and creates the
Airflow admin account.

Database passwords are read from the local `.env`; no database password is stored in Compose or
the SQL file. The SQL init directory is processed only when `postgres_data` is empty. Editing the
SQL file or `.env` does not rotate passwords in an existing database. After changing database
passwords, either rotate each PostgreSQL role explicitly or intentionally recreate the development
volume as described below. Inspect the result with Adminer, or from the terminal:

```bash
docker compose exec postgres psql -U postgres -c '\\l'
docker compose exec postgres psql -U postgres -c '\\du'
```

For a normal restart, preserve the database:

```bash
make down
make up
```

To intentionally rebuild all local Postgres data, remove only its named volume. This deletes
MLflow history, Airflow metadata and API users and cannot be undone:

```bash
make down-all
docker volume ls | grep postgres_data
# Confirm the exact name shown above, then:
docker volume rm <compose-project>_postgres_data
make up-all
```

Do not use `docker compose down -v` casually: it also removes MinIO, Prometheus and Grafana
volumes.

## MinIO, MLflow and DVC storage

`minio-init` is a one-shot Compose service. Once MinIO is healthy, it idempotently creates:

- `mlflow-artifacts` for MLflow model/run artifacts;
- `dvc-data` for DVC-tracked datasets and model outputs.

The committed `.dvc/config` points at the local `dvc-data` bucket. Credentials belong in the
gitignored `.dvc/config.local`; configure them once per machine:

```bash
uv run dvc remote modify --local minio access_key_id "$MINIO_ROOT_USER"
uv run dvc remote modify --local minio secret_access_key "$MINIO_ROOT_PASSWORD"
uv run dvc pull   # download DVC-tracked files
uv run dvc push   # upload new DVC-tracked files
```

This MinIO instance is a local-development remote, not a team-wide or production backup.
For multiple machines, point DVC at a reachable S3/MinIO bucket and store its credentials in
CI secrets or the host secret manager.

## Secrets and passwords

All Postgres, MinIO, Grafana, Airflow and initial API-user credentials are required through the
gitignored root `.env`. Start from the committed variable-name template:

```bash
cp .env.example .env
openssl rand -hex 32   # run separately for every password variable
```

Replace every `replace-with-...` value; never reuse one password across services. The Compose
configuration fails immediately with a clear message when a required variable is absent. The
separate Airflow Compose project must be invoked through the Makefile, or explicitly with the same
root env file:

```bash
cd airflow_dst
docker compose --env-file ../.env up --build -d
```

`.env` is appropriate for local development. In production, inject the same variables from the
cloud/platform secret manager instead of copying a plaintext `.env` onto the server.

## Makefile command reference

Run these commands from the repository root:

| Command | What it does |
|---------|--------------|
| `make install` | Creates/synchronizes the Python 3.12 environment from `uv.lock`. |
| `make setup-data` | Downloads the Kaggle dataset into `data/raw/`. |
| `make prepare-splits` | Runs `dvc repro prepare_splits`. |
| `make docker-up` | Verifies Docker; on macOS starts Docker Desktop and waits for it. |
| `make up` | Builds and starts the complete always-on serving/monitoring stack. |
| `make serve` | Starts only API and Streamlit (plus required dependencies), without MLflow. |
| `make airflow-up` | Builds and starts the separate Airflow stack. Start the root stack first. |
| `make up-all` | Runs `make up`, then `make airflow-up`. |
| `make down` | Stops the root Compose stack without deleting named volumes. |
| `make airflow-down` | Stops the Airflow stack. |
| `make down-all` | Stops both Compose stacks and preserves data volumes. |
| `make restart-all` | Stops and starts both stacks. |
| `make logs` | Follows root-stack container logs. |
| `make airflow-logs` | Follows Airflow logs. |
| `make health` | Checks MLflow, API and Streamlit health endpoints. |
| `make test` | Runs the complete pytest suite through `uv`. |
| `make manual-up` | Runs MLflow, API and Streamlit as local processes; Postgres still uses Docker. |
| `make manual-down` | Stops processes created by `manual-up`. |
| `make manual-up-all` | Starts manual serving processes plus Dockerized Airflow. |
| `make manual-down-all` | Stops manual serving processes and Airflow. |

`manual-up` and `make up` use the same host ports and deliberately refuse to run together.
Use Docker mode for the reproducible full stack; use manual mode while debugging Python code.

## Tests

```bash
make install
make test
# More detailed output:
uv run pytest tests/ -v
```

The suite currently contains **55 tests** across five files:

| File | Coverage |
|------|----------|
| `tests/test_api.py` | Health, login/logout, prediction, auth and retrain endpoints |
| `tests/test_drift.py` | Reference distribution, PSI thresholds and rolling window |
| `tests/test_predictor_architecture.py` | ConvNeXt serving/training head compatibility |
| `tests/test_retrain.py` | Airflow client mapping, conflicts and retrain state |
| `tests/test_promotion.py` | Candidate metric parsing, bootstrap and champion promotion gates |

Tests mock external model assets and Airflow calls; they need neither GPU, production weights,
nor a running Docker stack.

## GitHub Actions: CI and image publishing

The workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### What is stored where?

| Artifact | Destination | How it gets there |
|----------|-------------|-------------------|
| Git source code and configuration | **GitHub repository** | `git push origin <branch>` |
| Docker images | **GitHub Container Registry (GHCR)** | GitHub Actions publishes them after a successful `main` push |
| Datasets and model checkpoints | **DVC remote on MinIO/S3** | `uv run dvc push` |
| MLflow run and Model Registry artifacts | **MinIO** | MLflow writes them to the `mlflow-artifacts` bucket |

In short: GitHub stores the code, GHCR stores deployable containers, DVC/MinIO/S3
stores large versioned data and model files, and MLflow uses MinIO for experiment and
Registry artifacts. These files must not be mixed together or committed to Git.

### Pull request flow

1. Create a feature branch and push it to the GitHub `origin` remote.
2. Open a pull request targeting `main`.
3. GitHub Actions installs the locked dependencies and runs all 55 tests.
4. Only after tests pass, it builds `api`, `streamlit`, `airflow`, `trainer` and `promotion` images.
5. PR builds do **not** push images to a registry.
6. Require this workflow in the `main` branch protection rules before merging.

```bash
git checkout -b feature/my-change
git add .
git commit -m "Describe the change"
git push -u origin feature/my-change
```

### Main branch publish flow

After the pull request is merged, the `push` event on `main` repeats tests and builds. If all
checks pass, the workflow authenticates to GitHub Container Registry (GHCR) with the automatic
`GITHUB_TOKEN` and publishes five packages:

```text
ghcr.io/suozdemir/rakuten-api:latest
ghcr.io/suozdemir/rakuten-streamlit:latest
ghcr.io/suozdemir/rakuten-airflow:latest
ghcr.io/suozdemir/rakuten-trainer:latest
ghcr.io/suozdemir/rakuten-promotion:latest
```

Every image also receives an immutable `sha-<commit>` tag. Deploy the SHA tag for reproducible
releases; use `latest` only for a convenient development environment. GHCR creates each package
automatically on its first successful push—there is no empty registry repository to create by
hand. The job has only `contents: read` and `packages: write` permissions.

In GitHub, open the repository **Actions** tab to approve/run the workflow and the profile or
organization **Packages** page to see the images. Set package visibility to public if anonymous
pulls are desired; otherwise keep it private and authenticate deployment machines.

### Which remote receives what?

- Source code goes to the existing Git remote: `git push origin <branch>` → GitHub repository.
- Docker images go to GHCR: `docker push ghcr.io/suozdemir/rakuten-<service>:<tag>`.
- DVC data/model files go to the configured DVC/S3 remote: `uv run dvc push`.
- MLflow artifacts go to MinIO through the MLflow server; they do not belong in Git or GHCR.

These stores solve different problems. Never place datasets, model checkpoints or `.env` secrets
inside the Git repository or application images.

### Pulling images on a remote server

The deployment destination should be a controlled Linux host (for example an AWS EC2, GCP VM,
Azure VM or another VPS) with Docker Compose installed. For private packages, create a GitHub
fine-grained/classic token with at least `read:packages`, store it in the server's secret manager,
and log in once:

```bash
export CR_PAT='<token-with-read-packages>'
echo "$CR_PAT" | docker login ghcr.io -u SuOzdemir --password-stdin
docker pull ghcr.io/suozdemir/rakuten-api:sha-<commit>
docker pull ghcr.io/suozdemir/rakuten-streamlit:sha-<commit>
```

The current `docker-compose.yaml` is optimized for local development and contains `build:`
definitions. A production deployment should use a separate Compose override (or Kubernetes
manifests) whose `image:` values reference the immutable GHCR tags. Keep Postgres/MinIO data in
persistent managed storage, provide secrets outside Git, and make the model bundle available via
DVC/object storage before starting the API. Publishing an image is not itself a deployment; the
remote host still needs an explicit pull/restart step.

The repository currently publishes images but intentionally does not SSH into a server. Add that
deployment job only after the target host, domain/TLS strategy, secret manager and rollback policy
have been chosen.

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
