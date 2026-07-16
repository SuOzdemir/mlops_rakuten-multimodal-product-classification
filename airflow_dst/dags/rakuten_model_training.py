"""
rakuten_model_training
=======================
Prepares the train/val data splits (via DVC, in this container) and retrains
one of the two unimodal models (image or text) in a fresh, ephemeral
`trainer` container (see trainer/Dockerfile, docker-compose.yaml). Triggered
per-model via the Airflow REST API by the FastAPI `/retrain` endpoint (see
src/api/retrain.py) — `model` selects the script and `epochs` sets the
maximum number of training epochs.

Run order:
  prepare_data      (dvc repro prepare_splits, in this container)
      ↓
  launch_trainer    (DockerOperator: launches a new `trainer` container, which
                     runs dvc repro train_image|train_text, then exits and is
                     removed)
      ↓
  trigger_promotion (only on launch_trainer success: triggers
                     rakuten_model_promotion and waits for it, so a
                     successful retrain's weights land in the serving dir
                     without a separate manual step)

Why a separate container instead of a subprocess in this one: training a
full-size ConvNeXt-Base/CamemBERT model needs several GB of RAM. Running it
as a subprocess of the scheduler meant it competed for memory with every
other always-on service in the same container -- OOM-killed under any real
load. The trainer container gets its own mem_limit/shm_size (see
docker-compose.yaml), launched fresh per run and discarded after, and its
model/data never touch this container's disk (mlflow, pointed at MinIO,
i.e. s3-compatible storage, holds the artifact; DVC's own state lives on
the shared ../:/project mount both containers see).

Airflow talks to docker-socket-proxy (also in this compose file), not the
raw host Docker socket -- it can create/start/stop containers (what
DockerOperator needs) but can't touch volumes, exec into unrelated
containers, or otherwise control the whole host the way raw socket access
would.

Trigger examples
-----------------
  CLI  : airflow dags trigger rakuten_model_training --conf '{"model": "text", "epochs": 3}'
  UI   : Trigger DAG w/ config -> {"model": "image", "epochs": 3}
  REST : POST /api/v1/dags/rakuten_model_training/dagRuns
         body: {"conf": {"model": "text"}}

  launch_trainer always runs `dvc repro --force --single-item` inside the
  trainer container: a manual retrain should always actually retrain, even
  if DVC sees no changed deps since the last run. --single-item scopes
  --force to the train stage only, so it doesn't also re-run prepare_splits
  (already done by prepare_data).

  Epoch count is supplied per run. TRAIN_ROWS_OVERRIDE / VAL_ROWS_OVERRIDE
  can still be set in this DAG's environment to bound the data size.

Environment variables
---------------------
  RAKUTEN_PROJECT_DIR : project root as seen inside *this* container (default: /project)
  HOST_PROJECT_DIR    : project root as a real path on the HOST machine
                         (required -- DockerOperator needs this for the
                         trainer's bind mount; see the comment above)
  TRAINER_IMAGE        : trainer image tag (default: the compose-built one)
  TRAINER_NETWORK      : docker network the trainer container joins, so it
                          can reach mlflow/postgres/minio (default: the main
                          stack's compose network)
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_DIR = Path(os.environ.get("RAKUTEN_PROJECT_DIR", "/project"))

# DockerOperator talks to the HOST's Docker daemon (via docker-socket-proxy)
# to create a sibling container, not a child of this one -- so a bind mount
# source must be a path that exists on the HOST, not PROJECT_DIR (which is
# only valid inside *this* container, via its own ../:/project mount).
# There's no way to derive the host path from inside a container, so it's a
# required, separate env var.
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]

TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE", "mlops-rakuten-product-classification-trainer:latest"
)
# docker-compose names networks "<project-dir-name>_default" unless
# overridden; the main stack's project dir is "mlops-rakuten-product-classification".
TRAINER_NETWORK = os.environ.get(
    "TRAINER_NETWORK", "mlops-rakuten-product-classification_default"
)
DOCKER_PROXY_URL = os.environ.get("DOCKER_PROXY_URL", "tcp://docker-socket-proxy:2375")

# dvc.yaml stage names, keyed by the same "model" values the /retrain API uses
TRAIN_STAGES = {
    "image": "train_image",
    "text": "train_text",
}


@dag(
    dag_id="rakuten_model_training",
    description="Prepare data splits (DVC) and retrain the image or text model in an ephemeral trainer container.",
    schedule=None,  # triggered manually or via REST API, per model
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "model": Param("text", enum=list(TRAIN_STAGES)),
        "epochs": Param(3, type="integer", minimum=1, maximum=50),
    },
    tags=["mlops", "training", "rakuten"],
)
def rakuten_model_training():

    @task
    def prepare_data() -> str:
        subprocess.run(
            ["dvc", "repro", "prepare_splits"],
            cwd=PROJECT_DIR,
            check=True,
        )
        return "prepared"

    launch_trainer = DockerOperator(
        task_id="launch_trainer",
        image=TRAINER_IMAGE,
        docker_url=DOCKER_PROXY_URL,
        network_mode=TRAINER_NETWORK,
        auto_remove="success",
        mount_tmp_dir=False,
        # Same project-root mount the trainer service uses in docker-compose.yaml
        # (".:/project") -- DockerOperator needs the equivalent as an absolute
        # host path since it isn't launched through that compose file.
        mounts=[Mount(source=HOST_PROJECT_DIR, target="/project", type="bind")],
        # Matches the `trainer` service's compose config (mem_limit/shm_size)
        # -- DockerOperator creates the container directly via the Docker API
        # (through docker-socket-proxy), so it doesn't read docker-compose.yaml
        # and needs these set here too, or NUM_WORKERS>0 DataLoaders fail with
        # "unable to allocate shared memory" on Docker's 64MB default.
        mem_limit="8g",
        shm_size="2gb",
        environment={
            "MODEL": "{{ params.model }}",
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001"),
            "MLFLOW_REQUIRED": "true",
            "MAX_EPOCHS_OVERRIDE": "{{ params.epochs }}",
            "TRAIN_ROWS_OVERRIDE": os.environ.get("TRAIN_ROWS_OVERRIDE", ""),
            "VAL_ROWS_OVERRIDE": os.environ.get("VAL_ROWS_OVERRIDE", ""),
        },
    )

    # Runs only if launch_trainer succeeds (default trigger_rule=all_success),
    # so a failed/OOM-killed training run never promotes stale or half-written
    # weights. wait_for_completion=True means this dag_run's own "running"
    # state (what /retrain's list_jobs() lock check looks at) spans the
    # promotion too, so a second retrain can't start while promotion is
    # still copying files into the serving dir.
    trigger_promotion = TriggerDagRunOperator(
        task_id="trigger_promotion",
        trigger_dag_id="rakuten_model_promotion",
        wait_for_completion=True,
        poke_interval=10,
    )

    prepare_data() >> launch_trainer >> trigger_promotion


rakuten_model_training()
