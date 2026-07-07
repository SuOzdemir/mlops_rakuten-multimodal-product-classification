"""
rakuten_model_training
=======================
Prepares the train/val data splits (via DVC) and retrains one of the two
unimodal models (image or text). Triggered per-model via the Airflow REST
API by the FastAPI `/retrain` endpoint (see src/api/retrain.py) — the
`model` param selects which training script runs.

Run order:
  prepare_data (dvc repro prepare_splits)
      ↓
  train_model (dvc repro train_image|train_text, model chosen via params)

Both stages go through `dvc repro`, not a bare `python` call — DVC hashes
deps/outs (dvc.yaml) and skips the run if nothing changed since the last
successful repro, and records the new model weight's hash in dvc.lock
when it does run (see docs/methodology_phase2.md §3).

Trigger examples
-----------------
  CLI  : airflow dags trigger rakuten_model_training --conf '{"model": "text"}'
  UI   : Trigger DAG w/ config -> {"model": "image"}
  REST : POST /api/v1/dags/rakuten_model_training/dagRuns
         body: {"conf": {"model": "text"}}

  Smoke test (1 epoch, ~64 train / 16 val rows, minutes not hours — for
  validating this DAG's plumbing end-to-end, not model quality):
         body: {"conf": {"model": "text", "smoke_test": true}}
  Runs with `dvc repro --force` since SMOKE_TEST doesn't change any
  DVC-tracked dep, so a plain `dvc repro` would otherwise see "nothing
  changed" and skip the stage.

Environment variables
---------------------
  RAKUTEN_PROJECT_DIR : project root (default: /project)
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_DIR = Path(os.environ.get("RAKUTEN_PROJECT_DIR", "/project"))

# dvc.yaml stage names, keyed by the same "model" values the /retrain API uses
TRAIN_STAGES = {
    "image": "train_image",
    "text": "train_text",
}


@dag(
    dag_id="rakuten_model_training",
    description="Prepare data splits (DVC) and retrain the image or text model.",
    schedule=None,  # triggered manually or via REST API, per model
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "model": Param("text", enum=list(TRAIN_STAGES)),
        "smoke_test": Param(False, type="boolean"),
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

    @task
    def train_model(_: str, **context) -> str:
        model = context["params"]["model"]
        smoke_test = context["params"].get("smoke_test", False)
        if model not in TRAIN_STAGES:
            raise ValueError(f"Unknown model '{model}'. Choose one of: {list(TRAIN_STAGES)}")

        cmd = ["dvc", "repro", TRAIN_STAGES[model]]
        env = dict(os.environ)
        if smoke_test:
            cmd.append("--force")  # SMOKE_TEST doesn't touch any DVC-tracked dep
            env["SMOKE_TEST"] = "1"

        subprocess.run(cmd, cwd=PROJECT_DIR, env=env, check=True)
        return model

    train_model(prepare_data())


rakuten_model_training()
