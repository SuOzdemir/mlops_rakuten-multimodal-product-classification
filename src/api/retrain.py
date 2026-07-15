import os
import uuid
from enum import Enum

import requests

DAG_ID = "rakuten_model_training"

AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://host.docker.internal:8080")
AIRFLOW_API_USER = os.environ.get("AIRFLOW_API_USER", "admin")
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD", "admin")
_AUTH = (AIRFLOW_API_USER, AIRFLOW_API_PASSWORD)
_TIMEOUT = 10

_STATE_MAP = {
    "queued": "queued",
    "running": "running",
    "success": "completed",
    "failed": "failed",
}


class ModelName(str, Enum):
    image = "image"
    text = "text"


def start_retrain(model: ModelName) -> dict:
    # Blocks on ANY in-flight retrain, not just the same model: every retrain
    # runs `dvc repro` against the same shared project checkout, and DVC's
    # lock (.dvc/tmp/rwlock) is repo-wide, not per-stage. Two different
    # models retraining at once don't get parallelism -- they collide on
    # that lock and one (sometimes both) fails.
    for job in list_jobs():
        if job["status"] in ("queued", "running"):
            raise RuntimeError(
                f"A retrain job for '{job['model']}' is already running. "
                "Only one retrain (any model) can run at a time -- they share DVC's lock."
            )

    dag_run_id = f"retrain_{model.value}_{uuid.uuid4().hex[:8]}"
    resp = requests.post(
        f"{AIRFLOW_API_URL}/api/v1/dags/{DAG_ID}/dagRuns",
        auth=_AUTH,
        json={"dag_run_id": dag_run_id, "conf": {"model": model.value}},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return _public(resp.json(), model=model.value)


def get_status(job_id: str) -> dict | None:
    resp = requests.get(
        f"{AIRFLOW_API_URL}/api/v1/dags/{DAG_ID}/dagRuns/{job_id}",
        auth=_AUTH,
        timeout=_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _public(resp.json())


def list_jobs() -> list[dict]:
    resp = requests.get(
        f"{AIRFLOW_API_URL}/api/v1/dags/{DAG_ID}/dagRuns",
        auth=_AUTH,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return [_public(run) for run in resp.json().get("dag_runs", [])]


def _public(dag_run: dict, model: str | None = None) -> dict:
    conf = dag_run.get("conf") or {}
    return {
        "job_id": dag_run["dag_run_id"],
        "model": model or conf.get("model"),
        "status": _STATE_MAP.get(dag_run.get("state"), dag_run.get("state") or "unknown"),
        "started_at": dag_run.get("start_date") or dag_run.get("logical_date"),
    }
