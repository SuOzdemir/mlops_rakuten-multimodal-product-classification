import os
import uuid
from enum import Enum

import requests

DAG_ID = "rakuten_model_training"

AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://host.docker.internal:8080")
AIRFLOW_API_USER = os.environ.get("AIRFLOW_API_USER", "admin")
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD")
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


TRAINING_DEFAULTS = {
    ModelName.image: {
        "batch_size": 16,
        "learning_rate": 5e-5,
        "seed": 42,
        "early_stopping_patience": 6,
        "weight_decay": 0.05,
        "use_amp": True,
        "label_smoothing": 0.1,
        "dropout": 0.5,
    },
    ModelName.text: {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "seed": 42,
        "early_stopping_patience": 3,
        "weight_decay": 0.01,
        "use_amp": True,
    },
}

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32
MIN_LEARNING_RATE = 1e-7
MAX_LEARNING_RATE = 1e-2


def _training_params(
    model: ModelName,
    batch_size: int | None,
    learning_rate: float | None,
    seed: int | None,
    early_stopping_patience: int | None,
    weight_decay: float | None,
    use_amp: bool | None,
    label_smoothing: float | None,
    dropout: float | None,
) -> dict:
    defaults = TRAINING_DEFAULTS[model]
    params = {
        "batch_size": defaults["batch_size"] if batch_size is None else batch_size,
        "learning_rate": (
            defaults["learning_rate"] if learning_rate is None else learning_rate
        ),
        "seed": defaults["seed"] if seed is None else seed,
        "early_stopping_patience": (
            defaults["early_stopping_patience"]
            if early_stopping_patience is None
            else early_stopping_patience
        ),
        "weight_decay": (
            defaults["weight_decay"] if weight_decay is None else weight_decay
        ),
        "use_amp": defaults["use_amp"] if use_amp is None else use_amp,
    }
    if not MIN_BATCH_SIZE <= params["batch_size"] <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}."
        )
    if not MIN_LEARNING_RATE <= params["learning_rate"] <= MAX_LEARNING_RATE:
        raise ValueError(
            "learning_rate must be between "
            f"{MIN_LEARNING_RATE:g} and {MAX_LEARNING_RATE:g}."
        )
    if not 0 <= params["seed"] <= 2_147_483_647:
        raise ValueError("seed must be between 0 and 2147483647.")
    if not 1 <= params["early_stopping_patience"] <= 50:
        raise ValueError("early_stopping_patience must be between 1 and 50.")
    if not 0 <= params["weight_decay"] <= 1:
        raise ValueError("weight_decay must be between 0 and 1.")

    params["batch_size"] = int(params["batch_size"])
    params["learning_rate"] = float(params["learning_rate"])
    params["seed"] = int(params["seed"])
    params["early_stopping_patience"] = int(
        params["early_stopping_patience"]
    )
    params["weight_decay"] = float(params["weight_decay"])
    params["use_amp"] = bool(params["use_amp"])

    if model == ModelName.image:
        params["label_smoothing"] = (
            defaults["label_smoothing"]
            if label_smoothing is None
            else label_smoothing
        )
        params["dropout"] = (
            defaults["dropout"] if dropout is None else dropout
        )
        if not 0 <= params["label_smoothing"] <= 0.5:
            raise ValueError("label_smoothing must be between 0 and 0.5.")
        if not 0 <= params["dropout"] <= 0.9:
            raise ValueError("dropout must be between 0 and 0.9.")
        params["label_smoothing"] = float(params["label_smoothing"])
        params["dropout"] = float(params["dropout"])
    elif label_smoothing is not None or dropout is not None:
        raise ValueError(
            "label_smoothing and dropout are only configurable for the image model."
        )

    return params


def start_retrain(
    model: ModelName,
    epochs: int = 3,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    early_stopping_patience: int | None = None,
    weight_decay: float | None = None,
    use_amp: bool | None = None,
    label_smoothing: float | None = None,
    dropout: float | None = None,
) -> dict:
    model = ModelName(model)
    training_params = _training_params(
        model,
        batch_size,
        learning_rate,
        seed,
        early_stopping_patience,
        weight_decay,
        use_amp,
        label_smoothing,
        dropout,
    )

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
    conf = {
        "model": model.value,
        "epochs": epochs,
        **training_params,
    }
    resp = requests.post(
        f"{AIRFLOW_API_URL}/api/v1/dags/{DAG_ID}/dagRuns",
        auth=_AUTH,
        json={
            "dag_run_id": dag_run_id,
            "conf": conf,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return _public(resp.json(), requested_conf=conf)


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


def _public(dag_run: dict, requested_conf: dict | None = None) -> dict:
    conf = {**(requested_conf or {}), **(dag_run.get("conf") or {})}
    model = conf.get("model")
    defaults = TRAINING_DEFAULTS.get(ModelName(model)) if model else {}
    public = {
        "job_id": dag_run["dag_run_id"],
        "model": model,
        "epochs": conf.get("epochs", 3),
        "batch_size": conf.get("batch_size", defaults.get("batch_size")),
        "learning_rate": conf.get(
            "learning_rate", defaults.get("learning_rate")
        ),
        "seed": conf.get("seed", defaults.get("seed")),
        "early_stopping_patience": conf.get(
            "early_stopping_patience",
            defaults.get("early_stopping_patience"),
        ),
        "weight_decay": conf.get(
            "weight_decay", defaults.get("weight_decay")
        ),
        "use_amp": conf.get("use_amp", defaults.get("use_amp")),
        "status": _STATE_MAP.get(dag_run.get("state"), dag_run.get("state") or "unknown"),
        "started_at": dag_run.get("start_date") or dag_run.get("logical_date"),
    }
    if model == ModelName.image:
        public["label_smoothing"] = conf.get(
            "label_smoothing", defaults.get("label_smoothing")
        )
        public["dropout"] = conf.get("dropout", defaults.get("dropout"))
    return public
