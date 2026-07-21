"""Optional, out-of-band Evidently drift monitor.

This service reads privacy-safe prediction features from PostgreSQL and never
sits on FastAPI's request path. If it is stopped or unhealthy, live prediction,
the existing in-memory PSI metric, Prometheus, and Grafana continue to work.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from minio import Minio
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Gauge,
    generate_latest,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("evidently-monitor")

DATABASE_URL = os.environ["DATABASE_URL"]
REFERENCE_DATA_PATH = Path(
    os.environ.get("EVIDENTLY_REFERENCE_DATA", "/reference/train_split.csv")
)
REFERENCE_DISTRIBUTION_PATH = Path(
    os.environ.get(
        "EVIDENTLY_REFERENCE_DISTRIBUTION",
        "/app/reference_class_distribution.json",
    )
)
REPORT_DIR = Path(os.environ.get("EVIDENTLY_REPORT_DIR", "/reports"))
INTERVAL_SECONDS = max(10, int(os.environ.get("EVIDENTLY_INTERVAL_SECONDS", "60")))
WINDOW_SIZE = max(5, int(os.environ.get("EVIDENTLY_WINDOW_SIZE", "200")))
MIN_CURRENT_ROWS = max(5, int(os.environ.get("EVIDENTLY_MIN_CURRENT_ROWS", "5")))
REFERENCE_SAMPLE_SIZE = max(
    100,
    int(os.environ.get("EVIDENTLY_REFERENCE_SAMPLE_SIZE", "5000")),
)
PSI_THRESHOLD = float(os.environ.get("EVIDENTLY_PSI_THRESHOLD", "0.10"))
DRIFT_SHARE_THRESHOLD = float(
    os.environ.get("EVIDENTLY_DRIFT_SHARE_THRESHOLD", "0.50")
)

MONITOR_UP = Gauge(
    "evidently_monitor_up",
    "Whether the Evidently monitor process is running.",
)
LAST_RUN_SUCCESS = Gauge(
    "evidently_last_run_success",
    "Whether the most recent Evidently evaluation succeeded.",
)
LAST_RUN_TIMESTAMP = Gauge(
    "evidently_last_run_timestamp_seconds",
    "Unix timestamp of the most recent successful Evidently evaluation.",
)
CURRENT_WINDOW_ROWS = Gauge(
    "evidently_current_window_rows",
    "Prediction rows in the current Evidently evaluation window.",
)
DATASET_DRIFT_SHARE = Gauge(
    "evidently_dataset_drift_share",
    "Share of monitored columns detected as drifted by Evidently.",
)
DRIFTED_COLUMNS = Gauge(
    "evidently_drifted_columns",
    "Number of monitored columns detected as drifted by Evidently.",
)
COLUMN_DRIFT_SCORE = Gauge(
    "evidently_column_drift_score",
    "Evidently PSI score for a monitored column.",
    ["column"],
)
COLUMN_DRIFT_DETECTED = Gauge(
    "evidently_column_drift_detected",
    "One when a monitored column's Evidently PSI reaches its threshold.",
    ["column"],
)

_stop_event = threading.Event()
_monitor_thread: threading.Thread | None = None
_state_lock = threading.Lock()
_last_event_id: int | None = None
_last_error: str | None = None
_last_report_at: str | None = None
_reference_source: str | None = None


def _load_training_reference() -> tuple[pd.DataFrame, str]:
    if REFERENCE_DATA_PATH.exists():
        reference = pd.read_csv(
            REFERENCE_DATA_PATH,
            usecols=[
                "designation",
                "description",
                "image_path_local",
                "prdtypecode",
            ],
        )
        if len(reference) > REFERENCE_SAMPLE_SIZE:
            reference = reference.sample(
                n=REFERENCE_SAMPLE_SIZE,
                random_state=42,
            )
        prepared = pd.DataFrame(
            {
                "prediction": reference["prdtypecode"].astype(str),
                "designation_length": (
                    reference["designation"].fillna("").astype(str).str.strip().str.len()
                ),
                "description_length": (
                    reference["description"].fillna("").astype(str).str.strip().str.len()
                ),
            }
        )
        # A non-empty training image path represents an available image. Avoid
        # checking host-specific absolute paths from inside the container.
        prepared["has_image"] = (
            reference["image_path_local"].fillna("").astype(str).str.len() > 0
        ).astype(str)
        return prepared, str(REFERENCE_DATA_PATH)

    distribution = json.loads(
        REFERENCE_DISTRIBUTION_PATH.read_text(encoding="utf-8")
    )
    rows: list[str] = []
    allocations = {
        code: int(float(share) * REFERENCE_SAMPLE_SIZE)
        for code, share in distribution.items()
    }
    remainder = REFERENCE_SAMPLE_SIZE - sum(allocations.values())
    ranked = sorted(
        distribution,
        key=lambda code: (
            float(distribution[code]) * REFERENCE_SAMPLE_SIZE
            - allocations[code]
        ),
        reverse=True,
    )
    for code in ranked[:remainder]:
        allocations[code] += 1
    for code, count in allocations.items():
        rows.extend([str(code)] * count)
    return pd.DataFrame({"prediction": rows}), str(REFERENCE_DISTRIBUTION_PATH)


def _load_current_window() -> tuple[pd.DataFrame, int | None]:
    query = """
        SELECT
            id,
            predicted_code,
            designation_length,
            description_length,
            has_image
        FROM (
            SELECT
                id,
                predicted_code,
                designation_length,
                description_length,
                has_image,
                created_at
            FROM prediction_events
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        ) recent
        ORDER BY created_at ASC, id ASC
    """
    columns = [
        "id",
        "predicted_code",
        "designation_length",
        "description_length",
        "has_image",
    ]
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (WINDOW_SIZE,))
            current = pd.DataFrame(cursor.fetchall(), columns=columns)
    if current.empty:
        return current, None
    latest_event_id = int(current["id"].max())
    current = current.drop(columns=["id"])
    current["prediction"] = current.pop("predicted_code").astype(str)
    current["has_image"] = current["has_image"].astype(str)
    return current, latest_event_id


def _extract_metrics(snapshot: dict) -> tuple[float, float, dict[str, float]]:
    drift_count = 0.0
    drift_share = 0.0
    column_scores: dict[str, float] = {}
    for metric in snapshot.get("metrics", []):
        config = metric.get("config", {})
        metric_type = str(config.get("type", ""))
        if metric_type.endswith("DriftedColumnsCount"):
            value = metric.get("value") or {}
            drift_count = float(value.get("count", 0.0))
            drift_share = float(value.get("share", 0.0))
        elif metric_type.endswith("ValueDrift"):
            column = str(config["column"])
            column_scores[column] = float(metric["value"])
    return drift_count, drift_share, column_scores


def _as_evidently_dataset(frame: pd.DataFrame, columns: list[str]) -> Dataset:
    numerical = [
        column
        for column in ("designation_length", "description_length")
        if column in columns
    ]
    categorical = [
        column
        for column in ("prediction", "has_image")
        if column in columns
    ]
    return Dataset.from_pandas(
        frame[columns],
        data_definition=DataDefinition(
            numerical_columns=numerical,
            categorical_columns=categorical,
        ),
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _upload_to_minio(report_html: Path, report_json: Path) -> None:
    endpoint = os.environ.get("MINIO_ENDPOINT")
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    if not endpoint or not access_key or not secret_key:
        return

    bucket = os.environ.get("EVIDENTLY_MINIO_BUCKET", "evidently-reports")
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    for source, object_name, content_type in (
        (report_html, "latest/report.html", "text/html"),
        (report_json, "latest/report.json", "application/json"),
        (
            report_html,
            f"history/{report_html.name}",
            "text/html",
        ),
        (
            report_json,
            f"history/{report_json.name}",
            "application/json",
        ),
    ):
        client.fput_object(
            bucket,
            object_name,
            str(source),
            content_type=content_type,
        )


def run_evaluation() -> bool:
    global _last_error, _last_event_id, _last_report_at, _reference_source

    current, latest_event_id = _load_current_window()
    CURRENT_WINDOW_ROWS.set(len(current))
    if len(current) < MIN_CURRENT_ROWS:
        logger.info(
            "Waiting for prediction events: %d/%d rows.",
            len(current),
            MIN_CURRENT_ROWS,
        )
        return False
    if latest_event_id == _last_event_id:
        return False

    reference, reference_source = _load_training_reference()
    columns = [
        column
        for column in (
            "prediction",
            "designation_length",
            "description_length",
            "has_image",
        )
        if column in current.columns and column in reference.columns
    ]
    snapshot = Report(
        [
            DataDriftPreset(
                columns=columns,
                method="psi",
                threshold=PSI_THRESHOLD,
                drift_share=DRIFT_SHARE_THRESHOLD,
            )
        ],
        metadata={
            "reference_source": reference_source,
            "current_window_rows": str(len(current)),
        },
    ).run(
        _as_evidently_dataset(current, columns),
        _as_evidently_dataset(reference, columns),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_html = REPORT_DIR / f"report-{timestamp}.html"
    report_json = REPORT_DIR / f"report-{timestamp}.json"
    snapshot.save_html(str(report_html))
    snapshot.save_json(str(report_json))
    _atomic_copy(report_html, REPORT_DIR / "latest.html")
    _atomic_copy(report_json, REPORT_DIR / "latest.json")

    drift_count, drift_share, column_scores = _extract_metrics(snapshot.dict())
    DRIFTED_COLUMNS.set(drift_count)
    DATASET_DRIFT_SHARE.set(drift_share)
    for column, score in column_scores.items():
        COLUMN_DRIFT_SCORE.labels(column=column).set(score)
        COLUMN_DRIFT_DETECTED.labels(column=column).set(
            1 if score >= PSI_THRESHOLD else 0
        )

    try:
        _upload_to_minio(report_html, report_json)
    except Exception:
        # A MinIO outage must not invalidate the locally generated evaluation.
        logger.exception("Could not upload Evidently report to MinIO.")

    now = time.time()
    LAST_RUN_TIMESTAMP.set(now)
    LAST_RUN_SUCCESS.set(1)
    with _state_lock:
        _last_event_id = latest_event_id
        _last_error = None
        _last_report_at = datetime.fromtimestamp(
            now,
            tz=timezone.utc,
        ).isoformat()
        _reference_source = reference_source
    logger.info(
        "Evidently report created: rows=%d drift_share=%.3f columns=%s",
        len(current),
        drift_share,
        columns,
    )
    return True


def _monitor_loop() -> None:
    global _last_error
    MONITOR_UP.set(1)
    while not _stop_event.is_set():
        try:
            run_evaluation()
        except Exception as exc:
            LAST_RUN_SUCCESS.set(0)
            with _state_lock:
                _last_error = str(exc)
            logger.exception("Evidently evaluation failed.")
        _stop_event.wait(INTERVAL_SECONDS)
    MONITOR_UP.set(0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _monitor_thread
    _stop_event.clear()
    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        name="evidently-monitor",
        daemon=True,
    )
    _monitor_thread.start()
    yield
    _stop_event.set()
    _monitor_thread.join(timeout=5)


app = FastAPI(
    title="Rakuten Evidently Monitor",
    description="Optional out-of-band drift reports for the Rakuten classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def index():
    return {
        "service": "evidently-monitor",
        "health": "/health",
        "metrics": "/metrics",
        "latest_report": "/reports/latest.html",
    }


@app.get("/health")
def health():
    with _state_lock:
        return {
            "status": "ok" if _monitor_thread and _monitor_thread.is_alive() else "error",
            "last_report_at": _last_report_at,
            "last_error": _last_error,
            "reference_source": _reference_source,
        }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/reports/latest.html")
def latest_report():
    path = REPORT_DIR / "latest.html"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report is created after at least {MIN_CURRENT_ROWS} predictions.",
        )
    return FileResponse(path, media_type="text/html")


@app.get("/reports/latest.json")
def latest_report_json():
    path = REPORT_DIR / "latest.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report is created after at least {MIN_CURRENT_ROWS} predictions.",
        )
    return FileResponse(path, media_type="application/json")
