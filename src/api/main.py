import io
import json
import logging
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from requests.exceptions import RequestException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api import drift
from src.api.db import get_user, init_db, record_prediction_event, verify_password
from src.api.retrain import ModelName, get_status, list_jobs, start_retrain
from streamlit_app.services.raw_late_fusion_predictor import load_assets, predict

_tokens: dict[str, dict[str, str]] = {}  # token → {"username":..., "role":...}
_assets = None
_assets_deployment_id = None
_assets_lock = threading.Lock()
logger = logging.getLogger(__name__)
_deployment_manifest = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "rakuten_streamlit_predictor" / "deployment_manifest.json"
)


def _get_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    return auth[len("Bearer "):]


def _require_auth(request: Request) -> dict[str, str]:
    token = _get_token(request)
    if token not in _tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return _tokens[token]


def _require_admin(request: Request) -> dict[str, str]:
    user = _require_auth(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------
def _get_assets():
    global _assets, _assets_deployment_id
    deployment_id = None
    if _deployment_manifest.exists():
        try:
            manifest = json.loads(_deployment_manifest.read_text(encoding="utf-8"))
            deployment_id = (manifest.get("version"), manifest.get("run_id"))
        except (OSError, json.JSONDecodeError):
            # Promotion writes the manifest after an atomic directory swap. A
            # concurrent request may briefly see no/partial manifest; keep the
            # already-loaded model and retry on the next request.
            deployment_id = _assets_deployment_id

    if _assets is None or deployment_id != _assets_deployment_id:
        with _assets_lock:
            if _assets is None or deployment_id != _assets_deployment_id:
                new_assets = load_assets()
                _assets = new_assets
                _assets_deployment_id = deployment_id
    return _assets


def _record_prediction_event_safely(**event) -> None:
    """Monitoring is fail-open: its database write must never fail a prediction."""
    try:
        record_prediction_event(**event)
    except Exception:
        logger.exception("Could not persist prediction event for Evidently monitoring.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Rakuten Product Classifier API",
    description="Late-fusion (ConvNeXt-Base + CamemBERT) multimodal prediction service.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Top-1 predicted class confidence (0-1)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTION_DRIFT_PSI = Gauge(
    "prediction_drift_psi",
    "Population Stability Index of the live top-1 class distribution "
    "(last 200 predictions) vs. the training set's class distribution. "
    "<0.1 no significant drift, 0.1-0.25 moderate, >0.25 significant.",
)
PREDICTION_DRIFT_WINDOW_SIZE = Gauge(
    "prediction_drift_window_size",
    "Number of predictions currently in the drift rolling window (max 200).",
)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    print(f"Login attempt: {username}")
    user = get_user(username)
    if user is None or not verify_password(password, user):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = secrets.token_hex(32)
    _tokens[token] = {"username": user["username"], "role": user["role"]}
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


@app.post("/logout")
def logout(request: Request):
    token = _get_token(request)
    _tokens.pop(token, None)
    return {"status": "logged out"}


@app.post("/predict")
async def predict_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile | None = File(None),
    designation: str = Form(""),
    description: str = Form(""),
    image_weight: float = Form(0.45),
):
    _require_auth(request)

    img = None
    if image is not None:
        contents = await image.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot decode image file.")

    try:
        assets = _get_assets()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model assets not found: {exc}")

    try:
        result = predict(
            assets=assets,
            designation=designation,
            description=description,
            image=img,
            image_weight=image_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    PREDICTION_CONFIDENCE.observe(result["top3"][0]["confidence_float"] / 100.0)

    drift.record_prediction(result["top3"][0]["prdtypecode"])
    PREDICTION_DRIFT_WINDOW_SIZE.set(drift.window_size())
    psi = drift.current_psi()
    # NaN (not 0.0) while the window is still warming up -- 0.0 would read as
    # "no drift", but really means "not enough predictions yet to say."
    PREDICTION_DRIFT_PSI.set(psi if psi is not None else float("nan"))

    image_width, image_height = img.size if img is not None else (None, None)
    deployment_version = (
        str(_assets_deployment_id[0])
        if _assets_deployment_id and _assets_deployment_id[0] is not None
        else None
    )
    background_tasks.add_task(
        _record_prediction_event_safely,
        predicted_code=int(result["top3"][0]["prdtypecode"]),
        confidence=float(result["top3"][0]["confidence_float"]) / 100.0,
        designation_length=len(designation.strip()),
        description_length=len(description.strip()),
        has_image=img is not None,
        image_width=image_width,
        image_height=image_height,
        prediction_mode=str(result["mode"]),
        image_weight=float(result["image_weight"]),
        deployment_version=deployment_version,
    )

    return JSONResponse(content=result)


# -------------------------------------------------------------------
# Retrain — admin only
# -------------------------------------------------------------------
@app.post("/retrain")
def retrain_endpoint(
    request: Request,
    model: ModelName = Form(...),
    epochs: int = Form(3, ge=1, le=50),
    batch_size: int | None = Form(None, ge=1, le=32),
    learning_rate: float | None = Form(None, ge=1e-7, le=1e-2),
    seed: int | None = Form(None, ge=0, le=2_147_483_647),
    early_stopping_patience: int | None = Form(None, ge=1, le=50),
    weight_decay: float | None = Form(None, ge=0, le=1),
    use_amp: bool | None = Form(None),
    label_smoothing: float | None = Form(None, ge=0, le=0.5),
    dropout: float | None = Form(None, ge=0, le=0.9),
):
    _require_admin(request)
    try:
        job = start_retrain(
            model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            use_amp=use_amp,
            label_smoothing=label_smoothing,
            dropout=dropout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Airflow unreachable: {exc}")
    return job


@app.get("/retrain")
def retrain_list_endpoint(request: Request):
    _require_auth(request)
    try:
        return list_jobs()
    except RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Airflow unreachable: {exc}")


@app.get("/retrain/{job_id}")
def retrain_status_endpoint(request: Request, job_id: str):
    _require_auth(request)
    try:
        job = get_status(job_id)
    except RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Airflow unreachable: {exc}")
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job
