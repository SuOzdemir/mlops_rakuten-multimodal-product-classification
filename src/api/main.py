import io
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from requests.exceptions import RequestException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.db import get_user, init_db, verify_password
from src.api.retrain import ModelName, get_status, list_jobs, start_retrain
from streamlit_app.services.raw_late_fusion_predictor import load_assets, predict

_tokens: dict[str, dict[str, str]] = {}  # token → {"username":..., "role":...}
_assets = None


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
    global _assets
    if _assets is None:
        _assets = load_assets()
    return _assets


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

    return JSONResponse(content=result)


# -------------------------------------------------------------------
# Retrain — admin only
# -------------------------------------------------------------------
@app.post("/retrain")
def retrain_endpoint(request: Request, model: ModelName = Form(...), smoke_test: bool = Form(False)):
    _require_admin(request)
    try:
        job = start_retrain(model, smoke_test=smoke_test)
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
