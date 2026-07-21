"""
Unit tests for src/api/main.py

Strategy: mock user lookup (DB access) and model assets (heavy ML),
so tests run fast without disk access or GPU.
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.api.main as api_module
from src.api.main import app

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FAKE_USERS = {
    "admin": {"username": "admin", "password": "secret", "role": "admin"},
    "user": {"username": "user", "password": "pass", "role": "viewer"},
}

FAKE_ASSETS = MagicMock(name="assets")


def _fake_get_user(username: str):
    return FAKE_USERS.get(username)


def _fake_verify_password(password: str, user: dict) -> bool:
    return user["password"] == password


def test_get_assets_reloads_when_registry_deployment_changes(tmp_path, monkeypatch):
    manifest = tmp_path / "deployment_manifest.json"
    manifest.write_text('{"version": "1", "run_id": "run-1"}', encoding="utf-8")
    first_assets = object()
    second_assets = object()

    monkeypatch.setattr(api_module, "_deployment_manifest", manifest)
    monkeypatch.setattr(api_module, "_assets", None)
    monkeypatch.setattr(api_module, "_assets_deployment_id", None)
    loader = MagicMock(side_effect=[first_assets, second_assets])
    monkeypatch.setattr(api_module, "load_assets", loader)

    assert api_module._get_assets() is first_assets
    assert api_module._get_assets() is first_assets

    manifest.write_text('{"version": "2", "run_id": "run-2"}', encoding="utf-8")
    assert api_module._get_assets() is second_assets
    assert loader.call_count == 2

FAKE_PREDICT_RESULT = {
    "mode": "Text only",
    "text_used": "livre cuisine",
    "image_weight": 0.0,
    "text_weight": 1.0,
    "top3": [
        {
            "Rank": 1,
            "prdtypecode": 10,
            "Category name": "Books",
            "Confidence": "95.0%",
            "confidence_float": 95.0,
            "class_id": 0,
        },
        {
            "Rank": 2,
            "prdtypecode": 40,
            "Category name": "PC and console video games",
            "Confidence": "3.0%",
            "confidence_float": 3.0,
            "class_id": 1,
        },
        {
            "Rank": 3,
            "prdtypecode": 50,
            "Category name": "Video game accessories",
            "Confidence": "2.0%",
            "confidence_float": 2.0,
            "class_id": 2,
        },
    ],
}


def _make_jpeg_bytes() -> bytes:
    """Return a minimal valid JPEG image as bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """TestClient with user lookup mocked so lifespan doesn't touch the DB file."""
    with patch("src.api.main.init_db"), \
         patch("src.api.main.get_user", side_effect=_fake_get_user), \
         patch("src.api.main.verify_password", side_effect=_fake_verify_password), \
         patch("src.api.main.record_prediction_event"):
        with TestClient(app) as c:
            yield c
    # Reset module-level globals; assign new objects instead of mutating in-place
    api_module._tokens.clear()
    api_module._assets = None


@pytest.fixture()
def token(client):
    """A valid Bearer token obtained via /login."""
    resp = client.post("/login", data={"username": "admin", "password": "secret"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture()
def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def viewer_token(client):
    """A valid Bearer token for the non-admin demo user."""
    resp = client.post("/login", data={"username": "user", "password": "pass"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture()
def viewer_auth_headers(viewer_token):
    return {"Authorization": f"Bearer {viewer_token}"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /login
# ---------------------------------------------------------------------------


def test_login_success(client):
    resp = client.post("/login", data={"username": "admin", "password": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"
    assert len(body["access_token"]) == 64  # secrets.token_hex(32) → 64 hex chars


def test_login_wrong_password(client):
    resp = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


def test_login_unknown_user(client):
    resp = client.post("/login", data={"username": "ghost", "password": "secret"})
    assert resp.status_code == 401


def test_login_missing_username(client):
    resp = client.post("/login", data={"password": "secret"})
    assert resp.status_code == 422


def test_login_missing_password(client):
    resp = client.post("/login", data={"username": "admin"})
    assert resp.status_code == 422


def test_login_creates_unique_tokens(client):
    r1 = client.post("/login", data={"username": "admin", "password": "secret"})
    r2 = client.post("/login", data={"username": "admin", "password": "secret"})
    assert r1.json()["access_token"] != r2.json()["access_token"]


# ---------------------------------------------------------------------------
# /logout
# ---------------------------------------------------------------------------


def test_logout_success(client, token):
    resp = client.post("/logout", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "logged out"}


def test_logout_invalidates_token(client, token):
    client.post("/logout", headers={"Authorization": f"Bearer {token}"})
    # Token should no longer be accepted
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=FAKE_PREDICT_RESULT):
        resp = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {token}"},
            data={"designation": "test"},
        )
    assert resp.status_code == 401


def test_logout_no_auth_header(client):
    resp = client.post("/logout")
    assert resp.status_code == 401


def test_logout_malformed_header(client):
    resp = client.post("/logout", headers={"Authorization": "NotBearer abc"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /predict — authentication
# ---------------------------------------------------------------------------


def test_predict_no_auth(client):
    resp = client.post("/predict", data={"designation": "test"})
    assert resp.status_code == 401


def test_predict_invalid_token(client):
    resp = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalidtoken123"},
        data={"designation": "test"},
    )
    assert resp.status_code == 401


def test_predict_malformed_auth_header(client):
    resp = client.post(
        "/predict",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        data={"designation": "test"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /predict — input validation
# ---------------------------------------------------------------------------


def test_predict_no_text_no_image_raises_422(client, auth_headers):
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", side_effect=ValueError("Provide at least a title")):
        resp = client.post("/predict", headers=auth_headers, data={})
    assert resp.status_code == 422


def test_predict_invalid_image_raises_400(client, auth_headers):
    resp = client.post(
        "/predict",
        headers=auth_headers,
        data={"designation": "test"},
        files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "decode" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /predict — model not available
# ---------------------------------------------------------------------------


def test_predict_model_assets_missing_returns_503(client, auth_headers):
    with patch("src.api.main._get_assets", side_effect=FileNotFoundError("model missing")):
        resp = client.post("/predict", headers=auth_headers, data={"designation": "test"})
    assert resp.status_code == 503
    assert "Model assets not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /predict — successful predictions
# ---------------------------------------------------------------------------


def test_predict_text_only(client, auth_headers):
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=FAKE_PREDICT_RESULT):
        resp = client.post(
            "/predict",
            headers=auth_headers,
            data={"designation": "Livre de cuisine", "description": "Recettes"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "Text only"
    assert len(body["top3"]) == 3
    assert body["top3"][0]["Category name"] == "Books"
    assert body["top3"][0]["Rank"] == 1


def test_predict_with_valid_image(client, auth_headers):
    image_result = {**FAKE_PREDICT_RESULT, "mode": "Image only", "image_weight": 1.0, "text_weight": 0.0}
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=image_result):
        resp = client.post(
            "/predict",
            headers=auth_headers,
            files={"image": ("product.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "Image only"


def test_predict_late_fusion(client, auth_headers):
    fusion_result = {**FAKE_PREDICT_RESULT, "mode": "Late fusion", "image_weight": 0.45, "text_weight": 0.55}
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=fusion_result):
        resp = client.post(
            "/predict",
            headers=auth_headers,
            data={"designation": "Livre"},
            files={"image": ("product.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "Late fusion"
    assert body["image_weight"] == pytest.approx(0.45)
    assert body["text_weight"] == pytest.approx(0.55)


def test_predict_response_shape(client, auth_headers):
    """Verify every field the Streamlit app depends on is present."""
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=FAKE_PREDICT_RESULT):
        resp = client.post("/predict", headers=auth_headers, data={"designation": "test"})
    body = resp.json()
    assert "mode" in body
    assert "top3" in body
    for item in body["top3"]:
        assert "Rank" in item
        assert "prdtypecode" in item
        assert "Category name" in item
        assert "Confidence" in item


def test_predict_custom_image_weight(client, auth_headers):
    """image_weight form field is forwarded to predict()."""
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS) as _, \
         patch("src.api.main.predict", return_value=FAKE_PREDICT_RESULT) as mock_predict:
        client.post(
            "/predict",
            headers=auth_headers,
            data={"designation": "test", "image_weight": "0.7"},
        )
    mock_predict.assert_called_once()
    _, kwargs = mock_predict.call_args
    assert kwargs["image_weight"] == pytest.approx(0.7)


def test_predict_monitoring_write_is_fail_open(client, auth_headers):
    """A monitoring database outage must not break the live prediction path."""
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=FAKE_PREDICT_RESULT), \
         patch("src.api.main.record_prediction_event", side_effect=RuntimeError("db unavailable")):
        resp = client.post(
            "/predict",
            headers=auth_headers,
            data={"designation": "Livre", "description": "Recettes"},
        )
    assert resp.status_code == 200
    assert resp.json()["top3"][0]["prdtypecode"] == 10


def test_predict_persists_only_monitoring_features(client, auth_headers):
    with patch("src.api.main._get_assets", return_value=FAKE_ASSETS), \
         patch("src.api.main.predict", return_value=FAKE_PREDICT_RESULT), \
         patch("src.api.main.record_prediction_event") as event_writer:
        resp = client.post(
            "/predict",
            headers=auth_headers,
            data={"designation": "Livre", "description": "Recettes"},
        )
    assert resp.status_code == 200
    event_writer.assert_called_once()
    event = event_writer.call_args.kwargs
    assert event["predicted_code"] == 10
    assert event["designation_length"] == 5
    assert event["description_length"] == 8
    assert event["has_image"] is False
    assert "designation" not in event
    assert "description" not in event


# ---------------------------------------------------------------------------
# /retrain — admin only
# ---------------------------------------------------------------------------

FAKE_JOB = {
    "job_id": "retrain_text_abc123",
    "model": "text",
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "seed": 42,
    "early_stopping_patience": 3,
    "weight_decay": 0.01,
    "use_amp": True,
    "status": "running",
    "started_at": "2026-01-01T00:00:00+00:00",
}


def test_retrain_requires_auth(client):
    resp = client.post("/retrain", data={"model": "text"})
    assert resp.status_code == 401


def test_retrain_viewer_forbidden(client, viewer_auth_headers):
    resp = client.post("/retrain", headers=viewer_auth_headers, data={"model": "text"})
    assert resp.status_code == 403


def test_retrain_admin_starts_job(client, auth_headers):
    with patch("src.api.main.start_retrain", return_value=FAKE_JOB) as mock_start:
        resp = client.post("/retrain", headers=auth_headers, data={"model": "text"})
    assert resp.status_code == 200
    assert resp.json() == FAKE_JOB
    mock_start.assert_called_once_with(
        "text",
        epochs=3,
        batch_size=None,
        learning_rate=None,
        seed=None,
        early_stopping_patience=None,
        weight_decay=None,
        use_amp=None,
        label_smoothing=None,
        dropout=None,
    )


def test_retrain_admin_passes_custom_hyperparameters(client, auth_headers):
    with patch("src.api.main.start_retrain", return_value=FAKE_JOB) as mock_start:
        resp = client.post(
            "/retrain",
            headers=auth_headers,
            data={
                "model": "text",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 3e-5,
                "seed": 7,
                "early_stopping_patience": 5,
                "weight_decay": 0.02,
                "use_amp": "false",
            },
        )
    assert resp.status_code == 200
    mock_start.assert_called_once_with(
        "text",
        epochs=5,
        batch_size=16,
        learning_rate=3e-5,
        seed=7,
        early_stopping_patience=5,
        weight_decay=0.02,
        use_amp=False,
        label_smoothing=None,
        dropout=None,
    )


def test_retrain_rejects_invalid_epochs(client, auth_headers):
    resp = client.post(
        "/retrain",
        headers=auth_headers,
        data={"model": "text", "epochs": 0},
    )
    assert resp.status_code == 422


def test_retrain_rejects_invalid_batch_size(client, auth_headers):
    resp = client.post(
        "/retrain",
        headers=auth_headers,
        data={"model": "text", "batch_size": 64},
    )
    assert resp.status_code == 422


def test_retrain_rejects_invalid_learning_rate(client, auth_headers):
    resp = client.post(
        "/retrain",
        headers=auth_headers,
        data={"model": "text", "learning_rate": 0.1},
    )
    assert resp.status_code == 422


def test_retrain_unknown_model_raises_422(client, auth_headers):
    """model is a real enum now — FastAPI rejects unknown values before start_retrain runs."""
    resp = client.post("/retrain", headers=auth_headers, data={"model": "bogus"})
    assert resp.status_code == 422


def test_retrain_duplicate_job_raises_409(client, auth_headers):
    with patch("src.api.main.start_retrain", side_effect=RuntimeError("already running")):
        resp = client.post("/retrain", headers=auth_headers, data={"model": "text"})
    assert resp.status_code == 409


def test_retrain_list_requires_auth(client):
    resp = client.get("/retrain")
    assert resp.status_code == 401


def test_retrain_list_available_to_viewer(client, viewer_auth_headers):
    with patch("src.api.main.list_jobs", return_value=[FAKE_JOB]):
        resp = client.get("/retrain", headers=viewer_auth_headers)
    assert resp.status_code == 200
    assert resp.json() == [FAKE_JOB]


def test_retrain_status_unknown_job_returns_404(client, auth_headers):
    with patch("src.api.main.get_status", return_value=None):
        resp = client.get("/retrain/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404


def test_retrain_status_known_job(client, auth_headers):
    with patch("src.api.main.get_status", return_value=FAKE_JOB):
        resp = client.get(f"/retrain/{FAKE_JOB['job_id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == FAKE_JOB


def test_retrain_airflow_unreachable_returns_503(client, auth_headers):
    with patch("src.api.main.start_retrain", side_effect=RequestsConnectionError("no route")):
        resp = client.post("/retrain", headers=auth_headers, data={"model": "text"})
    assert resp.status_code == 503


def test_retrain_list_airflow_unreachable_returns_503(client, auth_headers):
    with patch("src.api.main.list_jobs", side_effect=RequestsConnectionError("no route")):
        resp = client.get("/retrain", headers=auth_headers)
    assert resp.status_code == 503
