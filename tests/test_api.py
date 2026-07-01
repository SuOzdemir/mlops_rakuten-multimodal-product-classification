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
         patch("src.api.main.verify_password", side_effect=_fake_verify_password):
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
