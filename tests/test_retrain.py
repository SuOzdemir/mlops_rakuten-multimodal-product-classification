"""
Unit tests for src/api/retrain.py

Strategy: mock requests.post/requests.get so no real Airflow server is needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api.retrain import DAG_ID, ModelName, get_status, list_jobs, start_retrain


def _response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def test_start_retrain_triggers_dag_run():
    list_resp = _response(200, {"dag_runs": []})
    post_resp = _response(200, {"dag_run_id": "retrain_text_abc", "state": "queued", "conf": {"model": "text"}})

    with patch("src.api.retrain.requests.get", return_value=list_resp), \
         patch("src.api.retrain.requests.post", return_value=post_resp) as mock_post:
        job = start_retrain(ModelName.text)

    assert job == {
        "job_id": "retrain_text_abc",
        "model": "text",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "seed": 42,
        "early_stopping_patience": 3,
        "weight_decay": 0.01,
        "use_amp": True,
        "status": "queued",
        "started_at": None,
    }
    called_url = mock_post.call_args.args[0]
    assert f"/dags/{DAG_ID}/dagRuns" in called_url
    assert mock_post.call_args.kwargs["json"]["conf"] == {
        "model": "text",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "seed": 42,
        "early_stopping_patience": 3,
        "weight_decay": 0.01,
        "use_amp": True,
    }


def test_start_retrain_passes_custom_hyperparameters():
    list_resp = _response(200, {"dag_runs": []})
    post_resp = _response(
        200,
        {
            "dag_run_id": "retrain_image_abc",
            "state": "queued",
            "conf": {
                "model": "image",
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 3e-5,
                "seed": 7,
                "early_stopping_patience": 4,
                "weight_decay": 0.02,
                "use_amp": False,
                "label_smoothing": 0.05,
                "dropout": 0.25,
            },
        },
    )

    with patch("src.api.retrain.requests.get", return_value=list_resp), \
         patch("src.api.retrain.requests.post", return_value=post_resp) as mock_post:
        start_retrain(
            ModelName.image,
            epochs=5,
            batch_size=8,
            learning_rate=3e-5,
            seed=7,
            early_stopping_patience=4,
            weight_decay=0.02,
            use_amp=False,
            label_smoothing=0.05,
            dropout=0.25,
        )

    assert mock_post.call_args.kwargs["json"]["conf"] == {
        "model": "image",
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "seed": 7,
        "early_stopping_patience": 4,
        "weight_decay": 0.02,
        "use_amp": False,
        "label_smoothing": 0.05,
        "dropout": 0.25,
    }


def test_start_retrain_blocks_duplicate_running_job():
    existing_run = {"dag_run_id": "retrain_text_old", "state": "running", "conf": {"model": "text"}}
    list_resp = _response(200, {"dag_runs": [existing_run]})

    with patch("src.api.retrain.requests.get", return_value=list_resp), \
         patch("src.api.retrain.requests.post") as mock_post:
        with pytest.raises(RuntimeError):
            start_retrain(ModelName.text)
    mock_post.assert_not_called()


def test_start_retrain_blocks_different_model_while_one_running():
    # A different model doesn't get parallelism: every retrain runs `dvc
    # repro` against the same shared project checkout, and DVC's lock is
    # repo-wide, not per-stage -- two models retraining at once collide on
    # it instead of actually running in parallel.
    existing_run = {"dag_run_id": "retrain_text_old", "state": "running", "conf": {"model": "text"}}
    list_resp = _response(200, {"dag_runs": [existing_run]})

    with patch("src.api.retrain.requests.get", return_value=list_resp), \
         patch("src.api.retrain.requests.post") as mock_post:
        with pytest.raises(RuntimeError):
            start_retrain(ModelName.image)
    mock_post.assert_not_called()


def test_get_status_maps_success_to_completed():
    resp = _response(200, {"dag_run_id": "retrain_text_abc", "state": "success", "conf": {"model": "text"}, "start_date": "2026-01-01T00:00:00Z"})
    with patch("src.api.retrain.requests.get", return_value=resp):
        job = get_status("retrain_text_abc")
    assert job["status"] == "completed"
    assert job["started_at"] == "2026-01-01T00:00:00Z"


def test_get_status_unknown_job_returns_none():
    resp = _response(404, {})
    with patch("src.api.retrain.requests.get", return_value=resp):
        job = get_status("does-not-exist")
    assert job is None


def test_list_jobs_maps_all_runs():
    runs = [
        {"dag_run_id": "retrain_text_1", "state": "failed", "conf": {"model": "text"}},
        {"dag_run_id": "retrain_image_1", "state": "running", "conf": {"model": "image"}},
    ]
    resp = _response(200, {"dag_runs": runs})
    with patch("src.api.retrain.requests.get", return_value=resp):
        jobs = list_jobs()
    assert [j["status"] for j in jobs] == ["failed", "running"]
    assert [j["model"] for j in jobs] == ["text", "image"]
