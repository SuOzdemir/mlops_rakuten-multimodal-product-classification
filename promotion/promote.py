"""Archive a retrained checkpoint, gate it, register it, and deploy champion."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("RAKUTEN_PROJECT_DIR", "/project"))
CAMEMBERT_BASE_DIR = Path(
    os.environ.get(
        "CAMEMBERT_BASE_DIR",
        str(PROJECT_DIR / "data/camembert_base"),
    )
)
IMAGE_WEIGHTS_SRC = (
    PROJECT_DIR / "models/Model_I12_ConvNeXt_Base_ModerateAug_Full/best_model_state_dict.pt"
)
TEXT_WEIGHTS_SRC = (
    PROJECT_DIR
    / "models/textModeling/Model_T8_CamemBERT_FullFineTune_L128/best_model_text.pt"
)
IMAGE_METADATA_SRC = PROJECT_DIR / "outputs/I12_ConvNeXt_Base_ModerateAug_Full/run_metadata.json"
TEXT_METADATA_SRC = PROJECT_DIR / "outputs/T8_CamemBERT_FullFineTune_L128/run_metadata.json"
LABEL2ID_SRC = PROJECT_DIR / "outputs/image_modeling/label2id.json"
CATEGORY_MAP_SRC = PROJECT_DIR / "data/rakuten_streamlit_predictor/prdtypecode_mapping.json"
SERVING_DIR = PROJECT_DIR / "data/rakuten_streamlit_predictor"
REGISTERED_MODEL_NAME = "rakuten-multimodal-classifier"
MODEL_ALIAS = "champion"

MODEL_CONFIG = {
    "image": {
        "weights": IMAGE_WEIGHTS_SRC,
        "metadata": IMAGE_METADATA_SRC,
        "metric_tag": "image_best_macro_f1",
        "dvc_target": "dvc.yaml:train_image",
    },
    "text": {
        "weights": TEXT_WEIGHTS_SRC,
        "metadata": TEXT_METADATA_SRC,
        "metric_tag": "text_best_macro_f1",
        "dvc_target": "dvc.yaml:train_text",
    },
}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set for promotion.")
    return value


def _read_best_metric(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Training metadata missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(payload["best_macro_f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid best_macro_f1 in {path}") from exc


def _copy_bundle(destination: Path) -> None:
    image_dir = destination / "image_model"
    text_dir = destination / "text_model"
    base_dir = text_dir / "camembert_run4"
    image_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    shutil.copytree(CAMEMBERT_BASE_DIR, base_dir)
    shutil.copy2(IMAGE_WEIGHTS_SRC, image_dir / "best_model_base.pt")
    shutil.copy2(TEXT_WEIGHTS_SRC, text_dir / "best_model_text.pt")
    shutil.copy2(LABEL2ID_SRC, destination / "label2id.json")
    shutil.copy2(LABEL2ID_SRC, base_dir / "label2id.json")
    shutil.copy2(CATEGORY_MAP_SRC, destination / "prdtypecode_mapping.json")


def _atomic_replace_directory(source: Path, destination: Path) -> None:
    incoming = destination.with_name(f"{destination.name}.incoming")
    previous = destination.with_name(f"{destination.name}.previous")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.rmtree(previous, ignore_errors=True)
    shutil.copytree(source, incoming)
    if destination.exists():
        destination.rename(previous)
    incoming.rename(destination)
    shutil.rmtree(previous, ignore_errors=True)


def _push_dvc_checkpoint(target: str) -> None:
    """Push the reproduced stage output; DVC owns the remote object layout."""
    command = ["dvc", "push", target]
    endpoint = os.environ.get("DVC_ENDPOINT_URL", "").strip()
    config_local = PROJECT_DIR / ".dvc/config.local"
    original = config_local.read_bytes() if config_local.exists() else None
    try:
        if endpoint:
            subprocess.run(
                ["dvc", "remote", "modify", "--local", "minio", "endpointurl", endpoint],
                cwd=PROJECT_DIR,
                check=True,
            )
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    finally:
        if original is None:
            config_local.unlink(missing_ok=True)
        else:
            config_local.write_bytes(original)


def _champion_version(client):
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    except Exception:
        return None


def _passes_gate(client, model: str, candidate_metric: float) -> tuple[bool, str]:
    champion = _champion_version(client)
    if champion is None:
        return True, "No champion exists; bootstrap promotion allowed."
    tag_name = MODEL_CONFIG[model]["metric_tag"]
    champion_value = champion.tags.get(tag_name)
    if champion_value is None:
        if os.environ.get("PROMOTION_ALLOW_UNTAGGED_CHAMPION", "false").lower() == "true":
            return True, f"Existing champion has no {tag_name}; one-time migration override allowed."
        return False, f"Existing champion has no comparable {tag_name} tag."
    minimum_gain = float(os.environ.get("PROMOTION_MIN_F1_GAIN", "0"))
    required = float(champion_value) + minimum_gain
    return (
        candidate_metric >= required,
        f"candidate={candidate_metric:.6f}, required={required:.6f} "
        f"(champion={float(champion_value):.6f}, min_gain={minimum_gain:.6f})",
    )


def promote() -> dict:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    model = _required_env("MODEL")
    if model not in MODEL_CONFIG:
        raise ValueError(f"MODEL must be one of {sorted(MODEL_CONFIG)}, got {model!r}")

    required = [
        IMAGE_WEIGHTS_SRC,
        TEXT_WEIGHTS_SRC,
        MODEL_CONFIG[model]["metadata"],
        LABEL2ID_SRC,
        CATEGORY_MAP_SRC,
        CAMEMBERT_BASE_DIR / "config.json",
        CAMEMBERT_BASE_DIR / "model.safetensors",
        CAMEMBERT_BASE_DIR / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required promotion assets missing:\n  " + "\n  ".join(missing))

    candidate_metric = _read_best_metric(MODEL_CONFIG[model]["metadata"])
    _push_dvc_checkpoint(MODEL_CONFIG[model]["dvc_target"])

    import mlflow
    from mlflow import MlflowClient
    from ml.rakuten_pyfunc import RakutenMultimodalModel

    tracking_uri = _required_env("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rakuten-production-bundles")
    client = MlflowClient(tracking_uri=tracking_uri)
    gate_passed, gate_reason = _passes_gate(client, model, candidate_metric)

    with tempfile.TemporaryDirectory(dir=PROJECT_DIR / "data") as temp:
        bundle = Path(temp) / "serving_assets"
        _copy_bundle(bundle)
        with mlflow.start_run(run_name="multimodal-production-candidate") as run:
            mlflow.log_params({
                "retrained_component": model,
                "image_architecture": "convnext_base",
                "text_architecture": "camembert-base",
                "image_weight": 0.45,
                "promotion_gate_passed": gate_passed,
                "promotion_gate_reason": gate_reason,
            })
            mlflow.log_metric(MODEL_CONFIG[model]["metric_tag"], candidate_metric)
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RakutenMultimodalModel(),
                artifacts={"serving_assets": str(bundle)},
                code_paths=[
                    str(PROJECT_DIR / "ml"),
                    str(PROJECT_DIR / "streamlit_app"),
                ],
                registered_model_name=REGISTERED_MODEL_NAME,
                pip_requirements=[
                    "mlflow>=2.16.0", "torch", "torchvision", "transformers>=5.0.0",
                    "sentencepiece>=0.2.0", "pillow>=10.0.0", "numpy", "pandas>=2.0.0",
                ],
            )
            run_id = run.info.run_id

        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}' and run_id='{run_id}'"
        )
        if not versions:
            raise RuntimeError("MLflow did not create a registered model version")
        version = max(versions, key=lambda item: int(item.version)).version
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "candidate_metric", candidate_metric)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "retrained_component", model)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "promotion_gate_reason", gate_reason)

        # Preserve both component metrics on each champion version when known.
        previous = _champion_version(client)
        for component, config in MODEL_CONFIG.items():
            value = candidate_metric if component == model else (
                previous.tags.get(config["metric_tag"]) if previous else None
            )
            if value is not None:
                client.set_model_version_tag(REGISTERED_MODEL_NAME, version, config["metric_tag"], value)

        if not gate_passed:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "deployment_status", "rejected")
            return {"status": "rejected", "version": str(version), "reason": gate_reason}

        client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "deployment_status", "production")
        downloaded_model = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        ))
        registered_bundle = downloaded_model / "artifacts/serving_assets"
        if not registered_bundle.exists():
            raise FileNotFoundError(f"Registry bundle missing: {registered_bundle}")
        _atomic_replace_directory(registered_bundle, SERVING_DIR)

    manifest = {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(version),
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "retrained_component": model,
        "candidate_metric": candidate_metric,
        "promotion_gate_reason": gate_reason,
    }
    (SERVING_DIR / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(promote(), indent=2))
