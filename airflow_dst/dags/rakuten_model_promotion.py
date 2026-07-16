"""Register a complete serving bundle in MLflow and deploy its champion alias."""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_DIR = Path(os.environ.get("RAKUTEN_PROJECT_DIR", "/project"))
sys.path.insert(0, str(PROJECT_DIR))

CAMEMBERT_BASE_DIR = Path(
    os.environ.get(
        "CAMEMBERT_BASE_DIR",
        str(
            PROJECT_DIR / "data" / "rakuten_streamlit_predictor"
            / "text_model" / "camembert_run4"
        ),
    )
)
IMAGE_WEIGHTS_SRC = (
    PROJECT_DIR / "models" / "Model_I12_ConvNeXt_Base_ModerateAug_Full"
    / "best_model_state_dict.pt"
)
TEXT_WEIGHTS_SRC = (
    PROJECT_DIR / "models" / "textModeling"
    / "Model_T8_CamemBERT_FullFineTune_L128" / "best_model_text.pt"
)
LABEL2ID_SRC = PROJECT_DIR / "outputs" / "image_modeling" / "label2id.json"
CATEGORY_MAP_SRC = (
    PROJECT_DIR / "data" / "rakuten_streamlit_predictor"
    / "prdtypecode_mapping.json"
)
SERVING_DIR = PROJECT_DIR / "data" / "rakuten_streamlit_predictor"
REGISTERED_MODEL_NAME = "rakuten-multimodal-classifier"
MODEL_ALIAS = "champion"


def _copy_bundle(destination: Path) -> None:
    """Assemble all files needed by raw_late_fusion_predictor.load_assets."""
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
    """Replace serving assets without exposing a half-copied directory."""
    incoming = destination.with_name(f"{destination.name}.incoming")
    previous = destination.with_name(f"{destination.name}.previous")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.rmtree(previous, ignore_errors=True)
    shutil.copytree(source, incoming)
    if destination.exists():
        destination.rename(previous)
    incoming.rename(destination)
    shutil.rmtree(previous, ignore_errors=True)


@dag(
    dag_id="rakuten_model_promotion",
    description="Register the trained multimodal bundle in MLflow and deploy the champion alias.",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "registry", "deployment", "rakuten"],
)
def rakuten_model_promotion():

    @task
    def register_and_deploy() -> dict:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        import mlflow
        from mlflow import MlflowClient
        from ml.rakuten_pyfunc import RakutenMultimodalModel

        required = [
            IMAGE_WEIGHTS_SRC,
            TEXT_WEIGHTS_SRC,
            LABEL2ID_SRC,
            CATEGORY_MAP_SRC,
            CAMEMBERT_BASE_DIR / "config.json",
            CAMEMBERT_BASE_DIR / "model.safetensors",
            CAMEMBERT_BASE_DIR / "tokenizer_config.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Required promotion assets missing:\n  " + "\n  ".join(missing))

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("rakuten-production-bundles")

        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / "data") as temp:
            bundle = Path(temp) / "serving_assets"
            _copy_bundle(bundle)

            with mlflow.start_run(run_name="multimodal-production-candidate") as run:
                mlflow.log_params({
                    "image_architecture": "convnext_base",
                    "text_architecture": "camembert-base",
                    "image_weight": 0.45,
                })
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=RakutenMultimodalModel(),
                    artifacts={"serving_assets": str(bundle)},
                    code_paths=[str(PROJECT_DIR / "ml"), str(PROJECT_DIR / "streamlit_app")],
                    registered_model_name=REGISTERED_MODEL_NAME,
                    pip_requirements=[
                        "mlflow>=2.16.0", "torch", "torchvision", "transformers>=5.0.0",
                        "sentencepiece>=0.2.0", "pillow>=10.0.0", "numpy", "pandas>=2.0.0",
                    ],
                )
                run_id = run.info.run_id

            client = MlflowClient(tracking_uri=tracking_uri)
            versions = client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}' and run_id='{run_id}'"
            )
            if not versions:
                raise RuntimeError("MLflow did not create a registered model version")
            version = max(versions, key=lambda item: int(item.version)).version
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, version)
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME, version, "deployment_status", "production"
            )

            # Deployment is deliberately sourced back from Registry. This proves
            # that the API directory contains the exact version behind @champion,
            # rather than whichever checkpoint happens to be on local disk.
            downloaded_model = Path(mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
            ))
            registered_bundle = downloaded_model / "artifacts" / "serving_assets"
            if not registered_bundle.exists():
                raise FileNotFoundError(f"Registry bundle missing: {registered_bundle}")
            _atomic_replace_directory(registered_bundle, SERVING_DIR)

        manifest = {
            "registered_model": REGISTERED_MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": str(version),
            "run_id": run_id,
            "model_uri": model_info.model_uri,
        }
        (SERVING_DIR / "deployment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    register_and_deploy()


rakuten_model_promotion()
