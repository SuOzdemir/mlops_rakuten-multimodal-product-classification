"""
rakuten_model_promotion
=======================
Promotes trained model weights from the training output directories
to the serving directory consumed by the FastAPI service.

Run order (training must be complete before triggering this DAG):
  1. models/I12_ConvNeXt_Base_ModerateAug_Full/  →  image weights ready
  2. models/T8_CamemBERT_FullFineTune_L128/      →  text weights ready
  3. Trigger this DAG manually (schedule=None)

What this DAG does:
  check_artifacts
      ↓
  promote_image_model  ──┐
  promote_text_model   ──┤──→  validate_serving_dir
  promote_tokenizer    ──┤
  promote_label_maps   ──┘

Environment variables
---------------------
  RAKUTEN_PROJECT_DIR   : project root (default: /project)
  CAMEMBERT_BASE_DIR    : path to the locally downloaded camembert-base model
                          (same dir that config.LOCAL_MODEL_PATH points to in
                           Model_T8_CamemBERT_FullFineTune_L128/config.py).
                          Must contain model.safetensors, config.json,
                          sentencepiece.bpe.model, tokenizer_config.json,
                          special_tokens_map.json.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_DIR = Path(os.environ.get("RAKUTEN_PROJECT_DIR", "/project"))

# Full CamemBERT base model directory (architecture + tokenizer files).
# Set CAMEMBERT_BASE_DIR to wherever you downloaded camembert-base locally,
# e.g. the path stored in config.LOCAL_MODEL_PATH in the T8 training config.
CAMEMBERT_BASE_DIR = Path(
    os.environ.get("CAMEMBERT_BASE_DIR", str(PROJECT_DIR / "streamlit_app" / "models" / "camembert_run4"))
)

# Source paths (training outputs)
IMAGE_WEIGHTS_SRC = PROJECT_DIR / "models" / "Model_I12_ConvNeXt_Base_ModerateAug_Full" / "best_model_state_dict.pt"
TEXT_WEIGHTS_SRC  = PROJECT_DIR / "models" / "textModeling" / "Model_T8_CamemBERT_FullFineTune_L128" / "best_model_text.pt"
LABEL2ID_SRC      = PROJECT_DIR / "outputs" / "image_modeling" / "label2id.json"
# prdtypecode_mapping.json is a static prdtypecode->category lookup, not a
# training/DVC output (no dvc.yaml stage produces it) -- it already lives
# permanently at CATEGORY_MAP_DST, so there's no separate source to promote
# it from. The old source (streamlit_app/config/prdtypecode_mapping.json)
# was deleted in a repo reorg (cc7512a) once this dir became the live copy.

# Serving paths (read by raw_late_fusion_predictor.py)
SERVING_DIR         = PROJECT_DIR / "data" / "rakuten_streamlit_predictor"
IMAGE_WEIGHTS_DST   = SERVING_DIR / "image_model" / "best_model_base.pt"
TEXT_WEIGHTS_DST    = SERVING_DIR / "text_model"  / "best_model_text.pt"
TOKENIZER_DST_DIR   = SERVING_DIR / "text_model"  / "camembert_base"
LABEL2ID_DST        = SERVING_DIR / "label2id.json"
CATEGORY_MAP_DST    = SERVING_DIR / "prdtypecode_mapping.json"

REQUIRED_SERVING_FILES = [
    IMAGE_WEIGHTS_DST,
    TEXT_WEIGHTS_DST,
    TOKENIZER_DST_DIR / "config.json",
    TOKENIZER_DST_DIR / "tokenizer_config.json",
    LABEL2ID_DST,
    CATEGORY_MAP_DST,
]


@dag(
    dag_id="rakuten_model_promotion",
    description="Promote trained image/text model weights to the FastAPI serving directory.",
    schedule=None,   # triggered manually after training
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "promotion", "rakuten"],
)
def rakuten_model_promotion():

    @task
    def check_artifacts() -> dict:
        missing = []
        for path in [IMAGE_WEIGHTS_SRC, TEXT_WEIGHTS_SRC, LABEL2ID_SRC, CATEGORY_MAP_DST]:
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "Required training artifacts not found:\n  " + "\n  ".join(missing)
            )
        return {
            "image_weights_bytes": IMAGE_WEIGHTS_SRC.stat().st_size,
            "text_weights_bytes":  TEXT_WEIGHTS_SRC.stat().st_size,
        }

    @task
    def promote_image_model(artifact_info: dict) -> str:
        IMAGE_WEIGHTS_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(IMAGE_WEIGHTS_SRC, IMAGE_WEIGHTS_DST)
        print(f"Image model promoted: {IMAGE_WEIGHTS_SRC} → {IMAGE_WEIGHTS_DST}")
        return str(IMAGE_WEIGHTS_DST)

    @task
    def promote_text_model(artifact_info: dict) -> str:
        TEXT_WEIGHTS_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEXT_WEIGHTS_SRC, TEXT_WEIGHTS_DST)
        print(f"Text model promoted: {TEXT_WEIGHTS_SRC} → {TEXT_WEIGHTS_DST}")
        return str(TEXT_WEIGHTS_DST)

    @task
    def promote_tokenizer(artifact_info: dict) -> str:
        if not CAMEMBERT_BASE_DIR.exists():
            raise FileNotFoundError(
                f"CamemBERT base model directory not found: {CAMEMBERT_BASE_DIR}\n"
                "Set the CAMEMBERT_BASE_DIR environment variable to the directory "
                "used as LOCAL_MODEL_PATH in the T8 training config."
            )
        TOKENIZER_DST_DIR.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in CAMEMBERT_BASE_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, TOKENIZER_DST_DIR / f.name)
                copied.append(f.name)
        print(f"CamemBERT base model promoted from {CAMEMBERT_BASE_DIR} → {TOKENIZER_DST_DIR}")
        print(f"  Files copied: {copied}")
        return str(TOKENIZER_DST_DIR)

    @task
    def promote_label_maps(artifact_info: dict):
        SERVING_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LABEL2ID_SRC, LABEL2ID_DST)
        print(f"label2id.json → {LABEL2ID_DST}")
        # prdtypecode_mapping.json isn't copied here: it's a static file with
        # no training-output source, already permanent at CATEGORY_MAP_DST
        # (see the comment by CATEGORY_MAP_DST's definition above).

    @task
    def validate_serving_dir(
        image_dst: str, text_dst: str, tokenizer_dst: str
    ):
        missing = [str(p) for p in REQUIRED_SERVING_FILES if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Serving directory is incomplete after promotion:\n  "
                + "\n  ".join(missing)
            )
        sizes = {str(p.name): p.stat().st_size for p in REQUIRED_SERVING_FILES}
        print("Serving directory validated. File sizes:")
        for name, size in sizes.items():
            print(f"  {name}: {size:,} bytes")
        return sizes

    # ----------------------------------------------------------------
    # Wire up the tasks
    # ----------------------------------------------------------------
    artifacts    = check_artifacts()
    image_dst    = promote_image_model(artifacts)
    text_dst     = promote_text_model(artifacts)
    tokenizer    = promote_tokenizer(artifacts)
    promote_label_maps(artifacts)
    validate_serving_dir(image_dst, text_dst, tokenizer)


rakuten_model_promotion()
