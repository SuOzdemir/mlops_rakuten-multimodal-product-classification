# ============================================================
# config.py
# All hyperparameters and paths for:
#   Model_T8_CamemBERT_FullFineTune_L128
#
# Architecture: CamemBERT (RoBERTa-based, French-optimised)
#   - Full fine-tuning of all transformer layers
#   - Input: designation + description concatenated (max 128 tokens)
#   - AMP enabled for faster training
#   - Feature export: 768d CLS-token embeddings + logits for fusion
#
# Windows-specific notes:
#   - HF_HUB env fixes are applied at the top of train.py
#   - EXPORT_NUM_WORKERS = 0 is safer on Windows after a crash
#
# To start a new experiment, copy this file, adjust values,
# and pass the new config to train.py.
# ============================================================

import os
import sys
from pathlib import Path

# ------------------------------------------------------------
# Windows / HuggingFace Hub fixes
# These must be applied before any transformers import.
# They are set here so train.py can import config first.
# ------------------------------------------------------------
os.environ["HF_HUB_DISABLE_XET_BACKEND"]      = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]  = "1"
sys.modules["hf_xet"] = None   # Prevent hf_xet import attempt

# mlflow/protobuf>=7.34.0 incompatibility fix — must be set before mlflow import
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------
RUN_NAME = "T8_CamemBERT_FullFineTune_L128"

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
SEED = 42
MAX_LENGTH  = 128         # Token sequence length
BATCH_SIZE  = 32
NUM_WORKERS = 4

MAX_EPOCHS             = 10
EARLY_STOPPING_PATIENCE = 3   # Text models converge faster than image models

# MAX_EPOCHS_OVERRIDE bounds epoch count for a real (non-subsampled-model)
# sanity run without waiting hours -- e.g. a real single-epoch check.
if os.environ.get("MAX_EPOCHS_OVERRIDE"):
    MAX_EPOCHS = int(os.environ["MAX_EPOCHS_OVERRIDE"])

# TRAIN_ROWS_OVERRIDE / VAL_ROWS_OVERRIDE: subsample to a chosen size for a
# real (full-size model, real architecture) run that still finishes in
# bounded time instead of the full dataset.
TRAIN_ROWS_OVERRIDE = int(os.environ["TRAIN_ROWS_OVERRIDE"]) if os.environ.get("TRAIN_ROWS_OVERRIDE") else None
VAL_ROWS_OVERRIDE = int(os.environ["VAL_ROWS_OVERRIDE"]) if os.environ.get("VAL_ROWS_OVERRIDE") else None

LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01

# Scheduler
SCHEDULER_MODE     = "max"
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 1        # Aggressive LR reduction for text fine-tuning
SCHEDULER_MIN_LR   = 1e-7

# Mixed Precision Training (AMP)
USE_AMP = True

# ------------------------------------------------------------
# Feature export  (for fusion models)
# ------------------------------------------------------------
FEATURE_DIM          = 768    # CamemBERT hidden size (CLS token dimension)
EXPORT_FEATURES      = True
EXPORT_NUM_WORKERS   = 0      # 0 is safer on Windows after a crash/restart

# ------------------------------------------------------------
# Resume training
# ------------------------------------------------------------
RESUME_TRAINING   = False
CHECKPOINT_SOURCE = "local_last"   # "local_last" | "local_best"

# ------------------------------------------------------------
# Project directory structure — override with RAKUTEN_PROJECT_DIR /
# CAMEMBERT_BASE_DIR if running against a different data location
# (same env vars used by airflow_dst/dags/rakuten_model_promotion.py).
# ------------------------------------------------------------
PROJECT_DIR       = Path(
    os.environ.get("RAKUTEN_PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent.parent))
)
LOCAL_MODEL_PATH  = Path(
    os.environ.get(
        "CAMEMBERT_BASE_DIR",
        str(PROJECT_DIR / "data" / "rakuten_streamlit_predictor" / "text_model" / "camembert_run4"),
    )
)

OUTPUT_DIR  = PROJECT_DIR / "outputs"
FIGURE_DIR  = PROJECT_DIR / "figures"
MODEL_DIR   = PROJECT_DIR / "models"
SPLIT_DIR   = OUTPUT_DIR  / "image_modeling"

# Run-specific output folders (auto-created by train.py)
LOCAL_OUTPUT_ROOT = OUTPUT_DIR / RUN_NAME
# Checkpoints are colocated with this file (models/textModeling/Model_T8_.../),
# not MODEL_DIR/RUN_NAME — that name doesn't match this directory's real name
# and would silently write checkpoints where dvc.yaml/promotion DAG don't look.
LOCAL_MODEL_ROOT  = Path(__file__).resolve().parent
LOCAL_FIG_ROOT    = FIGURE_DIR / RUN_NAME

# Checkpoint & output file paths (derived, do not change)
LAST_CKPT_LOCAL     = LOCAL_MODEL_ROOT  / "last_checkpoint.pt"
BEST_CKPT_LOCAL     = LOCAL_MODEL_ROOT  / "best_checkpoint.pt"
BEST_WEIGHTS_LOCAL  = LOCAL_MODEL_ROOT  / "best_model_text.pt"
HISTORY_JSON_LOCAL  = LOCAL_OUTPUT_ROOT / "history.json"
METADATA_JSON_LOCAL = LOCAL_OUTPUT_ROOT / "run_metadata.json"
