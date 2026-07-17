# ============================================================
# config.py
# All hyperparameters and paths for:
#   Model_I12_ConvNeXt_Base_ModerateAug_Full
#
# Key differences vs. I9 (ConvNeXt-Tiny):
#   - Architecture  : ConvNeXt-Base (larger, 1024d features vs 768d)
#   - Lower LRs     : backbone 5e-6, head 5e-5 (more conservative)
#   - Smaller batch : 16 (Base has larger memory footprint)
#   - More workers  : 4 (compensates for smaller batch)
#   - Feature export: exports 1024d features + logits for fusion
#   - cudnn.benchmark = True (throughput-optimised for fixed input size)
#
# To start a new experiment, copy this file, adjust values,
# and pass the new config to train.py.
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------
RUN_NAME = "I12_ConvNeXt_Base_ModerateAug_Full"

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16           # ConvNeXt-Base requires smaller batches than Tiny
NUM_WORKERS = 4

if os.environ.get("SEED_OVERRIDE"):
    SEED = int(os.environ["SEED_OVERRIDE"])
if os.environ.get("BATCH_SIZE_OVERRIDE"):
    BATCH_SIZE = int(os.environ["BATCH_SIZE_OVERRIDE"])

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 6

if os.environ.get("EARLY_STOPPING_PATIENCE_OVERRIDE"):
    EARLY_STOPPING_PATIENCE = int(
        os.environ["EARLY_STOPPING_PATIENCE_OVERRIDE"]
    )

# MAX_EPOCHS_OVERRIDE bounds epoch count for a real (non-subsampled-model)
# sanity run without waiting hours -- e.g. a real single-epoch check.
if os.environ.get("MAX_EPOCHS_OVERRIDE"):
    MAX_EPOCHS = int(os.environ["MAX_EPOCHS_OVERRIDE"])

# TRAIN_ROWS_OVERRIDE / VAL_ROWS_OVERRIDE: subsample to a chosen size for a
# real (full-size model, real architecture) run that still finishes in
# bounded time instead of all ~72k/~13k rows.
TRAIN_ROWS_OVERRIDE = int(os.environ["TRAIN_ROWS_OVERRIDE"]) if os.environ.get("TRAIN_ROWS_OVERRIDE") else None
VAL_ROWS_OVERRIDE = int(os.environ["VAL_ROWS_OVERRIDE"]) if os.environ.get("VAL_ROWS_OVERRIDE") else None

# Differential LRs: very conservative to protect large pretrained backbone
LR_BACKBONE  = 5e-6
LR_HEAD      = 5e-5
WEIGHT_DECAY = 0.05

# Retrain exposes one LR control. It maps to the classifier-head LR while
# preserving the original 10:1 head/backbone ratio for safe fine-tuning.
if os.environ.get("LEARNING_RATE_OVERRIDE"):
    LR_HEAD = float(os.environ["LEARNING_RATE_OVERRIDE"])
    LR_BACKBONE = LR_HEAD / 10
if os.environ.get("WEIGHT_DECAY_OVERRIDE"):
    WEIGHT_DECAY = float(os.environ["WEIGHT_DECAY_OVERRIDE"])

LABEL_SMOOTHING = 0.1
DROPOUT = 0.5

if os.environ.get("LABEL_SMOOTHING_OVERRIDE"):
    LABEL_SMOOTHING = float(os.environ["LABEL_SMOOTHING_OVERRIDE"])
if os.environ.get("DROPOUT_OVERRIDE"):
    DROPOUT = float(os.environ["DROPOUT_OVERRIDE"])

# Scheduler
SCHEDULER_MODE     = "max"
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR   = 1e-7

# Mixed Precision Training (AMP)
USE_AMP = True

if os.environ.get("USE_AMP_OVERRIDE"):
    USE_AMP = os.environ["USE_AMP_OVERRIDE"].lower() in {
        "1", "true", "yes", "on",
    }

# cuDNN – benchmark=True is safe here because input size is fixed (224×224)
CUDNN_BENCHMARK   = True
CUDNN_DETERMINISTIC = False   # Determinism and benchmark are mutually exclusive

# ------------------------------------------------------------
# Feature export  (for fusion models)
# ------------------------------------------------------------
FEATURE_DIM = 1024            # ConvNeXt-Base intermediate feature size
EXPORT_FEATURES = True        # Run feature export after training

# ------------------------------------------------------------
# Resume training
# ------------------------------------------------------------
RESUME_TRAINING   = False
CHECKPOINT_SOURCE = "local_last"   # "local_last" | "local_best"

# ------------------------------------------------------------
# Project directory structure — override with RAKUTEN_PROJECT_DIR if
# running against a different data location.
# ------------------------------------------------------------
PROJECT_DIR = Path(
    os.environ.get("RAKUTEN_PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent))
)

DATA_DIR        = PROJECT_DIR / "data"
RAW_IMG_DIR     = DATA_DIR   / "raw" / "images"

OUTPUT_DIR      = PROJECT_DIR / "outputs"
FIGURE_DIR      = PROJECT_DIR / "figures"
MODEL_DIR       = PROJECT_DIR / "models"

SPLIT_DIR             = OUTPUT_DIR  / "image_modeling"
LOCAL_IMAGE_TRAIN_DIR = RAW_IMG_DIR / "image_train"
LOCAL_IMAGE_TEST_DIR  = RAW_IMG_DIR / "image_test"

# Run-specific output folders (auto-created by train.py)
LOCAL_OUTPUT_ROOT = OUTPUT_DIR / RUN_NAME
# Checkpoints are colocated with this file (models/Model_I12_.../), not
# MODEL_DIR/RUN_NAME — that name doesn't match this directory's real name
# and would silently write checkpoints where dvc.yaml/promotion DAG don't look.
LOCAL_MODEL_ROOT  = Path(__file__).resolve().parent
LOCAL_FIG_ROOT    = FIGURE_DIR / RUN_NAME

# Checkpoint & output file paths (derived, do not change)
LAST_CKPT_LOCAL     = LOCAL_MODEL_ROOT  / "last_checkpoint.pt"
BEST_CKPT_LOCAL     = LOCAL_MODEL_ROOT  / "best_checkpoint.pt"
BEST_WEIGHTS_LOCAL  = LOCAL_MODEL_ROOT  / "best_model_state_dict.pt"
HISTORY_JSON_LOCAL  = LOCAL_OUTPUT_ROOT / "history.json"
METADATA_JSON_LOCAL = LOCAL_OUTPUT_ROOT / "run_metadata.json"
