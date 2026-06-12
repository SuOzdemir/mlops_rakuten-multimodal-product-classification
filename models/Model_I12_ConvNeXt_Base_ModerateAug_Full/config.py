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

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 6

# Differential LRs: very conservative to protect large pretrained backbone
LR_BACKBONE  = 5e-6
LR_HEAD      = 5e-5
WEIGHT_DECAY = 0.05

LABEL_SMOOTHING = 0.1
DROPOUT = 0.5

# Scheduler
SCHEDULER_MODE     = "max"
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR   = 1e-7

# Mixed Precision Training (AMP)
USE_AMP = True

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
# Project directory structure  –  adjust PROJECT_DIR to your machine
# ------------------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\felix\Documents\DS_MLE_MasterSystem\06_PROJECTS\Project_01_Rakuten_Multimodal"
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
LOCAL_MODEL_ROOT  = MODEL_DIR  / RUN_NAME
LOCAL_FIG_ROOT    = FIGURE_DIR / RUN_NAME

# Checkpoint & output file paths (derived, do not change)
LAST_CKPT_LOCAL     = LOCAL_MODEL_ROOT  / "last_checkpoint.pt"
BEST_CKPT_LOCAL     = LOCAL_MODEL_ROOT  / "best_checkpoint.pt"
BEST_WEIGHTS_LOCAL  = LOCAL_MODEL_ROOT  / "best_model_state_dict.pt"
HISTORY_JSON_LOCAL  = LOCAL_OUTPUT_ROOT / "history.json"
METADATA_JSON_LOCAL = LOCAL_OUTPUT_ROOT / "run_metadata.json"
