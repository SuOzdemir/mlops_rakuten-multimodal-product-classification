# ============================================================
# config.py
# All paths and settings for:
#   MM_CamemBERT_ConvNeXtBase_LateFusion
#
# This model has NO training of its own.
# It loads pre-exported logits from:
#   - Model_I12_ConvNeXt_Base_ModerateAug_Full  (image logits, 27d)
#   - Model_T8_CamemBERT_FullFineTune_L128      (text logits,  27d)
# and finds the optimal weighted combination via grid search.
#
# Run order:
#   1. train.py  (I12)  → exports val_logits.npy
#   2. train.py  (T8)   → exports text_val_logits.npy
#   3. evaluate.py (this model)
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------
RUN_NAME = "MM_CamemBERT_ConvNeXtBase_LateFusion"

# ------------------------------------------------------------
# Fusion hyperparameters
# ------------------------------------------------------------
# Grid search: alpha controls the image model weight.
#   fused_logits = alpha * img_logits + (1 - alpha) * txt_logits
ALPHA_STEPS = 21          # Number of steps between 0 and 1 (0.05 increments)

# ------------------------------------------------------------
# Project directory structure  –  adjust PROJECT_DIR to your machine
# ------------------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\felix\Documents\DS_MLE_MasterSystem\06_PROJECTS\Project_01_Rakuten_Multimodal"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
SPLIT_DIR  = OUTPUT_DIR  / "image_modeling"

# Source model output directories
IMAGE_MODEL_OUTPUT_DIR = OUTPUT_DIR / "I12_ConvNeXt_Base_ModerateAug_Full"
TEXT_MODEL_OUTPUT_DIR  = OUTPUT_DIR / "T8_CamemBERT_FullFineTune_L128"

# Input logit files (produced by I12 and T8 feature export)
VAL_LOGITS_IMAGE = IMAGE_MODEL_OUTPUT_DIR / "val_logits.npy"
VAL_LOGITS_TEXT  = TEXT_MODEL_OUTPUT_DIR  / "text_val_logits.npy"

# Run-specific output folder (auto-created by evaluate.py)
LOCAL_OUTPUT_ROOT = OUTPUT_DIR / RUN_NAME
LOCAL_FIG_ROOT    = FIGURE_DIR / RUN_NAME

# Output file paths (derived, do not change)
METADATA_JSON_LOCAL      = LOCAL_OUTPUT_ROOT / "run_metadata.json"
FUSION_REPORT_LOCAL      = LOCAL_OUTPUT_ROOT / "fusion_classification_report.txt"
