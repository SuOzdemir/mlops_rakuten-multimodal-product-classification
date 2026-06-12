# Model I12 — ConvNeXt-Base, Moderate Augmentation, Full Fine-Tuning

ConvNeXt-Base classifier (1024-dim features) with **full fine-tuning**, **AMP**, and
**differential learning rates**. This model also **exports 1024-d feature vectors and logits**
that are required as inputs for the multimodal fusion models (MM_IntermediateFusion, MM_LateFusion).

## Run order

```
1. config.py          ← review / adjust hyperparameters before anything else
2. train.py           ← trains the model, saves best checkpoint + exports features/logits
3. evaluate.py        ← loads best checkpoint, prints metrics, saves confusion matrix
4. recover_logits.py  ← (optional) re-export features/logits without retraining
```

`dataset.py` and `utils.py` are imported automatically — do not run them directly.

### When to use `recover_logits.py`

Run this script if the feature export during `train.py` failed or was skipped, or if you need
to regenerate features for a different data split — without repeating the full training run.

## Where to adjust the training setup

All hyperparameters live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BACKBONE_LR` | `5e-6` | Very conservative — ConvNeXt-Base is larger than Tiny |
| `HEAD_LR` | `5e-5` | Learning rate for the classification head |
| `BATCH_SIZE` | `16` | Smaller than I9 due to larger model |
| `IMAGE_SIZE` | `224` | Input resolution |
| `MAX_EPOCHS` | `18` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `6` | Epochs without improvement before stopping |
| `WEIGHT_DECAY` | `0.05` | ConvNeXt default |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `SEED` | `42` | Reproducibility seed |

Feature export settings (paths for `.npy` outputs) are also in `config.py`.
Data paths and the output directory (`RUN_NAME = "I12_ConvNeXt_Base_ModerateAug_Full"`) are set there too.

> **Note for fusion models:** The exported files `train_features_1024d.npy`, `val_features_1024d.npy`,
> and `val_logits.npy` must exist before running MM_IntermediateFusion or MM_LateFusion.
