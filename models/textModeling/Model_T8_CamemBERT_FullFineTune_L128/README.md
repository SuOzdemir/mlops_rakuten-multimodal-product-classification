# Model T8 — CamemBERT, Full Fine-Tuning, Max Length 128

CamemBERT (French RoBERTa-based transformer) for **text classification** on product
designations + descriptions. Full fine-tuning of all layers. **AMP** enabled.
Exports **768-d CLS-token embeddings and logits** required by the multimodal fusion models.

> **Windows note:** `config.py` sets HuggingFace environment variables needed on Windows.
> It must be imported before any HuggingFace call — `train.py` already handles this.

## Run order

```
1. config.py            ← review / adjust hyperparameters before anything else
2. train.py             ← fine-tunes CamemBERT, saves best checkpoint + exports features/logits
3. evaluate.py          ← loads best checkpoint, prints metrics, saves confusion matrix
4. recover_features.py  ← (optional) re-export CLS features/logits without retraining
```

`dataset.py` and `utils.py` are imported automatically — do not run them directly.

### When to use `recover_features.py`

Run this script if the feature export during `train.py` failed or was skipped, or if you need
to regenerate features for a different split — without repeating the full training run.

## Where to adjust the training setup

All hyperparameters live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MAX_LENGTH` | `128` | Maximum tokenised sequence length |
| `LEARNING_RATE` | (see config) | Fine-tuning LR for all layers |
| `BATCH_SIZE` | (see config) | Training batch size |
| `MAX_EPOCHS` | (see config) | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | (see config) | Epochs without improvement before stopping |
| `EXPORT_NUM_WORKERS` | `0` | Set to 0 for safe export on Windows |
| `SEED` | `42` | Reproducibility seed |

Feature export paths (for `.npy` outputs) and the output directory
(`RUN_NAME = "T8_CamemBERT_FullFineTune_L128"`) are also set in `config.py`.

> **Note for fusion models:** The exported files `text_train_features_768d.npy`,
> `text_val_features_768d.npy`, and `text_val_logits.npy` must exist before running
> MM_IntermediateFusion or MM_LateFusion.
