# MM — CamemBERT + ConvNeXt-Base, Late Fusion

**No model training.** Loads pre-exported validation logits from I12 and T8, then finds the
optimal weighted combination via a grid search over alpha:

```
fused_logits = alpha * image_logits + (1 - alpha) * text_logits
```

## Prerequisites

The following exported logit files must exist before running:

| Source model | Required files |
|---|---|
| **I12** (ConvNeXt-Base) | `val_logits.npy` |
| **T8** (CamemBERT) | `text_val_logits.npy` |

Run `I12/train.py` and `T8/train.py` (or their `recover_*.py` scripts) first.

## Run order

```
1. config.py     ← set logit file paths and adjust alpha search settings
2. evaluate.py   ← grid-searches alpha, evaluates best fusion, saves plots and report
```

`utils.py` is imported automatically by `evaluate.py` — do not run it directly.

## Where to adjust the setup

All settings live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ALPHA_STEPS` | `21` | Number of alpha values to test (0 → 1 in 0.05 steps) |
| logit file paths | (see config) | Paths to `val_logits.npy` and `text_val_logits.npy` |

To search at finer granularity, increase `ALPHA_STEPS` (e.g. `101` for 0.01 steps).

**Outputs** saved by `evaluate.py`:
- Classification report (accuracy, macro F1, weighted F1)
- Confusion matrix PNG
- Alpha sweep plot (accuracy vs alpha)
- Model comparison plot
- `run_metadata.json` with the best alpha and final metrics
