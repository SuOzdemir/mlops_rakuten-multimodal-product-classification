# Methodology — Phase 1: Modeling

## Objective

Classify Rakuten France e-commerce products into 27 categories (`prdtypecode`) using product title, description, and image. Phase 1 focuses on selecting and training the two unimodal models that feed into the multimodal late fusion system used in production.

---

## Dataset

| Property | Value |
|----------|-------|
| Train samples | 84 916 |
| Test samples | 13 812 |
| Categories | 27 (`prdtypecode`) |
| Missing descriptions | ~35% |

**Inputs per product:**
- `designation` — product title (always present)
- `description` — optional product description
- Product image (JPEG)

**Evaluation metric:** Macro F1 — reflects true performance across all 27 classes equally, not skewed by class imbalance.

---

## 1. Text Model — CamemBERT (T8)

### Preprocessing

- HTML tag removal, lowercasing, special character cleaning
- Combined field: `designation + " " + description`
- Fallback to `designation` only when description is missing

### Architecture

| Property | Value |
|----------|-------|
| Base model | `camembert-base` (RoBERTa-based, French) |
| Input | `designation + description`, max 128 tokens |
| Fine-tuning | Full fine-tuning (all layers) |
| Output | 768d CLS embeddings + logits |
| AMP | Mixed precision (GradScaler + autocast) |
| Optimizer | AdamW |

### Training

```
mlflow.set_experiment("rakuten-text-modeling")

Logged per epoch:
  - train_loss, val_loss
  - val_accuracy, val_f1 (macro)

Artifacts:
  - confusion_matrix.png
  - learning_curves.png
  - classification_report.txt
```

### Result

| Metric | Value |
|--------|-------|
| Val Macro F1 | **0.871** |
| Val Accuracy | **0.877** |

---

## 2. Image Model — ConvNeXt-Base (I12)

### Architecture

| Property | Value |
|----------|-------|
| Base model | ConvNeXt-Base (ImageNet-1K pretrained) |
| Fine-tuning | Full fine-tuning (all layers) |
| Input size | 224×224 |
| Output | 1024d features + logits |

### Augmentation

```
RandomResizedCrop(224, scale=(0.7, 1.0))
RandomHorizontalFlip()
TrivialAugmentWide()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR backbone | 5e-6 (conservative — large pretrained backbone) |
| LR head | 5e-5 |
| Weight decay | 0.05 |
| Batch size | 16 |
| Max epochs | 20 |
| Early stopping patience | 6 |
| Label smoothing | 0.1 |
| Dropout | 0.5 |
| AMP | Mixed precision |

### Training

```
mlflow.set_experiment("rakuten-image-modeling")

Logged per epoch:
  - train_loss, val_loss
  - val_accuracy, val_f1 (macro)

Artifacts:
  - confusion_matrix.png
  - learning_curves.png
  - classification_report.txt
  - val_logits.npy (for fusion)
```

### Result

| Metric | Value |
|--------|-------|
| Val Macro F1 | **0.692** |
| Val Accuracy | **0.720** |
| Best epoch | 20 |

---

## 3. Multimodal — Late Fusion

### Why late fusion

- No additional training needed after unimodal models converge
- Robust to missing modality (text-only or image-only fallback)
- Simple, interpretable, easy to tune

### Formula

```
final_logits = α × image_logits + (1 − α) × text_logits
top3 = softmax(final_logits).argsort(descending=True)[:3]
```

### Weight tuning

Alpha (image weight) was swept on the validation set. Best result at **α = 0.45**:

| Mode | Val Macro F1 |
|------|--------------|
| CamemBERT alone | 0.871 |
| ConvNeXt-Base alone | 0.692 |
| Late fusion (α = 0.45) | **~0.893** |

### Prediction modes (automatic)

| Available input | Mode |
|-----------------|------|
| text + image | Late fusion |
| text only | CamemBERT |
| image only | ConvNeXt-Base |

---

## 4. Challenge Result

Final late fusion submission reached **top 5 on the public leaderboard** of the Rakuten France Multimodal Product Data Classification challenge.

Public leaderboard: https://challengedata.ens.fr/participants/challenges/35/ranking/public
