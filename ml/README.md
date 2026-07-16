# ml/ — MLflow (Chapter 11)

Experiment tracking + Model Registry for the project. See `docs/methodology_phase2.md`
§1 and §5 for the overall architecture (mlflow service, ports, env vars).

## Running the MLflow server

```bash
make up          # full stack: mlflow -> api -> streamlit
# or just:
docker compose up -d mlflow
```

UI: http://localhost:5001

## Backfilling the existing (already-trained) runs

`train.py` for I12 and T8 already logs to MLflow live (see
`models/Model_I12_ConvNeXt_Base_ModerateAug_Full/train.py` and
`models/textModeling/Model_T8_CamemBERT_FullFineTune_L128/train.py`) — but as
committed, those scripts still point at a sibling repo
(`Project_01_Rakuten_Multimodal`) via a hardcoded `PROJECT_DIR` in their
`config.py`, and use import paths that don't resolve in this repo. They were
not re-run for this backfill.

Instead, `ml/mlflow_backfill.py` logs the **already-completed** runs from
`data/model_artifacts/` (gitignored — see below) into MLflow without
retraining, then registers the LateFusion run and applies the promotion rule.

```bash
docker compose up -d mlflow     # or a local `mlflow server` on :5001
uv run python ml/mlflow_backfill.py
```

### `data/model_artifacts/` layout

Not committed (`data/` is gitignored). Populated by copying artifacts out of
`Project_01_Rakuten_Multimodal` on the machine that trained them:

```
data/model_artifacts/
├── I12_ConvNeXt_Base_ModerateAug_Full/
│   ├── best_model_state_dict.pt
│   ├── model_metadata.json          # params + final metrics
│   ├── history.csv                  # per-epoch train/val loss & f1
│   ├── val_classification_report.txt
│   └── val_confusion_matrix.png / learning_curves.png
├── T8_CamemBERT_FullFineTune_L128/
│   ├── best_model_text.pt
│   ├── model_metadata.json
│   ├── val_classification_report.txt   # regenerated — see note below
│   └── confusion_matrix.png / training_curves.png
└── MM_CamemBERT_ConvNeXtBase_LateFusion/
    ├── run_metadata.json            # best_alpha, macro_f1, accuracy, ...
    ├── fusion_classification_report.txt
    ├── confusion_matrix.png
    └── val_predictions.csv
```

Note: T8's confusion matrix / classification report didn't exist on disk in
the source repo — they were regenerated once by running that repo's own
`evaluate.py` against the existing `best_model_text.pt` checkpoint (inference
only, no retraining), then copied over.

## Experiments

| Experiment | Used by | Why |
|---|---|---|
| `rakuten-image-modeling` | I12 (and future image models) | Per-family comparison, own param/metric schema |
| `rakuten-text-modeling` | T8 (and future text models) | Same as above, for text models |
| `rakuten-production-bundles` | Promotion DAG | Complete deployable multimodal bundles |

## Model Registry & promotion (11.6 / 11.7)

`rakuten_model_promotion` registers a real, loadable MLflow pyfunc named
`rakuten-multimodal-classifier`. It contains both checkpoints, the tokenizer
and label/category mappings. The DAG assigns the `champion` alias and deploys
the assets downloaded from `models:/rakuten-multimodal-classifier@champion`.
`ml/mlflow_backfill.py` is only a legacy experiment-history importer and is
not part of production promotion.

## Known env quirks (Windows)

- `mlflow` import fails with `TypeError: Descriptors cannot be created
  directly` against this project's pinned `protobuf>=7.34.0`. Both
  `ml/mlflow_backfill.py` and the `mlflow` docker image itself need
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (already set at the top of
  the script for local runs).
- `uv sync` / `uv run` may need `--native-tls` behind a corporate proxy
  (`invalid peer certificate: UnknownIssuer` otherwise).
