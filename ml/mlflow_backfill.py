"""
ml/mlflow_backfill.py
======================
Chapter 11 — MLflow: Experiment Tracking and Model Registry.

Backfills MLflow with the already-completed I12 (image), T8 (text) and
LateFusion (multimodal) runs from data/model_artifacts/, without retraining
(checkpoints and metrics already exist on disk — see ml/README.md for how
that folder is populated).

This is a legacy history-import tool only. Production registration and
deployment are handled by the Airflow promotion DAG.

Experiment naming:
  - "rakuten-image-modeling" / "rakuten-text-modeling"
        Same experiments Sumeyra's train.py already logs to (per-family
        comparison — image and text runs have very different param/metric
        schemas, so keeping them separate avoids a messy mixed leaderboard).
  - "rakuten-multimodal-classifier"
        The experiment name required verbatim by the task sheet (11.2).
        Used for the final fusion run and the Model Registry entry.

Usage:
    uv run python ml/mlflow_backfill.py

Env:
    MLFLOW_TRACKING_URI   default: http://localhost:5001
                          (matches docker-compose's `mlflow` service; point
                          it at http://mlflow:5001 if running inside compose)
"""

import json
import os
import re

# mlflow's bundled protobuf-generated code is incompatible with this
# project's pinned protobuf>=7.34.0 (raises "Descriptors cannot be created
# directly" on import). Fall back to the pure-Python implementation.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys

# mlflow prints emoji status lines (e.g. "\U0001f3c3 View run ...") which crash
# on Windows' default cp1252 console encoding. Force UTF-8 stdout/stderr.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path

import mlflow

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "data" / "model_artifacts"
REGISTERED_MODEL_NAME = "rakuten-multimodal-classifier"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))

_MACRO_AVG_RE = re.compile(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def parse_macro_avg(report_path: Path) -> dict:
    """Pull macro-avg precision/recall/f1 out of a sklearn classification_report.txt."""
    text = report_path.read_text(encoding="utf-8")
    m = _MACRO_AVG_RE.search(text)
    if not m:
        print(f"  [warn] could not find 'macro avg' row in {report_path}")
        return {}
    precision, recall, f1 = (float(x) for x in m.groups())
    return {"precision_macro": precision, "recall_macro": recall, "f1_macro_from_report": f1}


def log_backfilled_run(
    experiment: str,
    run_name: str,
    model_dir: Path,
    params: dict,
    metrics: dict,
    artifact_files: list[str],
    history_csv: Path | None = None,
    history_cols: dict | None = None,
) -> str:
    """Open one MLflow run and log params/metrics/artifacts for an already-completed model."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "backfilled": "true",
            "source_repo": "Project_01_Rakuten_Multimodal",
            "note": "Logged from pre-existing training artifacts, not a live training run.",
        })

        if history_csv is not None and history_csv.exists() and history_cols:
            import csv
            with open(history_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    epoch = int(row["epoch"])
                    step_metrics = {
                        mlflow_name: float(row[col_name])
                        for mlflow_name, col_name in history_cols.items()
                        if col_name in row
                    }
                    mlflow.log_metrics(step_metrics, step=epoch)

        for fname in artifact_files:
            fpath = model_dir / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath))
            else:
                print(f"  [skip] artifact not found: {fpath}")

        run_id = run.info.run_id
    print(f"  logged run '{run_name}' -> {experiment}  (run_id={run_id})")
    return run_id


def backfill_i12() -> tuple[str, dict]:
    print("[I12 image model]")
    d = ARTIFACT_ROOT / "I12_ConvNeXt_Base_ModerateAug_Full"
    meta = json.loads((d / "model_metadata.json").read_text(encoding="utf-8"))
    report_stats = parse_macro_avg(d / "val_classification_report.txt")

    params = {
        "model_type": "image",
        "image_model_name": meta["architecture"],
        "learning_rate_backbone": meta["learning_rates"]["backbone"],
        "learning_rate_head": meta["learning_rates"]["classifier_head"],
        "batch_size": meta["batch_size"],
        "num_epochs": meta["max_epochs"],
        "weight_decay": meta["weight_decay"],
        "image_size": meta["image_size"],
    }
    metrics = {
        "accuracy": meta["accuracy"],
        "macro_f1": meta["macro_f1"],
        "weighted_f1": meta["weighted_f1"],
        **report_stats,
    }
    run_id = log_backfilled_run(
        "rakuten-image-modeling",
        meta["run_name"],
        d,
        params,
        metrics,
        artifact_files=[
            "best_model_state_dict.pt",
            "val_classification_report.txt",
            "val_confusion_matrix.png",
            "learning_curves.png",
        ],
        history_csv=d / "history.csv",
        history_cols={"train_loss": "t_loss", "train_f1": "t_f1", "val_loss": "v_loss", "val_f1": "v_f1"},
    )
    return run_id, metrics


def backfill_t8() -> tuple[str, dict]:
    print("[T8 text model]")
    d = ARTIFACT_ROOT / "T8_CamemBERT_FullFineTune_L128"
    meta = json.loads((d / "model_metadata.json").read_text(encoding="utf-8"))
    report_stats = parse_macro_avg(d / "val_classification_report.txt")

    params = {
        "model_type": "text",
        "text_model_name": meta["architecture"],
        "learning_rate": meta["learning_rate"],
        "batch_size": meta["batch_size"],
        "num_epochs": meta["max_epochs"],
        "weight_decay": meta["weight_decay"],
        "max_length": meta["max_length"],
    }
    metrics = {
        "accuracy": meta["accuracy"],
        "macro_f1": meta["macro_f1"],
        "weighted_f1": meta["weighted_f1"],
        **report_stats,
    }
    # No per-epoch history was preserved on disk for this run (see model_metadata.json notes) —
    # only final metrics are logged.
    run_id = log_backfilled_run(
        "rakuten-text-modeling",
        meta["run_name"],
        d,
        params,
        metrics,
        artifact_files=[
            "best_model_text.pt",
            "val_classification_report.txt",
            "confusion_matrix.png",
            "training_curves.png",
        ],
    )
    return run_id, metrics


def backfill_and_register_latefusion(image_run_id: str, text_run_id: str) -> tuple[str, dict]:
    print("[MM_CamemBERT_ConvNeXtBase_LateFusion]")
    d = ARTIFACT_ROOT / "MM_CamemBERT_ConvNeXtBase_LateFusion"
    meta = json.loads((d / "run_metadata.json").read_text(encoding="utf-8"))
    report_stats = parse_macro_avg(d / "fusion_classification_report.txt")

    params = {
        "model_type": "multimodal_fusion",
        "fusion_strategy": "late_fusion_weighted_logits",
        "image_model_name": "ConvNeXt-Base (I12)",
        "text_model_name": "CamemBERT-base (T8)",
        "alpha_image_weight": meta["best_alpha"],
        "source_image_run_id": image_run_id,
        "source_text_run_id": text_run_id,
    }
    metrics = {
        "accuracy": meta["accuracy"],
        "macro_f1": meta["best_macro_f1"],
        "weighted_f1": meta["weighted_f1"],
        "f1_image_only": meta["f1_image_only"],
        "f1_text_only": meta["f1_text_only"],
        **report_stats,
    }

    mlflow.set_experiment(REGISTERED_MODEL_NAME)
    with mlflow.start_run(run_name=meta["model_name"]) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "backfilled": "true",
            "source_repo": "Project_01_Rakuten_Multimodal",
        })
        for fname in ["confusion_matrix.png", "fusion_classification_report.txt", "val_predictions.csv"]:
            fpath = d / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath))

        run_id = run.info.run_id
    print(f"  logged historical run '{meta['model_name']}' (run_id={run_id})")
    return run_id, metrics


def main() -> None:
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}\n")

    image_run_id, _ = backfill_i12()
    text_run_id, _ = backfill_t8()
    backfill_and_register_latefusion(image_run_id, text_run_id)

    print("\nDone. Open the MLflow UI to inspect the imported historical runs.")


if __name__ == "__main__":
    main()
