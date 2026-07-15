"""
Prediction drift: compares the live rolling window of predicted classes
against the training set's class distribution (src/api/reference_class_distribution.json),
using the Population Stability Index (PSI) -- a standard categorical-drift
statistic, sum[(actual - expected) * ln(actual / expected)] over classes.

Read as: <0.1 no significant drift, 0.1-0.25 moderate, >0.25 significant.

Keyed by prdtypecode (the raw Rakuten product-type code), not label_id/class_id
-- label_id is an arbitrary index tied to whichever label2id.json produced it,
and the serving predictor's copy isn't guaranteed to match the one used to
build the reference distribution. prdtypecode is stable across both.

Window is in-memory (a deque of the last WINDOW_SIZE predicted prdtypecodes)
-- resets on API restart, and doesn't need a database. Fine for a
single-process demo; a multi-worker/multi-replica deployment would need a
shared store (Redis, Postgres) instead.
"""

import json
import math
from collections import Counter, deque
from pathlib import Path

REFERENCE_PATH = Path(__file__).resolve().parent / "reference_class_distribution.json"
WINDOW_SIZE = 200
MIN_PREDICTIONS_FOR_SCORE = 30  # below this, the window is too noisy to score
_EPSILON = 1e-4  # avoids log(0)/div-by-0 for classes absent from a window

with open(REFERENCE_PATH, encoding="utf-8") as f:
    _raw_reference = json.load(f)
REFERENCE_DISTRIBUTION: dict[int, float] = {int(k): v for k, v in _raw_reference.items()}

_window: deque[int] = deque(maxlen=WINDOW_SIZE)


def record_prediction(prdtypecode: int) -> None:
    _window.append(prdtypecode)


def current_psi() -> float | None:
    """None until MIN_PREDICTIONS_FOR_SCORE predictions have been recorded."""
    if len(_window) < MIN_PREDICTIONS_FOR_SCORE:
        return None

    counts = Counter(_window)
    total = len(_window)

    psi = 0.0
    for prdtypecode, expected_pct in REFERENCE_DISTRIBUTION.items():
        actual_pct = counts.get(prdtypecode, 0) / total
        actual_pct = max(actual_pct, _EPSILON)
        expected_pct = max(expected_pct, _EPSILON)
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)

    return psi


def window_size() -> int:
    return len(_window)
