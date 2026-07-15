"""
Unit tests for src/api/drift.py (prediction-drift PSI tracking).
"""

import importlib
import random

import pytest

from src.api import drift


@pytest.fixture(autouse=True)
def _reset_window():
    """Each test gets a fresh rolling window."""
    importlib.reload(drift)
    yield
    importlib.reload(drift)


def test_reference_distribution_sums_to_one():
    assert drift.REFERENCE_DISTRIBUTION
    assert sum(drift.REFERENCE_DISTRIBUTION.values()) == pytest.approx(1.0, abs=1e-3)


def test_psi_none_below_minimum_predictions():
    for _ in range(drift.MIN_PREDICTIONS_FOR_SCORE - 1):
        drift.record_prediction(next(iter(drift.REFERENCE_DISTRIBUTION)))
    assert drift.current_psi() is None


def test_psi_near_zero_when_matching_reference():
    """Feeding predictions in exactly the reference proportions should score ~0 drift.

    Allocates exactly WINDOW_SIZE picks (largest-remainder rounding, so the
    counts sum to WINDOW_SIZE precisely) and interleaves classes before
    inserting: the window is a maxlen=WINDOW_SIZE deque, so inserting one
    class at a time would let later classes evict earlier ones and leave the
    window unrepresentative of the mix.
    """
    items = list(drift.REFERENCE_DISTRIBUTION.items())
    raw = [(code, pct * drift.WINDOW_SIZE) for code, pct in items]
    counts = {code: int(share) for code, share in raw}
    remainder = drift.WINDOW_SIZE - sum(counts.values())
    for code, share in sorted(raw, key=lambda x: x[1] - int(x[1]), reverse=True)[:remainder]:
        counts[code] += 1
    assert sum(counts.values()) == drift.WINDOW_SIZE

    predictions = [code for code, n in counts.items() for _ in range(n)]
    random.Random(42).shuffle(predictions)
    for code in predictions:
        drift.record_prediction(code)

    psi = drift.current_psi()
    assert psi is not None
    assert psi < 0.05


def test_psi_high_when_one_class_dominates():
    """All predictions landing on a single (low-frequency) class should score high drift."""
    rare_code = min(drift.REFERENCE_DISTRIBUTION, key=drift.REFERENCE_DISTRIBUTION.get)
    for _ in range(50):
        drift.record_prediction(rare_code)
    psi = drift.current_psi()
    assert psi is not None
    assert psi > 0.25


def test_window_size_reports_current_fill():
    assert drift.window_size() == 0
    drift.record_prediction(next(iter(drift.REFERENCE_DISTRIBUTION)))
    assert drift.window_size() == 1


def test_window_caps_at_max_size():
    code = next(iter(drift.REFERENCE_DISTRIBUTION))
    for _ in range(drift.WINDOW_SIZE + 20):
        drift.record_prediction(code)
    assert drift.window_size() == drift.WINDOW_SIZE
