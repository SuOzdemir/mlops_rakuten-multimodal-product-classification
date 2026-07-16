import json
from types import SimpleNamespace

import pytest

from promotion.promote import _passes_gate, _read_best_metric


class FakeClient:
    def __init__(self, champion=None):
        self.champion = champion

    def get_model_version_by_alias(self, _name, _alias):
        if self.champion is None:
            raise RuntimeError("alias not found")
        return self.champion


def test_read_best_metric(tmp_path):
    metadata = tmp_path / "run_metadata.json"
    metadata.write_text(json.dumps({"best_macro_f1": 0.8123}), encoding="utf-8")
    assert _read_best_metric(metadata) == pytest.approx(0.8123)


def test_gate_allows_first_champion(monkeypatch):
    monkeypatch.delenv("PROMOTION_MIN_F1_GAIN", raising=False)
    passed, reason = _passes_gate(FakeClient(), "text", 0.8)
    assert passed is True
    assert "bootstrap" in reason.lower()


def test_gate_rejects_champion_without_comparable_metric(monkeypatch):
    monkeypatch.delenv("PROMOTION_MIN_F1_GAIN", raising=False)
    monkeypatch.delenv("PROMOTION_ALLOW_UNTAGGED_CHAMPION", raising=False)
    champion = SimpleNamespace(tags={})
    passed, reason = _passes_gate(FakeClient(champion), "image", 0.7)
    assert passed is False
    assert "no comparable" in reason.lower()


def test_gate_can_migrate_an_untagged_champion_once(monkeypatch):
    monkeypatch.setenv("PROMOTION_ALLOW_UNTAGGED_CHAMPION", "true")
    champion = SimpleNamespace(tags={})
    passed, reason = _passes_gate(FakeClient(champion), "text", 0.8)
    assert passed is True
    assert "migration override" in reason.lower()


def test_gate_requires_configured_improvement(monkeypatch):
    monkeypatch.setenv("PROMOTION_MIN_F1_GAIN", "0.01")
    champion = SimpleNamespace(tags={"text_best_macro_f1": "0.80"})
    assert _passes_gate(FakeClient(champion), "text", 0.809)[0] is False
    assert _passes_gate(FakeClient(champion), "text", 0.81)[0] is True
