from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.evaluator import compute_rouge, tokenize_vietnamese_for_rouge


def test_vietnamese_tokenizer_preserves_accents_and_normalizes_nfc() -> None:
    composed = "Việt Nam, PHÁT triển 2026!"
    decomposed = "Vie\u0323\u0302t Nam, PHA\u0301T trie\u0302\u0309n 2026!"

    expected = ["việt", "nam", "phát", "triển", "2026"]
    assert tokenize_vietnamese_for_rouge(composed) == expected
    assert tokenize_vietnamese_for_rouge(decomposed) == expected
    assert tokenize_vietnamese_for_rouge("Hà Hạ Ha") == ["hà", "hạ", "ha"]


def test_vietnamese_tokenizer_does_not_create_false_ascii_overlap() -> None:
    prediction_tokens = set(tokenize_vietnamese_for_rouge("Hà Nội đẹp"))
    reference_tokens = set(
        tokenize_vietnamese_for_rouge("Hồ Chí Minh lớn")
    )

    assert prediction_tokens == {"hà", "nội", "đẹp"}
    assert reference_tokens == {"hồ", "chí", "minh", "lớn"}
    assert prediction_tokens.isdisjoint(reference_tokens)


def test_compute_rouge_forwards_unicode_tokenizer_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRougeMetric:
        def compute(self, **kwargs: object) -> dict[str, float]:
            captured.update(kwargs)
            return {"rouge1": 0.12345, "rouge2": 0.5, "rougeL": 1.0}

    def fake_load(metric_name: str) -> FakeRougeMetric:
        captured["metric_name"] = metric_name
        return FakeRougeMetric()

    monkeypatch.setitem(sys.modules, "evaluate", SimpleNamespace(load=fake_load))

    predictions = ["Bệnh nhân đã ổn định"]
    references = ["Người bệnh ổn định"]
    scores = compute_rouge(predictions, references)

    assert captured["metric_name"] == "rouge"
    assert captured["predictions"] == predictions
    assert captured["references"] == references
    assert captured["use_stemmer"] is False
    assert captured["tokenizer"] is tokenize_vietnamese_for_rouge
    assert scores == {"rouge1": 12.35, "rouge2": 50.0, "rougeL": 100.0}
