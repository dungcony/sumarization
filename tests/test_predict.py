from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

import src.predict as predict_module
from src.config import (
    DataConfig,
    GenerationConfig,
    ModelConfig,
    SummarizationConfig,
)


class FakeBatch(dict[str, Any]):
    def to(self, device: object) -> "FakeBatch":
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, texts: object, **kwargs: object) -> FakeBatch:
        self.calls.append((texts, kwargs))
        batch_size = 1 if isinstance(texts, str) else len(texts)  # type: ignore[arg-type]
        return FakeBatch(input_ids=[[1]] * batch_size)

    def decode(self, tokens: object, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "  tóm tắt đơn  "

    def batch_decode(
        self,
        outputs: list[list[int]],
        skip_special_tokens: bool,
    ) -> list[str]:
        assert skip_special_tokens is True
        return [f"  tóm tắt {index}  " for index in range(len(outputs))]


class FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.eval_called = False

    def to(self, device: object) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        return [[99] for _ in input_ids]  # type: ignore[union-attr]


def _config() -> SummarizationConfig:
    return SummarizationConfig(
        model=ModelConfig(
            name_or_path="base-from-config",
            use_fast_tokenizer=False,
        ),
        data=DataConfig(
            source_prefix="medical: ",
            max_source_length=321,
        ),
        generation=GenerationConfig(
            max_length=87,
            min_length=7,
            num_beams=2,
            length_penalty=0.8,
            no_repeat_ngram_size=4,
            repetition_penalty=1.17,
            early_stopping=False,
        ),
    )


def _generation_kwargs(call: dict[str, object]) -> dict[str, object]:
    names = {
        "max_length",
        "min_length",
        "num_beams",
        "length_penalty",
        "no_repeat_ngram_size",
        "repetition_penalty",
        "early_stopping",
    }
    return {name: call[name] for name in names}


def test_generation_override_is_a_copy_and_does_not_mutate_config() -> None:
    config = _config()
    original = asdict(config.generation)

    resolved = predict_module._resolve_generation_config(
        config,
        num_beams=6,
        max_length=None,
        min_length=None,
        repetition_penalty=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
    )

    assert resolved is not config.generation
    assert resolved.num_beams == 6
    assert resolved.max_length == config.generation.max_length
    assert asdict(config.generation) == original


def test_single_and_batch_use_same_config_and_forward_adapter_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    original_config = asdict(config)
    load_calls: list[dict[str, object]] = []
    models: list[FakeModel] = []
    tokenizers: list[FakeTokenizer] = []

    def fake_load_model_for_inference(
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        adapter_path: str | None = None,
    ) -> tuple[FakeTokenizer, FakeModel]:
        load_calls.append(
            {
                "model_config": model_config,
                "generation_config": generation_config,
                "adapter_path": adapter_path,
            }
        )
        tokenizer = FakeTokenizer()
        model = FakeModel()
        tokenizers.append(tokenizer)
        models.append(model)
        return tokenizer, model

    monkeypatch.setattr(
        predict_module,
        "load_model_for_inference",
        fake_load_model_for_inference,
    )
    monkeypatch.setattr(predict_module.torch.cuda, "is_available", lambda: False)

    single = predict_module.summarize(
        "Bệnh nhân đang hồi phục.",
        base_model_path="phase-1-best",
        adapter_path="phase-2-adapter",
        config=config,
    )
    batch = predict_module.summarize_batch(
        ["Ca bệnh thứ nhất.", "Ca bệnh thứ hai."],
        base_model_path="phase-1-best",
        adapter_path="phase-2-adapter",
        config=config,
        batch_size=2,
    )

    assert single == "tóm tắt đơn"
    assert batch == ["tóm tắt 0", "tóm tắt 1"]
    assert len(load_calls) == 2
    assert all(
        call["model_config"].name_or_path == "phase-1-best"  # type: ignore[union-attr]
        for call in load_calls
    )
    assert [call["adapter_path"] for call in load_calls] == [
        "phase-2-adapter",
        "phase-2-adapter",
    ]
    assert _generation_kwargs(models[0].generate_calls[0]) == (
        _generation_kwargs(models[1].generate_calls[0])
    )
    assert tokenizers[0].calls[0][1]["max_length"] == 321
    assert tokenizers[1].calls[0][1]["max_length"] == 321
    assert asdict(config) == original_config


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"model_path": "legacy-base"}, "legacy-base"),
        ({"base_model_path": "explicit-base"}, "explicit-base"),
    ],
)
def test_base_model_aliases_resolve_without_loading(
    kwargs: dict[str, str],
    expected: str,
) -> None:
    assert predict_module._resolve_base_model_path(
        kwargs.get("model_path"),
        kwargs.get("base_model_path"),
        None,
    ) == expected


def test_conflicting_base_model_aliases_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("model loader must not be called")

    monkeypatch.setattr(
        predict_module,
        "load_model_for_inference",
        should_not_load,
    )

    with pytest.raises(ValueError, match="Chỉ truyền một"):
        predict_module.summarize(
            "Văn bản",
            model_path="legacy-base",
            base_model_path="explicit-base",
        )
