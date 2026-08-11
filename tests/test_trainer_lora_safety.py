from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import LoraConfig, SummarizationConfig, validate_config
from src.trainer import (
    _build_adapter_manifest,
    _verify_lora_resume_manifest,
    _verify_lora_trainable_parameters,
)
from src.utils import save_json


class TinyPeftLikeModel(torch.nn.Module):
    """Small local module whose parameter names resemble a PEFT model."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.lora_A = torch.nn.Parameter(torch.ones(2, 1))
        self.lora_B = torch.nn.Parameter(torch.ones(1, 2))


def test_lora_safety_accepts_only_lora_parameters_as_trainable() -> None:
    model = TinyPeftLikeModel()
    model.base.requires_grad_(False)

    stats = _verify_lora_trainable_parameters(model)

    assert stats == {
        "total": 8,
        "trainable": 4,
        "frozen": 4,
        "trainable_percent": 50.0,
        "trainable_tensor_count": 2,
    }


def test_lora_safety_rejects_a_trainable_base_parameter() -> None:
    model = TinyPeftLikeModel()

    with pytest.raises(RuntimeError, match=r"base\.weight"):
        _verify_lora_trainable_parameters(model)


def test_lora_safety_rejects_when_nothing_is_trainable() -> None:
    model = TinyPeftLikeModel()
    model.requires_grad_(False)

    with pytest.raises(RuntimeError, match="không có tham số"):
        _verify_lora_trainable_parameters(model)


def test_config_rejects_freeze_encoder_combined_with_lora() -> None:
    config = SummarizationConfig(lora=LoraConfig(enabled=True))
    config.training.freeze_encoder = True

    with pytest.raises(ValueError, match="freeze_encoder=false"):
        validate_config(config)


def _resume_fixture(tmp_path: Path) -> tuple[SummarizationConfig, dict]:
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    config = SummarizationConfig(lora=LoraConfig(enabled=True))
    config.phase.name = "phase_2_lora"
    config.training.output_dir = str(output_dir)
    config.training.resume_from_checkpoint = str(checkpoint)
    fingerprint = {
        "algorithm": "sha256",
        "digest": "abc123",
        "files": [{"name": "model.safetensors", "size": 3}],
    }
    manifest = _build_adapter_manifest(
        config,
        output_dir / "best",
        fingerprint,
        {"total": 10, "trainable": 1},
    )
    save_json(manifest, output_dir / "adapter_manifest.json")
    return config, fingerprint


def test_lora_resume_accepts_same_base_and_adapter_config(tmp_path: Path) -> None:
    config, fingerprint = _resume_fixture(tmp_path)

    _verify_lora_resume_manifest(
        config,
        Path(config.training.output_dir),
        fingerprint,
    )


def test_lora_resume_rejects_different_base_fingerprint(tmp_path: Path) -> None:
    config, _ = _resume_fixture(tmp_path)
    wrong = {"algorithm": "sha256", "digest": "wrong", "files": []}

    with pytest.raises(ValueError, match="Phase 1 base khác"):
        _verify_lora_resume_manifest(
            config,
            Path(config.training.output_dir),
            wrong,
        )
