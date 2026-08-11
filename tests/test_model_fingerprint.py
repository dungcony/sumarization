from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model import (
    fingerprint_full_checkpoint,
    verify_adapter_base_dependency,
)


def _write_base(path: Path, weights: bytes = b"phase-one-weights") -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        '{"model_type": "t5"}', encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(weights)
    return path


def _write_adapter(path: Path, fingerprint: dict[str, object]) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "base_model_dependency": {"fingerprint": fingerprint},
    }
    (path / "adapter_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def test_fingerprint_is_path_independent_and_verifies_matching_base(
    tmp_path: Path,
) -> None:
    first = _write_base(tmp_path / "first")
    relocated = _write_base(tmp_path / "relocated")

    fingerprint = fingerprint_full_checkpoint(first)
    assert fingerprint is not None
    assert fingerprint_full_checkpoint(relocated) == fingerprint

    adapter = _write_adapter(tmp_path / "adapter", fingerprint)
    assert verify_adapter_base_dependency(relocated, adapter) is True


def test_base_adapter_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    expected_base = _write_base(tmp_path / "expected", b"expected")
    wrong_base = _write_base(tmp_path / "wrong", b"wrong")
    fingerprint = fingerprint_full_checkpoint(expected_base)
    assert fingerprint is not None
    adapter = _write_adapter(tmp_path / "adapter", fingerprint)

    with pytest.raises(ValueError, match="không thuộc checkpoint base"):
        verify_adapter_base_dependency(wrong_base, adapter)


def test_fingerprinted_adapter_rejects_unverifiable_hub_base(
    tmp_path: Path,
) -> None:
    expected_base = _write_base(tmp_path / "expected")
    fingerprint = fingerprint_full_checkpoint(expected_base)
    assert fingerprint is not None
    adapter = _write_adapter(tmp_path / "adapter", fingerprint)

    with pytest.raises(ValueError, match="không phải full checkpoint local"):
        verify_adapter_base_dependency("VietAI/vit5-base", adapter)


def test_legacy_adapter_without_manifest_remains_loadable(
    tmp_path: Path,
) -> None:
    base = _write_base(tmp_path / "base")
    adapter = tmp_path / "legacy-adapter"
    adapter.mkdir()

    assert verify_adapter_base_dependency(base, adapter) is False
