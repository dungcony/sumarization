"""
Model & Tokenizer Loading (Tải Mô hình & Bộ tạo token)
=========================

Tải các mô hình seq2seq tiền huấn luyện và các bộ tạo token (tokenizer) cho việc tóm tắt tiếng Việt.

Các mô hình được hỗ trợ:
    - VietAI/vit5-base              (T5 tiếng Việt, kết quả tốt nhất)
    - VietAI/vit5-base-vietnews-summarization  (khởi động ấm - warm-start)
    - vinai/bartpho-syllable        (BART tiếng Việt)
    - google/mt5-base               (T5 đa ngôn ngữ)

Các tính năng:
    - Tự động phát hiện loại tokenizer (SentencePiece vs fast)
    - LoRA (Low-Rank Adaptation) thông qua PEFT
    - Ghi đè Dropout
    - Checkpoint gradient (Gradient checkpointing)
    - Đóng băng bộ mã hóa (Encoder freezing)
    - Xác thực số lượng tham số (<3B tham số)

Ví dụ:
    >>> from src.model import load_tokenizer, load_model
    >>> tokenizer = load_tokenizer(config.model)
    >>> model = load_model(config.model, tokenizer)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

from src.config import GenerationConfig as GenConfigDC
from src.config import LoraConfig, ModelConfig
from src.utils import count_parameters, format_number, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Tải tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_config: ModelConfig) -> Any:
    """Tải tokenizer cho một mô hình đã được tiền huấn luyện.

    Xử lý các trường hợp đặc biệt:
        - Các mô hình T5/ViT5: Sử dụng tokenizer SentencePiece chậm khi
          use_fast_tokenizer=False (tránh các vấn đề tương thích đã biết).
        - Các mô hình BART: Sử dụng AutoTokenizer tiêu chuẩn.

    Tham số:
        model_config: Cấu hình mô hình với name_or_path và các tùy chọn.

    Trả về:
        Đối tượng tokenizer của HuggingFace.

    Ví dụ:
        >>> tokenizer = load_tokenizer(config.model)
        >>> tokens = tokenizer("Xin chào Việt Nam")
    """
    model_name = model_config.name_or_path
    use_fast = model_config.use_fast_tokenizer

    # Xử lý đặc biệt cho SentencePiece tokenizer của T5/ViT5
    is_t5_model = _is_t5_family(model_name)
    if is_t5_model and not use_fast:
        logger.info(f"Đang tải T5 SentencePiece tokenizer cho: {model_name}")
        try:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                legacy=True,
                cache_dir=model_config.cache_dir,
            )
        except Exception as e:
            logger.warning(
                f"T5Tokenizer thất bại ({e}), chuyển về AutoTokenizer"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                cache_dir=model_config.cache_dir,
            )
    else:
        logger.info(f"Đang tải tokenizer cho: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=model_config.cache_dir,
        )

    logger.info(
        f"Đã tải tokenizer: vocab_size={tokenizer.vocab_size}, "
        f"type={type(tokenizer).__name__}"
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Tải mô hình
# ---------------------------------------------------------------------------

def load_model(
    model_config: ModelConfig,
    tokenizer: Any,
    generation_config: GenConfigDC | None = None,
) -> Any:
    """Tải một mô hình seq2seq tiền huấn luyện.

    Các bước:
        1. Tải cấu hình mô hình (AutoConfig)
        2. Ghi đè dropout nếu được chỉ định
        3. Tải các trọng số của mô hình (AutoModelForSeq2SeqLM)
        4. Sửa token pad nếu bị thiếu
        5. Thay đổi kích thước (resize) embeddings để khớp với tokenizer
        6. Cài đặt các tham số mặc định cho quá trình sinh (generation)
        7. Xác thực số lượng tham số

    Tham số:
        model_config: Cấu hình mô hình.
        tokenizer: Tokenizer đã tải (cần thiết cho việc thay đổi kích thước embedding).
        generation_config: Tùy chọn các tham số cho quá trình sinh.

    Trả về:
        Mô hình PyTorch đã được tải.

    Ngoại lệ:
        ValueError: Nếu mô hình vượt quá giới hạn max_parameters.
    """
    model_name = model_config.name_or_path
    logger.info(f"Đang tải mô hình: {model_name}")

    # Bước 1: Tải cấu hình
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
    )

    # Bước 2: Ghi đè dropout nếu được chỉ định
    if model_config.dropout is not None:
        _set_dropout(config, model_config.dropout)
        logger.info(f"Đã ghi đè dropout thành: {model_config.dropout}")

    # Bước 3: Tải mô hình
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
    )

    # Bước 4: Sửa token pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        logger.info("Đã đặt pad_token thành eos_token")

    # Bước 5: Đổi kích thước embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Bước 6: Đặt cấu hình sinh (generation config)
    if generation_config:
        gen_cfg = getattr(model, "generation_config", GenerationConfig())
        gen_cfg.max_length = generation_config.max_length
        gen_cfg.max_new_tokens = generation_config.max_new_tokens
        gen_cfg.min_length = generation_config.min_length
        gen_cfg.num_beams = generation_config.num_beams
        gen_cfg.length_penalty = generation_config.length_penalty
        gen_cfg.no_repeat_ngram_size = generation_config.no_repeat_ngram_size
        gen_cfg.repetition_penalty = generation_config.repetition_penalty
        gen_cfg.do_sample = generation_config.do_sample
        gen_cfg.early_stopping = generation_config.early_stopping
        
        # Sửa lỗi mất decoder_start_token_id khi khởi tạo generation_config mới
        if gen_cfg.decoder_start_token_id is None:
            gen_cfg.decoder_start_token_id = getattr(model.config, "decoder_start_token_id", tokenizer.pad_token_id)
            
        model.generation_config = gen_cfg

    # Bước 7: Xác thực số lượng tham số
    params = count_parameters(model)
    logger.info(
        f"Đã tải mô hình: {format_number(params['total'])} tổng số tham số, "
        f"{format_number(params['trainable'])} có thể huấn luyện "
        f"({params['trainable_percent']}%)"
    )

    if params["total"] > model_config.max_parameters:
        raise ValueError(
            f"Mô hình có {format_number(params['total'])} tham số, "
            f"vượt quá giới hạn {format_number(model_config.max_parameters)}"
        )

    return model


def is_lora_adapter_checkpoint(path: str | Path) -> bool:
    """Kiểm tra một thư mục cục bộ có phải PEFT/LoRA adapter không.

    PEFT lưu metadata bắt buộc trong ``adapter_config.json``. Hàm này
    chỉ kiểm tra đường dẫn cục bộ; HuggingFace Hub model ID không bị
    truy cập mạng chỉ để phát hiện loại checkpoint.
    """
    checkpoint_path = Path(path).expanduser()
    return (
        checkpoint_path.is_dir()
        and (checkpoint_path / "adapter_config.json").is_file()
    )


def fingerprint_full_checkpoint(
    path: str | Path,
) -> dict[str, Any] | None:
    """Tạo SHA-256 fingerprint cho config, tokenizer và weight shards local.

    Trả về ``None`` với HuggingFace Hub ID/đường dẫn không phải thư mục local.
    Fingerprint không chứa absolute path nên vẫn giữ nguyên khi artifact được
    chuyển từ Kaggle Output sang Kaggle Dataset hoặc máy deploy.
    """
    checkpoint_dir = Path(path).expanduser()
    if not checkpoint_dir.is_dir():
        return None

    config_path = checkpoint_dir / "config.json"
    weight_paths = sorted({
        *checkpoint_dir.glob("model*.safetensors"),
        *checkpoint_dir.glob("pytorch_model*.bin"),
    })
    weight_paths = [item for item in weight_paths if item.is_file()]
    if not config_path.is_file() or not weight_paths:
        return None

    tokenizer_paths = [
        checkpoint_dir / name
        for name in (
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        if (checkpoint_dir / name).is_file()
    ]
    files = [config_path, *tokenizer_paths, *weight_paths]
    digest = hashlib.sha256()
    records: list[dict[str, str | int]] = []
    for file_path in files:
        relative_name = file_path.relative_to(checkpoint_dir).as_posix()
        size = file_path.stat().st_size
        records.append({"name": relative_name, "size": size})
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "files": records,
    }


def verify_adapter_base_dependency(
    base_model_path: str | Path,
    adapter_path: str | Path,
) -> bool:
    """Xác minh adapter đang được gắn đúng full checkpoint đã dùng để train.

    Adapter cũ không có ``adapter_manifest.json`` vẫn được nạp để giữ tương
    thích, nhưng hàm log cảnh báo và trả về ``False``. Adapter do pipeline mới
    tạo có fingerprint; mismatch là lỗi cứng trước khi model lớn được tải.
    """
    adapter_dir = Path(adapter_path).expanduser()
    manifest_path = adapter_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        logger.warning(
            "Adapter không có adapter_manifest.json; không thể xác minh đúng "
            "checkpoint base. Chỉ dùng artifact này nếu provenance đã được "
            "kiểm tra bằng cách khác."
        )
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Không đọc được adapter manifest '{manifest_path}': {exc}"
        ) from exc

    dependency = manifest.get("base_model_dependency", {})
    expected = dependency.get("fingerprint")
    if not isinstance(expected, dict) or not expected.get("digest"):
        logger.warning(
            "Adapter manifest không có base fingerprint; không thể xác minh "
            "đúng cặp base-adapter."
        )
        return False

    actual = fingerprint_full_checkpoint(base_model_path)
    if actual is None:
        raise ValueError(
            "Adapter có base fingerprint nhưng base_model_path không phải "
            "full checkpoint local có thể xác minh. Hãy truyền đúng thư mục "
            "Phase 1 best/, không dùng Hub ID hoặc adapter path."
        )

    if (
        expected.get("algorithm") != actual["algorithm"]
        or expected.get("digest") != actual["digest"]
    ):
        raise ValueError(
            "LoRA adapter không thuộc checkpoint base đã chọn: fingerprint "
            f"mong đợi={expected.get('digest')}, thực tế={actual['digest']}."
        )

    logger.info("Xác minh fingerprint base-adapter thành công")
    return True


def load_model_for_inference(
    model_config: ModelConfig,
    generation_config: GenConfigDC | None = None,
    adapter_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Tải tokenizer và model cho inference, có thể gắn LoRA adapter.

    ``model_config.name_or_path`` luôn phải trỏ tới full base model (ví dụ
    checkpoint Phase 1). Khi ``adapter_path`` được cung cấp, base model được
    giữ nguyên và adapter được nạp bằng ``PeftModel``; hàm không merge
    adapter vào base weights.

    Tham số:
        model_config: Cấu hình của full base model/checkpoint.
        generation_config: Thiết lập generation tùy chọn.
        adapter_path: Thư mục LoRA cục bộ có ``adapter_config.json``.

    Trả về:
        Tuple ``(tokenizer, model)`` sẵn sàng cho inference.

    Ngoại lệ:
        ValueError: Khi base path thực chất là adapter, hoặc adapter không
            có metadata bắt buộc.
        FileNotFoundError: Khi ``adapter_path`` không tồn tại.
        RuntimeError: Khi adapter không thể gắn vào base model.

    Ví dụ:
        >>> tokenizer, model = load_model_for_inference(
        ...     phase_1_model_config,
        ...     generation_config,
        ...     adapter_path="outputs_phase_2/vit5_base/best",
        ... )
    """
    base_model_path = model_config.name_or_path
    if is_lora_adapter_checkpoint(base_model_path):
        raise ValueError(
            f"Đã phát hiện LoRA adapter tại '{base_model_path}', nhưng tham số "
            "model/base phải trỏ tới full checkpoint. Hãy truyền checkpoint "
            "Phase 1 làm base và truyền thư mục này qua adapter_path."
        )

    adapter_dir: Path | None = None
    if adapter_path is not None:
        # Xác thực adapter trước khi tải base model lớn để fail fast.
        adapter_dir = Path(adapter_path).expanduser()
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục LoRA adapter: '{adapter_dir}'."
            )
        if not adapter_dir.is_dir():
            raise ValueError(
                f"LoRA adapter phải là một thư mục, nhưng nhận được: "
                f"'{adapter_dir}'."
            )

        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise ValueError(
                f"'{adapter_dir}' không phải LoRA adapter hợp lệ: thiếu "
                "adapter_config.json. Nếu đây là full checkpoint, hãy truyền nó "
                "qua model_path/base_model_path và bỏ adapter_path."
            )

        verify_adapter_base_dependency(base_model_path, adapter_dir)

    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config, tokenizer, generation_config)

    if adapter_dir is None:
        model.eval()
        return tokenizer, model

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Cần cài package 'peft' để nạp LoRA adapter."
        ) from exc

    logger.info(
        f"Đã phát hiện LoRA adapter: {adapter_dir}; "
        f"đang gắn vào base model: {base_model_path}"
    )
    try:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            is_trainable=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Không thể nạp LoRA adapter '{adapter_dir}' vào base model "
            f"'{base_model_path}'. Hãy kiểm tra adapter đã được train từ "
            f"đúng checkpoint Phase 1. Chi tiết: {exc}"
        ) from exc

    model.eval()
    logger.info("Nạp LoRA adapter thành công (không merge vào base model)")
    return tokenizer, model


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation)
# ---------------------------------------------------------------------------

def apply_lora(model: Any, lora_config: LoraConfig) -> Any:
    """Áp dụng bộ chuyển đổi LoRA vào mô hình để huấn luyện hiệu quả tham số.

    LoRA đóng băng các trọng số ban đầu của mô hình và thêm các ma trận
    phân rã hạng (rank-decomposition) nhỏ có thể huấn luyện vào các lớp được chỉ định. Điều này làm giảm đáng kể
    số lượng tham số có thể huấn luyện.

    Tham số:
        model: Mô hình cơ sở để thêm các bộ chuyển đổi LoRA.
        lora_config: Cấu hình LoRA (rank, alpha, các module mục tiêu).

    Trả về:
        Mô hình PEFT có chứa các bộ chuyển đổi LoRA.

    Ví dụ:
        >>> model = load_model(config.model, tokenizer)
        >>> model = apply_lora(model, config.lora)
        >>> # Bây giờ chỉ khoảng ~2% tham số có thể huấn luyện
    """
    if not lora_config.enabled:
        logger.info("LoRA bị vô hiệu hóa, sử dụng fine-tuning toàn bộ (full fine-tuning)")
        return model

    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    # Xác định các module mục tiêu
    target_modules = _get_lora_target_modules(
        model, lora_config.target_modules
    )

    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)

    params = count_parameters(model)
    logger.info(
        f"Đã áp dụng LoRA (rank={lora_config.r}): "
        f"{format_number(params['trainable'])} tham số có thể huấn luyện "
        f"({params['trainable_percent']}% trong số {format_number(params['total'])})"
    )

    return model


# ---------------------------------------------------------------------------
# Các hàm trợ giúp cấu hình mô hình
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: Any) -> None:
    """Kích hoạt checkpoint gradient để giảm việc sử dụng bộ nhớ.

    Đánh đổi tính toán để lấy bộ nhớ: tính toán lại các kích hoạt (activations) trong bước backward
    thay vì lưu trữ chúng. Giảm khoảng ~40% bộ nhớ nhưng làm huấn luyện chậm đi khoảng ~20%.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Đã kích hoạt gradient checkpointing")
    else:
        logger.warning("Mô hình không hỗ trợ gradient checkpointing")


def freeze_encoder(model: Any) -> None:
    """Đóng băng các tham số của bộ mã hóa (chỉ huấn luyện bộ giải mã + cross-attention).

    Hữu ích khi:
        - Sử dụng một bộ mã hóa tiền huấn luyện mạnh (ví dụ: ViT5 VietNews)
        - Muốn giảm thời gian huấn luyện
        - Có bộ nhớ GPU hạn chế
    """
    if hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen = sum(
            p.numel() for p in model.encoder.parameters()
        )
        logger.info(f"Đã đóng băng bộ mã hóa: {format_number(frozen)} tham số")
    else:
        logger.warning("Mô hình không có thuộc tính 'encoder' để đóng băng")


# ---------------------------------------------------------------------------
# Các hàm trợ giúp nội bộ
# ---------------------------------------------------------------------------

def _is_t5_family(model_name: str) -> bool:
    """Kiểm tra xem mô hình có thuộc dòng T5 không (T5, ViT5, mT5)."""
    name_lower = model_name.lower()
    return any(keyword in name_lower for keyword in ["t5", "mt5"])


def _set_dropout(config: Any, dropout: float) -> None:
    """Cài đặt dropout trên cấu hình mô hình (xử lý các dòng mô hình khác nhau)."""
    # Các mô hình dòng T5
    if hasattr(config, "dropout_rate"):
        config.dropout_rate = dropout
    # Các mô hình dòng BART
    if hasattr(config, "dropout"):
        config.dropout = dropout
    if hasattr(config, "attention_dropout"):
        config.attention_dropout = dropout
    if hasattr(config, "activation_dropout"):
        config.activation_dropout = dropout


def _get_lora_target_modules(
    model: Any,
    target_modules: str,
) -> list[str]:
    """Xác định các module nào cần áp dụng LoRA.

    Tự động phát hiện dựa trên kiến trúc mô hình:
        - T5/mT5: ['q', 'v'] (các phép chiếu query và value của attention)
        - BART:    ['q_proj', 'v_proj']
    """
    if target_modules != "auto":
        return [m.strip() for m in target_modules.split(",")]

    # Tự động phát hiện dựa trên loại mô hình
    model_type = getattr(model.config, "model_type", "").lower()

    if model_type in ("t5", "mt5"):
        modules = ["q", "v"]
    elif "bart" in model_type:
        modules = ["q_proj", "v_proj"]
    else:
        # Dự phòng: cố gắng tìm các module attention
        modules = ["q_proj", "v_proj"]
        logger.warning(
            f"Loại mô hình không xác định '{model_type}', dùng mặc định là {modules}"
        )

    logger.info(f"Các module mục tiêu của LoRA (tự động): {modules}")
    return modules
