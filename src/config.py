"""
Configuration System (Hệ thống cấu hình)
====================

Tải, xác thực và quản lý các cấu hình huấn luyện.

Các file cấu hình ở định dạng YAML với 5 phần:
    - model:      Mô hình tiền huấn luyện nào sẽ được sử dụng
    - data:       Đường dẫn tập dữ liệu và các tham số cho tokenizer
    - training:   Các siêu tham số (hyperparameters như epoch, learning rate, batch size...)
    - generation: Cài đặt giải mã (beam search, penalties...)
    - lora:       Cài đặt LoRA tùy chọn cho việc huấn luyện hiệu quả tham số

Ví dụ:
    >>> config = load_config("configs/vit5_base.yaml")
    >>> print(config.model.name_or_path)
    'VietAI/vit5-base'
    >>> config.training.learning_rate
    3e-05
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Định nghĩa Dataclass — một dataclass cho mỗi phần của cấu hình
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Mô hình nào cần tải và cách cấu hình nó."""

    name_or_path: str = "VietAI/vit5-base"
    """ID của mô hình HuggingFace hoặc đường dẫn cục bộ.
    Ví dụ: 'VietAI/vit5-base', 'vinai/bartpho-syllable', 'google/mt5-base'"""

    use_fast_tokenizer: bool = True
    """Sử dụng tokenizer nhanh (Rust). Đặt False cho ViT5/T5 (vấn đề của SentencePiece)."""

    trust_remote_code: bool = False
    """Cho phép sử dụng mã mô hình tùy chỉnh từ HuggingFace Hub."""

    cache_dir: Optional[str] = None
    """Thư mục để lưu trữ bộ đệm (cache) các mô hình đã tải về."""

    max_parameters: int = 3_000_000_000
    """Số lượng tham số tối đa cho phép của mô hình (kiểm tra an toàn)."""

    dropout: Optional[float] = None
    """Ghi đè tỷ lệ dropout mặc định. None = sử dụng mặc định của mô hình."""


@dataclass
class DataConfig:
    """Đường dẫn tập dữ liệu và các tham số cho tokenizer."""

    train_file: str = ""
    """Đường dẫn đến dữ liệu huấn luyện (định dạng parquet). Các cột bắt buộc: 'article', 'summary'."""

    valid_file: str = ""
    """Đường dẫn đến dữ liệu xác thực (validation) (định dạng parquet)."""

    test_file: str = ""
    """Đường dẫn đến dữ liệu kiểm tra (test) (định dạng parquet). Tùy chọn."""

    source_prefix: str = "summarize: "
    """Tiền tố được thêm vào văn bản đầu vào. Dùng 'summarize: ' cho các mô hình T5, '' cho BART."""

    max_source_length: int = 768
    """Chiều dài token tối đa của đầu vào. Các bài viết dài hơn sẽ bị cắt ngắn."""

    max_target_length: int = 160
    """Chiều dài token tối đa của bản tóm tắt."""

    max_train_samples: Optional[int] = None
    """Giới hạn số mẫu huấn luyện (dùng để gỡ lỗi/kiểm tra nhanh). None = dùng tất cả."""

    max_eval_samples: Optional[int] = None
    """Giới hạn số mẫu đánh giá (validation). None = dùng tất cả."""


@dataclass
class TrainingConfig:
    """Các siêu tham số huấn luyện."""

    output_dir: str = "outputs/default"
    """Nơi lưu trữ các checkpoint và kết quả."""

    seed: int = 42
    """Random seed để đảm bảo tính tái lập (reproducibility)."""

    # --- Epoch & Các bước huấn luyện (Steps) ---
    num_train_epochs: int = 3
    """Số lượng epoch huấn luyện."""

    max_steps: int = -1
    """Số bước huấn luyện (steps) tối đa. -1 = sử dụng num_train_epochs để thay thế."""

    # --- Batch size ---
    per_device_train_batch_size: int = 4
    """Kích thước batch (batch size) huấn luyện trên mỗi GPU."""

    per_device_eval_batch_size: int = 8
    """Kích thước batch (batch size) đánh giá trên mỗi GPU."""

    gradient_accumulation_steps: int = 2
    """Tích lũy gradient qua N bước trước khi cập nhật.
    Batch thực tế = per_device_train_batch_size * gradient_accumulation_steps * num_gpus."""

    # --- Bộ tối ưu (Optimizer) ---
    learning_rate: float = 3e-5
    """Tốc độ học (learning rate) lớn nhất."""

    weight_decay: float = 0.01
    """Sức mạnh chuẩn hóa L2 (L2 regularization strength)."""

    warmup_ratio: float = 0.1
    """Tỷ lệ tổng số bước dùng để khởi động tốc độ học (learning rate warmup)."""

    lr_scheduler_type: str = "cosine"
    """Lịch trình tốc độ học: 'cosine', 'linear', 'constant'."""

    optim: str = "adamw_torch"
    """Bộ tối ưu (Optimizer). Dùng 'adafactor' cho mT5 (tiết kiệm bộ nhớ)."""

    # --- Chuẩn hóa (Regularization) ---
    label_smoothing_factor: float = 0.05
    """Làm mịn nhãn (Label smoothing) cho mất mát cross-entropy. 0.0 = không làm mịn."""

    # --- Độ chính xác (Precision) ---
    precision: str = "auto"
    """Độ chính xác huấn luyện: 'auto', 'fp16', 'bf16', 'fp32'.
    'auto' tự động phát hiện khả năng của GPU."""

    # --- Lưu Checkpoint ---
    gradient_checkpointing: bool = False
    """Đánh đổi tính toán để lấy bộ nhớ. Kích hoạt nếu hết bộ nhớ GPU."""

    freeze_encoder: bool = False
    """Đóng băng các trọng số của bộ mã hóa (encoder). Chỉ huấn luyện bộ giải mã (decoder) + cross-attention."""

    # --- Đánh giá & Lưu trữ (Evaluation & Saving) ---
    eval_strategy: str = "steps"
    """Khi nào đánh giá: 'steps', 'epoch', 'no'."""

    eval_steps: int = 500
    """Đánh giá sau mỗi N bước (khi eval_strategy='steps')."""

    save_strategy: str = "steps"
    """Khi nào lưu checkpoint: 'steps', 'epoch', 'no'."""

    save_steps: int = 500
    """Lưu checkpoint sau mỗi N bước."""

    save_total_limit: int = 2
    """Chỉ giữ lại N checkpoint gần nhất."""

    logging_steps: int = 100
    """Ghi log các chỉ số sau mỗi N bước."""

    # --- Lựa chọn mô hình tốt nhất ---
    metric_for_best_model: str = "rougeL"
    """Chỉ số để xác định checkpoint tốt nhất."""

    greater_is_better: bool = True
    """Xác định xem chỉ số cao hơn có đồng nghĩa với mô hình tốt hơn hay không."""

    load_best_model_at_end: bool = True
    """Tải checkpoint tốt nhất sau khi kết thúc quá trình huấn luyện."""

    # --- Dừng sớm (Early stopping) ---
    early_stopping_patience: int = 5
    """Dừng nếu chỉ số không cải thiện trong N lần đánh giá. 0 = vô hiệu hóa."""

    # --- Tiếp tục (Resume) ---
    resume_from_checkpoint: Optional[str] = None
    """Đường dẫn đến checkpoint để tiếp tục huấn luyện."""

    # --- Multi-GPU ---
    ddp_find_unused_parameters: Optional[bool] = None
    """Cài đặt DDP. Đặt thành False cho các mô hình chuẩn."""


@dataclass
class GenerationConfig:
    """Các tham số sinh văn bản (giải mã)."""

    max_length: int = 200
    """Độ dài tối đa của chuỗi được sinh ra."""

    max_new_tokens: Optional[int] = None
    """Số lượng token MỚI tối đa để sinh (lựa chọn thay thế cho max_length)."""

    min_length: int = 30
    """Độ dài tối thiểu của bản tóm tắt."""

    num_beams: int = 4
    """Chiều rộng của beam search. 1 = greedy decoding."""

    length_penalty: float = 1.0
    """Phạt độ dài beam search. >1.0 = ưu tiên chuỗi dài hơn, <1.0 = ưu tiên chuỗi ngắn hơn."""

    no_repeat_ngram_size: int = 3
    """Chặn việc lặp lại n-gram có kích thước này."""

    repetition_penalty: float = 1.0
    """Hình phạt cho việc lặp lại token. 1.0 = không phạt."""

    do_sample: bool = False
    """Sử dụng lấy mẫu (sampling) thay vì beam search."""

    early_stopping: bool = True
    """Dừng beam search khi tất cả các beam hoàn thành."""


@dataclass
class LoraConfig:
    """Cài đặt LoRA (Low-Rank Adaptation) để huấn luyện hiệu quả tham số."""

    enabled: bool = False
    """Bật LoRA. Khi True, chỉ các tham số LoRA mới được huấn luyện."""

    r: int = 16
    """Hạng (rank) của LoRA. Thấp hơn = ít tham số hơn, cao hơn = sức chứa lớn hơn. Tiêu biểu: 8, 16, 32."""

    lora_alpha: int = 32
    """Hệ số tỷ lệ LoRA. Thường là 2 * r."""

    lora_dropout: float = 0.05
    """Dropout cho các lớp LoRA."""

    target_modules: str = "auto"
    """Các lớp nào sẽ áp dụng LoRA.
    'auto' = phát hiện dựa trên kiến trúc mô hình:
      - T5/mT5: ['q', 'v']
      - BART: ['q_proj', 'v_proj']
    Hoặc chỉ định rõ ràng: 'q,v,k' hoặc 'q_proj,v_proj'"""


@dataclass
class SummarizationConfig:
    """Toàn bộ cấu hình cho một thử nghiệm tóm tắt.

    Kết hợp tất cả các cấu hình phụ thành một đối tượng duy nhất.

    Ví dụ:
        >>> config = load_config("configs/vit5_base.yaml")
        >>> config.model.name_or_path
        'VietAI/vit5-base'
        >>> config.training.learning_rate
        3e-05
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)


# ---------------------------------------------------------------------------
# Tải và thao tác với cấu hình
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> SummarizationConfig:
    """Tải một file cấu hình YAML và trả về một SummarizationConfig đã được xác thực.

    Tham số:
        config_path: Đường dẫn tới file cấu hình YAML.

    Trả về:
        SummarizationConfig đã được điền đầy đủ với các giá trị mặc định cho các trường bị thiếu.

    Ngoại lệ:
        FileNotFoundError: Nếu file cấu hình không tồn tại.
        ValueError: Nếu các trường bắt buộc không hợp lệ.

    Ví dụ:
        >>> config = load_config("configs/vit5_base.yaml")
        >>> config.model.name_or_path
        'VietAI/vit5-base'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _build_config(raw)


def apply_overrides(
    config: SummarizationConfig,
    overrides: dict[str, Any],
) -> SummarizationConfig:
    """Áp dụng các ghi đè key-value cho một cấu hình hiện có.

    Tham số:
        config: Cấu hình cơ sở để sửa đổi.
        overrides: Dictionary các khóa theo định dạng đường dẫn có dấu chấm tới các giá trị.
                   Ví dụ: {'training.learning_rate': 1e-4, 'training.num_train_epochs': 5}

    Trả về:
        Cấu hình mới đã áp dụng các ghi đè (cấu hình gốc không bị sửa đổi).

    Ví dụ:
        >>> config = load_config("configs/vit5_base.yaml")
        >>> config = apply_overrides(config, {'training.learning_rate': 1e-4})
    """
    config = copy.deepcopy(config)

    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Khóa ghi đè phải ở định dạng 'section.field', nhận được: '{key}'"
            )

        section_name, field_name = parts
        section = getattr(config, section_name, None)
        if section is None:
            raise ValueError(
                f"Không rõ phần cấu hình: '{section_name}'. "
                f"Các phần hợp lệ: model, data, training, generation, lora"
            )

        if not hasattr(section, field_name):
            raise ValueError(
                f"Không rõ trường '{field_name}' trong phần '{section_name}'"
            )

        # Chuyển đổi kiểu (type) để khớp với kiểu mong đợi của trường
        current_value = getattr(section, field_name)
        converted_value = _convert_type(value, current_value, field_name)
        setattr(section, field_name, converted_value)

    return config


def config_to_dict(config: SummarizationConfig) -> dict[str, Any]:
    """Chuyển đổi một SummarizationConfig thành một dictionary thông thường.

    Hữu ích để lưu các cấu hình đã được phân giải hoặc ghi log.
    """
    from dataclasses import asdict
    return asdict(config)


# ---------------------------------------------------------------------------
# Các hàm hỗ trợ nội bộ
# ---------------------------------------------------------------------------

def _build_config(raw: dict[str, Any]) -> SummarizationConfig:
    """Xây dựng một SummarizationConfig từ một dictionary gốc."""
    return SummarizationConfig(
        model=_build_section(ModelConfig, raw.get("model", {})),
        data=_build_section(DataConfig, raw.get("data", {})),
        training=_build_section(TrainingConfig, raw.get("training", {})),
        generation=_build_section(GenerationConfig, raw.get("generation", {})),
        lora=_build_section(LoraConfig, raw.get("lora", {})),
    )


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    """Xây dựng một phần dataclass, bỏ qua các trường không xác định."""
    if not raw:
        return cls()

    # Chỉ truyền các trường mà dataclass thực sự mong đợi
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return cls(**filtered)


def _convert_type(value: Any, current: Any, field_name: str) -> Any:
    """Chuyển đổi một giá trị để khớp với kiểu của giá trị trường hiện tại."""
    if current is None:
        return value

    target_type = type(current)

    if target_type == bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    try:
        return target_type(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Không thể chuyển đổi '{value}' thành {target_type.__name__} cho trường '{field_name}'"
        )
