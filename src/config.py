"""
Configuration System (Hệ thống cấu hình)
====================

Tải, xác thực và quản lý các cấu hình huấn luyện.

Các file cấu hình ở định dạng YAML với 6 phần:
    - phase:      Tên và mô tả giai đoạn huấn luyện
    - model:      Mô hình tiền huấn luyện nào sẽ được sử dụng
    - data:       Đường dẫn tập dữ liệu và các tham số cho tokenizer
    - training:   Các siêu tham số (hyperparameters như epoch, learning rate, batch size...)
    - generation: Cài đặt giải mã (beam search, penalties...)
    - lora:       Cài đặt LoRA tùy chọn cho việc huấn luyện hiệu quả tham số

Ví dụ:
    >>> config = load_config("configs/vit5_base_phase_1.yaml")
    >>> config.phase.name
    'phase_1'
    >>> print(config.model.name_or_path)
    VietAI/vit5-base
    >>> config.training.learning_rate
    3e-05
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml


# ---------------------------------------------------------------------------
# Định nghĩa Dataclass — một dataclass cho mỗi phần của cấu hình
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Thông tin nhận diện một giai đoạn trong quy trình huấn luyện."""

    name: str = "default"
    """Tên phase, ví dụ: 'phase_1' hoặc 'phase_2'."""

    description: str = ""
    """Mô tả ngắn gọn mục tiêu của phase."""


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
    """File hoặc glob CSV/Parquet huấn luyện. Các cột bắt buộc: 'article', 'summary'."""

    valid_file: str = ""
    """File hoặc glob CSV/Parquet dùng cho validation."""

    test_file: str = ""
    """File hoặc glob CSV/Parquet dùng cho test. Tùy chọn."""

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
    """Kích thước batch huấn luyện trên mỗi thiết bị (GPU/TPU core)."""

    per_device_eval_batch_size: int = 8
    """Kích thước batch đánh giá trên mỗi thiết bị (GPU/TPU core)."""

    gradient_accumulation_steps: int = 2
    """Tích lũy gradient qua N bước trước khi cập nhật.
    Batch toàn cục = batch mỗi thiết bị * gradient accumulation * số thiết bị."""

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
    """Bộ tối ưu. 'adafactor' tiết kiệm bộ nhớ khi train dòng T5 trên TPU."""

    # --- Chuẩn hóa (Regularization) ---
    label_smoothing_factor: float = 0.05
    """Làm mịn nhãn (Label smoothing) cho mất mát cross-entropy. 0.0 = không làm mịn."""

    # --- Độ chính xác (Precision) ---
    precision: str = "auto"
    """Độ chính xác huấn luyện: 'auto', 'fp16', 'bf16', 'fp32'.
    'auto' tự động phát hiện khả năng của GPU/TPU."""

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
        >>> config = load_config("configs/vit5_base_phase_1.yaml")
        >>> config.phase.name
        'phase_1'
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
    phase: PhaseConfig = field(default_factory=PhaseConfig)


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
        >>> config = load_config("configs/vit5_base_phase_1.yaml")
        >>> config.phase.name
        'phase_1'
        >>> config.model.name_or_path
        'VietAI/vit5-base'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            raw = {} if loaded is None else loaded
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML không hợp lệ trong '{config_path}': {exc}") from exc

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
        >>> config = load_config("configs/vit5_base_phase_1.yaml")
        >>> config = apply_overrides(config, {'training.learning_rate': 1e-4})
    """
    config = copy.deepcopy(config)
    valid_sections = {item.name for item in fields(SummarizationConfig)}

    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Khóa ghi đè phải ở định dạng 'section.field', nhận được: '{key}'"
            )

        section_name, field_name = parts
        if section_name not in valid_sections:
            raise ValueError(
                f"Không rõ phần cấu hình: '{section_name}'. "
                f"Các phần hợp lệ: phase, model, data, training, generation, lora"
            )
        section = getattr(config, section_name)

        valid_fields = {item.name for item in fields(type(section))}
        if field_name not in valid_fields:
            raise ValueError(
                f"Không rõ trường '{field_name}' trong phần '{section_name}'"
            )

        # Chuyển đổi kiểu (type) để khớp với kiểu mong đợi của trường
        current_value = getattr(section, field_name)
        expected_type = get_type_hints(type(section)).get(field_name)
        converted_value = _convert_type(
            value,
            current_value,
            f"{section_name}.{field_name}",
            expected_type,
        )
        setattr(section, field_name, converted_value)

    return validate_config(config)


def config_to_dict(config: SummarizationConfig) -> dict[str, Any]:
    """Chuyển đổi một SummarizationConfig thành một dictionary thông thường.

    Hữu ích để lưu các cấu hình đã được phân giải hoặc ghi log.
    """
    from dataclasses import asdict
    return asdict(config)


def validate_config(config: SummarizationConfig) -> SummarizationConfig:
    """Xác thực các giá trị và quan hệ chéo quan trọng trong cấu hình.

    Hàm trả lại chính ``config`` để có thể dùng trực tiếp trong pipeline. Tất cả
    lỗi được gom vào một ``ValueError`` thay vì để quá trình train hỏng muộn hơn.
    Không kiểm tra checkpoint cục bộ có tồn tại hay không vì checkpoint phase 2
    có thể chỉ được tạo sau khi phase 1 hoàn tất trên Kaggle.
    """
    errors: list[str] = []

    def non_empty(path: str, value: str) -> None:
        if not value or not value.strip():
            errors.append(f"'{path}' không được để trống")

    def at_least(path: str, value: int, minimum: int) -> None:
        if value < minimum:
            errors.append(f"'{path}' phải >= {minimum}, nhận được {value}")

    def in_range(
        path: str,
        value: float,
        minimum: float,
        maximum: float,
        *,
        include_maximum: bool = True,
    ) -> None:
        upper_ok = value <= maximum if include_maximum else value < maximum
        if value < minimum or not upper_ok:
            right = "]" if include_maximum else ")"
            errors.append(
                f"'{path}' phải nằm trong [{minimum}, {maximum}{right}, nhận được {value}"
            )

    non_empty("phase.name", config.phase.name)
    non_empty("model.name_or_path", config.model.name_or_path)
    at_least("model.max_parameters", config.model.max_parameters, 1)
    if config.model.dropout is not None:
        in_range("model.dropout", config.model.dropout, 0.0, 1.0, include_maximum=False)

    at_least("data.max_source_length", config.data.max_source_length, 1)
    at_least("data.max_target_length", config.data.max_target_length, 1)
    if config.data.max_train_samples is not None:
        at_least("data.max_train_samples", config.data.max_train_samples, 1)
    if config.data.max_eval_samples is not None:
        at_least("data.max_eval_samples", config.data.max_eval_samples, 1)

    tc = config.training
    non_empty("training.output_dir", tc.output_dir)
    at_least("training.num_train_epochs", tc.num_train_epochs, 1)
    if tc.max_steps == 0 or tc.max_steps < -1:
        errors.append("'training.max_steps' phải là -1 hoặc một số nguyên dương")
    at_least("training.per_device_train_batch_size", tc.per_device_train_batch_size, 1)
    at_least("training.per_device_eval_batch_size", tc.per_device_eval_batch_size, 1)
    at_least("training.gradient_accumulation_steps", tc.gradient_accumulation_steps, 1)
    if tc.learning_rate <= 0:
        errors.append("'training.learning_rate' phải > 0")
    if tc.weight_decay < 0:
        errors.append("'training.weight_decay' phải >= 0")
    in_range("training.warmup_ratio", tc.warmup_ratio, 0.0, 1.0)
    in_range(
        "training.label_smoothing_factor",
        tc.label_smoothing_factor,
        0.0,
        1.0,
        include_maximum=False,
    )
    if tc.precision not in {"auto", "fp16", "bf16", "fp32"}:
        errors.append(
            "'training.precision' phải là một trong: auto, fp16, bf16, fp32"
        )
    non_empty("training.lr_scheduler_type", tc.lr_scheduler_type)
    non_empty("training.optim", tc.optim)

    valid_strategies = {"steps", "epoch", "no"}
    if tc.eval_strategy not in valid_strategies:
        errors.append("'training.eval_strategy' phải là: steps, epoch hoặc no")
    if tc.save_strategy not in valid_strategies:
        errors.append("'training.save_strategy' phải là: steps, epoch hoặc no")
    if tc.eval_strategy == "steps":
        at_least("training.eval_steps", tc.eval_steps, 1)
    if tc.save_strategy == "steps":
        at_least("training.save_steps", tc.save_steps, 1)
    at_least("training.save_total_limit", tc.save_total_limit, 1)
    at_least("training.logging_steps", tc.logging_steps, 1)
    at_least("training.early_stopping_patience", tc.early_stopping_patience, 0)

    if tc.early_stopping_patience > 0 and tc.eval_strategy == "no":
        errors.append("early stopping yêu cầu 'training.eval_strategy' khác 'no'")

    if tc.load_best_model_at_end:
        if tc.eval_strategy == "no" or tc.save_strategy == "no":
            errors.append(
                "load_best_model_at_end yêu cầu cả eval_strategy và save_strategy"
            )
        elif tc.eval_strategy != tc.save_strategy:
            errors.append(
                "load_best_model_at_end yêu cầu eval_strategy == save_strategy"
            )
        elif (
            tc.eval_strategy == "steps"
            and tc.eval_steps > 0
            and tc.save_steps % tc.eval_steps != 0
        ):
            errors.append(
                "load_best_model_at_end yêu cầu save_steps là bội số của eval_steps"
            )
        non_empty("training.metric_for_best_model", tc.metric_for_best_model)

    gc = config.generation
    at_least("generation.max_length", gc.max_length, 1)
    if gc.max_new_tokens is not None:
        at_least("generation.max_new_tokens", gc.max_new_tokens, 1)
    at_least("generation.min_length", gc.min_length, 0)
    if gc.min_length > gc.max_length:
        errors.append("'generation.min_length' không được lớn hơn max_length")
    at_least("generation.num_beams", gc.num_beams, 1)
    at_least("generation.no_repeat_ngram_size", gc.no_repeat_ngram_size, 0)
    if gc.repetition_penalty <= 0:
        errors.append("'generation.repetition_penalty' phải > 0")

    lc = config.lora
    in_range("lora.lora_dropout", lc.lora_dropout, 0.0, 1.0, include_maximum=False)
    if lc.enabled:
        at_least("lora.r", lc.r, 1)
        at_least("lora.lora_alpha", lc.lora_alpha, 1)
        non_empty("lora.target_modules", lc.target_modules)

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Cấu hình không hợp lệ:\n{details}")

    return config


# ---------------------------------------------------------------------------
# Các hàm hỗ trợ nội bộ
# ---------------------------------------------------------------------------

def _build_config(raw: dict[str, Any]) -> SummarizationConfig:
    """Xây dựng một SummarizationConfig từ một dictionary gốc."""
    if not isinstance(raw, dict):
        raise ValueError("Nội dung gốc của config YAML phải là một mapping/object")

    section_types = {
        "phase": PhaseConfig,
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "generation": GenerationConfig,
        "lora": LoraConfig,
    }
    unknown_sections = sorted(set(raw) - set(section_types))
    if unknown_sections:
        raise ValueError(
            "Không rõ phần cấu hình: " + ", ".join(repr(name) for name in unknown_sections)
        )

    config = SummarizationConfig(
        phase=_build_section(PhaseConfig, raw.get("phase", {}), "phase"),
        model=_build_section(ModelConfig, raw.get("model", {}), "model"),
        data=_build_section(DataConfig, raw.get("data", {}), "data"),
        training=_build_section(TrainingConfig, raw.get("training", {}), "training"),
        generation=_build_section(
            GenerationConfig,
            raw.get("generation", {}),
            "generation",
        ),
        lora=_build_section(LoraConfig, raw.get("lora", {}), "lora"),
    )
    return validate_config(config)


def _build_section(cls: type, raw: dict[str, Any], section_name: str) -> Any:
    """Xây dựng một section dataclass và từ chối key sai chính tả."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"Phần '{section_name}' phải là một mapping/object")
    if not raw:
        return cls()

    valid_fields = {item.name for item in fields(cls)}
    unknown_fields = sorted(set(raw) - valid_fields)
    if unknown_fields:
        names = ", ".join(repr(name) for name in unknown_fields)
        raise ValueError(f"Không rõ trường trong phần '{section_name}': {names}")

    defaults = cls()
    type_hints = get_type_hints(cls)
    converted = {
        name: _convert_type(
            value,
            getattr(defaults, name),
            f"{section_name}.{name}",
            type_hints.get(name),
        )
        for name, value in raw.items()
    }
    return cls(**converted)


def _convert_type(
    value: Any,
    current: Any,
    field_name: str,
    expected_type: Any = None,
) -> Any:
    """Chuyển đổi một giá trị để khớp với kiểu của giá trị trường hiện tại."""
    optional = False
    origin = get_origin(expected_type)
    if origin in (Union, UnionType):
        args = get_args(expected_type)
        optional = type(None) in args
        concrete_types = [arg for arg in args if arg is not type(None)]
        if len(concrete_types) == 1:
            expected_type = concrete_types[0]

    if optional and (
        value is None
        or (isinstance(value, str) and value.strip().lower() in {"none", "null"})
    ):
        return None
    if value is None:
        raise ValueError(f"Trường '{field_name}' không được nhận giá trị null")

    target_type = type(current) if current is not None else expected_type
    if target_type in (None, Any):
        return value

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(
            f"Không thể chuyển đổi '{value}' thành bool cho trường '{field_name}'"
        )

    if target_type in (int, float) and isinstance(value, bool):
        raise ValueError(
            f"Không thể chuyển đổi bool thành số cho trường '{field_name}'"
        )

    if target_type is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"Không thể chuyển đổi '{value}' thành int cho trường '{field_name}'"
            )

    if target_type is str:
        return str(value)

    try:
        return target_type(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Không thể chuyển đổi '{value}' thành {target_type.__name__} cho trường '{field_name}'"
        ) from exc
