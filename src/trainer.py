from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import SummarizationConfig, config_to_dict
from src.data import load_and_preprocess
from src.evaluator import build_compute_metrics
from src.model import (
    apply_lora,
    enable_gradient_checkpointing,
    fingerprint_full_checkpoint,
    freeze_encoder,
    load_model,
    load_tokenizer,
)
from src.utils import (
    count_parameters,
    detect_precision,
    format_duration,
    format_number,
    get_device_info,
    load_json,
    save_json,
    set_seed,
    setup_logger,
)

logger = setup_logger(__name__)


def train(config: SummarizationConfig) -> dict[str, float]:
    """Hàm chạy toàn bộ quy trình huấn luyện AI từ A-Z.
    
    Quy trình 5 bước cốt lõi:
      1. Khởi tạo môi trường (Device, Seed, Thư mục)
      2. Tải não bộ AI (Model) & Dữ liệu (Data)
      3. Lắp ráp các linh kiện vào Tổng tư lệnh (Trainer)
      4. Bấm nút Khởi động (Train)
      5. Lưu bài & Chấm điểm cuối kỳ (Evaluate)
    """
    start_time = time.time()
    tc = config.training

    if config.lora.enabled and tc.freeze_encoder:
        raise ValueError(
            "Không kết hợp training.freeze_encoder=true với LoRA. PEFT đã "
            "đóng băng toàn bộ base model; hãy đặt freeze_encoder=false."
        )

    logger.info("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN")

    # ==========================================
    # BƯỚC 1: KHỞI TẠO MÔI TRƯỜNG
    # ==========================================
    # Lấy thông tin Card đồ họa (GPU/TPU)
    device = get_device_info()
    logger.info(f"💻 Thiết bị: {device['device'].upper()} | Số lượng: {device['num_gpus']}")
    
    # Tạo thư mục lưu kết quả (ví dụ: outputs/vit5_base)
    output_dir = Path(tc.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Khóa ngẫu nhiên (Seed) để đảm bảo kết quả luôn giống nhau nếu chạy lại
    set_seed(tc.seed)

    base_checkpoint_fingerprint = None
    if config.lora.enabled:
        base_checkpoint_fingerprint = fingerprint_full_checkpoint(
            config.model.name_or_path
        )
        if base_checkpoint_fingerprint is None:
            logger.warning(
                "Không tạo được fingerprint cho base model (có thể là Hub ID). "
                "Adapter manifest sẽ không thể xác minh tuyệt đối dependency."
            )
        if tc.resume_from_checkpoint:
            _verify_lora_resume_manifest(
                config,
                output_dir,
                base_checkpoint_fingerprint,
            )

    # ==========================================
    # BƯỚC 2: TẢI MODEL & DATA (Nguyên liệu)
    # ==========================================
    logger.info("🧠 Đang tải Mô hình và Từ điển...")
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, tokenizer, config.generation)

    # LoRA, Freeze, Gradient Checkpoint
    if tc.gradient_checkpointing: enable_gradient_checkpointing(model)
    if tc.freeze_encoder: freeze_encoder(model)
    model = apply_lora(model, config.lora)

    lora_parameter_stats: dict[str, int | float] | None = None
    adapter_manifest: dict[str, Any] | None = None
    if config.lora.enabled:
        lora_parameter_stats = _verify_lora_trainable_parameters(model)
        adapter_manifest = _build_adapter_manifest(
            config,
            output_dir / "best",
            base_checkpoint_fingerprint,
            lora_parameter_stats,
        )
        # Ghi dependency trước trainer.train(): một run bị ngắt vẫn có đủ
        # provenance để lần resume kế tiếp xác minh đúng Phase 1 base.
        save_json(adapter_manifest, output_dir / "adapter_manifest.json")

    logger.info("📚 Đang xử lý Dữ liệu văn bản...")
    # Test là holdout cuối cùng: không cần tải/tokenize trong lúc train và
    # tuyệt đối không được dùng để chọn checkpoint. Nó chỉ được chấm tường
    # minh sau khi đã khóa model bằng ``evaluate_checkpoint(split='test')``.
    training_data_config = replace(config.data, test_file="")
    datasets = load_and_preprocess(tokenizer, training_data_config)

    # ==========================================
    # BƯỚC 3: LẮP RÁP TRAINER (Tổng tư lệnh)
    # ==========================================
    # Tạo Bảng điều khiển (Training Arguments)
    training_args = build_training_args(config)

    # Tạo Người xếp gạch (tự động đệm khoảng trắng cho các bài báo dài bằng nhau)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100, # Bỏ qua khoảng trắng khi chấm điểm
    )

    # Chốt chặn thông minh: Tự dừng nếu AI học vẹt (điểm không tăng sau N lần)
    callbacks = []
    if tc.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=tc.early_stopping_patience))

    # Lắp ráp mọi thứ vào Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer), # Giám thị chấm thi ROUGE
        callbacks=callbacks,
    )

    # ==========================================
    # BƯỚC 4: BẤM NÚT KHỞI ĐỘNG (TRAIN)
    # ==========================================
    logger.info(f"🔥 Đang Train... (Epochs: {tc.num_train_epochs}, Lr: {tc.learning_rate})")
    
    # NÚT BẤM KÍCH HOẠT (Nếu có bản lưu cũ thì học tiếp, không thì học từ đầu)
    train_result = trainer.train(resume_from_checkpoint=tc.resume_from_checkpoint)

    # ==========================================
    # BƯỚC 5: LƯU BÀI & ĐÁNH GIÁ (EVALUATE)
    # ==========================================
    logger.info("💾 Đang lưu mô hình xịn nhất...")
    best_dir = output_dir / "best"

    if base_checkpoint_fingerprint is not None:
        fingerprint_after_training = fingerprint_full_checkpoint(
            config.model.name_or_path
        )
        if fingerprint_after_training != base_checkpoint_fingerprint:
            raise RuntimeError(
                "Checkpoint base Phase 1 đã thay đổi trong lúc train LoRA; "
                "dừng trước khi xuất adapter."
            )

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    if adapter_manifest is not None:
        # Lưu cả cạnh adapter để artifact vẫn tự mô tả khi được copy riêng,
        # và ở thư mục run để notebook dễ tìm thấy dependency manifest.
        save_json(adapter_manifest, best_dir / "adapter_manifest.json")
        save_json(adapter_manifest, output_dir / "adapter_manifest.json")

    logger.info("📝 Đang làm bài kiểm tra cuối kỳ (Evaluation)...")
    eval_results = trainer.evaluate(metric_key_prefix="eval")

    # Lưu lại sổ điểm (json)
    train_metrics = train_result.metrics
    train_metrics["train_runtime_formatted"] = format_duration(train_metrics.get("train_runtime", 0))
    save_json(train_metrics, output_dir / "train_results.json")
    save_json(eval_results, output_dir / "eval_results.json")
    save_json(config_to_dict(config), output_dir / "resolved_config.json")

    # In kết quả ra màn hình
    logger.info(f"✅ HOÀN THÀNH! Tổng thời gian: {format_duration(time.time() - start_time)}")
    return eval_results


def _build_adapter_manifest(
    config: SummarizationConfig,
    adapter_path: Path,
    base_fingerprint: dict[str, Any] | None,
    parameter_stats: dict[str, int | float],
) -> dict[str, Any]:
    """Tạo manifest dùng chung trước train và cạnh best adapter."""
    return {
        "artifact_type": "peft_lora_adapter",
        "phase": config.phase.name,
        "base_model_dependency": {
            "name_or_path": config.model.name_or_path,
            "required_for_loading": True,
            "fingerprint": base_fingerprint,
        },
        "adapter_path": str(adapter_path),
        "merged_into_base_model": False,
        "lora": config_to_dict(config)["lora"],
        "parameters": parameter_stats,
    }


def _verify_lora_resume_manifest(
    config: SummarizationConfig,
    output_dir: Path,
    current_base_fingerprint: dict[str, Any] | None,
) -> None:
    """Chặn resume LoRA bằng nhầm base hoặc nhầm cấu hình adapter."""
    resume_path = Path(config.training.resume_from_checkpoint or "").resolve()
    if output_dir.resolve() not in resume_path.parents:
        raise ValueError(
            "LoRA resume checkpoint phải nằm trong đúng training.output_dir."
        )
    if not (resume_path / "trainer_state.json").is_file():
        raise FileNotFoundError(
            f"LoRA resume checkpoint thiếu trainer_state.json: {resume_path}"
        )

    manifest_path = output_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Không thể resume LoRA an toàn vì run cũ thiếu "
            f"adapter_manifest.json: {manifest_path}"
        )

    manifest = load_json(manifest_path)
    expected_fingerprint = manifest.get("base_model_dependency", {}).get(
        "fingerprint"
    )
    if expected_fingerprint is None or current_base_fingerprint is None:
        raise ValueError(
            "Không thể resume LoRA an toàn vì base fingerprint bị thiếu."
        )
    if expected_fingerprint != current_base_fingerprint:
        raise ValueError(
            "Checkpoint resume thuộc một Phase 1 base khác; fingerprint "
            "không khớp base hiện tại."
        )

    current_lora_config = config_to_dict(config)["lora"]
    if manifest.get("lora") != current_lora_config:
        raise ValueError(
            "Cấu hình LoRA hiện tại khác run cần resume; không được đổi "
            "rank/alpha/dropout/target_modules giữa run."
        )
    if manifest.get("phase") != config.phase.name:
        raise ValueError("Phase hiện tại khác phase trong adapter manifest.")

    logger.info("Xác minh manifest resume LoRA thành công")


def _verify_lora_trainable_parameters(model: Any) -> dict[str, int | float]:
    """Dừng train nếu PEFT chưa đóng băng hoàn toàn trọng số base model.

    Kiểm tra này cố ý chạy ngay sau ``apply_lora`` và trước khi tải dữ liệu/
    khởi tạo Trainer. Nhờ vậy một cấu hình PEFT sai không thể âm thầm biến
    Phase 2 thành full fine-tuning và ghi đè kiến thức của Phase 1.
    """
    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "LoRA đã bật nhưng không có tham số nào có thể huấn luyện. "
            "Hãy kiểm tra lora.target_modules."
        )

    unexpected_trainable = [
        name for name, _ in trainable_parameters if "lora_" not in name
    ]
    if unexpected_trainable:
        preview = ", ".join(unexpected_trainable[:10])
        suffix = " ..." if len(unexpected_trainable) > 10 else ""
        raise RuntimeError(
            "Phát hiện trọng số không thuộc LoRA vẫn có thể huấn luyện; "
            f"dừng Phase 2 để bảo vệ checkpoint Phase 1: {preview}{suffix}"
        )

    parameter_counts = count_parameters(model)
    trainable_tensor_count = len(trainable_parameters)
    logger.info(
        "Kiểm tra an toàn LoRA đạt: chỉ %s tensor LoRA (%s tham số, %.4f%%) "
        "được huấn luyện; %s tham số base đã được đóng băng.",
        trainable_tensor_count,
        format_number(parameter_counts["trainable"]),
        100 * parameter_counts["trainable"] / parameter_counts["total"],
        format_number(parameter_counts["frozen"]),
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return {
        "total": parameter_counts["total"],
        "trainable": parameter_counts["trainable"],
        "frozen": parameter_counts["frozen"],
        "trainable_percent": parameter_counts["trainable_percent"],
        "trainable_tensor_count": trainable_tensor_count,
    }


# ==============================================================================
# HÀM PHỤ TRỢ: TẠO BẢNG ĐIỀU KHIỂN (TRAINING ARGUMENTS)
# ==============================================================================

def build_training_args(config: SummarizationConfig) -> Seq2SeqTrainingArguments:
    """Đọc cấu hình của bạn và chuyển nó thành Bảng điều khiển (TrainingArguments) mà thư viện Transformers hiểu được."""
    tc = config.training

    # Tự động dò xem máy bạn hỗ trợ fp16 (GPU cũ) hay bf16 (TPU / GPU mới)
    precision = detect_precision() if tc.precision == "auto" else tc.precision
    
    return Seq2SeqTrainingArguments(
        output_dir=tc.output_dir,
        seed=tc.seed,

        # Nhóm Tốc độ & Vòng lặp
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=tc.optim, # GPU dùng adamw, TPU dùng adafactor

        # Nhóm RAM (Kích thước lô)
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,

        # Nhóm Độ chính xác (Tự động bật tính năng chống tràn số dựa vào chip của bạn)
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),

        # Nhóm Kiểm tra bài & Lưu trữ
        eval_strategy=tc.eval_strategy,
        eval_steps=tc.eval_steps,
        save_strategy=tc.save_strategy,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit, # Chỉ giữ 2 bản lưu mới nhất tránh đầy ổ cứng
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,

        # Báo hiệu đây là mô hình tóm tắt văn bản (Cần sinh chữ khi chấm điểm)
        predict_with_generate=True,
        generation_max_length=config.generation.max_length,

        # Cấu hình phụ trợ khác
        label_smoothing_factor=tc.label_smoothing_factor,
        report_to=["tensorboard"],
        logging_dir=str(Path(tc.output_dir) / "logs"),
        logging_steps=tc.logging_steps,
        ddp_find_unused_parameters=tc.ddp_find_unused_parameters,
    )
