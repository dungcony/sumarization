"""
Training Pipeline (Quy trình huấn luyện)
=================

Quy trình huấn luyện seq2seq hoàn chỉnh cho tóm tắt văn bản tiếng Việt.

Luồng huấn luyện:
    1. Tải cấu hình → 2. Cài đặt seed → 3. Tải tokenizer & mô hình
    → 4. Áp dụng LoRA (tùy chọn) → 5. Tải & tiền xử lý dữ liệu
    → 6. Xây dựng Seq2SeqTrainer → 7. Huấn luyện (Train) → 8. Lưu mô hình tốt nhất
    → 9. Đánh giá (Evaluate) → 10. Lưu các chỉ số (metrics)

Hỗ trợ:
    - Fine-tuning toàn bộ hoặc huấn luyện hiệu quả tham số bằng LoRA
    - Huấn luyện đa GPU (Multi-GPU) qua HuggingFace Accelerate
    - Độ chính xác hỗn hợp (Mixed precision) (fp16/bf16/fp32 với tính năng tự động phát hiện)
    - Dừng sớm (Early stopping) dựa trên ROUGE-L trên tập đánh giá (validation)
    - Tiếp tục (Resume) từ checkpoint
    - Gradient checkpointing để tiết kiệm bộ nhớ

Ví dụ:
    >>> from src.trainer import train
    >>> from src.config import load_config
    >>> config = load_config("configs/vit5_base.yaml")
    >>> train(config)  # Chạy toàn bộ quy trình huấn luyện
"""

from __future__ import annotations

import time
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
    freeze_encoder,
    load_model,
    load_tokenizer,
)
from src.utils import (
    detect_precision,
    format_duration,
    format_number,
    get_device_info,
    save_json,
    set_seed,
    setup_logger,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Hàm huấn luyện chính
# ---------------------------------------------------------------------------

def train(config: SummarizationConfig) -> dict[str, float]:
    """Chạy toàn bộ quy trình huấn luyện.

    Đây là điểm khởi đầu chính cho việc huấn luyện. Nó xử lý toàn bộ
    tiến trình từ việc tải dữ liệu cho đến lưu mô hình cuối cùng và các chỉ số (metrics).

    Tham số:
        config: Cấu hình tóm tắt hoàn chỉnh.

    Trả về:
        Dictionary chứa các chỉ số đánh giá cuối cùng (các điểm số ROUGE).

    Ví dụ:
        >>> from src.config import load_config
        >>> config = load_config("configs/vit5_base.yaml")
        >>> metrics = train(config)
        >>> print(f"ROUGE-L: {metrics['eval_rougeL']:.4f}")
    """
    start_time = time.time()
    tc = config.training  # shorthand

    # --- Bước 1: Thiết lập ---
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN")
    logger.info("=" * 60)

    # In thông tin thiết bị
    device_info = get_device_info()
    logger.info(f"Thiết bị: {device_info['device']}, Số GPU: {device_info['num_gpus']}")
    for gpu_name in device_info.get("gpu_names", []):
        logger.info(f"  GPU: {gpu_name}")

    # Tạo thư mục đầu ra
    output_dir = Path(tc.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Bước 2: Thiết lập seed ---
    set_seed(tc.seed)
    logger.info(f"Random seed: {tc.seed}")

    # --- Bước 3: Tải tokenizer & mô hình ---
    logger.info("-" * 40)
    logger.info("Đang tải tokenizer và mô hình...")
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, tokenizer, config.generation)

    # --- Bước 4: Áp dụng các cấu hình tùy chọn ---
    if tc.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    if tc.freeze_encoder:
        freeze_encoder(model)

    model = apply_lora(model, config.lora)

    # --- Bước 5: Tải và tiền xử lý dữ liệu ---
    logger.info("-" * 40)
    logger.info("Đang tải và tiền xử lý dữ liệu...")
    datasets = load_and_preprocess(tokenizer, config.data)

    # --- Bước 6: Xây dựng trainer ---
    logger.info("-" * 40)
    logger.info("Đang xây dựng trainer...")
    training_args = build_training_args(config)

    # Data collator xử lý đệm động (dynamic padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Tính toán các chỉ số ROUGE
    compute_metrics = build_compute_metrics(tokenizer)

    # Các callback
    callbacks = []
    if tc.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=tc.early_stopping_patience,
            )
        )
        logger.info(f"Đã bật early stopping: patience={tc.early_stopping_patience}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # --- Bước 7: Huấn luyện (Train) ---
    logger.info("-" * 40)
    logger.info("Bắt đầu huấn luyện...")
    logger.info(f"  Epoch: {tc.num_train_epochs}")
    logger.info(f"  Batch size (trên mỗi thiết bị): {tc.per_device_train_batch_size}")
    logger.info(f"  Tích lũy gradient (Gradient accumulation): {tc.gradient_accumulation_steps}")
    logger.info(f"  Tốc độ học (Learning rate): {tc.learning_rate}")
    logger.info(f"  Đầu ra (Output): {tc.output_dir}")

    train_result = trainer.train(
        resume_from_checkpoint=tc.resume_from_checkpoint,
    )

    # --- Bước 8: Lưu mô hình tốt nhất ---
    logger.info("-" * 40)
    best_dir = output_dir / "best"
    logger.info(f"Đang lưu mô hình tốt nhất tới: {best_dir}")
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # --- Bước 9: Đánh giá cuối cùng ---
    logger.info("-" * 40)
    logger.info("Chạy đánh giá cuối cùng...")
    eval_results = trainer.evaluate(
        metric_key_prefix="eval",
    )

    # --- Bước 10: Lưu các chỉ số ---
    train_metrics = train_result.metrics
    train_metrics["train_runtime_formatted"] = format_duration(
        train_metrics.get("train_runtime", 0)
    )

    save_json(train_metrics, output_dir / "train_results.json")
    save_json(eval_results, output_dir / "eval_results.json")
    save_json(config_to_dict(config), output_dir / "resolved_config.json")

    # --- Tóm tắt ---
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH HUẤN LUYỆN")
    logger.info(f"  Tổng thời gian: {format_duration(elapsed)}")
    logger.info(f"  Mô hình tốt nhất: {best_dir}")
    for key, value in sorted(eval_results.items()):
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 60)

    return eval_results


# ---------------------------------------------------------------------------
# Bộ xây dựng tham số huấn luyện (Training arguments builder)
# ---------------------------------------------------------------------------

def build_training_args(
    config: SummarizationConfig,
) -> Seq2SeqTrainingArguments:
    """Xây dựng Seq2SeqTrainingArguments từ cấu hình của chúng ta.

    Xử lý tự động phát hiện:
        - Độ chính xác hỗn hợp (Mixed precision) (fp16/bf16 dựa trên GPU)
        - Các chiến lược đánh giá và sinh chuỗi (evaluation and generation strategies)

    Tham số:
        config: Cấu hình tóm tắt hoàn chỉnh.

    Trả về:
        Seq2SeqTrainingArguments của HuggingFace.
    """
    tc = config.training

    # Xác định độ chính xác
    if tc.precision == "auto":
        precision = detect_precision()
    else:
        precision = tc.precision

    fp16 = precision == "fp16"
    bf16 = precision == "bf16"
    logger.info(f"Độ chính xác huấn luyện: {precision} (fp16={fp16}, bf16={bf16})")

    args = Seq2SeqTrainingArguments(
        output_dir=tc.output_dir,
        seed=tc.seed,

        # Epochs & steps
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,

        # Batch size
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,

        # Optimizer
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=tc.optim,

        # Chuẩn hóa (Regularization)
        label_smoothing_factor=tc.label_smoothing_factor,

        # Precision
        fp16=fp16,
        bf16=bf16,

        # Đánh giá & lưu (Evaluation & saving)
        eval_strategy=tc.eval_strategy,
        eval_steps=tc.eval_steps,
        save_strategy=tc.save_strategy,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        logging_steps=tc.logging_steps,

        # Mô hình tốt nhất (Best model)
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        load_best_model_at_end=tc.load_best_model_at_end,

        # Sinh văn bản trong quá trình đánh giá (Generation during evaluation)
        predict_with_generate=True,
        generation_max_length=config.generation.max_length,

        # Báo cáo
        report_to=["tensorboard"],
        logging_dir=str(Path(tc.output_dir) / "logs"),

        # DDP
        ddp_find_unused_parameters=tc.ddp_find_unused_parameters,
    )

    return args
