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
    get_device_info,
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

    logger.info("📚 Đang xử lý Dữ liệu văn bản...")
    datasets = load_and_preprocess(tokenizer, config.data)

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
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

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
