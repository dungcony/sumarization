"""
Evaluation & ROUGE Metrics (Đánh giá & Các chỉ số ROUGE)
==========================

Đánh giá các mô hình đã được fine-tune và tính toán điểm số ROUGE.

Các chỉ số ROUGE:
    - ROUGE-1:  Trùng lặp unigram (word-level recall - độ hồi tưởng cấp độ từ)
    - ROUGE-2:  Trùng lặp bigram (phrase-level quality - chất lượng cấp độ cụm từ)
    - ROUGE-L:  Chuỗi chung dài nhất (Longest common subsequence - độ trôi chảy)

Tất cả các điểm số được chuẩn hóa theo thang 0-100 để dễ đọc.

Các tính năng:
    - Đánh giá một checkpoint mô hình đã lưu
    - Xuất các dự đoán (predictions) dưới dạng JSONL (văn bản gốc, tham chiếu, dự đoán)
    - Tổng hợp kết quả qua nhiều lần chạy vào định dạng CSV/markdown
    - Xây dựng hàm compute_metrics để sử dụng trong quá trình huấn luyện

Ví dụ:
    >>> from src.evaluator import evaluate_checkpoint
    >>> metrics = evaluate_checkpoint("outputs/vit5_base/best", config)
    >>> print(f"ROUGE-L: {metrics['rougeL']:.2f}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.config import SummarizationConfig, config_to_dict
from src.data import load_dataset_from_files, preprocess_for_seq2seq
from src.model import load_model, load_tokenizer
from src.utils import format_number, save_json, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Các chỉ số ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Tính toán các điểm số ROUGE-1, ROUGE-2, và ROUGE-L.

    Tham số:
        predictions: Danh sách các bản tóm tắt được sinh ra.
        references: Danh sách các bản tóm tắt tham chiếu (ground truth).

    Trả về:
        Dictionary chứa các điểm số rouge1, rouge2, rougeL (thang 0-100).

    Ví dụ:
        >>> scores = compute_rouge(
        ...     predictions=["Việt Nam là đất nước đẹp"],
        ...     references=["Việt Nam là quốc gia xinh đẹp"],
        ... )
        >>> print(f"ROUGE-L: {scores['rougeL']:.2f}")
    """
    import evaluate

    rouge_metric = evaluate.load("rouge")
    results = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,
    )

    return {
        "rouge1": round(results["rouge1"] * 100, 2),
        "rouge2": round(results["rouge2"] * 100, 2),
        "rougeL": round(results["rougeL"] * 100, 2),
    }


def build_compute_metrics(tokenizer: Any) -> Callable:
    """Xây dựng một hàm compute_metrics cho Seq2SeqTrainer.

    Trả về một hàm thực hiện các bước:
        1. Giải mã (decode) các token ID dự đoán và token ID nhãn (labels)
        2. Chuẩn hóa các token (giới hạn các giá trị âm, thay thế OOV bằng pad)
        3. Tính toán các điểm số ROUGE
        4. Báo cáo độ dài sinh chuỗi trung bình (average generation length)

    Tham số:
        tokenizer: Tokenizer dùng để giải mã các dự đoán.

    Trả về:
        Một hàm tương thích với compute_metrics của HuggingFace Trainer.
    """
    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred

        # Chuẩn hóa các ID dự đoán (giới hạn các giá trị âm thành token pad)
        pad_id = tokenizer.pad_token_id or 0
        if isinstance(predictions, np.ndarray):
            predictions = np.where(predictions < 0, pad_id, predictions)
            predictions = np.where(
                predictions >= tokenizer.vocab_size, pad_id, predictions
            )

        # Giải mã (decode) các dự đoán
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Chuẩn hóa các ID nhãn (thay thế -100 bằng token pad)
        if isinstance(labels, np.ndarray):
            labels = np.where(labels == -100, pad_id, labels)

        # Giải mã (decode) các nhãn
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Xóa các khoảng trắng thừa
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Tính toán ROUGE
        scores = compute_rouge(decoded_preds, decoded_labels)

        # Thêm các thống kê về độ dài chuỗi sinh ra
        gen_lengths = [
            np.count_nonzero(pred != pad_id)
            for pred in (predictions if isinstance(predictions, np.ndarray)
                        else [predictions])
        ]
        scores["gen_len"] = round(np.mean(gen_lengths), 1)

        return scores

    return compute_metrics


# ---------------------------------------------------------------------------
# Đánh giá checkpoint
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    model_path: str | Path,
    config: SummarizationConfig,
    output_dir: Optional[str | Path] = None,
    export_predictions: bool = True,
) -> dict[str, float]:
    """Đánh giá một checkpoint mô hình đã lưu trên dữ liệu đánh giá (validation).

    Tham số:
        model_path: Đường dẫn tới mô hình đã lưu (thư mục chứa các file mô hình).
        config: Cấu hình (sử dụng các thiết lập về dữ liệu và generation).
        output_dir: Nơi lưu kết quả. Mặc định là thư mục cha của model_path.
        export_predictions: Có lưu các dự đoán dưới dạng JSONL hay không.

    Trả về:
        Dictionary chứa các chỉ số ROUGE.

    Ví dụ:
        >>> metrics = evaluate_checkpoint(
        ...     "outputs/vit5_base/best",
        ...     config=load_config("configs/vit5_base.yaml"),
        ... )
    """
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    model_path = Path(model_path)
    output_dir = Path(output_dir or model_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Đang đánh giá checkpoint: {model_path}")

    # Kiểm tra xem đây có phải là LoRA adapter không
    is_lora = (model_path / "adapter_config.json").exists()

    # Tải tokenizer và mô hình
    tokenizer = load_tokenizer(config.model)

    if is_lora:
        from peft import PeftModel
        logger.info("Phát hiện LoRA adapter, đang tải mô hình cơ sở + adapter")
        base_model = load_model(config.model, tokenizer, config.generation)
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        # Ghi đè đường dẫn mô hình để tải checkpoint
        from dataclasses import replace
        model_config = replace(config.model, name_or_path=str(model_path))
        model = load_model(model_config, tokenizer, config.generation)

    # Tải và tiền xử lý dữ liệu
    datasets = load_dataset_from_files(
        train_file=config.data.train_file,
        valid_file=config.data.valid_file,
    )
    tokenized = preprocess_for_seq2seq(datasets, tokenizer, config.data)

    # Xây dựng trainer đánh giá
    eval_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=config.generation.max_length,
        fp16=False,  # Sử dụng độ chính xác đầy đủ (full precision) cho việc đánh giá
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    compute_metrics_fn = build_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # Chạy đánh giá
    logger.info("Đang chạy đánh giá...")
    predict_output = trainer.predict(
        test_dataset=tokenized["validation"],
    )

    metrics = predict_output.metrics
    logger.info("Kết quả đánh giá:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")

    # Lưu các chỉ số (metrics)
    save_json(metrics, output_dir / "validation_metrics.json")

    # Xuất các dự đoán
    if export_predictions:
        _export_predictions(
            predictions=predict_output.predictions,
            labels=predict_output.label_ids,
            tokenizer=tokenizer,
            dataset=datasets["validation"],
            output_path=output_dir / "predictions_valid.jsonl",
        )

    return metrics


# ---------------------------------------------------------------------------
# Tổng hợp kết quả
# ---------------------------------------------------------------------------

def summarize_results(output_root: str | Path) -> Optional[Path]:
    """Tổng hợp kết quả từ nhiều lần huấn luyện thành một bảng tóm tắt.

    Quét thư mục output_root để tìm các thư mục chứa eval_results.json hoặc
    validation_metrics.json, và tạo ra:
        - summary_results.csv  (định dạng đọc bằng máy)
        - summary_results.md   (bảng markdown dễ đọc)
        - best_run.json        (lần chạy tốt nhất dựa trên ROUGE-L)

    Tham số:
        output_root: Thư mục gốc chứa kết quả các lần huấn luyện.

    Trả về:
        Đường dẫn tới file tóm tắt CSV, hoặc None nếu không tìm thấy kết quả.
    """
    import csv

    output_root = Path(output_root)
    if not output_root.exists():
        logger.warning(f"Không tìm thấy thư mục gốc: {output_root}")
        return None

    # Thu thập kết quả
    results = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics = None
        for metrics_file in ["eval_results.json", "validation_metrics.json"]:
            metrics_path = run_dir / metrics_file
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                break

        if metrics is None:
            continue

        results.append({
            "run": run_dir.name,
            "rouge1": metrics.get("eval_rouge1", metrics.get("rouge1", 0)),
            "rouge2": metrics.get("eval_rouge2", metrics.get("rouge2", 0)),
            "rougeL": metrics.get("eval_rougeL", metrics.get("rougeL", 0)),
        })

    if not results:
        logger.info("Không tìm thấy kết quả nào để tóm tắt")
        return None

    # Sắp xếp theo ROUGE-L (giảm dần)
    results.sort(key=lambda x: x["rougeL"], reverse=True)

    # Lưu file CSV
    csv_path = output_root / "summary_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "rouge1", "rouge2", "rougeL"])
        writer.writeheader()
        writer.writerows(results)

    # Lưu bảng markdown
    md_path = output_root / "summary_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Summarization Results\n\n")
        f.write("| Run | ROUGE-1 | ROUGE-2 | ROUGE-L |\n")
        f.write("|-----|---------|---------|--------|\n")
        for r in results:
            f.write(
                f"| {r['run']} | {r['rouge1']:.2f} | "
                f"{r['rouge2']:.2f} | {r['rougeL']:.2f} |\n"
            )

    # Lưu lần chạy tốt nhất
    best = results[0]
    save_json(best, output_root / "best_run.json")

    logger.info(f"Đã lưu tóm tắt kết quả tới: {csv_path}")
    logger.info(f"Lần chạy tốt nhất: {best['run']} (ROUGE-L: {best['rougeL']:.2f})")

    return csv_path


# ---------------------------------------------------------------------------
# Các hàm hỗ trợ nội bộ
# ---------------------------------------------------------------------------

def _export_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    tokenizer: Any,
    dataset: Any,
    output_path: Path,
) -> None:
    """Xuất các dự đoán dưới dạng file JSONL chứa văn bản gốc, tham chiếu và dự đoán."""
    pad_id = tokenizer.pad_token_id or 0

    # Giải mã (Decode)
    predictions = np.where(predictions < 0, pad_id, predictions)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels == -100, pad_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Ghi ra JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (pred, ref) in enumerate(zip(decoded_preds, decoded_labels)):
            article = dataset[i]["article"] if i < len(dataset) else ""
            record = {
                "index": i,
                "article": article[:500],  # Cắt ngắn để dễ đọc
                "reference": ref.strip(),
                "prediction": pred.strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Đã xuất các dự đoán: {output_path} ({len(decoded_preds)} mẫu)")
