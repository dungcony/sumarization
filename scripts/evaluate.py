#!/usr/bin/env python3
"""
Evaluation Script (Script đánh giá)
=================

Đánh giá checkpoint đầy đủ hoặc LoRA adapter trên tập validation/test.

Sử dụng:
    # Đánh giá cơ bản:
    python scripts/evaluate.py \
        --model outputs/vit5_base/best \
        --config configs/vit5_base.yaml \
        --split validation

    # Đánh giá LoRA adapter trên test bằng checkpoint Phase 1 làm base:
    python scripts/evaluate.py eval \
        --model outputs/phase_2_lora/best \
        --base-model outputs/phase_1/best \
        --config configs/vit5_base_phase_2_lora.yaml \
        --split test

    # Đánh giá với thư mục đầu ra tùy chỉnh:
    python scripts/evaluate.py \
        --model outputs/vit5_base/best \
        --config configs/vit5_base.yaml \
        --output-dir outputs/evaluation_results

    # Tổng hợp tất cả các kết quả:
    python scripts/evaluate.py --summarize outputs/
"""

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào đường dẫn (path)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.evaluator import evaluate_checkpoint, summarize_results


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá một mô hình tóm tắt tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Các lệnh có sẵn")

    # --- Lệnh eval ---
    eval_parser = subparsers.add_parser(
        "eval", help="Đánh giá một checkpoint mô hình",
    )
    eval_parser.add_argument(
        "--model", required=True,
        help="Đường dẫn tới thư mục checkpoint mô hình",
    )
    eval_parser.add_argument(
        "--config", required=True,
        help="Đường dẫn tới file cấu hình YAML",
    )
    eval_parser.add_argument(
        "--output-dir",
        help="Thư mục để lưu kết quả đánh giá",
    )
    eval_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Phần dữ liệu cần đánh giá (mặc định: validation)",
    )
    eval_parser.add_argument(
        "--base-model",
        help=(
            "Checkpoint mô hình nền (ví dụ Phase 1) khi --model là LoRA adapter"
        ),
    )
    eval_parser.add_argument(
        "--no-predictions", action="store_true",
        help="Bỏ qua việc xuất các dự đoán dạng JSONL",
    )

    # --- Lệnh summarize ---
    summary_parser = subparsers.add_parser(
        "summarize", help="Tổng hợp kết quả từ nhiều lần chạy",
    )
    summary_parser.add_argument(
        "--root", required=True,
        help="Thư mục gốc chứa các đầu ra (outputs) của quá trình huấn luyện",
    )

    # Đồng thời hỗ trợ các đối số phẳng (flat arguments) để sử dụng đơn giản
    parser.add_argument("--model", help="Đường dẫn tới checkpoint mô hình")
    parser.add_argument("--config", help="Đường dẫn tới file cấu hình YAML")
    parser.add_argument("--output-dir", help="Thư mục đầu ra")
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Phần dữ liệu cần đánh giá (mặc định: validation)",
    )
    parser.add_argument(
        "--base-model",
        help="Checkpoint mô hình nền khi --model là LoRA adapter",
    )
    parser.add_argument(
        "--no-predictions", action="store_true",
        help="Bỏ qua việc xuất các dự đoán dạng JSONL",
    )
    parser.add_argument("--summarize", help="Thư mục gốc cần tổng hợp")

    args = parser.parse_args()

    # Xử lý các lệnh phụ (subcommands)
    if args.command == "eval":
        config = load_config(args.config)
        metrics = evaluate_checkpoint(
            model_path=args.model,
            config=config,
            output_dir=args.output_dir,
            export_predictions=not args.no_predictions,
            split=args.split,
            base_model_path=args.base_model,
        )

        print("\n" + "=" * 50)
        print("KẾT QUẢ ĐÁNH GIÁ:")
        print("=" * 50)
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        print("=" * 50)

    elif args.command == "summarize":
        summarize_results(args.root)

    elif args.summarize:
        summarize_results(args.summarize)

    elif args.model and args.config:
        config = load_config(args.config)
        metrics = evaluate_checkpoint(
            model_path=args.model,
            config=config,
            output_dir=args.output_dir,
            export_predictions=not args.no_predictions,
            split=args.split,
            base_model_path=args.base_model,
        )

        print("\n" + "=" * 50)
        print("KẾT QUẢ ĐÁNH GIÁ:")
        print("=" * 50)
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        print("=" * 50)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
