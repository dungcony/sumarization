#!/usr/bin/env python3
"""
Training Script (Script huấn luyện)
===============

Huấn luyện một mô hình tóm tắt văn bản tiếng Việt.

Sử dụng:
    # Huấn luyện cơ bản với một file cấu hình:
    python scripts/train.py --config configs/vit5_base.yaml

    # Ghi đè các cài đặt cụ thể:
    python scripts/train.py --config configs/vit5_base.yaml \
        --output-dir outputs/my_experiment \
        --epochs 5 \
        --learning-rate 0.0001

    # Kiểm tra nhanh với dữ liệu giới hạn:
    python scripts/train.py --config configs/vit5_base.yaml \
        --max-steps 10 \
        --max-train-samples 32 \
        --max-eval-samples 16

    # Tiếp tục (resume) từ một checkpoint:
    python scripts/train.py --config configs/vit5_base.yaml \
        --resume outputs/vit5_base/checkpoint-500
"""

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào đường dẫn (path)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, apply_overrides
from src.trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện một mô hình tóm tắt văn bản tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ví dụ:
  python scripts/train.py --config configs/vit5_base.yaml
  python scripts/train.py --config configs/vit5_base.yaml --epochs 5
  python scripts/train.py --config configs/vit5_base.yaml --max-steps 10
        """,
    )

    # Bắt buộc
    parser.add_argument(
        "--config", required=True,
        help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/vit5_base.yaml)",
    )

    # Ghi đè cấu hình dữ liệu
    parser.add_argument(
        "--data-dir",
        help="Thư mục chứa các file parquet train/valid",
    )
    parser.add_argument(
        "--train-file",
        help="Đường dẫn tới file parquet huấn luyện",
    )
    parser.add_argument(
        "--valid-file",
        help="Đường dẫn tới file parquet đánh giá",
    )

    # Ghi đè cấu hình huấn luyện
    parser.add_argument(
        "--output-dir",
        help="Thư mục đầu ra cho các checkpoint và kết quả",
    )
    parser.add_argument(
        "--epochs", type=int,
        help="Số lượng epoch huấn luyện",
    )
    parser.add_argument(
        "--max-steps", type=int,
        help="Số bước (steps) huấn luyện tối đa (ghi đè epochs)",
    )
    parser.add_argument(
        "--learning-rate", type=float,
        help="Tốc độ học (learning rate)",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Kích thước batch (batch size) huấn luyện trên mỗi thiết bị",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Random seed",
    )

    # Giới hạn dữ liệu (để kiểm tra nhanh)
    parser.add_argument(
        "--max-train-samples", type=int,
        help="Giới hạn số mẫu huấn luyện (để kiểm tra nhanh)",
    )
    parser.add_argument(
        "--max-eval-samples", type=int,
        help="Giới hạn số mẫu đánh giá (evaluation)",
    )

    # Tiếp tục huấn luyện (Resume)
    parser.add_argument(
        "--resume",
        help="Đường dẫn tới checkpoint để tiếp tục huấn luyện",
    )

    # Ghi đè nâng cao
    parser.add_argument(
        "--set", nargs="*", metavar="KEY=VALUE",
        help="Ghi đè bất kỳ giá trị cấu hình nào (ví dụ: --set training.warmup_ratio=0.05)",
    )

    args = parser.parse_args()

    # Tải cấu hình
    config = load_config(args.config)

    # Áp dụng các ghi đè từ CLI
    overrides = {}

    if args.data_dir:
        data_dir = Path(args.data_dir)
        # Tìm các file parquet trong thư mục dữ liệu
        train_files = list(data_dir.glob("train*.parquet"))
        valid_files = list(data_dir.glob("valid*.parquet"))
        if train_files:
            overrides["data.train_file"] = str(train_files[0])
        if valid_files:
            overrides["data.valid_file"] = str(valid_files[0])

    if args.train_file:
        overrides["data.train_file"] = args.train_file
    if args.valid_file:
        overrides["data.valid_file"] = args.valid_file
    if args.output_dir:
        overrides["training.output_dir"] = args.output_dir
    if args.epochs:
        overrides["training.num_train_epochs"] = args.epochs
    if args.max_steps:
        overrides["training.max_steps"] = args.max_steps
    if args.learning_rate:
        overrides["training.learning_rate"] = args.learning_rate
    if args.batch_size:
        overrides["training.per_device_train_batch_size"] = args.batch_size
    if args.seed:
        overrides["training.seed"] = args.seed
    if args.max_train_samples:
        overrides["data.max_train_samples"] = args.max_train_samples
    if args.max_eval_samples:
        overrides["data.max_eval_samples"] = args.max_eval_samples
    if args.resume:
        overrides["training.resume_from_checkpoint"] = args.resume

    # Phân tích các ghi đè --set
    if args.set:
        for item in args.set:
            if "=" not in item:
                parser.error(f"Định dạng --set không hợp lệ: '{item}'. Hãy dùng KEY=VALUE")
            key, value = item.split("=", 1)
            overrides[key] = value

    if overrides:
        config = apply_overrides(config, overrides)

    # Chạy huấn luyện
    metrics = train(config)

    # In các chỉ số cuối cùng
    print("\n" + "=" * 50)
    print("CÁC CHỈ SỐ CUỐI CÙNG:")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
