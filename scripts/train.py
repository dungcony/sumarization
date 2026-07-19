#!/usr/bin/env python3
"""
Training Script (Script huấn luyện)
===============

Huấn luyện một mô hình tóm tắt văn bản tiếng Việt.

Sử dụng:
    # Huấn luyện cơ bản với một file cấu hình YAML:
    python scripts/train.py --config configs/vit5_base.yaml

    # Ghi đè nhanh một số cài đặt trên dòng lệnh:
    python scripts/train.py --config configs/vit5_base.yaml \
        --epochs 5 \
        --batch-size 8 \
        --learning-rate 0.0001
"""

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào đường dẫn (path) để python nhận diện được thư mục 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, apply_overrides
from src.trainer import train

def main():
    """Điểm khởi đầu của chương trình. Được thiết kế rút gọn để dễ đọc."""
    
    # Bước 1: Thu thập và xử lý các tham số người dùng gõ từ Terminal
    args, overrides = parse_arguments()

    # Bước 2: Tải file cấu hình YAML gốc
    config = load_config(args.config)

    # Bước 3: Nếu người dùng gõ thêm các cờ (như --epochs), ghi đè chúng vào config
    if overrides:
        config = apply_overrides(config, overrides)

    # Bước 4: Khởi chạy quá trình huấn luyện AI! (Gọi hàm train trong src/trainer.py)
    metrics = train(config)

    # Bước 5: Báo cáo điểm số khi học xong
    print("\n" + "=" * 50)
    print("CÁC CHỈ SỐ CUỐI CÙNG (FINAL METRICS):")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("=" * 50)


def parse_arguments() -> tuple[argparse.Namespace, dict]:
    """Hàm xử lý việc đọc các tham số dòng lệnh (argparse).
    Trả về:
        args: Chứa tham số cơ bản (như file config)
        overrides: Từ điển chứa các thông số cần ghi đè (vd: {"training.num_train_epochs": 5})
    """
    parser = argparse.ArgumentParser(
        description="Huấn luyện một mô hình tóm tắt văn bản tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Bắt buộc phải có file config
    parser.add_argument(
        "--config", required=True,
        help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/vit5_base.yaml)",
    )

    # Các tham số ghi đè tùy chọn (không bắt buộc)
    parser.add_argument("--data-dir", help="Thư mục chứa các file parquet train/valid")
    parser.add_argument("--train-file", help="Đường dẫn tới file parquet huấn luyện")
    parser.add_argument("--valid-file", help="Đường dẫn tới file parquet đánh giá")
    parser.add_argument("--output-dir", help="Thư mục đầu ra cho model")
    parser.add_argument("--epochs", type=int, help="Số lượng vòng học (epochs)")
    parser.add_argument("--max-steps", type=int, help="Số bước huấn luyện tối đa (ghi đè epochs)")
    parser.add_argument("--learning-rate", type=float, help="Tốc độ học (learning rate)")
    parser.add_argument("--batch-size", type=int, help="Số lượng bài báo học cùng lúc")
    parser.add_argument("--seed", type=int, help="Hạt giống ngẫu nhiên (seed)")
    parser.add_argument("--resume", help="Đường dẫn tới bản lưu (checkpoint) để chạy tiếp")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", 
                        help="Ghi đè nâng cao (vd: --set training.warmup_ratio=0.05)")

    args = parser.parse_args()
    overrides = {}

    # Chuyển đổi cờ dòng lệnh thành Dictionary để ghi đè config
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_files = list(data_dir.glob("train*.parquet"))
        valid_files = list(data_dir.glob("valid*.parquet"))
        if train_files: overrides["data.train_file"] = str(train_files[0])
        if valid_files: overrides["data.valid_file"] = str(valid_files[0])

    if args.train_file: overrides["data.train_file"] = args.train_file
    if args.valid_file: overrides["data.valid_file"] = args.valid_file
    if args.output_dir: overrides["training.output_dir"] = args.output_dir
    if args.epochs: overrides["training.num_train_epochs"] = args.epochs
    if args.max_steps: overrides["training.max_steps"] = args.max_steps
    if args.learning_rate: overrides["training.learning_rate"] = args.learning_rate
    if args.batch_size: overrides["training.per_device_train_batch_size"] = args.batch_size
    if args.seed: overrides["training.seed"] = args.seed
    if args.resume: overrides["training.resume_from_checkpoint"] = args.resume

    # Xử lý ghi đè nâng cao bằng cờ --set
    if args.set:
        for item in args.set:
            if "=" not in item:
                parser.error(f"Định dạng --set không hợp lệ: '{item}'. Hãy dùng KEY=VALUE")
            key, value = item.split("=", 1)
            overrides[key] = value

    return args, overrides


if __name__ == "__main__":
    main()
