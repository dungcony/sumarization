#!/usr/bin/env python3
"""
Prediction Script (Script suy luận)
=================

Sinh các bản tóm tắt bằng cách sử dụng một mô hình đã được huấn luyện.

Sử dụng:
    # Tóm tắt văn bản trực tiếp:
    python scripts/predict.py \
        --model outputs/vit5_base/best \
        --text "Bài viết dài cần tóm tắt..."

    # Tóm tắt từ một file:
    python scripts/predict.py \
        --model outputs/vit5_base/best \
        --file article.txt

    # Nhận dữ liệu từ stdin (Pipe):
    cat article.txt | python scripts/predict.py --model outputs/vit5_base/best

    # Sử dụng cấu hình generation tùy chỉnh:
    python scripts/predict.py \
        --model outputs/vit5_base/best \
        --text "..." \
        --beams 6 \
        --max-length 300
"""

import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào đường dẫn (path)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.predict import main

if __name__ == "__main__":
    main()
