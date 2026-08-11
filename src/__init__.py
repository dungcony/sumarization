"""
src - Vietnamese Text Summarization (Tóm tắt văn bản tiếng Việt)
=================================================

Một framework rõ ràng và theo mô-đun để fine-tune các mô hình seq2seq
(ViT5, BARTpho, mT5) cho tác vụ tóm tắt văn bản tiếng Việt (abstractive summarization).

Các mô-đun:
    config      - Tải và xác thực cấu hình
    data        - Tải và tiền xử lý tập dữ liệu
    model       - Quản lý mô hình và bộ tạo token (tokenizer)
    trainer     - Quy trình huấn luyện (Training pipeline)
    evaluator   - Đánh giá và các chỉ số ROUGE
    predict     - Suy luận (inference) trên một văn bản đơn lẻ
    utils       - Các tiện ích dùng chung

Bắt đầu nhanh (Quick Start):
    from src.config import load_config
    from src.trainer import train

    config = load_config("configs/vit5_base.yaml")
    train(config)
"""

__version__ = "1.0.0"
__author__ = "Vietnamese NLP Team"
