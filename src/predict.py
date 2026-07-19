"""
Single-Text Inference (Suy luận văn bản đơn lẻ)
=====================

Sinh bản tóm tắt cho từng văn bản bằng cách sử dụng một mô hình đã được huấn luyện.

Mô-đun này cung cấp một giao diện đơn giản để chạy suy luận (inference)
mà không cần đến toàn bộ quy trình huấn luyện.

Ví dụ (Python):
    >>> from src.predict import summarize
    >>> summary = summarize(
    ...     text="Bài viết dài về kinh tế Việt Nam...",
    ...     model_path="outputs/vit5_base/best",
    ... )
    >>> print(summary)

Ví dụ (CLI):
    $ python -m vn_summarization.predict \\
        --model outputs/vit5_base/best \\
        --text "Bài viết cần tóm tắt..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from src.config import (
    GenerationConfig,
    ModelConfig,
    SummarizationConfig,
    load_config,
)
from src.data import clean_text
from src.model import load_model, load_tokenizer
from src.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Hàm suy luận cốt lõi (Core inference function)
# ---------------------------------------------------------------------------

def summarize(
    text: str,
    model_path: str | Path,
    config: Optional[SummarizationConfig] = None,
    source_prefix: str = "summarize: ",
    max_source_length: int = 768,
    num_beams: int = 4,
    max_length: int = 200,
    min_length: int = 30,
    repetition_penalty: float = 1.05,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
) -> str:
    """Sinh bản tóm tắt cho một văn bản đơn lẻ.

    Tham số:
        text: Văn bản đầu vào (tiếng Việt).
        model_path: Đường dẫn tới checkpoint mô hình đã lưu.
        config: Cấu hình đầy đủ (tùy chọn). Nếu None, dùng cài đặt mặc định.
        source_prefix: Tiền tố cho đầu vào (ví dụ: 'summarize: ' cho T5).
        max_source_length: Chiều dài token tối đa của đầu vào.
        num_beams: Kích thước beam search (num_beams).
        max_length: Độ dài tối đa của bản tóm tắt.
        min_length: Độ dài tối thiểu của bản tóm tắt.
        repetition_penalty: Phạt cho việc lặp lại token.
        length_penalty: Phạt độ dài cho beam search.
        no_repeat_ngram_size: Chặn lặp lại các n-gram.

    Trả về:
        Chuỗi bản tóm tắt được sinh ra.

    Ví dụ:
        >>> text = "Ngày 15/7, Thủ tướng Chính phủ đã chủ trì cuộc họp..."
        >>> summary = summarize(text, "outputs/vit5_base/best")
        >>> print(summary)
    """
    model_path = Path(model_path)

    # Xây dựng cấu hình nếu không được cung cấp
    if config is None:
        model_config = ModelConfig(name_or_path=str(model_path))
        # Phát hiện nếu thuộc dòng T5 để thiết lập cờ (flags) cho tokenizer
        name_lower = str(model_path).lower()
        if any(k in name_lower for k in ["t5", "mt5", "vit5"]):
            model_config.use_fast_tokenizer = False

        config = SummarizationConfig(
            model=model_config,
            generation=GenerationConfig(
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ),
        )

    # Tải mô hình và tokenizer
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, tokenizer, config.generation)

    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Chuẩn bị đầu vào
    cleaned = clean_text(text)
    input_text = source_prefix + cleaned

    inputs = tokenizer(
        input_text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Sinh chuỗi (Generate)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.generation.max_length,
            min_length=config.generation.min_length,
            num_beams=config.generation.num_beams,
            length_penalty=config.generation.length_penalty,
            no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
            repetition_penalty=config.generation.repetition_penalty,
            early_stopping=config.generation.early_stopping,
        )

    # Giải mã (Decode)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()


# ---------------------------------------------------------------------------
# Suy luận theo batch (Batch inference)
# ---------------------------------------------------------------------------

def summarize_batch(
    texts: list[str],
    model_path: str | Path,
    config: Optional[SummarizationConfig] = None,
    source_prefix: str = "summarize: ",
    max_source_length: int = 768,
    batch_size: int = 8,
) -> list[str]:
    """Sinh bản tóm tắt cho một batch các văn bản.

    Hiệu quả hơn so với việc gọi summarize() lặp đi lặp lại vì mô hình
    chỉ được tải một lần.

    Tham số:
        texts: Danh sách các bài viết đầu vào.
        model_path: Đường dẫn tới checkpoint mô hình đã lưu.
        config: Cấu hình đầy đủ (tùy chọn).
        source_prefix: Tiền tố cho các đầu vào.
        max_source_length: Chiều dài token tối đa của đầu vào.
        batch_size: Số lượng văn bản cần xử lý cùng một lúc.

    Trả về:
        Danh sách các bản tóm tắt được sinh ra.
    """
    model_path = Path(model_path)

    if config is None:
        config = SummarizationConfig(
            model=ModelConfig(name_or_path=str(model_path)),
        )

    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, tokenizer, config.generation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    summaries = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Chuẩn bị đầu vào
        inputs_text = [
            source_prefix + clean_text(text) for text in batch_texts
        ]

        inputs = tokenizer(
            inputs_text,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Sinh chuỗi (Generate)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.generation.max_length,
                min_length=config.generation.min_length,
                num_beams=config.generation.num_beams,
                length_penalty=config.generation.length_penalty,
                no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
                repetition_penalty=config.generation.repetition_penalty,
                early_stopping=config.generation.early_stopping,
            )

        # Giải mã (Decode)
        batch_summaries = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        summaries.extend([s.strip() for s in batch_summaries])

        logger.info(f"Đã xử lý {min(i + batch_size, len(texts))}/{len(texts)} văn bản")

    return summaries


# ---------------------------------------------------------------------------
# Điểm vào CLI (CLI entry point)
# ---------------------------------------------------------------------------

def main() -> None:
    """Điểm vào CLI cho việc dự đoán một văn bản đơn lẻ."""
    parser = argparse.ArgumentParser(
        description="Sinh một bản tóm tắt văn bản tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Từ văn bản:
  python -m src.predict --model outputs/vit5_base/best --text "Bài viết..."

  # Từ file:
  python -m src.predict --model outputs/vit5_base/best --file article.txt

  # Thông qua stdin:
  cat article.txt | python -m src.predict --model outputs/vit5_base/best
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help="Đường dẫn tới checkpoint mô hình đã lưu",
    )
    parser.add_argument(
        "--config",
        help="Đường dẫn tới file cấu hình YAML (tùy chọn)",
    )
    parser.add_argument(
        "--text",
        help="Văn bản cần tóm tắt",
    )
    parser.add_argument(
        "--file",
        help="File chứa văn bản cần tóm tắt",
    )
    parser.add_argument(
        "--prefix", default="summarize: ",
        help="Tiền tố nguồn (mặc định: 'summarize: ')",
    )
    parser.add_argument(
        "--beams", type=int, default=4,
        help="Số lượng beams (mặc định: 4)",
    )
    parser.add_argument(
        "--max-length", type=int, default=200,
        help="Độ dài bản tóm tắt tối đa (mặc định: 200)",
    )

    args = parser.parse_args()

    # Nhận văn bản đầu vào
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Hãy cung cấp văn bản thông qua --text, --file, hoặc stdin")

    # Tải cấu hình nếu được cung cấp
    config = load_config(args.config) if args.config else None

    # Sinh bản tóm tắt
    summary = summarize(
        text=text,
        model_path=args.model,
        config=config,
        source_prefix=args.prefix,
        num_beams=args.beams,
        max_length=args.max_length,
    )

    print("\n" + "=" * 60)
    print("BẢN TÓM TẮT:")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
