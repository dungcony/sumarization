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

    >>> medical_summary = summarize(
    ...     text="Bài viết y tế cần tóm tắt...",
    ...     base_model_path="outputs_phase_1/vit5_base/best",
    ...     adapter_path="outputs_phase_2_lora/vit5_base/best",
    ... )

Ví dụ (CLI):
    $ python -m src.predict \\
        --base-model outputs_phase_1/vit5_base/best \\
        --adapter outputs_phase_2_lora/vit5_base/best \\
        --text "Bài viết cần tóm tắt..."
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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
from src.model import load_model_for_inference
from src.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Hàm suy luận cốt lõi (Core inference function)
# ---------------------------------------------------------------------------

def _resolve_base_model_path(
    model_path: str | Path | None,
    base_model_path: str | Path | None,
    config: Optional[SummarizationConfig],
) -> str:
    """Chuẩn hóa alias cũ ``model_path`` và tên rõ nghĩa mới.

    ``model_path`` được giữ lại để không phá vỡ code cũ. Khi nạp LoRA,
    cả hai tên đều mang nghĩa full base checkpoint, không phải adapter.
    """
    if model_path is not None and base_model_path is not None:
        raise ValueError(
            "Chỉ truyền một trong hai tham số model_path hoặc "
            "base_model_path; hai tham số này là alias của nhau."
        )

    resolved = base_model_path if base_model_path is not None else model_path
    if resolved is None and config is not None:
        resolved = config.model.name_or_path

    if resolved is None or not str(resolved).strip():
        raise ValueError(
            "Thiếu full base checkpoint. Hãy truyền model_path hoặc "
            "base_model_path (ví dụ checkpoint Phase 1)."
        )

    return str(resolved)


def _resolve_generation_config(
    config: Optional[SummarizationConfig],
    *,
    num_beams: int | None,
    max_length: int | None,
    min_length: int | None,
    repetition_penalty: float | None,
    length_penalty: float | None,
    no_repeat_ngram_size: int | None,
) -> GenerationConfig:
    """Tạo bản copy generation config và chỉ áp dụng override rõ ràng.

    Khi không có config, các giá trị fallback giữ nguyên hành vi cũ của
    ``summarize``. Khi có config, ``None`` mang nghĩa là dùng giá trị YAML.
    """
    if config is None:
        resolved = GenerationConfig(
            num_beams=4,
            max_length=200,
            min_length=30,
            repetition_penalty=1.05,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
    else:
        resolved = replace(config.generation)

    overrides = {
        "num_beams": num_beams,
        "max_length": max_length,
        "min_length": min_length,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    explicit_overrides = {
        name: value
        for name, value in overrides.items()
        if value is not None
    }
    if explicit_overrides:
        resolved = replace(resolved, **explicit_overrides)

    return resolved


def summarize(
    text: str,
    model_path: str | Path | None = None,
    config: Optional[SummarizationConfig] = None,
    source_prefix: str | None = None,
    max_source_length: int | None = None,
    num_beams: int | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
    repetition_penalty: float | None = None,
    length_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    adapter_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
) -> str:
    """Sinh bản tóm tắt cho một văn bản đơn lẻ.

    Tham số:
        text: Văn bản đầu vào (tiếng Việt).
        model_path: Full checkpoint mô hình đã lưu. Đây là tên cũ,
            được giữ để tương thích; khi dùng LoRA nó là base Phase 1.
        config: Cấu hình đầy đủ (tùy chọn). Nếu None, dùng cài đặt mặc định.
        source_prefix: Tiền tố cho đầu vào. ``None`` dùng giá trị trong
            config, hoặc ``'summarize: '`` khi không có config.
        max_source_length: Chiều dài token tối đa của đầu vào. ``None``
            dùng giá trị trong config, hoặc 768 khi không có config.
        num_beams: Kích thước beam search. ``None`` dùng config hoặc 4.
        max_length: Độ dài tối đa. ``None`` dùng config hoặc 200.
        min_length: Độ dài tối thiểu. ``None`` dùng config hoặc 30.
        repetition_penalty: Phạt lặp token. ``None`` dùng config hoặc 1.05.
        length_penalty: Phạt độ dài. ``None`` dùng config hoặc 1.0.
        no_repeat_ngram_size: Chặn lặp n-gram. ``None`` dùng config hoặc 3.
        adapter_path: Thư mục LoRA adapter tùy chọn, phải có
            ``adapter_config.json``. Adapter được gắn khi inference và không merge.
        base_model_path: Alias rõ nghĩa của ``model_path``. Không truyền
            đồng thời cả hai.

    Trả về:
        Chuỗi bản tóm tắt được sinh ra.

    Ví dụ:
        >>> text = "Ngày 15/7, Thủ tướng Chính phủ đã chủ trì cuộc họp..."
        >>> summary = summarize(text, "outputs/vit5_base/best")
        >>> print(summary)
    """
    resolved_base_path = _resolve_base_model_path(
        model_path,
        base_model_path,
        config,
    )

    resolved_source_prefix = (
        source_prefix
        if source_prefix is not None
        else (config.data.source_prefix if config is not None else "summarize: ")
    )
    resolved_max_source_length = (
        max_source_length
        if max_source_length is not None
        else (config.data.max_source_length if config is not None else 768)
    )
    resolved_generation = _resolve_generation_config(
        config,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    # Xây dựng cấu hình nếu không được cung cấp
    if config is None:
        model_config = ModelConfig(name_or_path=resolved_base_path)
        # Phát hiện nếu thuộc dòng T5 để thiết lập cờ (flags) cho tokenizer
        name_lower = resolved_base_path.lower()
        if any(k in name_lower for k in ["t5", "mt5", "vit5"]):
            model_config.use_fast_tokenizer = False

        config = SummarizationConfig(
            model=model_config,
            generation=resolved_generation,
        )
    else:
        # Không mutate config của caller; notebook có thể tái sử dụng config
        # để so sánh base và base + adapter trong cùng một process.
        model_config = replace(
            config.model,
            name_or_path=resolved_base_path,
        )
    
    # Tải full checkpoint, sau đó gắn adapter tùy chọn mà không merge.
    tokenizer, model = load_model_for_inference(
        model_config,
        resolved_generation,
        adapter_path=adapter_path,
    )

    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Chuẩn bị đầu vào
    cleaned = clean_text(text)
    input_text = resolved_source_prefix + cleaned

    inputs = tokenizer(
        input_text,
        max_length=resolved_max_source_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Sinh chuỗi (Generate)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=resolved_generation.max_length,
            min_length=resolved_generation.min_length,
            num_beams=resolved_generation.num_beams,
            length_penalty=resolved_generation.length_penalty,
            no_repeat_ngram_size=resolved_generation.no_repeat_ngram_size,
            repetition_penalty=resolved_generation.repetition_penalty,
            early_stopping=resolved_generation.early_stopping,
        )

    # Giải mã (Decode)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()


# ---------------------------------------------------------------------------
# Suy luận theo batch (Batch inference)
# ---------------------------------------------------------------------------

def summarize_batch(
    texts: list[str],
    model_path: str | Path | None = None,
    config: Optional[SummarizationConfig] = None,
    source_prefix: str | None = None,
    max_source_length: int | None = None,
    batch_size: int = 8,
    adapter_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
) -> list[str]:
    """Sinh bản tóm tắt cho một batch các văn bản.

    Hiệu quả hơn so với việc gọi summarize() lặp đi lặp lại vì mô hình
    chỉ được tải một lần.

    Tham số:
        texts: Danh sách các bài viết đầu vào.
        model_path: Full checkpoint mô hình đã lưu (tên cũ, tương thích
            ngược). Khi dùng LoRA đây là base Phase 1.
        config: Cấu hình đầy đủ (tùy chọn).
        source_prefix: Tiền tố cho các đầu vào. ``None`` dùng config
            hoặc ``'summarize: '`` nếu không có config.
        max_source_length: Chiều dài token tối đa của đầu vào.
            ``None`` dùng config hoặc 768 nếu không có config.
        batch_size: Số lượng văn bản cần xử lý cùng một lúc.
        adapter_path: Thư mục LoRA adapter tùy chọn.
        base_model_path: Alias rõ nghĩa của ``model_path``; không truyền
            đồng thời cả hai.

    Trả về:
        Danh sách các bản tóm tắt được sinh ra.
    """
    resolved_base_path = _resolve_base_model_path(
        model_path,
        base_model_path,
        config,
    )
    resolved_source_prefix = (
        source_prefix
        if source_prefix is not None
        else (config.data.source_prefix if config is not None else "summarize: ")
    )
    resolved_max_source_length = (
        max_source_length
        if max_source_length is not None
        else (config.data.max_source_length if config is not None else 768)
    )

    if config is None:
        model_config = ModelConfig(name_or_path=resolved_base_path)
        name_lower = resolved_base_path.lower()
        if any(k in name_lower for k in ["t5", "mt5", "vit5"]):
            model_config.use_fast_tokenizer = False

        config = SummarizationConfig(
            model=model_config,
        )
    else:
        model_config = replace(
            config.model,
            name_or_path=resolved_base_path,
        )

    tokenizer, model = load_model_for_inference(
        model_config,
        config.generation,
        adapter_path=adapter_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    summaries = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Chuẩn bị đầu vào
        inputs_text = [
            resolved_source_prefix + clean_text(text) for text in batch_texts
        ]

        inputs = tokenizer(
            inputs_text,
            max_length=resolved_max_source_length,
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
  # Full checkpoint (cách gọi cũ vẫn hoạt động):
  python -m src.predict --model outputs/vit5_base/best --text "Bài viết..."

  # Phase 1 full checkpoint + Phase 2 LoRA adapter:
  python -m src.predict \\
    --base-model outputs_phase_1/vit5_base/best \\
    --adapter outputs_phase_2_lora/vit5_base/best \\
    --text "Bài viết y tế..."

  # Từ file:
  python -m src.predict --model outputs/vit5_base/best --file article.txt

  # Thông qua stdin:
  cat article.txt | python -m src.predict --model outputs/vit5_base/best
        """,
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        dest="model_path",
        help="Full checkpoint; tên tham số cũ được giữ để tương thích",
    )
    model_group.add_argument(
        "--base-model",
        dest="base_model_path",
        help="Full base checkpoint (ví dụ checkpoint Phase 1)",
    )
    parser.add_argument(
        "--adapter",
        help=(
            "Thư mục LoRA adapter tùy chọn, phải chứa "
            "adapter_config.json; adapter không bị merge"
        ),
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
        "--prefix",
        default=None,
        help=(
            "Ghi đè tiền tố nguồn; mặc định dùng data.source_prefix "
            "trong config hoặc 'summarize: '"
        ),
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=None,
        help="Ghi đè num_beams; mặc định dùng config hoặc fallback 4",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Ghi đè max_length; mặc định dùng config hoặc fallback 200",
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
    try:
        summary = summarize(
            text=text,
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            adapter_path=args.adapter,
            config=config,
            source_prefix=args.prefix,
            num_beams=args.beams,
            max_length=args.max_length,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print("\n" + "=" * 60)
    print("BẢN TÓM TẮT:")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
