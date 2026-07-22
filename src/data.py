"""
Data Loading & Preprocessing (Tải dữ liệu và Tiền xử lý)
============================

Tải các tập dữ liệu tóm tắt văn bản tiếng Việt và chuẩn bị chúng cho quá trình huấn luyện.

Định dạng dữ liệu:
    - Các file Apache Parquet
    - Các cột bắt buộc: 'article' (văn bản nguồn), 'summary' (văn bản đích)
    - Tập dữ liệu tiêu biểu: ~10,000 mẫu huấn luyện, ~1,300 mẫu đánh giá (validation)

Quy trình (Pipeline):
    1. Tải các file parquet → HuggingFace Dataset
    2. Làm sạch văn bản (Chuẩn hóa Unicode NFC + chuẩn hóa khoảng trắng)
    3. Tokenize với tiền tố (source prefix) cho các mô hình seq2seq
    4. Tạo input_ids + labels cho việc huấn luyện
"""

from __future__ import annotations

from glob import glob
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict

from src.config import DataConfig
from src.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Làm sạch văn bản (Text cleaning)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Chuẩn hóa và làm sạch văn bản tiếng Việt.

    Các bước:
        1. Chuẩn hóa Unicode NFC (chuẩn hóa các dấu tiếng Việt)
        2. Gộp nhiều khoảng trắng/tab thành một khoảng trắng duy nhất
        3. Cắt bỏ khoảng trắng ở đầu/cuối chuỗi

    Tham số:
        text: Văn bản đầu vào thô.

    Trả về:
        Chuỗi văn bản đã được làm sạch.

    Ví dụ:
        >>> clean_text("  Xin   chào   Việt  Nam  ")
        'Xin chào Việt Nam'
    """
    if not text:
        return ""
    # Chuẩn hóa Unicode (quan trọng cho các dấu tiếng Việt)
    text = unicodedata.normalize("NFC", text)
    # Gộp khoảng trắng
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tải dữ liệu (Data loading)
# ---------------------------------------------------------------------------

def load_dataset_from_files(
        train_file: str | Path,
        valid_file: str | Path,
        test_file: Optional[str | Path] = None,
) -> DatasetDict:
    """Tải toàn bộ file CSV/Parquet khớp với mỗi pattern vào DatasetDict.
    Trả về:
        DatasetDict chứa các phần chia 'train', 'validation', và 'test'.
    """
    from datasets import concatenate_datasets, load_dataset

    patterns = {
        "train": str(train_file),
        "validation": str(valid_file),
    }
    if test_file:
        patterns["test"] = str(test_file)

    def resolve_files(split_name: str, pattern: str) -> list[Path]:
        paths = sorted(
            {Path(match) for match in glob(pattern, recursive=True) if Path(match).is_file()}
        )
        if not paths:
            raise FileNotFoundError(
                f"Không tìm thấy file cho split '{split_name}' với pattern: {pattern}"
            )
        unsupported = [
            path
            for path in paths
            if path.suffix.casefold() not in {".csv", ".parquet", ".pq"}
        ]
        if unsupported:
            names = ", ".join(str(path) for path in unsupported)
            raise ValueError(
                f"Split '{split_name}' có định dạng không hỗ trợ: {names}. "
                "Chỉ hỗ trợ CSV, Parquet hoặc PQ."
            )
        return paths

    def validate_columns(split_name: str, split_data: Dataset) -> Dataset:
        if "article" not in split_data.column_names:
            raise ValueError(
                f"Thiếu cột 'article' trong tập {split_name}. "
                f"Các cột hiện có: {split_data.column_names}. "
                "Hãy chạy scripts/clean_data.py để chuẩn hóa dữ liệu trước."
            )
        if "summary" not in split_data.column_names:
            raise ValueError(
                f"Thiếu cột 'summary' trong tập {split_name}. "
                f"Các cột hiện có: {split_data.column_names}. "
                "Hãy chạy scripts/clean_data.py để chuẩn hóa dữ liệu trước."
            )
        return split_data

    resolved_files = {
        split_name: resolve_files(split_name, pattern)
        for split_name, pattern in patterns.items()
    }
    logger.info(
        "Đang tải dữ liệu: "
        + "; ".join(
            f"{split}={len(paths)} file" for split, paths in resolved_files.items()
        )
    )

    splits: dict[str, Dataset] = {}
    for split_name, paths in resolved_files.items():
        files_by_format: dict[str, list[str]] = {"csv": [], "parquet": []}
        for path in paths:
            file_format = "csv" if path.suffix.casefold() == ".csv" else "parquet"
            files_by_format[file_format].append(str(path))

        loaded_parts: list[Dataset] = []
        for file_format in ("csv", "parquet"):
            files = files_by_format[file_format]
            if not files:
                continue
            loaded = load_dataset(
                file_format,
                data_files={split_name: files},
                split=split_name,
            )
            loaded_parts.append(validate_columns(split_name, loaded))

        splits[split_name] = (
            loaded_parts[0]
            if len(loaded_parts) == 1
            else concatenate_datasets(loaded_parts)
        )

    dataset = DatasetDict(splits)

    logger.info(
        f"Đã tải tập dữ liệu: "
        f"{len(dataset.get('train', []))} train, "
        f"{len(dataset.get('validation', []))} validation"
        + (f", {len(dataset.get('test', []))} test" if 'test' in dataset else "")
    )

    return dataset


# ---------------------------------------------------------------------------
# Tiền xử lý (Tokenization)
# ---------------------------------------------------------------------------

def preprocess_for_seq2seq(
        dataset: DatasetDict,
        tokenizer: Any,
        data_config: DataConfig,
) -> DatasetDict:
    """Tokenize tập dữ liệu cho việc huấn luyện seq2seq.

    Cho mỗi mẫu (example):
        - Đầu vào (Input):  tiền tố (source_prefix) + clean(article)
        - Đích (Target):    clean(summary)

    Bộ tokenizer sẽ tạo ra:
        - input_ids:      chuỗi nguồn đã được token hóa
        - attention_mask:  attention mask cho chuỗi nguồn
        - labels:          chuỗi đích đã được token hóa (với các padding token được đặt thành -100)

    Tham số:
        dataset: DatasetDict thô với các cột 'article' và 'summary'.
        tokenizer: HuggingFace tokenizer.
        data_config: Cấu hình dữ liệu với các độ dài và tiền tố.

    Trả về:
        DatasetDict đã được token hóa, sẵn sàng cho việc huấn luyện.
    """
    prefix = data_config.source_prefix or ""

    def tokenize_function(examples: dict[str, list]) -> dict[str, list]:
        """Tokenize một batch các mẫu."""
        # Làm sạch và chuẩn bị các đầu vào
        inputs = [
            prefix + clean_text(article)
            for article in examples["article"]
        ]
        targets = [
            clean_text(summary)
            for summary in examples["summary"]
        ]

        # Tokenize đầu vào
        model_inputs = tokenizer(
            inputs,
            max_length=data_config.max_source_length,
            truncation=True,
            padding=False,  # Đệm động (Dynamic padding) ở data collator
        )

        # Tokenize đầu ra mục tiêu (targets)
        labels = tokenizer(
            text_target=targets,
            max_length=data_config.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Áp dụng việc token hóa cho từng phần chia
    tokenized = DatasetDict()

    for split_name, split_data in dataset.items():
        # Tùy chọn giới hạn số mẫu (dùng để gỡ lỗi)
        max_samples = None
        if split_name == "train" and data_config.max_train_samples:
            max_samples = min(data_config.max_train_samples, len(split_data))
        elif split_name in ("validation", "test") and data_config.max_eval_samples:
            max_samples = min(data_config.max_eval_samples, len(split_data))

        if max_samples:
            split_data = split_data.select(range(max_samples))
            logger.info(f"Đang giới hạn phần {split_name} xuống còn {max_samples} mẫu")

        tokenized[split_name] = split_data.map(
            tokenize_function,
            batched=True,
            remove_columns=split_data.column_names,
            desc=f"Tokenizing {split_name}",
        )

        logger.info(
            f"{split_name}: đã tokenize {len(tokenized[split_name])} mẫu"
        )

    return tokenized


# ---------------------------------------------------------------------------
# Hàm tiện ích (Convenience function)
# ---------------------------------------------------------------------------

def load_and_preprocess(
        tokenizer: Any,
        data_config: DataConfig,
) -> DatasetDict:
    """Tải dữ liệu và tiền xử lý trong một bước.

    Đây là điểm khởi đầu chính cho việc tải dữ liệu.

    Tham số:
        tokenizer: HuggingFace tokenizer.
        data_config: Cấu hình dữ liệu.

    Trả về:
        DatasetDict đã được token hóa, sẵn sàng cho việc huấn luyện.

    Ví dụ:
        >>> datasets = load_and_preprocess(tokenizer, config.data)
        >>> train_dataset = datasets['train']
    """
    # Tải dữ liệu thô
    dataset = load_dataset_from_files(
        train_file=data_config.train_file,
        valid_file=data_config.valid_file,
        test_file=data_config.test_file if data_config.test_file else None,
    )

    # Tokenize
    return preprocess_for_seq2seq(dataset, tokenizer, data_config)
