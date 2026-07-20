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

Ví dụ:
    >>> from src.data import load_and_preprocess
    >>> datasets = load_and_preprocess(tokenizer, config.data)
    >>> print(datasets['train'][0].keys())
    dict_keys(['input_ids', 'attention_mask', 'labels'])
"""

from __future__ import annotations

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
    """Tải dữ liệu từ các file hoặc thư mục (hỗ trợ CSV/Parquet) vào DatasetDict.

    Tham số:
        train_file: Đường dẫn (hoặc mẫu glob) đến dữ liệu huấn luyện.
        valid_file: Đường dẫn (hoặc mẫu glob) đến dữ liệu đánh giá.
        test_file: Tùy chọn đường dẫn (hoặc mẫu glob) đến dữ liệu kiểm tra.

    Trả về:
        DatasetDict chứa các phần chia 'train', 'validation', và 'test'.
    """
    from datasets import load_dataset
    
    data_files = {
        "train": str(train_file),
        "validation": str(valid_file),
    }
    if test_file:
        data_files["test"] = str(test_file)
        
    # Tự động nhận diện định dạng dựa trên đuôi file
    file_format = "csv" if "csv" in str(train_file).lower() else "parquet"
    
    logger.info(f"Đang tải dữ liệu (định dạng: {file_format}) từ các đường dẫn: {data_files}")
    dataset = load_dataset(file_format, data_files=data_files)

    # Đổi tên cột nếu người dùng gõ sai chính tả (sumary -> summary)
    for split_name, split_data in dataset.items():
        if "sumary" in split_data.column_names:
            logger.info(f"Tự động đổi tên cột 'sumary' thành 'summary' trong tập {split_name}")
            dataset[split_name] = split_data.rename_column("sumary", "summary")
            split_data = dataset[split_name]
            
        if "Document" in split_data.column_names:
            dataset[split_name] = split_data.rename_column("Document", "article")
            split_data = dataset[split_name]
            
        if "Summary" in split_data.column_names:
            dataset[split_name] = split_data.rename_column("Summary", "summary")
            split_data = dataset[split_name]

    # Xác thực các cột bắt buộc
    for split_name, split_data in dataset.items():
        if "article" not in split_data.column_names:
            raise ValueError(
                f"Thiếu cột 'article' trong tập {split_name}. "
                f"Các cột hiện có: {split_data.column_names}"
            )
        if "summary" not in split_data.column_names:
            raise ValueError(
                f"Thiếu cột 'summary' trong tập {split_name}. "
                f"Các cột hiện có: {split_data.column_names}"
            )

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
