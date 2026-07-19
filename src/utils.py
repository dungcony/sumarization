"""
Shared Utilities (Các tiện ích chung)
================

Các hàm hỗ trợ chung được sử dụng trong toàn bộ package:
    - Thiết lập logging
    - Quản lý random seed
    - I/O File (YAML, JSON)
    - Phân giải đường dẫn (Path resolution)
    - Đếm tham số
    - Nhận diện độ chính xác (Precision detection)
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Ghi log (Logging)
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "src",
    level: int = logging.INFO,
) -> logging.Logger:
    """Tạo một logger với định dạng nhất quán.

    Tham số:
        name: Tên logger (xuất hiện trong log output).
        level: Cấp độ logging (DEBUG, INFO, WARNING, ERROR).

    Trả về:
        Đối tượng logger đã được cấu hình.

    Ví dụ:
        >>> logger = setup_logger()
        >>> logger.info("Training started")
        [INFO] src: Training started
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Khả năng tái tạo (Reproducibility)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Thiết lập random seed cho khả năng tái tạo trên tất cả các framework.

    Thiết lập seed cho: Python random, NumPy, PyTorch (CPU + CUDA).

    Tham số:
        seed: Giá trị seed (số nguyên).

    Ví dụ:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Đảm bảo hành vi xác định (tốn một chút hiệu năng)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# I/O File
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Tải một file YAML và trả về dưới dạng dictionary.

    Tham số:
        path: Đường dẫn tới file YAML.

    Trả về:
        Nội dung YAML đã được phân tích dưới dạng dictionary.

    Ngoại lệ:
        FileNotFoundError: Nếu file không tồn tại.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file YAML: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:
    """Lưu dữ liệu thành một file JSON có định dạng.

    Tham số:
        data: Dữ liệu cần tuần tự hóa (dict, list, v.v.).
        path: Đường dẫn file đầu ra.
        indent: Mức độ thụt lề JSON.

    Trả về:
        Đường dẫn tới file đã lưu.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return path


def load_json(path: str | Path) -> Any:
    """Tải một file JSON.

    Tham số:
        path: Đường dẫn tới file JSON.

    Trả về:
        Nội dung JSON đã được phân tích.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phân giải đường dẫn (Path resolution)
# ---------------------------------------------------------------------------

def resolve_data_path(
    file_path: str,
    search_dirs: Optional[list[str | Path]] = None,
) -> Path:
    """Tìm một file dữ liệu, có thể tìm kiếm ở nhiều thư mục nếu cần.

    Thứ tự phân giải:
        1. Đường dẫn tuyệt đối (absolute path) (trả về nguyên dạng nếu tồn tại)
        2. Tương đối so với thư mục làm việc hiện tại (CWD)
        3. Từng thư mục trong search_dirs

    Tham số:
        file_path: Đường dẫn cần tìm (tuyệt đối hoặc tương đối).
        search_dirs: Các thư mục bổ sung để tìm kiếm.

    Trả về:
        Đường dẫn tuyệt đối đã được phân giải.

    Ngoại lệ:
        FileNotFoundError: Nếu không tìm thấy file.
    """
    path = Path(file_path)

    # 1. Đã là đường dẫn tuyệt đối và tồn tại
    if path.is_absolute() and path.exists():
        return path

    # 2. Tương đối so với CWD
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    # 3. Tìm kiếm ở các thư mục bổ sung
    for search_dir in (search_dirs or []):
        candidate = Path(search_dir) / path
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu: '{file_path}'\n"
        f"Đã tìm trong:\n"
        f"  - {path} (tuyệt đối)\n"
        f"  - {cwd_path} (tương đối so với CWD)\n"
        + "\n".join(f"  - {Path(d) / path}" for d in (search_dirs or []))
    )


# ---------------------------------------------------------------------------
# Các tiện ích mô hình (Model utilities)
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Đếm các tham số của mô hình.

    Tham số:
        model: Mô hình PyTorch.

    Trả về:
        Dictionary chứa số lượng tham số 'total' (tổng), 'trainable' (có thể huấn luyện) và 'frozen' (đã đóng băng).

    Ví dụ:
        >>> counts = count_parameters(model)
        >>> print(f"Trainable: {counts['trainable']:,}")
        Trainable: 223,000,000
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": round(100 * trainable / total, 2) if total > 0 else 0,
    }


def detect_precision() -> str:
    """Tự động phát hiện độ chính xác huấn luyện tốt nhất cho phần cứng hiện tại.

    Trả về:
        'bf16' nếu GPU hỗ trợ bfloat16 (Ampere+),
        'fp16' nếu CUDA có sẵn,
        'fp32' nếu không hỗ trợ (CPU).
    """
    if not torch.cuda.is_available():
        return "fp32"

    if torch.cuda.is_bf16_supported():
        return "bf16"

    return "fp16"


def get_device_info() -> dict[str, Any]:
    """Nhận thông tin thiết bị hiện tại.

    Trả về:
        Dictionary với loại thiết bị, số lượng GPU, tên GPU, v.v.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "precision": detect_precision(),
    }

    for i in range(info["num_gpus"]):
        info["gpu_names"].append(torch.cuda.get_device_name(i))

    return info


# ---------------------------------------------------------------------------
# Định dạng văn bản (Text formatting)
# ---------------------------------------------------------------------------

def format_number(n: int) -> str:
    """Định dạng một số lớn với dấu phẩy để dễ đọc.

    Ví dụ:
        >>> format_number(223000000)
        '223,000,000'
    """
    return f"{n:,}"


def format_duration(seconds: float) -> str:
    """Định dạng số giây thành một khoảng thời gian dễ đọc.

    Ví dụ:
        >>> format_duration(3661)
        '1h 1m 1s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)
