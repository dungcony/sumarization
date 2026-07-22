# Vietnamese Text Summarization - Fine-tuning Framework

> **Tóm tắt văn bản tiếng Việt** — Framework fine-tuning mô hình seq2seq cho bài toán tóm tắt văn bản tiếng Việt.

## 🎯 Tổng quan / Overview

Framework này giúp fine-tune ViT5 trên dữ liệu tóm tắt văn bản tiếng Việt. Kết quả tốt nhất đạt **ROUGE-L ≈ 0.49** với ViT5-base full fine-tuning.

This framework fine-tunes seq2seq language models for Vietnamese abstractive text summarization. Best result: **ROUGE-L ≈ 0.49** with ViT5-base full fine-tuning.

## 📊 Kết quả / Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Type |
|-------|---------|---------|---------|------|
| **ViT5-base (full)** | 74.22 | 46.75 | **48.89** | Full fine-tuning |
| ViT5 warm-start | 71.61 | 44.26 | 47.28 | Full fine-tuning |

## 🚀 Bắt đầu nhanh / Quick Start

### 1. Cài đặt / Install

```bash
cd src/text-sumarization
pip install -e .
```

### 2. Huấn luyện / Train

```bash
# ViT5-base full fine-tuning (khuyến nghị / recommended)
python scripts/train.py --config configs/vit5_base.yaml

# ViT5 warm-start từ checkpoint đã học summarization
python scripts/train.py --config configs/vit5_warmstart.yaml

# Test nhanh (chạy 10 bước / quick test with 10 steps)
python scripts/train.py --config configs/vit5_base.yaml \
    --max-steps 10 --max-train-samples 32
```

### 3. Đánh giá / Evaluate

```bash
python scripts/evaluate.py \
    --model outputs/vit5_base/best \
    --config configs/vit5_base.yaml
```

### 4. Dự đoán / Predict

```bash
# Từ text / From text
python scripts/predict.py \
    --model outputs/vit5_base/best \
    --text "Bài viết cần tóm tắt..."

# Từ file / From file
python scripts/predict.py \
    --model outputs/vit5_base/best \
    --file article.txt
```

## 📁 Cấu trúc dự án / Project Structure

```
text-sumarization/
├── configs/                      # Cấu hình YAML / YAML configurations
│   ├── vit5_base.yaml            # ViT5 full fine-tuning ⭐ (best)
│   └── vit5_warmstart.yaml       # ViT5 warm-start (đã học summarization)
│
├── vn_summarization/             # Package chính / Main Python package
│   ├── __init__.py               # Package init
│   ├── config.py                 # Hệ thống cấu hình / Config system
│   ├── data.py                   # Tải & tiền xử lý dữ liệu / Data loading
│   ├── model.py                  # Tải model & tokenizer / Model loading
│   ├── trainer.py                # Pipeline huấn luyện / Training pipeline
│   ├── evaluator.py              # Đánh giá & ROUGE / Evaluation
│   ├── predict.py                # Suy luận / Inference
│   └── utils.py                  # Tiện ích chung / Shared utilities
│
├── scripts/                      # CLI scripts
│   ├── train.py                  # Huấn luyện / Training entry point
│   ├── evaluate.py               # Đánh giá / Evaluation entry point
│   ├── predict.py                # Dự đoán / Prediction entry point
│   └── check_tokenizer.py        # Kiểm tra tokenizer / Tokenizer check
│
├── pyproject.toml                # Dependencies & package definition
└── README.md                     # Tài liệu này / This file
```

## 🔧 Mô hình hỗ trợ / Supported Models

| Model | HuggingFace ID | Params | Ghi chú / Notes |
|-------|----------------|--------|------------------|
| **ViT5-base** | `VietAI/vit5-base` | 223M | ⭐ Best overall |
| ViT5 VietNews | `VietAI/vit5-base-vietnews-summarization` | 223M | Warm-start |

## ⚙️ Cấu hình / Configuration

Mỗi config YAML có 5 phần / Each YAML config has 5 sections:

```yaml
model:        # Mô hình nào / Which model
data:         # Dữ liệu ở đâu / Data paths
training:     # Siêu tham số / Hyperparameters
generation:   # Cài đặt sinh text / Decoding settings
lora:         # Cài đặt LoRA / LoRA settings (optional)
```

Override từ dòng lệnh / Override from CLI:
```bash
python scripts/train.py --config configs/vit5_base.yaml \
    --epochs 5 \
    --learning-rate 0.0001 \
    --batch-size 8
```

## 📋 Định dạng dữ liệu / Data Format

Dữ liệu ở định dạng Apache Parquet với 2 cột bắt buộc:

| Column | Type | Description |
|--------|------|-------------|
| `article` | string | Bài viết gốc / Source article |
| `summary` | string | Tóm tắt / Target summary |

Nguồn gốc nằm trong `data/original`. Script làm sạch tạo đúng hai bộ cho hai
phase và giữ nguyên định dạng:

```
data/clean/
├── parquet/                  # Phase 1: tin tức ngẫu nhiên
│   ├── train/cleaned.parquet
│   ├── validation/cleaned.parquet
│   └── test/cleaned.parquet
└── medical/                  # Phase 2: chủ đề y tế
    ├── train/cleaned.csv
    ├── validation/cleaned.csv
    └── test/cleaned.csv
```

Kiểm tra dữ liệu mà không ghi file / Audit without writing files:

```bash
python scripts/clean_data.py --audit-only \
    --dedupe-repeated-sentences \
    --near-duplicate-threshold 0.90
```

Tạo cả hai bộ sạch (không sửa dữ liệu gốc):

```bash
python scripts/clean_data.py \
    --dedupe-repeated-sentences \
    --near-duplicate-threshold 0.90
```

Phase 2 chỉ lấy `herding_512_bio_medicine.csv`. Sáu CSV y tế còn lại là các
biến thể của cùng source nên không được gộp. `data_summary.csv` không thuộc
pipeline hai phase hiện tại.

## 🧪 Sử dụng trong Python / Python API

```python
from src.config import load_config
from src.trainer import train
from src.predict import summarize

# Huấn luyện / Train
config = load_config("configs/vit5_base.yaml")
metrics = train(config)
print(f"ROUGE-L: {metrics['eval_rougeL']:.4f}")

# Dự đoán / Predict
summary = summarize(
    text="Bài viết dài về kinh tế Việt Nam...",
    model_path="outputs/vit5_base/best",
)
print(summary)
```

## 💡 Tips

- **Bắt đầu với ViT5-base** — cho kết quả tốt nhất / Start with ViT5-base for best results
- **Thử ViT5 warm-start** — dùng checkpoint đã học summarization làm điểm khởi đầu
- **Test nhanh trước khi train dài** — dùng `--max-steps 10` / Quick test before long training
- **Kiểm tra tokenizer trước** — chạy `scripts/check_tokenizer.py` / Check tokenizer first

## 📚 Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.3.0
- Transformers ≥ 4.51.0
- PEFT ≥ 0.12.0 (for LoRA)
- Datasets ≥ 2.20.0
- GPU with ≥ 8GB VRAM (16GB+ recommended)
