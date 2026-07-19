# Vietnamese Text Summarization - Fine-tuning Framework

> **Tóm tắt văn bản tiếng Việt** — Framework fine-tuning mô hình seq2seq cho bài toán tóm tắt văn bản tiếng Việt.

## 🎯 Tổng quan / Overview

Framework này giúp fine-tune các mô hình ngôn ngữ lớn (ViT5, BARTpho, mT5) trên dữ liệu tóm tắt văn bản tiếng Việt. Kết quả tốt nhất đạt **ROUGE-L ≈ 0.49** với ViT5-base full fine-tuning.

This framework fine-tunes seq2seq language models for Vietnamese abstractive text summarization. Best result: **ROUGE-L ≈ 0.49** with ViT5-base full fine-tuning.

## 📊 Kết quả / Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Type |
|-------|---------|---------|---------|------|
| **ViT5-base (full)** | 74.22 | 46.75 | **48.89** | Full fine-tuning |
| BARTpho-syllable | 73.47 | 46.17 | 48.07 | Full fine-tuning |
| ViT5 warm-start | 71.61 | 44.26 | 47.28 | Full fine-tuning |
| ViT5-base (LoRA) | 72.62 | 44.08 | 46.63 | LoRA r=16 |

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

# ViT5 với LoRA (tiết kiệm bộ nhớ / memory-efficient)
python scripts/train.py --config configs/vit5_base_lora.yaml

# BARTpho
python scripts/train.py --config configs/bartpho.yaml

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
│   ├── vit5_base_lora.yaml       # ViT5 + LoRA (tiết kiệm memory)
│   ├── vit5_warmstart.yaml       # ViT5 warm-start (đã pre-trained)
│   ├── bartpho.yaml              # BARTpho Vietnamese BART
│   └── mt5_base.yaml             # mT5 multilingual T5
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
| BARTpho | `vinai/bartpho-syllable` | 140M | No prefix needed |
| mT5-base | `google/mt5-base` | 580M | Multilingual |

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

Files mặc định nằm ở / Default data location:
```
data/fine-turn/
├── train-00000-of-00001.parquet   # ~10,000 samples
├── valid-00000-of-00001.parquet   # ~1,300 samples
└── test-00000-of-00001.parquet    # ~1,300 samples (optional)
```

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
- **Dùng LoRA nếu ít VRAM** — chỉ cần 8GB / Use LoRA if limited GPU memory (8GB+)
- **Test nhanh trước khi train dài** — dùng `--max-steps 10` / Quick test before long training
- **Kiểm tra tokenizer trước** — chạy `scripts/check_tokenizer.py` / Check tokenizer first

## 📚 Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.3.0
- Transformers ≥ 4.51.0
- PEFT ≥ 0.12.0 (for LoRA)
- Datasets ≥ 2.20.0
- GPU with ≥ 8GB VRAM (16GB+ recommended)
