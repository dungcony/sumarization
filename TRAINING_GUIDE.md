# Hướng dẫn train lại ViT5 với hai bộ dữ liệu hiện có

> Mục tiêu: giữ nguyên `data/phase_1` và `data/phase_2`, train Phase 1 bằng
> full fine-tuning, sau đó học Phase 2 bằng LoRA mà không ghi đè checkpoint
> Phase 1.

Luồng này đã được triển khai trong:

- `configs/vit5_base_phase_1.yaml`;
- `configs/vit5_base_phase_2_lora.yaml`;
- `notebook/vit5_two_phase_lora_train.ipynb`;
- `src/trainer.py`, `src/evaluator.py`, `src/model.py`, `src/predict.py`;
- `scripts/evaluate.py`, `scripts/predict.py`.

Không tiếp tục dùng cell Phase 2 full fine-tuning trong
`notebook/vit5_base_train.ipynb` cho flow mới.

---

## 1. Dữ liệu không thay đổi

| Giai đoạn | Train | Validation | Test |
|---|---:|---:|---:|
| Phase 1 | 10.775 | 1.348 | 1.344 |
| Phase 2 | 6.909 | 859 | 871 |

Các config vẫn đọc đúng các split hiện có:

```text
data/phase_1/train_*.**
data/phase_1/validation_*.**
data/phase_1/test_*.**

data/phase_2/train_*.*
data/phase_2/validation_*.*
data/phase_2/test_*.*
```

Notebook mới kiểm tra cứng số dòng trên trước khi train. Nếu file hoặc split
thay đổi, notebook dừng thay vì âm thầm train trên dữ liệu khác.

Trong tài liệu này, “Phase 1” và “Phase 2” là tên hai phân phối dữ liệu hiện
có. Kết quả chỉ chứng minh chất lượng trên các phân phối đó.

---

## 2. Chiến lược train chính thức

```text
VietAI/vit5-base
        │
        │ full fine-tuning trên data/phase_1
        ▼
Phase 1 full checkpoint W₁
        │
        ├── đóng băng toàn bộ W₁
        └── gắn và train LoRA Δ₂ trên data/phase_2
                ▼
        Phase 2 LoRA adapter
```

Khi inference:

```text
Phase 1: W₁
Phase 2: W₁ + Δ₂
```

Không dùng lại hướng `freeze encoder → unfreeze all`: bước unfreeze vẫn sửa
trọng số Phase 1 và có thể gây catastrophic forgetting.

LoRA bảo toàn **artifact Phase 1** vì chỉ adapter được cập nhật. Nó không đảm
bảo output Phase 1 vẫn giống hệt khi adapter đang bật. Vì vậy:

- input Phase 1 dùng base Phase 1, không nạp adapter;
- input Phase 2 dùng Phase 1 + adapter;
- nếu muốn adapter luôn bật cho mọi input, phải áp retention gate ở mục 8.

Không merge adapter trong lúc train/evaluate. Chỉ merge một bản copy sau khi
đã khóa kết quả và thực sự cần một checkpoint duy nhất để deploy.

---

## 3. Artifact bắt buộc

Phase 1 là full model:

```text
outputs_phase_1/vit5_base/
├── best/
│   ├── config.json
│   ├── model.safetensors (hoặc pytorch_model.bin)
│   └── tokenizer files
├── train_results.json
├── eval_results.json          # validation, không phải test
└── resolved_config.json
```

Phase 2 chỉ là adapter:

```text
outputs_phase_2_lora/vit5_base/
├── best/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── adapter_manifest.json
├── adapter_manifest.json
├── train_results.json
├── eval_results.json          # validation, không phải test
└── resolved_config.json
```

Kết quả test và prediction được tách khỏi training artifact:

```text
evaluations_two_phase/
├── phase1_on_phase1_test/
├── phase1_lora_on_phase1_test/
├── phase1_on_phase2_test/
├── phase1_lora_on_phase2_test/
└── cross_phase_results.json
```

Khi đưa Phase 2 lên Kaggle Dataset hoặc server, phải version cả:

1. full checkpoint Phase 1;
2. LoRA adapter Phase 2;
3. `adapter_manifest.json` mô tả dependency.

Adapter riêng lẻ không phải một model hoàn chỉnh.

Pipeline tạo SHA-256 fingerprint từ `config.json`, tokenizer files và toàn bộ
weight shards của Phase 1. Fingerprint được lưu trong adapter manifest và
không chứa absolute path, nên vẫn dùng được sau khi chuyển artifact. Train
kiểm tra Phase 1 không thay đổi trước/sau Phase 2; evaluator và inference từ
chối một adapter nếu fingerprint không khớp base đang chọn. Adapter legacy
không có manifest chỉ được API thư viện nạp kèm cảnh báo để tương thích ngược;
notebook train chuẩn từ chối artifact đó cho tới khi provenance được tạo lại.

---

## 4. Cấu hình đang dùng

### Phase 1 full fine-tuning

File: `configs/vit5_base_phase_1.yaml`.

| Thiết lập | Giá trị |
|---|---:|
| Epoch | 3 |
| Batch train / eval | 2 / 1 |
| Gradient accumulation | 8 |
| Effective batch, 1 GPU | 16 |
| Learning rate | `3e-5` |
| Optimizer | Adafactor |
| Scheduler / warmup | cosine / 5% |
| Precision | auto |
| Gradient checkpointing | bật |
| Validation / save | mỗi epoch |
| Early stopping patience | 2 lần validation |
| Generation beams | 2 |
| Source / target cap | 768 / 256 token |
| LoRA | tắt |

### Phase 2 LoRA

File: `configs/vit5_base_phase_2_lora.yaml`.

| Thiết lập | Giá trị |
|---|---:|
| Base model | Phase 1 `best/` |
| Epoch | 3 |
| Batch train / eval | 2 / 1 |
| Gradient accumulation | 8 |
| Effective batch, 1 GPU | 16 |
| Learning rate | `1e-4` |
| Optimizer | AdamW |
| Scheduler / warmup | cosine / 5% |
| Validation / save | mỗi epoch |
| Generation beams | 2 |
| Source / target cap | 768 / 160 token |
| LoRA | `r=16`, `alpha=32`, dropout `0.05`, target `q,v` tự động |

`freeze_encoder` phải là `false` ở Phase 2. PEFT tự đóng băng toàn bộ base;
không kết hợp cơ chế freeze encoder cũ với LoRA.

`src/trainer.py` có chốt chặn trước khi load data/train:

- phải có ít nhất một tham số trainable;
- mọi tham số trainable phải có tên `lora_*`;
- nếu còn base weight trainable, run dừng bằng `RuntimeError`;
- manifest ghi base checkpoint, fingerprint, cấu hình LoRA và tỷ lệ tham số
  trainable.

Nếu OOM trên GPU 16 GB, ưu tiên đổi:

```text
batch 2 × accumulation 8
→ batch 1 × accumulation 16
```

Không đổi effective batch và learning rate cùng lúc nếu chưa có validation
experiment riêng.

---

## 5. Chạy notebook chuẩn

Mở `notebook/vit5_two_phase_lora_train.ipynb`. Notebook có năm mode:

| MODE | Tác dụng | Có xem test không? |
|---|---|---|
| `phase1_smoke` | 20 bước Phase 1, save và reload | Không |
| `phase1_full` | Full fine-tuning Phase 1 | Có, sau khi train nếu bật gate |
| `phase2_smoke` | 20 bước LoRA từ Phase 1, save và reload | Không |
| `phase2_full` | Full train adapter và ma trận đánh giá | Có, sau khi train |
| `evaluate` | Chỉ reload hai artifact và đánh giá | Có |

Thứ tự khuyến nghị:

```text
Phiên 1: phase1_smoke → phase1_full → version Phase 1 best/
Phiên 2: mount Phase 1 → phase2_smoke → phase2_full → version adapter
```

Mỗi lần đổi `MODE`, nên Restart Session rồi Run All từ đầu để giải phóng GPU
và tránh giữ model cũ trong RAM.

Các biến cần chú ý ở bảng điều khiển notebook:

```python
MODE = "phase1_smoke"
RUN_TAG = ""
MANUAL_DATA_ROOT = ""
MANUAL_PHASE1_BEST = ""
MANUAL_PHASE2_ADAPTER = ""
RESUME_CHECKPOINT = ""
RUN_TEST_AFTER_FULL_TRAIN = True
RUN_CROSS_PHASE_MATRIX = True
RUN_PHASE1_PRETRAINED_BASELINE = True
REQUIRE_PHASE1_VALIDATION_IMPROVEMENT = True
REQUIRE_PHASE2_VALIDATION_IMPROVEMENT = True
MAX_PHASE1_VALIDATION_ROUGEL_DROP = 1.0
ALLOW_CPU = False
EVAL_MAX_SAMPLES = None
```

Source project nằm trong repo:

```text
https://github.com/dungcony/text-sumarization.git
└── tuan 5-6/sumarization/
```

Trước khi chạy Kaggle, commit/push pipeline mới rồi đặt `REPO_REF` trong cell
setup bằng commit hash hoặc tag đó. Notebook in commit đang dùng để kết quả có
thể tái lập; không dựa vào một branch thay đổi theo thời gian.

Hai thư mục dữ liệu đang bị Git ignore, nên clone source **không mang theo
data**. Trên Kaggle, Add Input chứa cả `phase_1/` và `phase_2/`, rồi đặt:

```python
MANUAL_DATA_ROOT = "/kaggle/input/<dataset-name>/data"
```

Biến này phải trỏ tới thư mục cha trực tiếp của `phase_1/` và `phase_2/`.
Notebook có thể tự tìm nếu chỉ có đúng một data root hợp lệ trong
`/kaggle/input`; nếu có nhiều, nó dừng và yêu cầu chọn rõ. Sáu data path được
override trong resolved config, không copy hoặc sửa dữ liệu.

Trên Kaggle Phase 2, nên điền rõ `MANUAL_PHASE1_BEST` bằng đường dẫn `best/`
đã mount. Nếu có nhiều artifact, notebook dừng và yêu cầu chọn thay vì đoán.

Notebook mặc định là `phase1_smoke`. Full train chỉ bắt đầu khi người dùng tự
đổi `MODE`. Nếu output đã có dữ liệu, notebook cũng dừng; đặt `RUN_TAG` mới
hoặc resume đúng `checkpoint-N`, không trộn hai run.

Notebook mặc định yêu cầu CUDA GPU vì ViT5-base train/generate trên CPU rất
chậm. Chỉ đặt `ALLOW_CPU=True` khi bạn chủ động chấp nhận thời gian chạy đó.

---

## 6. Warm-start khác resume

Phase 2 warm-start từ Phase 1:

```yaml
model:
  name_or_path: "<phase_1_best>"
```

`resume_from_checkpoint` chỉ dùng khi **cùng một run** bị ngắt và cần khôi
phục model/adapter, optimizer, scheduler, global step và RNG state.

```text
Phase 1 → Phase 2: warm-start base, optimizer mới
Cùng Phase 2 bị ngắt: resume checkpoint-N của đúng output Phase 2
```

Không đặt adapter `best/` làm `model.name_or_path`. Khi resume Phase 2, vẫn
khởi tạo từ Phase 1 base, bật cùng cấu hình LoRA rồi resume `checkpoint-N`.

Với Phase 2, `adapter_manifest.json` được ghi ở run root **trước**
`trainer.train()`. Khi resume, code bắt buộc kiểm tra checkpoint nằm trong
đúng output run, có `trainer_state.json`, base fingerprint trùng khớp, đồng
thời rank/alpha/dropout/target modules và phase không đổi. Run LoRA cũ bị ngắt
mà không có manifest sẽ bị từ chối resume để tránh âm thầm gắn checkpoint vào
nhầm Phase 1 base.

Không resume full run từ smoke checkpoint vì tổng step và scheduler khác.

---

## 7. Đánh giá và inference

Đánh giá full checkpoint trên test:

```bash
python scripts/evaluate.py eval \
  --model outputs_phase_1/vit5_base/best \
  --config configs/vit5_base_phase_1.yaml \
  --split test \
  --output-dir evaluations_two_phase/phase1_on_phase1_test
```

Đánh giá adapter trên Phase 2 test:

```bash
python scripts/evaluate.py eval \
  --model outputs_phase_2_lora/vit5_base/best \
  --base-model outputs_phase_1/vit5_base/best \
  --config configs/vit5_base_phase_2_lora.yaml \
  --split test \
  --output-dir evaluations_two_phase/phase1_lora_on_phase2_test
```

Inference Phase 2:

```python
from src.config import load_config
from src.predict import summarize

config = load_config("configs/vit5_base_phase_2_lora.yaml")
summary = summarize(
    text=article,
    base_model_path="outputs_phase_1/vit5_base/best",
    adapter_path="outputs_phase_2_lora/vit5_base/best",
    config=config,
)
```

CLI tương đương:

```bash
python scripts/predict.py \
  --base-model outputs_phase_1/vit5_base/best \
  --adapter outputs_phase_2_lora/vit5_base/best \
  --text "Văn bản cần tóm tắt..."
```

Loader luôn nạp Phase 1 trước rồi gắn adapter ở chế độ inference; không merge
và không sửa config object của caller. Trước khi tải model lớn, loader đối
chiếu SHA-256 fingerprint trong manifest để ngăn gắn adapter vào nhầm một
checkpoint ViT5 có cùng kiến trúc.

Notebook so sánh cũ
`compare_phase_1_phase_2_on_phase_1_test.ipynb` chỉ dành cho hai full
checkpoint. Không dùng notebook đó để nạp LoRA; notebook mới đã chạy ma trận
đúng cho adapter.

`app.py` hiện vẫn tự nạp một full checkpoint bằng Transformers và chưa có hai
đầu vào `base_model_path + adapter_path`. Không trỏ `app.py` trực tiếp vào thư
mục adapter `best/`; dùng `src.predict`/CLI ở trên cho tới khi app được nâng
cấp riêng.

---

## 8. Protocol đánh giá

### Trong khi train

Chỉ dùng validation để:

- chọn best checkpoint/adapter;
- early stopping;
- chọn learning rate, epoch, rank hoặc generation settings.

`src/trainer.py` không còn tải/tokenize test trong quá trình train. Test chỉ
được gọi tường minh qua `evaluate_checkpoint(..., split="test")` sau khi model
đã khóa.

Trước khi mở Phase 2 test, notebook reload hai artifact và chấm Phase 1 so
với Phase 1 + LoRA trên Phase 2 validation. Với
`REQUIRE_PHASE2_VALIDATION_IMPROVEMENT=True`, notebook dừng nếu delta
ROUGE-L không dương; test sẽ không được dùng để cứu một cấu hình validation
không đạt.

Khi bật `RUN_CROSS_PHASE_MATRIX`, notebook cũng chấm Phase 1 và Phase 1 +
LoRA trên Phase 1 validation trước mọi test. Mặc định nó dừng nếu adapter làm
giảm ROUGE-L quá `MAX_PHASE1_VALIDATION_ROUGEL_DROP=1.0`. Nếu hệ thống có
router bảo đảm Phase 1 luôn tắt adapter, có thể đặt ngưỡng này thành `None`,
nhưng vẫn nên giữ kết quả retention làm diagnostic.

Tương tự, `phase1_full` mặc định chấm lại `VietAI/vit5-base` và Phase 1 trên
cùng Phase 1 validation bằng scorer Unicode. Nếu Phase 1 không cải thiện và
`REQUIRE_PHASE1_VALIDATION_IMPROVEMENT=True`, notebook dừng trước Phase 1
test. Có thể tắt `RUN_PHASE1_PRETRAINED_BASELINE` chỉ khi baseline mới đã được
chấm và lưu ở một run tái lập khác.

### Sau khi khóa config

| Dataset | Baseline | Candidate | Generation config |
|---|---|---|---|
| Phase 1 test | Phase 1 | Phase 1 + LoRA | cùng config Phase 1 |
| Phase 2 test | Phase 1 | Phase 1 + LoRA | cùng config Phase 2 |

Không so loss của hai dataset với nhau vì target length và phân phối khác
nhau. Chỉ so hai model trên cùng split, cùng tokenizer và cùng decoding.

ROUGE hiện dùng tokenizer Unicode tiếng Việt: chuẩn hóa NFC, `casefold` và giữ
đúng chữ có dấu. Tất cả điểm được tạo trước bản sửa này là **LEGACY**; phải
chấm lại Phase 1 và candidate bằng cùng scorer mới.

Evaluator xuất `predictions_validation.jsonl` hoặc `predictions_test.jsonl`
với toàn bộ `article`, `reference` và `prediction` của từng mẫu.

### Gate bắt buộc

Phase 1:

- validation ROUGE-L tốt hơn pretrained baseline trên cùng split;
- test không lệch bất thường so với validation;
- không có output rỗng, lặp vòng hoặc bị cắt hàng loạt.

Phase 2:

- Phase 1 + adapter cải thiện Phase 2 validation so với Phase 1;
- sau khi khóa adapter, Phase 2 test tốt hơn Phase 1 trên cùng test;
- không có `NaN`/`Inf`, output rỗng hoặc lặp nghiêm trọng;
- review factuality là hard gate: entity, số liệu, phủ định, thuốc/liều và
  quan hệ nguyên nhân phải đúng.

Retention:

- route Phase 1 không nạp adapter nên artifact Phase 1 được bảo toàn;
- nếu adapter luôn bật, đặt trước ngưỡng regression Phase 1 validation, ví dụ
  ROUGE-L không giảm quá 1 điểm, rồi mới xem test;
- LoRA không biến ROUGE thành bằng chứng an toàn y tế. Cần đọc prediction và
  review chuyên môn trước deploy.

Test hiện có đã được dùng trong các lần so sánh trước, nên nên gọi đây là
development benchmark, không tuyên bố là holdout hoàn toàn chưa từng xem.

---

## 9. Checklist trước full train

- [ ] Kernel đã Restart và notebook được Run All từ đầu.
- [ ] `phase1_smoke` đã save/reload full checkpoint thành công.
- [ ] Phase 1 dùng `lora.enabled=false`.
- [ ] Phase 2 trỏ đúng Phase 1 `best/` đã version.
- [ ] Fingerprint base-adapter được xác minh thành công.
- [ ] `phase2_smoke` đã save/reload adapter thành công.
- [ ] Phase 2 dùng `lora.enabled=true`, `freeze_encoder=false`.
- [ ] Log xác nhận chỉ `lora_*` trainable và tỷ lệ dưới 1%.
- [ ] Phase 1 và Phase 2 dùng hai output directory khác nhau.
- [ ] Không resume full run từ smoke run.
- [ ] Không dùng test để chọn hyperparameter/checkpoint.
- [ ] `EVAL_MAX_SAMPLES=None` khi chấm kết quả báo cáo.
- [ ] Không trộn điểm ROUGE legacy với scorer Unicode mới.
- [ ] Prediction JSONL được review lỗi factuality trước deploy.

---

## Kết luận

```text
Phase 1: full fine-tuning → full checkpoint cố định
Phase 2: Phase 1 frozen + LoRA → adapter riêng có thể bật/tắt
```

Đây là phép “cộng” theo dạng `base + adapter`, không phải tiếp tục sửa toàn bộ
base. Nhờ vậy Phase 1 không bị ghi đè; hiệu quả của phần cộng thêm được kiểm
tra bằng ma trận Phase 1/Phase 2 thay vì chỉ so với pretrained base.
