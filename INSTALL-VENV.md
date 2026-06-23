## 1. Cài môi trường

```bash
bash install.sh
```

Tự động hóa 3 bước theo đúng thứ tự:
1. Cài `torch==2.3.1+cu121` từ index riêng của PyTorch
2. Compile `mamba-ssm` với `--no-build-isolation` (cần thấy torch đã cài)
3. `pip install -r requirements.txt` cho phần còn lại

| Package | Lý do pin version |
|---|---|
| `timm==0.4.12` | Code dùng `timm.models.layers` — API bị xóa ở timm mới hơn |
| `transformers==4.46.3` | `mamba-ssm 2.1.0` import API bị xóa ở transformers 5.x |
| `torch==2.3.1+cu121` | Không có trên PyPI, chỉ có trên `download.pytorch.org/whl/cu121` |

---

## 2. Chuẩn bị dataset

```bash
bash setup_data.sh
```

Tự động: download Synapse từ Google Drive → unzip vào `data/` → rebuild `train.txt` đúng (file gốc trong zip chỉ có 3 entries).

---

## 3. Train

```bash
./venv/bin/python train_synapse.py
```
