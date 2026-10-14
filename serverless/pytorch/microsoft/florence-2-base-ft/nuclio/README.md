# Microsoft Florence-2-base-ft for CVAT Auto-Annotation

This directory configures **[microsoft/Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft/tree/main)** as a native **Serverless (Nuclio)** auto-annotation model for CVAT.

---

## 1. Overview

**Florence-2-base-ft** is Microsoft's unified vision foundation model (~0.23B parameters, ~230M) fine-tuned on a rich collection of downstream vision tasks. Its compact size delivers high inference speed while achieving exceptional accuracy for:
- **Object Detection (`<OD>`)**: Standard bounding box detection across common perception categories.
- **Phrase Grounding (`<CAPTION_TO_PHRASE_GROUNDING>`)**: Open-vocabulary localization of specific phrases or classes.
- **Dense Region Captioning (`<DENSE_REGION_CAPTION>`)**: Detecting objects and annotating them with descriptive captions.
- **OCR & OCR with Region (`<OCR>`, `<OCR_WITH_REGION>`)**.

---

## 2. Directory Structure

```
serverless/pytorch/microsoft/florence-2-base-ft/nuclio/
├── function.yaml          # Nuclio manifest for CPU execution
├── function-gpu.yaml      # Nuclio manifest for GPU execution (CUDA, fp16)
├── main.py               # Nuclio entrypoint (init_context & handler)
├── model_handler.py      # Florence-2 loader, inference & CVAT coordinate formatter
├── download_model.py     # Utility to pre-download model weights from Hugging Face
├── test_florence2.py     # Standalone local test & verification script
└── README.md             # This documentation
```

---

## 3. Quick Start & Deployment Guide

### Bước 1: Khởi động CVAT với Serverless (Nuclio)
Đảm bảo CVAT đang chạy với profile serverless kích hoạt:

```bash
cd /media/bang/Work/projects/cvat
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

Kiểm tra container `nuclio` đã ở trạng thái `Up`:
```bash
docker ps --filter "name=nuclio"
```

---

### Bước 2: Cài đặt công cụ dòng lệnh `nuctl`
Nếu máy chưa cài `nuctl`, tải phiên bản tương ứng với Nuclio dashboard (1.16.3):

```bash
wget https://github.com/nuclio/nuclio/releases/download/1.16.3/nuctl-1.16.3-linux-amd64
chmod +x nuctl-1.16.3-linux-amd64
sudo mv nuctl-1.16.3-linux-amd64 /usr/local/bin/nuctl
nuctl version
```

---

### Bước 3: Triển khai Mô hình lên CVAT

#### Tùy chọn A: Triển khai trên GPU (Khuyến nghị - Nhanh & Nhẹ với fp16)
Hệ thống có GPU NVIDIA (ví dụ: RTX 3070 Laptop GPU 8GB VRAM):

```bash
cd /media/bang/Work/projects/cvat

nuctl deploy --project-name cvat \
  --path serverless/pytorch/microsoft/florence-2-base-ft/nuclio \
  --file serverless/pytorch/microsoft/florence-2-base-ft/nuclio/function-gpu.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

Hoặc dùng helper script có sẵn của CVAT:
```bash
./serverless/deploy_gpu.sh serverless/pytorch/microsoft/florence-2-base-ft/nuclio
```

#### Tùy chọn B: Triển khai trên CPU
Nếu không dùng GPU:

```bash
cd /media/bang/Work/projects/cvat

nuctl deploy --project-name cvat \
  --path serverless/pytorch/microsoft/florence-2-base-ft/nuclio \
  --file serverless/pytorch/microsoft/florence-2-base-ft/nuclio/function.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

---

### Bước 4: Kiểm tra trạng thái Function
Chạy lệnh:
```bash
nuctl get functions --platform local
```
Kết quả mong muốn:
```
NAME                             PROJECT  STATE  NODE PORT  REPLICAS
pth-microsoft-florence-2-base-ft cvat     ready  ...        1/1
```

---

### Bước 5: Sử dụng trên Giao diện CVAT UI
1. Mở trình duyệt vào CVAT: `http://localhost:8080`.
2. Mở một **Task** hoặc **Job**.
3. Chọn menu **Actions** > **Automatic Annotation** (hoặc mở thanh công cụ bên trái > biểu tượng **AI Tools** / **Detectors**).
4. Trong danh sách dropdown model, chọn:
   **Florence-2 Base FT (Microsoft)**.
5. Khớp nhãn (Label matching) giữa task của bạn và nhãn của Florence-2.
6. Bấm **Annotate** để mô hình tự động gán nhãn toàn bộ khung hình!

---

## 4. Kiểm Thử Cục Bộ Không Cần Docker (Local Verification)

Bạn có thể kiểm tra trực tiếp khả năng suy luận của Florence-2 trên một file ảnh trước khi build:

```bash
# Cài đặt thư viện thử nghiệm nếu chưa có
pip install transformers timm einops pillow

# Chạy test với ảnh bất kỳ
python serverless/pytorch/microsoft/florence-2-base-ft/nuclio/test_florence2.py \
  --image /path/to/sample.jpg \
  --prompt "<OD>"
```

Để thử nghiệm **Phrase Grounding** (gán nhãn theo câu lệnh cụ thể):
```bash
python serverless/pytorch/microsoft/florence-2-base-ft/nuclio/test_florence2.py \
  --image /path/to/sample.jpg \
  --prompt "<CAPTION_TO_PHRASE_GROUNDING>" \
  --text "car. pedestrian. traffic light."
```
