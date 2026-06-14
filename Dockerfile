# FLUX.1-schnell on RunPod Serverless (text-to-image + image-to-image)
#
# Weights KHÔNG bake lúc build (vì RunPod GitHub build không có HF_TOKEN lúc build).
# Thay vào đó tải lúc runtime bằng HF_TOKEN (runtime env var) và cache vào
# Network Volume mount tại /runpod-volume -> các cold start sau dùng lại cache.
#
# Khi tạo endpoint nhớ:
#   1. Gắn một Network Volume (vd 50GB) -> nó mount ở /runpod-volume
#   2. Thêm runtime env var HF_TOKEN = token (tài khoản đã accept license FLUX.1-schnell)

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ID=black-forest-labs/FLUX.1-schnell
# HF_HOME do handler.py tự quyết lúc runtime (dùng /runpod-volume nếu có, không thì cache mặc định).

WORKDIR /app

# System libs needed by Pillow / image IO
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- App code (weights tải lúc runtime, xem handler.py) ---
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
