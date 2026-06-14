# FLUX.1-schnell on RunPod Serverless (text-to-image + image-to-image)
#
# Deploy via RunPod GitHub integration:
#   Thêm build environment variable HF_TOKEN trong RunPod -> nó được truyền vào ARG bên dưới.
#
# (Hoặc build local:
#   docker build --build-arg HF_TOKEN=$HF_TOKEN -t <user>/flux-schnell-runpod:latest . )
#
# HF_TOKEN phải là token từ tài khoản đã accept license FLUX.1-schnell tại
#   https://huggingface.co/black-forest-labs/FLUX.1-schnell

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=black-forest-labs/FLUX.1-schnell \
    MODEL_DIR=/models/FLUX.1-schnell

WORKDIR /app

# System libs needed by Pillow / opencv-style image IO
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (faster downloads with hf_transfer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install hf_transfer && \
    pip install -r requirements.txt

# --- Bake model weights into the image (build-time, via HF_TOKEN build arg) ---
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
COPY builder/download_weights.py /app/builder/download_weights.py
RUN python /app/builder/download_weights.py

# --- App code ---
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
