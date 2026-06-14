# FLUX.1-schnell on RunPod Serverless (text-to-image + image-to-image)
#
# Build:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t <your-dockerhub-user>/flux-schnell-runpod:latest .
#
# (HF_TOKEN must be a token from an account that has accepted the FLUX.1-schnell
#  license at https://huggingface.co/black-forest-labs/FLUX.1-schnell)

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

# --- Bake model weights into the image (build-time, using HF token secret) ---
COPY builder/download_weights.py /app/builder/download_weights.py
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || echo "$HF_TOKEN")" \
    python /app/builder/download_weights.py

# --- App code ---
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
