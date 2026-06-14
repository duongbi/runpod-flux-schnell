"""
Download FLUX.1-schnell weights at BUILD TIME so they are baked into the image.
This avoids slow cold starts and runtime dependency on the HF Hub.

Requires a HF token (the model is gated -> you must accept the license once on
huggingface.co, then create a read token). Pass it as a build arg / secret.
"""

import os

from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
TARGET = os.environ.get("MODEL_DIR", f"/models/{MODEL_ID.split('/')[-1]}")

# Only grab what the diffusers FluxPipeline needs; skip duplicate / non-diffusers
# checkpoints to keep the image smaller.
ALLOW = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "transformer/**",
    "vae/**",
]
IGNORE = ["*.bin", "*.onnx", "*.pt", "flux1-schnell.safetensors", "*.gguf"]

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[download] {MODEL_ID} -> {TARGET}")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET,
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        token=token,
    )
    print("[download] done.")
