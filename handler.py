"""
RunPod Serverless handler for FLUX.1-schnell.

Supports two modes in a single worker:
  - text-to-image  (no `image` in input)
  - image-to-image (base64 `image` provided)

Both modes share the same model weights (loaded once), so VRAM usage
does not increase by supporting img2img.

Input schema (job["input"]):
  prompt              (str, required)   - text prompt
  image               (str, optional)   - base64-encoded init image -> enables img2img
  strength            (float, optional) - img2img only, 0..1 (default 0.6); higher = more change
  num_inference_steps (int, optional)   - default 4 (schnell is distilled, keep low)
  guidance_scale      (float, optional) - default 0.0 (schnell ignores CFG)
  width               (int, optional)   - t2i only, default 1024 (multiple of 16)
  height              (int, optional)   - t2i only, default 1024 (multiple of 16)
  num_images          (int, optional)   - default 1
  seed                (int, optional)   - reproducibility; omit for random
  max_sequence_length (int, optional)   - default 256

Output:
  { "images": [ "<base64 png>", ... ], "seed": <int>, "mode": "t2i"|"i2i" }
"""

import base64
import io
import os
import random
import traceback

import torch
import runpod
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
# HF token đọc lúc runtime (RunPod runtime env var hoặc Secret).
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
MAX_DIM = int(os.environ.get("MAX_DIM", "1536"))

# ---------------------------------------------------------------------------
# Load pipelines once at cold start. img2img reuses t2i components -> no extra VRAM.
# Weights tải từ HF (gated) bằng HF_TOKEN, cache vào HF_HOME (Network Volume).
# Lần cold start đầu tải ~24GB; các lần sau dùng lại cache trên volume.
# ---------------------------------------------------------------------------
if not HF_TOKEN:
    print("[init] CẢNH BÁO: chưa có HF_TOKEN -> sẽ fail nếu model gated. "
          "Đặt env var HF_TOKEN cho endpoint.")

print(f"[init] Loading {MODEL_ID} on {DEVICE} ({DTYPE}) ...")
txt2img_pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, token=HF_TOKEN)

if DEVICE == "cuda":
    # Keep everything on GPU for lowest latency. If you hit OOM on a small GPU,
    # swap the next line for: txt2img_pipe.enable_model_cpu_offload()
    txt2img_pipe.to("cuda")
    try:
        txt2img_pipe.enable_vae_tiling()
    except Exception:
        pass

# Share the loaded modules with the img2img pipeline (no second download / no extra VRAM).
img2img_pipe = FluxImg2ImgPipeline(**txt2img_pipe.components)
print("[init] Pipelines ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(b64: str) -> Image.Image:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]  # strip data URL prefix
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _round16(x: int) -> int:
    x = max(256, min(int(x), MAX_DIM))
    return x - (x % 16)


def _clamp(val, lo, hi, default):
    try:
        v = type(default)(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(v, hi))


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    try:
        inp = job.get("input") or {}

        prompt = inp.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"error": "Field 'prompt' (string) is required."}

        steps = _clamp(inp.get("num_inference_steps", 4), 1, MAX_STEPS, 4)
        guidance = float(inp.get("guidance_scale", 0.0))
        num_images = _clamp(inp.get("num_images", 1), 1, MAX_IMAGES, 1)
        max_seq = _clamp(inp.get("max_sequence_length", 256), 64, 512, 256)

        seed = inp.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        common = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
            max_sequence_length=max_seq,
            generator=generator,
        )

        init_b64 = inp.get("image")
        if init_b64:
            # -------- image-to-image --------
            mode = "i2i"
            init_img = _decode_image(init_b64)
            strength = _clamp(inp.get("strength", 0.6), 0.05, 1.0, 0.6)
            result = img2img_pipe(image=init_img, strength=strength, **common)
        else:
            # -------- text-to-image --------
            mode = "t2i"
            width = _round16(inp.get("width", 1024))
            height = _round16(inp.get("height", 1024))
            result = txt2img_pipe(width=width, height=height, **common)

        images = [_encode_image(im) for im in result.images]
        return {"images": images, "seed": seed, "mode": mode}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
