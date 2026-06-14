"""
RunPod Serverless handler for FLUX.1-schnell.

Hai chế độ trong một worker:
  - text-to-image  (không có `image` trong input)
  - image-to-image (có `image` base64)

Model nạp LAZY ở request đầu tiên (không nạp lúc import) -> worker không bị
crash-loop; nếu nạp lỗi (thiếu token, OOM, ...) sẽ trả lỗi rõ ràng cho client.

Input (job["input"]):
  prompt, image?, strength?, num_inference_steps?, guidance_scale?,
  width?, height?, num_images?, seed?, max_sequence_length?
Output:
  { "images": ["<base64 png>", ...], "seed": <int>, "mode": "t2i"|"i2i" }
"""

import base64
import io
import os
import random
import traceback

# --- Cache dir: dùng Network Volume nếu có, không thì fallback cache mặc định ---
# (Phải set TRƯỚC khi import diffusers/huggingface_hub.)
_VOL = "/runpod-volume"
if os.path.isdir(_VOL) and os.access(_VOL, os.W_OK):
    os.environ["HF_HOME"] = f"{_VOL}/hf-cache"
    print(f"[init] HF_HOME = {os.environ['HF_HOME']} (Network Volume)")
else:
    # Không có volume: bỏ HF_HOME nếu nó đang trỏ vào /runpod-volume (tránh ghi vào path không tồn tại).
    if os.environ.get("HF_HOME", "").startswith(_VOL):
        os.environ.pop("HF_HOME", None)
    print("[init] Không thấy /runpod-volume (chưa gắn Network Volume?) -> "
          "dùng cache mặc định trong container; mỗi cold start sẽ tải lại weights.")

import torch
import runpod
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
MAX_DIM = int(os.environ.get("MAX_DIM", "1536"))

# Lazy state: nạp 1 lần, lỗi cache lại để trả về cho mọi request (khỏi tải lại liên tục).
_STATE = {"t2i": None, "i2i": None, "load_error": None}


def _load_pipes():
    """Nạp model lần đầu. Lỗi -> lưu vào _STATE['load_error'] (không raise)."""
    if _STATE["t2i"] is not None or _STATE["load_error"] is not None:
        return

    if not HF_TOKEN:
        print("[init] CẢNH BÁO: chưa có HF_TOKEN -> model gated sẽ tải lỗi (401).")

    try:
        print(f"[init] Loading {MODEL_ID} on {DEVICE} ({DTYPE}) ... (lần đầu có thể tải ~24GB)")
        t2i = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, token=HF_TOKEN)

        if DEVICE == "cuda":
            t2i.to("cuda")
            try:
                t2i.enable_vae_tiling()
            except Exception:
                pass

        # img2img dùng chung modules -> không tốn thêm VRAM, không tải lại.
        i2i = FluxImg2ImgPipeline(**t2i.components)

        _STATE["t2i"], _STATE["i2i"] = t2i, i2i
        print("[init] Pipelines ready.")
    except Exception as e:
        msg = f"Load model thất bại: {type(e).__name__}: {e}"
        if not HF_TOKEN:
            msg += " | Nhiều khả năng do THIẾU env var HF_TOKEN trên endpoint."
        elif "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            msg += " | HF_TOKEN sai, hoặc tài khoản chưa Accept license FLUX.1-schnell."
        _STATE["load_error"] = msg
        print("[init] " + msg)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(b64: str) -> Image.Image:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
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
        _load_pipes()
        if _STATE["load_error"]:
            return {"error": _STATE["load_error"]}

        txt2img_pipe = _STATE["t2i"]
        img2img_pipe = _STATE["i2i"]

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
            mode = "i2i"
            init_img = _decode_image(init_b64)
            strength = _clamp(inp.get("strength", 0.6), 0.05, 1.0, 0.6)
            result = img2img_pipe(image=init_img, strength=strength, **common)
        else:
            mode = "t2i"
            width = _round16(inp.get("width", 1024))
            height = _round16(inp.get("height", 1024))
            result = txt2img_pipe(width=width, height=height, **common)

        images = [_encode_image(im) for im in result.images]
        return {"images": images, "seed": seed, "mode": mode}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
