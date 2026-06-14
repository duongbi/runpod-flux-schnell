"""
RunPod Serverless handler for FLUX.1-schnell (text-to-image + image-to-image).

Thiết kế chống crash-câm:
  - import `runpod` TRƯỚC -> serverless loop luôn khởi động được, log được forward.
  - các import nặng (torch/diffusers) bọc trong try/except -> lỗi import không giết
    worker câm lặng mà được trả về cho client + hiện trong log.
  - model nạp LAZY ở request đầu; lỗi nạp -> trả về JSON lỗi rõ ràng.
"""

import base64
import io
import os
import random
import traceback

import runpod  # phải import trước

# --- Cache dir: dùng Network Volume nếu có, không thì cache mặc định ---
_VOL = "/runpod-volume"
if os.path.isdir(_VOL) and os.access(_VOL, os.W_OK):
    os.environ["HF_HOME"] = f"{_VOL}/hf-cache"
    print(f"[init] HF_HOME = {os.environ['HF_HOME']} (Network Volume)", flush=True)
else:
    if os.environ.get("HF_HOME", "").startswith(_VOL):
        os.environ.pop("HF_HOME", None)
    print("[init] Không có /runpod-volume -> dùng cache mặc định (cold start tải lại weights).", flush=True)

# --- Heavy imports: bọc try/except để lỗi import hiện ra thay vì chết câm ---
_IMPORT_ERROR = None
try:
    import torch
    from PIL import Image
    from diffusers import FluxPipeline, FluxImg2ImgPipeline
    print("[init] Import thư viện OK.", flush=True)
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = f"Import thư viện lỗi: {type(e).__name__}: {e}"
    print("[init] " + _IMPORT_ERROR, flush=True)
    traceback.print_exc()

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
MAX_DIM = int(os.environ.get("MAX_DIM", "1536"))

# Lazy state: nạp 1 lần; lỗi cache lại để trả cho mọi request.
_STATE = {"t2i": None, "i2i": None, "load_error": None, "device": None}


def _load_pipes():
    if _STATE["t2i"] is not None or _STATE["load_error"] is not None:
        return
    try:
        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _STATE["device"] = device

        if not HF_TOKEN:
            print("[init] CẢNH BÁO: chưa có HF_TOKEN -> model gated sẽ 401.", flush=True)

        print(f"[init] Loading {MODEL_ID} on {device} ({dtype}) ... (lần đầu tải ~24GB)", flush=True)
        t2i = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=HF_TOKEN)

        if device == "cuda":
            t2i.to("cuda")
            try:
                t2i.enable_vae_tiling()
            except Exception:
                pass

        i2i = FluxImg2ImgPipeline(**t2i.components)
        _STATE["t2i"], _STATE["i2i"] = t2i, i2i
        print("[init] Pipelines ready.", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = f"Load model thất bại: {type(e).__name__}: {e}"
        low = str(e).lower()
        if not HF_TOKEN:
            msg += " | Nhiều khả năng THIẾU env var HF_TOKEN."
        elif "401" in low or "gated" in low or "restricted" in low:
            msg += " | HF_TOKEN sai, hoặc chưa Accept license FLUX.1-schnell."
        _STATE["load_error"] = msg
        print("[init] " + msg, flush=True)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(b64: str):
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(img) -> str:
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
        if _IMPORT_ERROR:
            return {"error": _IMPORT_ERROR}

        _load_pipes()
        if _STATE["load_error"]:
            return {"error": _STATE["load_error"]}

        txt2img_pipe = _STATE["t2i"]
        img2img_pipe = _STATE["i2i"]
        device = _STATE["device"]

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
        generator = torch.Generator(device=device).manual_seed(seed)

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

    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
