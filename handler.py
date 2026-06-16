"""
RunPod Serverless handler cho FLUX.1-Kontext-dev (image EDITING + text-to-image).

Mục tiêu: từ 1 ảnh gốc, GIỮ nhân vật/chủ thể rồi đặt vào bối cảnh khác theo câu
lệnh text (instruction-based editing) — thay cho img2img-by-strength cũ vốn làm
biến dạng nhân vật.

Thiết kế chống crash-câm (giữ nguyên triết lý bản cũ):
  - import `runpod` TRƯỚC -> serverless loop luôn khởi động, log được forward.
  - import nặng (torch/diffusers) bọc trong try/except -> lỗi import được trả về
    client + hiện trong log thay vì chết câm.
  - model nạp LAZY ở request đầu; lỗi nạp -> trả JSON lỗi rõ ràng.

Lưu ý về tham số (KHÁC schnell):
  - Kontext-dev KHÔNG phải model turbo. Cần nhiều bước hơn (~28) và guidance ~2.5.
  - KHÔNG dùng `strength` nữa. Việc giữ/đổi do prompt + guidance quyết định.
  - Handler tự nâng steps/guidance về mức hợp lý nếu caller cũ gửi steps=4,
    guidance=0 -> backend KHÔNG cần sửa ngay vẫn ra ảnh đẹp.
"""

import base64
import io
import os
import random
import traceback

import runpod  # phải import trước

# Giảm phân mảnh VRAM. Phải set TRƯỚC khi import torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    from diffusers import FluxKontextPipeline
    print("[init] Import thư viện OK.", flush=True)
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = f"Import thư viện lỗi: {type(e).__name__}: {e}"
    print("[init] " + _IMPORT_ERROR, flush=True)
    traceback.print_exc()

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "40"))
MAX_DIM = int(os.environ.get("MAX_DIM", "1536"))

# Defaults hợp lý cho Kontext-dev (model guidance-distilled, KHÔNG phải turbo).
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "28"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "2.5"))
# Nếu caller cũ gửi steps quá thấp (vd 4 của schnell), nâng lên sàn này để khỏi ra ảnh lỗi.
MIN_STEPS = int(os.environ.get("MIN_STEPS", "20"))

# Lazy state: nạp 1 lần; lỗi cache lại để trả cho mọi request.
_STATE = {"pipe": None, "load_error": None, "device": None}


def _load_pipes():
    if _STATE["pipe"] is not None or _STATE["load_error"] is not None:
        return
    try:
        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _STATE["device"] = device

        if not HF_TOKEN:
            print("[init] CẢNH BÁO: chưa có HF_TOKEN -> model gated sẽ 401.", flush=True)

        print(f"[init] Loading {MODEL_ID} on {device} ({dtype}) ... (lần đầu tải ~24GB)", flush=True)
        pipe = FluxKontextPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=HF_TOKEN)

        if device == "cuda":
            # Kontext bf16 đầy đủ > 24GB GPU -> model cpu offload: đẩy từng module lên
            # GPU khi cần, đủ chạy trên 24GB (kể cả 16GB). KHÔNG gọi .to("cuda").
            pipe.enable_model_cpu_offload()
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass

        _STATE["pipe"] = pipe
        print("[init] Pipeline ready.", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = f"Load model thất bại: {type(e).__name__}: {e}"
        low = str(e).lower()
        if not HF_TOKEN:
            msg += " | Nhiều khả năng THIẾU env var HF_TOKEN."
        elif "401" in low or "gated" in low or "restricted" in low:
            msg += " | HF_TOKEN sai, hoặc chưa Accept license FLUX.1-Kontext-dev."
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

        pipe = _STATE["pipe"]

        inp = job.get("input") or {}

        prompt = inp.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"error": "Field 'prompt' (string) is required."}

        # Steps: nâng về sàn nếu caller cũ gửi quá thấp (schnell dùng 4).
        steps = _clamp(inp.get("num_inference_steps", DEFAULT_STEPS), 1, MAX_STEPS, DEFAULT_STEPS)
        if steps < MIN_STEPS:
            steps = DEFAULT_STEPS

        # Guidance: Kontext cần ~2.5. Caller cũ gửi 0.0 -> coi như chưa set, dùng default.
        try:
            guidance = float(inp.get("guidance_scale", DEFAULT_GUIDANCE))
        except (TypeError, ValueError):
            guidance = DEFAULT_GUIDANCE
        if guidance <= 0:
            guidance = DEFAULT_GUIDANCE

        num_images = _clamp(inp.get("num_images", 1), 1, MAX_IMAGES, 1)
        max_seq = _clamp(inp.get("max_sequence_length", 512), 64, 512, 512)

        seed = inp.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        # Với model cpu offload, generator để trên CPU là an toàn (theo mẫu FLUX).
        generator = torch.Generator(device="cpu").manual_seed(seed)

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
            # --- EDIT: giữ nhân vật trong ảnh gốc, áp prompt làm chỉ dẫn đổi cảnh ---
            mode = "edit"
            init_img = _decode_image(init_b64)
            # Kontext tự suy ra kích thước theo ảnh vào; có thể ép qua width/height nếu gửi.
            extra = {}
            if inp.get("width"):
                extra["width"] = _round16(inp.get("width"))
            if inp.get("height"):
                extra["height"] = _round16(inp.get("height"))
            result = pipe(image=init_img, **common, **extra)
        else:
            # --- T2I: sinh ảnh từ text (Kontext vẫn generate được, không bắt buộc ảnh vào) ---
            mode = "t2i"
            width = _round16(inp.get("width", 1024))
            height = _round16(inp.get("height", 1024))
            result = pipe(width=width, height=height, **common)

        images = [_encode_image(im) for im in result.images]
        return {"images": images, "seed": seed, "mode": mode}

    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
