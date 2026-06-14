# FLUX.1-schnell — RunPod Serverless

Một worker duy nhất, hỗ trợ cả **text-to-image** và **image-to-image**, dùng chung weights (không tốn thêm VRAM). Model `black-forest-labs/FLUX.1-schnell` — license Apache 2.0, dùng thương mại được.

## Cấu trúc

```
runpod-flux-schnell/
├── Dockerfile              # base pytorch CUDA 12.4, bake weights vào image
├── handler.py              # RunPod handler: t2i + i2i
├── requirements.txt
├── builder/
│   └── download_weights.py # tải weights lúc build (cần HF token)
├── test_input.json         # input mẫu để test local
└── .dockerignore
```

## 1. Lấy HF token (bắt buộc — model gated)

1. Vào https://huggingface.co/black-forest-labs/FLUX.1-schnell, đăng nhập, bấm **Agree** chấp nhận điều kiện.
2. Tạo **read token** tại https://huggingface.co/settings/tokens
3. Export ra biến môi trường:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

## 2. Build & push image

Weights được **bake sẵn vào image** để cold start nhanh và không phụ thuộc HF lúc chạy.

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t <dockerhub-user>/flux-schnell-runpod:latest .

docker push <dockerhub-user>/flux-schnell-runpod:latest
```

> Image khá lớn (~30GB+ vì chứa weights). Build máy có mạng tốt, hoặc build trên RunPod GPU Cloud.

## 3. Tạo Serverless Endpoint trên RunPod

1. RunPod → **Serverless** → **New Endpoint**.
2. Container Image: `<dockerhub-user>/flux-schnell-runpod:latest`
3. GPU: chọn **24GB (RTX 4090 / A5000)** — đủ cho bf16. Nếu muốn rẻ hơn, dùng GPU 16GB và đổi `txt2img_pipe.to("cuda")` thành `enable_model_cpu_offload()` trong `handler.py`.
4. Container Disk: đặt ≥ 5GB (weights đã nằm trong image nên không cần volume).
5. **Active Workers**: để 0 để scale-to-zero (rẻ nhất), hoặc 1 nếu cần tránh cold start.
6. Max Workers: tùy tải.

> Không cần truyền HF_TOKEN ở runtime vì weights đã bake sẵn.

## 4. Gọi API

### Text-to-image

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a corgi astronaut on the moon, cinematic lighting",
      "num_inference_steps": 4,
      "width": 1024,
      "height": 1024
    }
  }'
```

### Image-to-image

Chỉ cần thêm `image` (base64) và `strength`:

```jsonc
{
  "input": {
    "prompt": "turn this sketch into a photorealistic painting",
    "image": "<base64-encoded-png>",   // có data URL prefix cũng được
    "strength": 0.65,                   // 0..1, cao = biến đổi nhiều
    "num_inference_steps": 4
  }
}
```

### Response

```jsonc
{
  "id": "...",
  "status": "COMPLETED",
  "output": {
    "images": ["<base64 png>"],
    "seed": 42,
    "mode": "t2i"
  }
}
```

## 5. Gọi từ backend Node.js

```js
async function generateImage(input) {
  const res = await fetch(
    `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input }),
    }
  );
  const data = await res.json();
  if (data.output?.error) throw new Error(data.output.error);
  return data.output.images; // mảng base64 PNG
}

// text-to-image
await generateImage({ prompt: "a neon city at night" });

// image-to-image
await generateImage({ prompt: "make it snowy", image: base64Png, strength: 0.6 });
```

> Với job lâu, dùng `/run` (async) thay cho `/runsync` rồi poll `/status/<id>` — hợp hơn cho luồng chat realtime (kết hợp WebSocket/SSE để đẩy ảnh về client khi xong).

## Tham số input

| Field | Default | Ghi chú |
|---|---|---|
| `prompt` | — | bắt buộc |
| `image` | — | base64 → bật chế độ img2img |
| `strength` | 0.6 | img2img, 0..1 |
| `num_inference_steps` | 4 | schnell distilled, giữ thấp (1–4) |
| `guidance_scale` | 0.0 | schnell bỏ qua CFG |
| `width` / `height` | 1024 | t2i, bội số 16 |
| `num_images` | 1 | tối đa 4 (đổi qua env `MAX_IMAGES`) |
| `seed` | random | để tái lập kết quả |

## Ghi chú vận hành

- **Cold start**: weights đã bake trong image → load thẳng từ disk, nhanh. Để giảm thêm độ trễ lượt đầu, đặt Active Workers = 1, hoặc bật FlashBoot trên RunPod.
- **Giảm VRAM**: muốn chạy GPU 16GB → đổi `.to("cuda")` thành `enable_model_cpu_offload()` (chậm hơn chút) hoặc dùng bản FP8 quantized.
- **An toàn nội dung**: thêm bước lọc prompt ở backend trước khi gọi, vì handler không tự kiểm duyệt.
```
