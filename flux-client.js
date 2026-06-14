/**
 * FLUX.1-schnell RunPod client (Node 18+, dùng global fetch).
 *
 * ENV cần có:
 *   RUNPOD_ENDPOINT_ID
 *   RUNPOD_API_KEY
 *
 * Cách dùng:
 *   const { textToImage, imageToImage } = require("./flux-client");
 *   const [b64] = await textToImage({ prompt: "a neon city at night" });
 */

const ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const API_KEY = process.env.RUNPOD_API_KEY;
const BASE = `https://api.runpod.ai/v2/${ENDPOINT_ID}`;

/** Gọi đồng bộ (/runsync). Phù hợp cho ảnh schnell vì chỉ mất vài giây. */
async function runSync(input, { timeoutMs = 120000 } = {}) {
  if (!ENDPOINT_ID || !API_KEY) {
    throw new Error("Thiếu RUNPOD_ENDPOINT_ID hoặc RUNPOD_API_KEY trong env.");
  }

  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${BASE}/runsync`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input }),
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`RunPod HTTP ${res.status}: ${await res.text()}`);
    }

    const data = await res.json();

    if (data.status === "FAILED") {
      throw new Error(`RunPod job FAILED: ${JSON.stringify(data.error ?? data)}`);
    }
    if (data.output?.error) {
      throw new Error(`Handler error: ${data.output.error}`);
    }

    return data.output; // { images: [base64...], seed, mode }
  } finally {
    clearTimeout(t);
  }
}

/** Text-to-image. Trả về mảng base64 PNG. */
async function textToImage(opts = {}) {
  const { prompt, width = 1024, height = 1024, numImages = 1, seed, steps = 4 } = opts;
  if (!prompt) throw new Error("prompt là bắt buộc.");
  const out = await runSync({
    prompt,
    width,
    height,
    num_images: numImages,
    num_inference_steps: steps,
    ...(seed !== undefined ? { seed } : {}),
  });
  return out.images;
}

/**
 * Image-to-image. `image` là base64 PNG/JPG (có data URL prefix cũng được).
 * strength: 0..1, cao = biến đổi nhiều (mặc định 0.6).
 */
async function imageToImage(opts = {}) {
  const { prompt, image, strength = 0.6, numImages = 1, seed, steps = 4 } = opts;
  if (!prompt) throw new Error("prompt là bắt buộc.");
  if (!image) throw new Error("image (base64) là bắt buộc cho img2img.");
  const out = await runSync({
    prompt,
    image,
    strength,
    num_images: numImages,
    num_inference_steps: steps,
    ...(seed !== undefined ? { seed } : {}),
  });
  return out.images;
}

module.exports = { runSync, textToImage, imageToImage };
