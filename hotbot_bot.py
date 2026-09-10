#!/usr/bin/env python3
"""
hotbot_bot.py — Telegram AI bot using OFFICIAL / documented provider APIs.

Providers used right now:
  💬 Chat               -> Groq (console.groq.com), OpenAI-compatible /chat/completions
  🎙️ Voice-in (STT)      -> Groq Whisper, OpenAI-compatible /audio/transcriptions
  🔊 Voice-out (TTS)     -> Microsoft Edge TTS (edge-tts library), free, no key needed
  👁️ Vision / 📄 PDF      -> Gemini (Google), generateContent (multimodal)
  🖼️ Image generate/edit -> Cloudflare Workers AI (free tier, two different
                            models — one tuned for generation, one for img2img)
  🆓 Image backup #1     -> Ashlynn community worker (generation only)
  🆓 Image backup #2     -> "still-queen" community worker (generation only)
  🎬 Video generate      -> "Video Studio" community worker (best-effort —
                            see video_generate() docstring)
  🔍 Web search          -> TeCoxBeta (txb.latdlabs.in), documented REST API

Everything provider-specific lives in the CONFIG block below. Secrets come
ONLY from environment variables / your host's "Secrets" panel — nothing is
hardcoded in this file.

Required environment variables:
  TELEGRAM_BOT_TOKEN     - from @BotFather
  GROQ_API_KEY           - console.groq.com (chat + voice transcription)
  GEMINI_API_KEY         - aistudio.google.com (vision + PDF)
  CLOUDFLARE_ACCOUNT_ID  - dash.cloudflare.com (image generate + edit)
  CLOUDFLARE_API_TOKEN   - dash.cloudflare.com

Run:
    pip install -r requirements.txt
    python hotbot_bot.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from PIL import Image

import edge_tts
import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================================
# CONFIG — the ONLY place you should need to edit when swapping providers
# ============================================================================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]  # required, no default

# --- Chat provider (Groq — console.groq.com, OpenAI-compatible chat API) --
CHAT_PROVIDER = {
    "name": "groq",
    "api_key": os.environ.get("GROQ_API_KEY", ""),
    "base_url": os.environ.get("CHAT_BASE_URL", "https://api.groq.com/openai/v1"),
    "model": os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b"),
}
# Voice-in (speech-to-text) reuses the same Groq account/key, different endpoint.
WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# --- Voice-out (text-to-speech) — Microsoft Edge TTS, free, no API key -----
# Change TTS_VOICE to any edge-tts voice name, e.g. bn-BD-PradeepNeural
# (male) instead of the default bn-BD-NabanitaNeural (female). Full list:
# run `edge-tts --list-voices` locally, or see the edge-tts PyPI page.
TTS_VOICE = os.environ.get("TTS_VOICE", "bn-BD-NabanitaNeural")

# --- Vision + PDF provider (Gemini / Google) -------------------------------
GEMINI_PROVIDER = {
    "name": "gemini",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "base_url": os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    ),
    "vision_model": os.environ.get("GEMINI_VISION_MODEL", "gemini-3.6-flash"),
}
# NOTE: Gemini is used for VISION (asking questions about a photo) and PDF
# summarization only. gemini-2.5-flash-image effectively has no usable
# free-tier quota (Google requires billing enabled for it), so image
# generation/editing lives on Cloudflare Workers AI below instead.

# --- Image generate + edit provider (Cloudflare Workers AI, free tier) ----
# No Worker needs to be deployed — we call Cloudflare's REST API directly.
# You need two values from dash.cloudflare.com:
#   1) Account ID -> right sidebar of any Workers & Pages page
#   2) API Token  -> My Profile -> API Tokens -> Create Token -> "Workers AI"
#
# Two DIFFERENT models are used on purpose:
#   - generate_model: stable-diffusion-xl-base-1.0, best quality for fresh
#     text-to-image generation.
#   - edit_model: runwayml/stable-diffusion-v1-5-img2img, a model built
#     specifically for img2img. The XL model's img2img path is flaky
#     ("Beta", frequent 500 InferenceResponse errors); the dedicated
#     img2img model is far more reliable for editing an existing photo.
CLOUDFLARE_PROVIDER = {
    "account_id": os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
    "api_token": os.environ.get("CLOUDFLARE_API_TOKEN", ""),
    "generate_model": os.environ.get(
        "CLOUDFLARE_IMAGE_MODEL", "@cf/stabilityai/stable-diffusion-xl-base-1.0"
    ),
    "edit_model": os.environ.get(
        "CLOUDFLARE_EDIT_MODEL", "@cf/runwayml/stable-diffusion-v1-5-img2img"
    ),
}

# --- Backup / community text-to-image providers (generation only) ---------
# Independent third-party Cloudflare-Workers-based projects. No account, no
# key. Used as fallbacks / secondary "free" generators, never for editing.
# SAFETY IS ALWAYS FORCED ON for the Ashlynn backend (see backup_image_generate)
# and is never exposed as a togglable option anywhere in this bot.
BACKUP_IMAGE_PROVIDER = {
    "name": "ashlynn-community",
    "enabled": os.environ.get("BACKUP_IMAGE_ENABLED", "true").lower() == "true",
    "base_url": os.environ.get(
        "BACKUP_IMAGE_BASE_URL", "https://death-image.ashlynn.workers.dev"
    ),
    "steps": int(os.environ.get("BACKUP_IMAGE_STEPS", "8")),
}
STILLQUEEN_PROVIDER = {
    "name": "still-queen-community",
    "enabled": os.environ.get("STILLQUEEN_ENABLED", "true").lower() == "true",
    "base_url": os.environ.get(
        "STILLQUEEN_BASE_URL", "https://still-queen-2b05.glitchy.workers.dev"
    ),
}

# --- Video generation (community worker, generation only, best-effort) ----
# The exact request/response field names are inferred from this worker's
# public UI, not a published API spec. If the provider changes its shape,
# video_generate() below is the only place that needs adjusting — errors
# are surfaced with the raw response text to make that easy to diagnose.
VIDEO_PROVIDER = {
    "enabled": os.environ.get("VIDEO_ENABLED", "true").lower() == "true",
    "base_url": os.environ.get("VIDEO_BASE_URL", "https://amphvid.glitchy.workers.dev"),
}

# --- Web search (TeCoxBeta — documented REST API) --------------------------
SEARCH_PROVIDER = {
    "enabled": os.environ.get("SEARCH_ENABLED", "true").lower() == "true",
    "base_url": os.environ.get("SEARCH_BASE_URL", "https://txb.latdlabs.in"),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "bot_memory.json")
MEMORY_MAX_MESSAGES = int(os.environ.get("MEMORY_MAX_MESSAGES", "60"))
MEMORY_MAX_CHARS = int(os.environ.get("MEMORY_MAX_CHARS", "60000"))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("bot")

# ============================================================================
# Memory (per-user, persisted to disk)
# ============================================================================

MEMORY: dict[int, dict[str, Any]] = {}


def _load_memory() -> None:
    global MEMORY
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            MEMORY = {int(k): v for k, v in json.load(f).items()}
    except Exception:
        MEMORY = {}


def _save_memory() -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(MEMORY, f)
    except Exception as e:
        log.warning("could not persist memory: %s", e)


_load_memory()


def _hist(user_id: int) -> dict[str, Any]:
    return MEMORY.setdefault(user_id, {"messages": [], "on": True, "voice": False})


def remember(user_id: int, role: str, text: str) -> None:
    h = _hist(user_id)
    if not h.get("on", True):
        return
    h["messages"].append({"role": role, "content": text})
    _trim(user_id)
    _save_memory()


def _trim(user_id: int) -> None:
    h = _hist(user_id)
    msgs = h["messages"]
    while len(msgs) > MEMORY_MAX_MESSAGES:
        msgs.pop(0)
    total = sum(len(m["content"]) for m in msgs)
    while total > MEMORY_MAX_CHARS and len(msgs) > 2:
        total -= len(msgs.pop(0)["content"])


def history_messages(user_id: int) -> list[dict[str, str]]:
    return list(_hist(user_id).get("messages", []))


def memory_stats(user_id: int) -> str:
    h = _hist(user_id)
    msgs = h.get("messages", [])
    total = sum(len(m["content"]) for m in msgs)
    return (
        f"🧠 *Memory*\non: {h.get('on', True)}\n"
        f"voice replies for text chat: {h.get('voice', False)}\n"
        f"messages: {len(msgs)}\n"
        f"chars kept: {total:,} / limit {MEMORY_MAX_CHARS:,}"
    )


def clear_memory(user_id: int) -> None:
    old = _hist(user_id)
    MEMORY[user_id] = {"messages": [], "on": True, "voice": old.get("voice", False)}
    _save_memory()


# ============================================================================
# Small helpers
# ============================================================================


class ProviderError(Exception):
    """Raised on any provider-side failure (bad key, rate limit, etc.)."""


def chunk_text(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text] if text else []
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 <= limit:
            cur = cur + "\n\n" + para if cur else para
        else:
            if cur:
                parts.append(cur)
            cur = para
            while len(cur) > limit:
                parts.append(cur[:limit])
                cur = cur[limit:]
    if cur:
        parts.append(cur)
    return parts


async def telegram_photo_to_b64(update: Update) -> tuple[str, str]:
    """Download the largest photo the user sent. Returns (base64_data, mime_type)."""
    photo = update.message.photo[-1]
    f = await photo.get_file()
    blob = await f.download_as_bytearray()
    return base64.b64encode(bytes(blob)).decode(), "image/jpeg"


# ============================================================================
# Chat provider adapter (Groq — OpenAI-compatible)
# ============================================================================


async def _call_chat(messages: list[dict[str, str]]) -> str:
    cfg = CHAT_PROVIDER
    if not cfg["api_key"]:
        raise ProviderError(
            "Chat provider API key is missing. Set the GROQ_API_KEY secret."
        )
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    body = {"model": cfg["model"], "messages": messages}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=body)
    if resp.status_code >= 400:
        raise ProviderError(f"Chat provider error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"Unexpected chat response shape: {json.dumps(data)[:300]}")


# ============================================================================
# Voice-in (STT) via Groq Whisper — OpenAI-compatible /audio/transcriptions
# ============================================================================


async def groq_transcribe(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    cfg = CHAT_PROVIDER  # same Groq account/key as chat
    if not cfg["api_key"]:
        raise ProviderError(
            "Voice transcription needs the GROQ_API_KEY secret to be set."
        )
    url = f"{cfg['base_url'].rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    files = {"file": (filename, audio_bytes, "audio/ogg")}
    data = {"model": WHISPER_MODEL}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, files=files, data=data)
    if resp.status_code >= 400:
        raise ProviderError(f"Voice transcription error {resp.status_code}: {resp.text[:300]}")
    out = resp.json()
    text = out.get("text")
    if not text:
        raise ProviderError(f"Unexpected transcription response: {json.dumps(out)[:300]}")
    return text.strip()


# ============================================================================
# Voice-out (TTS) via Microsoft Edge TTS (edge-tts library — free, no key)
# ============================================================================


async def tts_speak(text: str) -> bytes:
    """Returns MP3 bytes. Sent to Telegram via reply_audio (not reply_voice):
    a native Telegram "voice bubble" requires OGG/Opus, which would need an
    ffmpeg conversion step this bot intentionally avoids to stay dependency
    -light and reliable on free hosting. reply_audio plays fine either way,
    just with a normal audio-player bubble instead of a round voice note."""
    text = text.strip()
    if not text:
        raise ProviderError("Nothing to speak.")
    communicate = edge_tts.Communicate(text[:2000], TTS_VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            buf.write(chunk["data"])
    data = buf.getvalue()
    if not data:
        raise ProviderError("TTS produced no audio (check TTS_VOICE is a valid edge-tts voice name).")
    return data


# ============================================================================
# Vision + PDF provider adapter (Gemini)
# ============================================================================


async def _gemini_generate(
    model: str,
    text_prompt: str,
    image_b64: Optional[str] = None,
    image_mime: str = "image/jpeg",
) -> dict[str, Any]:
    cfg = GEMINI_PROVIDER
    if not cfg["api_key"]:
        raise ProviderError(
            "Vision/PDF provider API key is missing. Set the GEMINI_API_KEY secret."
        )
    url = f"{cfg['base_url'].rstrip('/')}/models/{model}:generateContent"
    parts: list[dict[str, Any]] = [{"text": text_prompt}]
    if image_b64:
        parts.append(
            {"inline_data": {"mime_type": image_mime, "data": image_b64}}
        )
    body = {"contents": [{"role": "user", "parts": parts}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": cfg["api_key"]}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, headers=headers, json=body)
    if resp.status_code >= 400:
        raise ProviderError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _gemini_text_from_response(data: dict[str, Any]) -> str:
    try:
        candidate = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in candidate)
        if not text:
            raise KeyError
        return text
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"Unexpected Gemini response: {json.dumps(data)[:300]}")


async def gemini_vision_answer(text_prompt: str, image_b64: str, image_mime: str) -> str:
    """Ask a vision-capable Gemini model a question about an image."""
    data = await _gemini_generate(
        GEMINI_PROVIDER["vision_model"], text_prompt, image_b64, image_mime
    )
    return _gemini_text_from_response(data)


async def gemini_pdf_summary(pdf_b64: str) -> str:
    """Summarize a PDF document using the same Gemini vision-capable model
    (Gemini accepts PDFs as inline_data, same as images)."""
    data = await _gemini_generate(
        GEMINI_PROVIDER["vision_model"],
        "এই PDF ডকুমেন্টটা বাংলায় সংক্ষেপে সারাংশ করে দাও। মূল পয়েন্টগুলো "
        "বুলেট আকারে লেখো, তারপর ১-২ লাইনে overall গিস্ট বলো।",
        image_b64=pdf_b64,
        image_mime="application/pdf",
    )
    return _gemini_text_from_response(data)


def _downscale_image(image_bytes: bytes, max_side: int = 896) -> bytes:
    """Shrink an image before sending it to Cloudflare's SD models.
    Large payloads are a common trigger for 5xx errors on the Beta models."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88)
        return out.getvalue()
    except Exception:
        return image_bytes  # fall back to the original if Pillow can't read it


# ============================================================================
# Image generate + edit (Cloudflare Workers AI)
# ============================================================================


async def cf_image_run(
    prompt: str, image_bytes: Optional[bytes] = None, strength: float = 0.6
) -> bytes:
    """Generate a new image, or edit `image_bytes` if provided, via Cloudflare
    Workers AI. Uses a DIFFERENT model for editing than for generation (see
    CLOUDFLARE_PROVIDER comment above) — this is the fix for edits failing
    while plain generation worked fine. Returns raw PNG/JPEG bytes."""
    cfg = CLOUDFLARE_PROVIDER
    if not cfg["account_id"] or not cfg["api_token"]:
        raise ProviderError(
            "Cloudflare image provider isn't configured. Set CLOUDFLARE_ACCOUNT_ID "
            "and CLOUDFLARE_API_TOKEN secrets."
        )
    model = cfg["edit_model"] if image_bytes else cfg["generate_model"]
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{cfg['account_id']}"
        f"/ai/run/{model}"
    )
    headers = {
        "Authorization": f"Bearer {cfg['api_token']}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {"prompt": prompt, "num_steps": 20}
    if image_bytes:
        body["image_b64"] = base64.b64encode(_downscale_image(image_bytes)).decode()
        body["strength"] = strength

    last_error = None
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=body)
        if resp.status_code < 400:
            content_type = resp.headers.get("content-type", "")
            if content_type.startswith("image/"):
                return resp.content
            try:
                data = resp.json()
                b64 = data.get("result", {}).get("image") or data.get("image")
                if b64:
                    return base64.b64decode(b64)
            except (ValueError, AttributeError):
                pass
            last_error = ProviderError(
                f"Unexpected Cloudflare response for {model}: {resp.text[:300]}"
            )
        elif resp.status_code >= 500:
            last_error = ProviderError(
                f"Cloudflare error {resp.status_code} on {model}: {resp.text[:300]}"
            )
            await asyncio.sleep(2)
            continue  # transient backend error — retry
        else:
            raise ProviderError(f"Cloudflare error {resp.status_code} on {model}: {resp.text[:300]}")
    raise last_error or ProviderError("Cloudflare image provider failed after retries.")


# ============================================================================
# Backup / community text-to-image generators (generation only)
# ============================================================================


async def backup_image_generate(prompt: str, dimensions: str = "1:1") -> str:
    """Ashlynn community backup. Returns a hosted image URL. SAFETY IS ALWAYS
    FORCED TRUE here — this is intentional and must never be made configurable."""
    cfg = BACKUP_IMAGE_PROVIDER
    if not cfg["enabled"]:
        raise ProviderError("Backup image provider #1 is disabled.")
    url = f"{cfg['base_url'].rstrip('/')}/generate"
    params = {
        "prompt": prompt,
        "image": 1,
        "dimensions": dimensions,
        "safety": "true",  # hardcoded — never read from config or user input
        "steps": cfg["steps"],
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.get(url, params=params)
    if resp.status_code >= 400:
        raise ProviderError(f"Backup image provider #1 error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    images = data.get("images") or []
    if not images:
        raise ProviderError(f"Backup provider #1 returned no images: {json.dumps(data)[:200]}")
    return images[0]


async def stillqueen_generate(prompt: str, ratio: str = "1:1") -> bytes:
    """"still-queen" community backup #2 (generation only)."""
    cfg = STILLQUEEN_PROVIDER
    if not cfg["enabled"]:
        raise ProviderError("Backup image provider #2 is disabled.")
    url = f"{cfg['base_url'].rstrip('/')}/api/generate-image"
    body = {"prompt": prompt, "ratio": ratio}
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(url, json=body)
    if resp.status_code >= 400:
        raise ProviderError(f"Backup image provider #2 error {resp.status_code}: {resp.text[:200]}")
    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("image/"):
        return resp.content
    try:
        data = resp.json()
    except ValueError:
        raise ProviderError(f"Unexpected backup provider #2 response: {resp.text[:200]}")
    b64 = data.get("image") or data.get("image_b64")
    if b64:
        return base64.b64decode(b64)
    image_url = data.get("url") or data.get("image_url") or ((data.get("images") or [None])[0])
    if image_url:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r2 = await client.get(image_url)
        if r2.status_code < 400:
            return r2.content
    raise ProviderError(f"Unexpected backup provider #2 response shape: {json.dumps(data)[:200]}")


# ============================================================================
# Video generation (community worker, best-effort — see VIDEO_PROVIDER note)
# ============================================================================


async def video_generate(
    prompt: str, resolution: str = "480p", aspect_ratio: str = "16:9", duration: int = 4
) -> bytes:
    cfg = VIDEO_PROVIDER
    if not cfg["enabled"]:
        raise ProviderError("Video generation is disabled.")
    base = cfg["base_url"].rstrip("/")
    body = {
        "prompt": prompt,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{base}/api/generate", json=body)
    if resp.status_code >= 400:
        raise ProviderError(f"Video provider error {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError:
        raise ProviderError(f"Unexpected video submit response: {resp.text[:300]}")
    job_id = (
        data.get("id") or data.get("jobId") or data.get("job_id") or data.get("requestId")
    )
    if not job_id:
        raise ProviderError(f"No job id in video submit response: {json.dumps(data)[:300]}")

    status_url = f"{base}/api/status/{job_id}"
    for _ in range(40):  # ~2 minutes of polling
        await asyncio.sleep(3)
        async with httpx.AsyncClient(timeout=30.0) as client:
            s = await client.get(status_url)
        if s.status_code >= 400:
            continue
        try:
            sdata = s.json()
        except ValueError:
            continue
        state = str(sdata.get("status") or sdata.get("state") or "").lower()
        if state in ("completed", "done", "ready", "succeeded", "success"):
            video_url = (
                sdata.get("url") or sdata.get("videoUrl") or sdata.get("video_url")
                or sdata.get("output") or sdata.get("result")
            )
            dl_url = video_url or f"{base}/api/download/{job_id}"
            async with httpx.AsyncClient(timeout=120.0) as client:
                v = await client.get(dl_url)
            if v.status_code < 400:
                return v.content
            raise ProviderError(f"Could not download finished video ({v.status_code}).")
        if state in ("failed", "error", "stopped"):
            raise ProviderError(f"Video render failed: {json.dumps(sdata)[:300]}")
    raise ProviderError("Video render timed out (still processing on the provider's side — try again in a bit).")


# ============================================================================
# Web search (TeCoxBeta — documented REST API)
# ============================================================================


async def web_search(query: str, depth: str = "standard") -> str:
    cfg = SEARCH_PROVIDER
    if not cfg["enabled"]:
        raise ProviderError("Web search is disabled.")
    url = f"{cfg['base_url'].rstrip('/')}/search"
    body = {
        "query": query,
        "depth": depth,
        "safe": True,
        "max_output_tokens": 1200,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(url, json=body)
    if resp.status_code >= 400:
        raise ProviderError(f"Search provider error {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError:
        raise ProviderError(f"Unexpected search response: {resp.text[:300]}")
    answer = data.get("answer") or data.get("result") or data.get("text") or data.get("summary")
    sources = data.get("sources") or data.get("citations") or []
    if not answer:
        raise ProviderError(f"Unexpected search response shape: {json.dumps(data)[:300]}")
    out = answer.strip()
    if sources:
        lines = []
        for s in sources[:5]:
            if isinstance(s, dict):
                title = s.get("title") or s.get("name") or "source"
                link = s.get("url") or s.get("link") or ""
                lines.append(f"• {title} — {link}" if link else f"• {title}")
            else:
                lines.append(f"• {s}")
        out += "\n\n🔗 Sources:\n" + "\n".join(lines)
    return out


# ============================================================================
# Preset prompt gallery — pick a ready-made edit style instead of typing one
# ============================================================================

PRESET_PROMPTS: list[dict[str, str]] = [
    {
        "id": "product_bg",
        "label": "📦 প্রোডাক্ট শোকেস",
        "prompt": "Place the main product on a clean white studio background with a soft realistic shadow, professional product photography lighting, high detail.",
    },
    {
        "id": "cinematic",
        "label": "🎬 সিনেমাটিক পোর্ট্রেট",
        "prompt": "Transform this photo into a cinematic portrait with dramatic lighting, shallow depth of field, and a moody film color grade.",
    },
    {
        "id": "anime",
        "label": "🌸 অ্যানিমে স্টাইল",
        "prompt": "Convert this photo into a vibrant Japanese anime illustration: clean line art, cel-shaded coloring, expressive style.",
    },
    {
        "id": "ghibli",
        "label": "🎨 ঘিবলি স্টাইল",
        "prompt": "Reimagine this image in a hand-painted Studio-Ghibli-inspired animation style: soft pastel colors, whimsical, painterly backgrounds.",
    },
    {
        "id": "3d_render",
        "label": "🧊 থ্রিডি রেন্ডার",
        "prompt": "Turn this into a polished 3D rendered illustration with Pixar-style character design and soft global illumination.",
    },
    {
        "id": "remove_bg",
        "label": "✂️ ব্যাকগ্রাউন্ড রিমুভ",
        "prompt": "Remove the background completely, keeping only the main subject sharply cut out on a plain transparent/white background.",
    },
    {
        "id": "vintage_poster",
        "label": "🖼️ ভিন্টেজ পোস্টার",
        "prompt": "Redesign this as a vintage travel poster illustration: bold flat colors, retro typography feel, 1950s aesthetic.",
    },
    {
        "id": "cyberpunk",
        "label": "🌃 নিয়ন সাইবারপাঙ্ক",
        "prompt": "Restyle this image with a cyberpunk aesthetic: neon lights, futuristic city glow, high-contrast saturated colors.",
    },
    {
        "id": "logo_mockup",
        "label": "🏷️ লোগো মকআপ",
        "prompt": "Place this logo/design onto a realistic professional mockup such as a business card, storefront sign, or product packaging.",
    },
    {
        "id": "golden_hour",
        "label": "🌅 গোল্ডেন আওয়ার লাইটিং",
        "prompt": "Relight this photo as if taken during golden hour sunset: warm tones, soft glowing light, long soft shadows.",
    },
    {
        "id": "watercolor",
        "label": "🎨 ওয়াটারকালার পেইন্টিং",
        "prompt": "Convert this photo into a delicate watercolor painting: soft bleeding edges, visible paper texture, artistic color washes.",
    },
    {
        "id": "wallpaper_hd",
        "label": "🖥️ ওয়ালপেপার এইচডি",
        "prompt": "Enhance and reimagine this as a high-detail desktop wallpaper: ultra sharp, vivid colors, epic wide composition.",
    },
]

PRESET_BY_ID = {p["id"]: p for p in PRESET_PROMPTS}


def presets_kb() -> InlineKeyboardMarkup:
    rows = []
    for i in range(0, len(PRESET_PROMPTS), 2):
        pair = PRESET_PROMPTS[i : i + 2]
        rows.append(
            [InlineKeyboardButton(p["label"], callback_data=f"preset:{p['id']}") for p in pair]
        )
    rows.append([InlineKeyboardButton("🔙 Menu", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


# ============================================================================
# UI
# ============================================================================


def main_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("💬 Chat", callback_data="chat"),
                InlineKeyboardButton("🖼️ Image", callback_data="image"),
            ],
            [
                InlineKeyboardButton("🎨 Presets", callback_data="presets"),
                InlineKeyboardButton("🧠 Memory", callback_data="memory"),
            ],
            [
                InlineKeyboardButton("ℹ️ Help", callback_data="help"),
            ],
        ]
    )


def done_kb(kind: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"🔄 Again ({kind})", callback_data=kind)],
            [InlineKeyboardButton("🔙 Menu", callback_data="menu")],
        ]
    )


HELP_TEXT = (
    "*ℹ️ সব কমান্ড*\n\n"
    "*💬 চ্যাট*\n"
    "• `/ask <প্রশ্ন>` — সরাসরি প্রশ্ন করো\n"
    "• স্বাভাবিক টেক্সট পাঠালেও চ্যাট হবে\n"
    "• একটা ভয়েস মেসেজ পাঠালে — আমি শুনে বুঝে টেক্সট + ভয়েস দুটোতেই রিপ্লাই দেব\n\n"
    "*🖼️ ছবি বানানো*\n"
    "• `/img <prompt>` — Cloudflare (প্রধান)\n"
    "• `/img2 <prompt>` — ফ্রি backup #1\n"
    "• `/img3 <prompt>` — ফ্রি backup #2\n\n"
    "*✏️ ছবি এডিট করা*\n"
    "• একটা ছবি পাঠাও, তারপর কীভাবে এডিট করতে চাও লিখে দাও\n"
    "• অথবা 🎨 *Presets* থেকে একটা রেডিমেড স্টাইল বেছে নিয়ে ছবি পাঠাও\n\n"
    "*🎬 ভিডিও*\n"
    "• `/video <prompt>` — টেক্সট থেকে ছোট ভিডিও বানায় (একটু সময় লাগে)\n\n"
    "*🔍 ওয়েব সার্চ*\n"
    "• `/search <প্রশ্ন>` — লাইভ ওয়েব সার্চ করে উত্তর দেয়\n\n"
    "*📄 পিডিএফ*\n"
    "• একটা PDF পাঠালে বাংলায় সারাংশ করে দেব\n\n"
    "*⚙️ অন্যান্য*\n"
    "• `/voice on` / `/voice off` — টেক্সট চ্যাটেও ভয়েস রিপ্লাই চালু/বন্ধ\n"
    "• `/memory` — মেমরি status, `/memory on` / `/memory off`\n"
    "• `/clear` — সেভ করা history মুছে ফেলো\n"
    "• `/cancel` — বট এখন যা চাইছে সেটা বাতিল করো"
)


# ============================================================================
# Commands
# ============================================================================


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*🧿 AI Bot*\n\n"
        "💬 *Chat* — টেক্সট বা ভয়েস দিয়ে কথা বলো (Groq)\n"
        "👁️ *Vision* — ছবি পাঠিয়ে প্রশ্ন করো (Gemini)\n"
        "🖼️ *Image* — নতুন ছবি বানাও বা পাঠানো ছবি এডিট করো (Cloudflare)\n"
        "🎨 *Presets* — রেডিমেড এডিট স্টাইল বেছে নাও\n"
        "🎬 *Video* — `/video <prompt>` দিয়ে ছোট ভিডিও বানাও\n"
        "🔍 *Search* — `/search <প্রশ্ন>` দিয়ে লাইভ ওয়েব সার্চ\n"
        "📄 *PDF* — পিডিএফ পাঠালে সারাংশ করে দেব\n"
        "🧠 *Memory* — chat history মনে রাখে, /clear দিয়ে মুছে ফেলা যায়\n\n"
        "সব কমান্ডের তালিকার জন্য `/help` লেখো।",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_kb(),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb())


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/ask your question`", parse_mode=ParseMode.MARKDOWN)
        return
    await run_chat(update, ctx, parts[1].strip())


async def cmd_img(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/img your prompt`", parse_mode=ParseMode.MARKDOWN)
        return
    await run_image(update, ctx, parts[1].strip(), image_b64=None)


async def cmd_img2(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Generation-only, via the free community backup #1 (safety forced on)."""
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/img2 your prompt`", parse_mode=ParseMode.MARKDOWN)
        return
    prompt = parts[1].strip()
    status = await update.message.reply_text("🆓 Generating (backup #1)…")
    try:
        image_url = await backup_image_generate(prompt)
        await status.delete()
        await update.message.reply_photo(photo=image_url, caption=f"🆓 {prompt[:900]}", reply_markup=main_kb())
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("backup image #1 failed")
        await status.edit_text(f"❌ Error: {e}")


async def cmd_img3(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Generation-only, via the free community backup #2."""
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/img3 your prompt`", parse_mode=ParseMode.MARKDOWN)
        return
    prompt = parts[1].strip()
    status = await update.message.reply_text("🆓 Generating (backup #2)…")
    try:
        image_bytes = await stillqueen_generate(prompt)
        await status.delete()
        await update.message.reply_photo(photo=image_bytes, caption=f"🆓 {prompt[:900]}", reply_markup=main_kb())
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("backup image #2 failed")
        await status.edit_text(f"❌ Error: {e}")


async def cmd_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/video your prompt`", parse_mode=ParseMode.MARKDOWN)
        return
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    prompt = parts[1].strip()
    status = await update.message.reply_text("🎬 Rendering video… এটাতে ১-২ মিনিট লাগতে পারে।")
    try:
        video_bytes = await video_generate(prompt)
        await status.delete()
        await update.message.reply_video(video=video_bytes, caption=f"🎬 {prompt[:900]}", reply_markup=main_kb())
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("video failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/search your question`", parse_mode=ParseMode.MARKDOWN)
        return
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    query = parts[1].strip()
    status = await update.message.reply_text("🔍 Searching the web…")
    try:
        answer = await web_search(query)
        await status.delete()
        for chunk in chunk_text(answer):
            await update.message.reply_text(chunk, disable_web_page_preview=True)
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("search failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False


async def cmd_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    parts = (update.message.text or "").split(maxsplit=1)
    h = _hist(user.id)
    if len(parts) == 2 and parts[1].strip().lower() in ("on", "off"):
        h["voice"] = parts[1].strip().lower() == "on"
        _save_memory()
        await update.message.reply_text(
            f"🔊 টেক্সট চ্যাটে ভয়েস রিপ্লাই {'চালু ✅' if h['voice'] else 'বন্ধ ⏸️'}।\n"
            f"(তুমি ভয়েস মেসেজ পাঠালে সবসময়ই ভয়েস রিপ্লাই পাবে, এই সেটিং শুধু টেক্সট চ্যাটের জন্য।)",
            reply_markup=main_kb(),
        )
        return
    await update.message.reply_text(
        f"🔊 টেক্সট চ্যাটে ভয়েস রিপ্লাই: {'on' if h.get('voice') else 'off'}\n"
        f"চালু/বন্ধ করতে: `/voice on` অথবা `/voice off`",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_kb(),
    )


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) == 2 and parts[1].strip().lower() in ("on", "off"):
        h = _hist(user.id)
        h["on"] = parts[1].strip().lower() == "on"
        _save_memory()
        await update.message.reply_text(
            f"🧠 Memory {'enabled ✅' if h['on'] else 'disabled ⏸️'}.", reply_markup=main_kb()
        )
        return
    await update.message.reply_text(memory_stats(user.id), parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb())


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    clear_memory(update.effective_user.id)
    await update.message.reply_text("🧹 Memory cleared.", reply_markup=main_kb())


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.clear()
    await update.message.reply_text("Cancelled.", reply_markup=main_kb())


# ============================================================================
# Callbacks (menu navigation)
# ============================================================================


async def cb_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data.clear()
    await q.edit_message_text("🧿 *Menu*", parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb())


async def cb_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb())


async def cb_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(memory_stats(update.effective_user.id), parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb())


async def cb_chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["waiting_for"] = "chat_prompt"
    await q.edit_message_text("💬 Send your message (text or voice). `/cancel` to abort.", parse_mode=ParseMode.MARKDOWN)


async def cb_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["waiting_for"] = "image_prompt"
    await q.edit_message_text(
        "🖼️ Describe the image you want — or send a photo first, then describe the edit.\n\n`/cancel` to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cb_presets(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(
        "🎨 *Preset Styles*\nএকটা স্টাইল বেছে নিন, তারপর যে ছবিটা এডিট করতে চান সেটা পাঠান।",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=presets_kb(),
    )


async def cb_preset_pick(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    preset_id = q.data.split(":", 1)[1]
    preset = PRESET_BY_ID.get(preset_id)
    if not preset:
        await q.edit_message_text("⚠️ Preset not found.", reply_markup=main_kb())
        return
    ctx.user_data["preset_prompt"] = preset["prompt"]
    ctx.user_data["waiting_for"] = "preset_awaiting_photo"
    await q.edit_message_text(
        f"🎨 *{preset['label']}* সিলেক্ট করা হয়েছে।\n\n📸 এখন যে ছবিটাতে এই স্টাইল অ্যাপ্লাই করতে চান, সেটা পাঠান।\n\n`/cancel` to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cb_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data.clear()
    await q.edit_message_text("Cancelled.", reply_markup=main_kb())


# ============================================================================
# Chat flow (text or voice input -> Groq -> optional TTS reply)
# ============================================================================


async def run_chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE, prompt: str, also_voice: bool = False):
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    user_id = update.effective_user.id
    status = await update.message.reply_text("💬 Thinking…")
    try:
        history = history_messages(user_id)
        messages = history + [{"role": "user", "content": prompt}]
        reply = await _call_chat(messages)
        remember(user_id, "user", prompt)
        remember(user_id, "assistant", reply)
        await status.delete()
        for chunk in chunk_text(reply):
            await update.message.reply_text(chunk)
        want_voice = also_voice or _hist(user_id).get("voice", False)
        if want_voice:
            try:
                audio_bytes = await tts_speak(reply)
                await update.message.reply_audio(audio=audio_bytes, filename="reply.mp3", title="🔊 Voice reply")
            except ProviderError as e:
                log.warning("tts skipped: %s", e)
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("chat failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False
        ctx.user_data.pop("waiting_for", None)


# ============================================================================
# Vision flow (photo + question)
# ============================================================================


async def photo_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Any photo the user sends: apply a pending preset if one was chosen,
    otherwise store it and ask what they want done with it."""
    b64, mime = await telegram_photo_to_b64(update)

    if ctx.user_data.get("waiting_for") == "preset_awaiting_photo":
        prompt = ctx.user_data.pop("preset_prompt", None)
        ctx.user_data.pop("waiting_for", None)
        if prompt:
            await run_image(update, ctx, prompt, image_b64=b64, image_mime=mime)
            return

    ctx.user_data["pending_image_b64"] = b64
    ctx.user_data["pending_image_mime"] = mime
    ctx.user_data["waiting_for"] = "photo_followup"
    await update.message.reply_text(
        "📸 Got it. Now tell me what you want:\n"
        "• ask a question about it (vision), or\n"
        "• describe how to edit it (image editing)\n\n`/cancel` to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def run_vision(update: Update, ctx: ContextTypes.DEFAULT_TYPE, prompt: str, image_b64: str, image_mime: str):
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    status = await update.message.reply_text("👁️ Looking at the image…")
    try:
        reply = await gemini_vision_answer(prompt, image_b64, image_mime)
        await status.delete()
        for chunk in chunk_text(reply):
            await update.message.reply_text(chunk)
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("vision failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False
        ctx.user_data.pop("waiting_for", None)
        ctx.user_data.pop("pending_image_b64", None)
        ctx.user_data.pop("pending_image_mime", None)


# ============================================================================
# Image generation / editing flow
# ============================================================================


async def run_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE, prompt: str, image_b64: Optional[str], image_mime: str = "image/jpeg"):
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    verb = "Editing" if image_b64 else "Generating"
    status = await update.message.reply_text(f"🖼️ {verb} image…")
    try:
        raw_bytes = base64.b64decode(image_b64) if image_b64 else None
        image_bytes = await cf_image_run(prompt, raw_bytes)
        await status.delete()
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"🖼️ {prompt[:900]}",
            reply_markup=done_kb("image"),
        )
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("image failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False
        ctx.user_data.pop("waiting_for", None)
        ctx.user_data.pop("pending_image_b64", None)
        ctx.user_data.pop("pending_image_mime", None)


# ============================================================================
# Voice message flow (STT -> chat -> TTS)
# ============================================================================


async def voice_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    status = await update.message.reply_text("🎙️ শুনছি…")
    try:
        voice = update.message.voice or update.message.audio
        f = await voice.get_file()
        blob = bytes(await f.download_as_bytearray())
        text = await groq_transcribe(blob)
        if not text.strip():
            await status.edit_text("🤔 কিছু বুঝতে পারলাম না, আবার বলো?")
            return
        await status.edit_text(f"🎙️ শুনলাম: _{text[:300]}_", parse_mode=ParseMode.MARKDOWN)
        ctx.user_data["busy"] = False  # let run_chat manage its own busy-lock
        await run_chat(update, ctx, text, also_voice=True)
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("voice input failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False


# ============================================================================
# PDF flow (Gemini summary)
# ============================================================================


async def document_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or (doc.mime_type != "application/pdf" and not (doc.file_name or "").lower().endswith(".pdf")):
        await update.message.reply_text(
            "📎 আমি শুধু PDF ফাইল পড়তে পারি এই মুহূর্তে।", reply_markup=main_kb()
        )
        return
    if ctx.user_data.get("busy"):
        await update.message.reply_text("⏳ Still working on your previous request…")
        return
    ctx.user_data["busy"] = True
    status = await update.message.reply_text("📄 পিডিএফ পড়ছি…")
    try:
        f = await doc.get_file()
        blob = bytes(await f.download_as_bytearray())
        b64 = base64.b64encode(blob).decode()
        summary = await gemini_pdf_summary(b64)
        await status.delete()
        for chunk in chunk_text(summary):
            await update.message.reply_text(chunk)
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("pdf summary failed")
        await status.edit_text(f"❌ Error: {e}")
    finally:
        ctx.user_data["busy"] = False


# ============================================================================
# Text router (dispatches based on what we're waiting for)
# ============================================================================


async def text_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    waiting = ctx.user_data.get("waiting_for")

    if waiting == "photo_followup":
        b64 = ctx.user_data.get("pending_image_b64")
        mime = ctx.user_data.get("pending_image_mime", "image/jpeg")
        edit_words = ("edit", "change", "remove", "replace", "add", "make it", "turn it into", "modify", "transform")
        if any(w in text.lower() for w in edit_words):
            await run_image(update, ctx, text, image_b64=b64, image_mime=mime)
        else:
            await run_vision(update, ctx, text, image_b64=b64, image_mime=mime)
        return

    if waiting == "image_prompt":
        await run_image(update, ctx, text, image_b64=None)
        return

    # default: plain chat (also covers waiting == "chat_prompt" and no state at all)
    await run_chat(update, ctx, text)


# ============================================================================
# Fallback / error handling
# ============================================================================


async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("update %s caused error %s", update, ctx.error)


async def unknown_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤔 Unknown command. Try `/help` for the full command list.", parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb()
    )


async def unknown_any(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            "👋 আমি টেক্সট, ভয়েস, ছবি আর PDF হ্যান্ডেল করতে পারি। `/help` চাপো।",
            reply_markup=main_kb(),
        )
    except Exception:
        pass


# ============================================================================
# Main
# ============================================================================


async def post_init(app: Application):
    await app.bot.set_my_commands(
        [
            ("start", "🧿 Main menu"),
            ("ask", "💬 চ্যাট — /ask <প্রশ্ন>"),
            ("img", "🖼️ ছবি বানাও — /img <prompt>"),
            ("img2", "🆓 ছবি বানাও (backup ১) — /img2 <prompt>"),
            ("img3", "🆓 ছবি বানাও (backup ২) — /img3 <prompt>"),
            ("video", "🎬 ভিডিও বানাও — /video <prompt>"),
            ("search", "🔍 ওয়েব সার্চ — /search <প্রশ্ন>"),
            ("voice", "🔊 চ্যাটে ভয়েস রিপ্লাই on/off"),
            ("memory", "🧠 মেমরি status / on / off"),
            ("clear", "🧹 আমার মেমরি মুছে ফেলো"),
            ("help", "ℹ️ সব কমান্ডের তালিকা"),
        ]
    )


def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.post_init = post_init

    # --- commands ---
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("img", cmd_img))
    app.add_handler(CommandHandler("img2", cmd_img2))
    app.add_handler(CommandHandler("img3", cmd_img3))
    app.add_handler(CommandHandler("video", cmd_video))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("cancel", cmd_cancel))

    # --- inline-button callbacks ---
    app.add_handler(CallbackQueryHandler(cb_menu, pattern=r"^menu$"))
    app.add_handler(CallbackQueryHandler(cb_help, pattern=r"^help$"))
    app.add_handler(CallbackQueryHandler(cb_memory, pattern=r"^memory$"))
    app.add_handler(CallbackQueryHandler(cb_chat, pattern=r"^chat$"))
    app.add_handler(CallbackQueryHandler(cb_image, pattern=r"^image$"))
    app.add_handler(CallbackQueryHandler(cb_presets, pattern=r"^presets$"))
    app.add_handler(CallbackQueryHandler(cb_preset_pick, pattern=r"^preset:"))
    app.add_handler(CallbackQueryHandler(cb_cancel, pattern=r"^cancel$"))

    # --- message types (order matters: specific -> general) ---
    app.add_handler(MessageHandler(filters.PHOTO, photo_input))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_input))
    app.add_handler(MessageHandler(filters.Document.ALL, document_input))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    app.add_handler(MessageHandler(filters.ALL, unknown_any))

    app.add_error_handler(error_handler)
    return app


class _HealthHandler(BaseHTTPRequestHandler):
    """Bare-bones handler so Render's free Web Service tier sees a live
    HTTP port and stops flagging the deploy as unhealthy. It has nothing
    to do with Telegram — the bot itself still runs on long polling."""

    def do_GET(self):  # noqa: N802 (stdlib method name)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"bot is running")

    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        pass  # silence per-request logging; the bot's own logger covers activity


def start_health_server() -> None:
    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    log.info("health check server listening on 0.0.0.0:%d", port)
    server.serve_forever()


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Bot — Chat/Voice: Groq  |  Vision/PDF: Gemini  |  Image: Cloudflare")
    print("  Waiting for messages… (/start)")
    print("=" * 60)
    # Render's free tier only exists for "Web Service", which requires
    # binding to $PORT. This thread satisfies that requirement while the
    # main thread runs the actual Telegram bot via long polling.
    threading.Thread(target=start_health_server, daemon=True).start()
    build_app().run_polling(allowed_updates=Update.ALL_TYPES)
