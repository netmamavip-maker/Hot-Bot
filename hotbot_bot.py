#!/usr/bin/env python3
"""
hotbot_bot.py — Telegram AI bot using OFFICIAL provider APIs.

Providers used right now:
  💬 Chat            -> Groq (console.groq.com), OpenAI-compatible /chat/completions
  👁️ Vision           -> Gemini (Google), generateContent (multimodal)
  🖼️ Image generate/edit -> Gemini (Google), image-capable model

Everything provider-specific lives in the CONFIG block below. To switch
providers later (e.g. move Chat to Claude, or Image to OpenAI), you only
need to edit CONFIG — no other code changes required, as long as the new
provider is OpenAI-chat-compatible (for CHAT) or you add a small adapter
function (for VISION/IMAGE, see the two "_call_*" functions).

Secrets come ONLY from environment variables / your host's "Secrets" panel.
Nothing is hardcoded in this file.

Required environment variables:
  TELEGRAM_BOT_TOKEN   - from @BotFather
  GROQ_API_KEY         - Groq (console.groq.com) API key
  GEMINI_API_KEY       - Google AI Studio (Gemini) API key

Run:
    pip install python-telegram-bot httpx
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

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
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
# To switch chat to a different OpenAI-compatible provider later, just change
# api_key / base_url / model above — _call_chat() below doesn't need to change.

# --- Vision + Image provider (currently Gemini / Google) ------------------
GEMINI_PROVIDER = {
    "name": "gemini",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "base_url": os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    ),
    "vision_model": os.environ.get("GEMINI_VISION_MODEL", "gemini-3.6-flash"),
}
# NOTE: Gemini is used for VISION (asking questions about a photo) only now.
# gemini-2.5-flash-image effectively has no usable free-tier quota (Google
# requires billing enabled for it), so image generation/editing has been
# moved to Cloudflare Workers AI below, which has a real, working free tier
# (Stable Diffusion XL supports image-to-image editing, not just generation).

# --- Image generate + edit provider (Cloudflare Workers AI, free tier) ----
# No Worker needs to be deployed — we call Cloudflare's REST API directly.
# You need two values from dash.cloudflare.com:
#   1) Account ID           -> right sidebar of any Workers & Pages page
#   2) API Token            -> My Profile -> API Tokens -> Create Token
#                              -> template "Workers AI" (or custom with
#                              "Workers AI: Edit" permission)
CLOUDFLARE_PROVIDER = {
    "account_id": os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
    "api_token": os.environ.get("CLOUDFLARE_API_TOKEN", ""),
    "model": os.environ.get(
        "CLOUDFLARE_IMAGE_MODEL", "@cf/stabilityai/stable-diffusion-xl-base-1.0"
    ),
}

# --- Backup / community text-to-image provider (generation only, no editing)
# Third-party community project on Cloudflare Workers AI. No account, no key.
# Independent of Gemini — used as a fallback / secondary "free" generator.
# SAFETY IS ALWAYS FORCED ON below (see backup_image_generate) and is never
# exposed as a togglable option anywhere in this bot.
# Set BACKUP_IMAGE_ENABLED=false to turn this feature off entirely.
BACKUP_IMAGE_PROVIDER = {
    "name": "ashlynn-community",
    "enabled": os.environ.get("BACKUP_IMAGE_ENABLED", "true").lower() == "true",
    "base_url": os.environ.get(
        "BACKUP_IMAGE_BASE_URL", "https://death-image.ashlynn.workers.dev"
    ),
    "steps": int(os.environ.get("BACKUP_IMAGE_STEPS", "8")),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "bot_memory.json")
MEMORY_MAX_MESSAGES = int(os.environ.get("MEMORY_MAX_MESSAGES", "60"))
MEMORY_MAX_CHARS = int(os.environ.get("MEMORY_MAX_CHARS", "60000"))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("bot")

S_INPUT = 1

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
    return MEMORY.setdefault(user_id, {"messages": [], "on": True})


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
        f"messages: {len(msgs)}\n"
        f"chars kept: {total:,} / limit {MEMORY_MAX_CHARS:,}"
    )


def clear_memory(user_id: int) -> None:
    MEMORY[user_id] = {"messages": [], "on": True}
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
# Vision + Image provider adapter (Gemini)
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
            "Vision/Image provider API key is missing. Set the GEMINI_API_KEY secret."
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


async def gemini_vision_answer(text_prompt: str, image_b64: str, image_mime: str) -> str:
    """Ask a vision-capable Gemini model a question about an image."""
    data = await _gemini_generate(
        GEMINI_PROVIDER["vision_model"], text_prompt, image_b64, image_mime
    )
    try:
        candidate = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in candidate)
        if not text:
            raise KeyError
        return text
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"Unexpected Gemini vision response: {json.dumps(data)[:300]}")


def _downscale_image(image_bytes: bytes, max_side: int = 896) -> bytes:
    """Shrink an image before sending it to Cloudflare's SDXL (Beta) model.
    Large payloads are a common trigger for its 'InferenceResponse' 500s."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88)
        return out.getvalue()
    except Exception:
        return image_bytes  # fall back to the original if Pillow can't read it


async def cf_image_generate(
    prompt: str, image_bytes: Optional[bytes] = None, strength: float = 0.75
) -> bytes:
    """Generate a new image, or edit `image_bytes` if provided, via Cloudflare
    Workers AI (Stable Diffusion XL — supports real image-to-image editing on
    the free tier, not just text-to-image). Returns raw PNG/JPEG bytes.

    The model is still "Beta" on Cloudflare's side and occasionally returns a
    500 'InferenceResponse' error unrelated to our request — a large input
    image is a common trigger, and a short retry often succeeds."""
    cfg = CLOUDFLARE_PROVIDER
    if not cfg["account_id"] or not cfg["api_token"]:
        raise ProviderError(
            "Cloudflare image provider isn't configured. Set CLOUDFLARE_ACCOUNT_ID "
            "and CLOUDFLARE_API_TOKEN secrets."
        )
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{cfg['account_id']}"
        f"/ai/run/{cfg['model']}"
    )
    headers = {
        "Authorization": f"Bearer {cfg['api_token']}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {"prompt": prompt, "num_steps": 20}
    if image_bytes:
        body["image"] = list(_downscale_image(image_bytes))
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
            last_error = ProviderError(f"Unexpected Cloudflare image response: {resp.text[:300]}")
        elif resp.status_code >= 500:
            last_error = ProviderError(f"Cloudflare image error {resp.status_code}: {resp.text[:300]}")
            await asyncio.sleep(2)
            continue  # transient backend error — retry
        else:
            raise ProviderError(f"Cloudflare image error {resp.status_code}: {resp.text[:300]}")
    raise last_error or ProviderError("Cloudflare image provider failed after retries.")


async def backup_image_generate(prompt: str, dimensions: str = "1:1") -> str:
    """Community/free text-to-image backup (generation only — cannot edit an
    existing photo). Returns a hosted image URL. SAFETY IS ALWAYS FORCED TRUE
    here — this is intentional and must never be made configurable."""
    cfg = BACKUP_IMAGE_PROVIDER
    if not cfg["enabled"]:
        raise ProviderError("Backup image provider is disabled.")
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
        raise ProviderError(f"Backup image provider error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    images = data.get("images") or []
    if not images:
        raise ProviderError(f"Backup provider returned no images: {json.dumps(data)[:200]}")
    return images[0]


# ============================================================================
# Preset prompt gallery — pick a ready-made edit style instead of typing one
# (same idea as meigen.ai's prompt cards; runs on our own Gemini pipeline)
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


# ============================================================================
# Commands
# ============================================================================


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*🧿 AI Bot*\n\n"
        "💬 *Chat* — text conversation (Groq)\n"
        "👁️ *Vision* — send a photo + a question, I'll look at it (Gemini)\n"
        "🖼️ *Image* — generate a new image, or send a photo + edit instructions (Gemini)\n"
        "🎨 *Presets* — pick a ready-made edit style, then send your photo\n"
        "🧠 *Memory* — remembers your chat history until you clear it\n\n"
        "Tap a button below, or use `/ask`, `/img`, `/memory`, `/clear`.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_kb(),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*ℹ️ Help*\n\n"
        "• `/ask <question>` — quick chat\n"
        "• `/img <prompt>` — quick image generation\n"
        "• `/img2 <prompt>` — backup generator (free community service)\n"
        "• send a photo — I'll ask what you want to know / do with it\n"
        "• `/memory` — see memory status, `/memory on` / `/memory off`\n"
        "• `/clear` — wipe your saved history\n"
        "• `/cancel` — abort whatever it's currently asking you for",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_kb(),
    )


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
    """Generation-only, via the free community backup provider (safety forced on)."""
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text("Usage: `/img2 your prompt`", parse_mode=ParseMode.MARKDOWN)
        return
    prompt = parts[1].strip()
    status = await update.message.reply_text("🆓 Generating (community backup)…")
    try:
        image_url = await backup_image_generate(prompt)
        await status.delete()
        await update.message.reply_photo(photo=image_url, caption=f"🆓 {prompt[:900]}", reply_markup=main_kb())
    except ProviderError as e:
        await status.edit_text(f"⚠️ {e}")
    except Exception as e:
        log.exception("backup image failed")
        await status.edit_text(f"❌ Error: {e}")


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
    await q.edit_message_text(
        "*ℹ️ Help*\n\n• `/ask <q>` chat\n• `/img <prompt>` image\n"
        "• send a photo for vision / editing\n• `/memory`, `/clear`",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_kb(),
    )


async def cb_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(memory_stats(update.effective_user.id), parse_mode=ParseMode.MARKDOWN, reply_markup=main_kb())


async def cb_chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["waiting_for"] = "chat_prompt"
    await q.edit_message_text("💬 Send your message. `/cancel` to abort.", parse_mode=ParseMode.MARKDOWN)
    return S_INPUT


async def cb_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data["waiting_for"] = "image_prompt"
    await q.edit_message_text(
        "🖼️ Describe the image you want — or send a photo first, then describe the edit.\n\n`/cancel` to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )
    return S_INPUT


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


# ============================================================================
# Chat flow
# ============================================================================


async def run_chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE, prompt: str):
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
            return ConversationHandler.END

    ctx.user_data["pending_image_b64"] = b64
    ctx.user_data["pending_image_mime"] = mime
    ctx.user_data["waiting_for"] = "photo_followup"
    await update.message.reply_text(
        "📸 Got it. Now tell me what you want:\n"
        "• ask a question about it (vision), or\n"
        "• describe how to edit it (image editing)\n\n`/cancel` to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )
    return S_INPUT


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
        image_bytes = await cf_image_generate(prompt, raw_bytes)
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
        return ConversationHandler.END

    if waiting == "image_prompt":
        await run_image(update, ctx, text, image_b64=None)
        return ConversationHandler.END

    # default: plain chat (also covers waiting == "chat_prompt" and no state at all)
    await run_chat(update, ctx, text)
    return ConversationHandler.END


# ============================================================================
# Cancel & fallback
# ============================================================================


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.clear()
    await update.message.reply_text("Cancelled.", reply_markup=main_kb())
    return ConversationHandler.END


async def cancel_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    ctx.user_data.clear()
    await q.edit_message_text("Cancelled.", reply_markup=main_kb())
    return ConversationHandler.END


async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("update %s caused error %s", update, ctx.error)


async def unknown_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤔 Unknown command. Try `/start` for the menu.", reply_markup=main_kb()
    )


async def unknown_any(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            "👋 I handle text and photos. Tap `/start` for the menu.", reply_markup=main_kb()
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
            ("ask", "💬 Quick chat — /ask <question>"),
            ("img", "🖼️ Quick image — /img <prompt>"),
            ("img2", "🆓 Backup image — /img2 <prompt>"),
            ("memory", "🧠 Memory status / on / off"),
            ("clear", "🧹 Clear my memory"),
            ("help", "ℹ️ Help"),
        ]
    )


def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.post_init = post_init

    conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(cb_chat, pattern=r"^chat$"),
            CallbackQueryHandler(cb_image, pattern=r"^image$"),
            MessageHandler(filters.PHOTO, photo_input),
        ],
        states={
            S_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, text_input),
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel_cb, pattern=r"^cancel$")],
        per_message=False,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("img", cmd_img))
    app.add_handler(CommandHandler("img2", cmd_img2))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(conv)

    # global fallbacks so any message always gets a response
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input))
    app.add_handler(MessageHandler(filters.PHOTO, photo_input))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    app.add_handler(MessageHandler(filters.ALL, unknown_any))

    app.add_handler(CallbackQueryHandler(cb_menu, pattern=r"^menu$"))
    app.add_handler(CallbackQueryHandler(cb_help, pattern=r"^help$"))
    app.add_handler(CallbackQueryHandler(cb_memory, pattern=r"^memory$"))
    app.add_handler(CallbackQueryHandler(cb_presets, pattern=r"^presets$"))
    app.add_handler(CallbackQueryHandler(cb_preset_pick, pattern=r"^preset:"))

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
    print("  AI Bot — Chat: Groq  |  Vision/Image: Gemini")
    print("  Waiting for messages… (/start)")
    print("=" * 60)
    # Render's free tier only exists for "Web Service", which requires
    # binding to $PORT. This thread satisfies that requirement while the
    # main thread runs the actual Telegram bot via long polling.
    threading.Thread(target=start_health_server, daemon=True).start()
    build_app().run_polling(allowed_updates=Update.ALL_TYPES)
