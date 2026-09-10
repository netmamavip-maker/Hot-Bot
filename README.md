# 🤖 AI Telegram Bot

A Telegram bot for chat, voice, vision, PDF summarization, image
generation/editing, and video generation — built on **official / documented
provider APIs**.

| Feature | Provider |
|---|---|
| 💬 Chat (multi-turn, memory) | [Groq](https://console.groq.com) |
| 🎙️ Voice-in (speech → text) | Groq Whisper |
| 🔊 Voice-out (text → speech) | Microsoft Edge TTS (`edge-tts`, free, no key) |
| 👁️ Vision (ask about a photo) | [Google Gemini](https://ai.google.dev) |
| 📄 PDF summarization | Google Gemini |
| 🖼️ Image generate | Cloudflare Workers AI (Stable Diffusion XL) |
| ✏️ Image edit | Cloudflare Workers AI (Stable Diffusion v1.5 img2img — a **different**, edit-specific model, see note below) |
| 🆓 Image generate backup #1 | Ashlynn community worker |
| 🆓 Image generate backup #2 | "still-queen" community worker |
| 🎬 Video generate | Video Studio community worker (best-effort) |
| 🔍 Web search | TeCoxBeta |
| 🧠 Memory | Local JSON, per-user, on/off toggle |

---

## 🛠️ What was fixed

**Image editing was failing while generation worked.** Both used to go
through the same Cloudflare model (`stable-diffusion-xl-base-1.0`), whose
img2img path is still "Beta" and prone to `500 InferenceResponse` errors.
Editing now uses a **dedicated img2img model**
(`@cf/runwayml/stable-diffusion-v1-5-img2img`), which is far more reliable
for editing an existing photo. Generation still uses SDXL for best quality.
If it still fails, the bot now surfaces Cloudflare's raw error text so it's
easy to see exactly what's wrong.

**Buttons occasionally not responding.** The old code mixed a
`ConversationHandler` with separate global fallback handlers doing the same
job — a known source of swallowed/duplicate updates in
`python-telegram-bot`. It's been removed in favor of the simpler manual
state tracking (`ctx.user_data["waiting_for"]`) the bot already used
internally, which is what actually drove the logic anyway.

---

## ✨ Commands

- `/start` — main menu with inline buttons
- `/ask <question>` — quick chat
- `/img <prompt>` — generate an image (Cloudflare)
- `/img2 <prompt>` — generate an image (free backup #1)
- `/img3 <prompt>` — generate an image (free backup #2)
- `/video <prompt>` — generate a short video
- `/search <question>` — live web search with sources
- `/voice on` / `/voice off` — also get a voice reply on **text** chats
  (voice messages you send always get a voice reply back)
- `/memory` — view memory status; `/memory on` / `/memory off`
- `/clear` — wipe your saved conversation history
- `/cancel` — abort whatever the bot is currently waiting for
- `/help` — full command list inside the bot

**Send a photo** → bot asks whether you want it *analyzed* (vision) or
*edited*, then does it. **Send a voice message** → bot transcribes it,
replies in chat, and also sends a spoken reply. **Send a PDF** → bot
replies with a Bengali summary.

---

## 🗂️ Project structure

```
.
├── hotbot_bot.py       # the entire bot
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version for Render
├── .env.example        # names of the secrets you need (no real values)
└── README.md
```

---

## 🔑 Required secrets

| Variable | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) on Telegram |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) — free |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `CLOUDFLARE_ACCOUNT_ID` | [dash.cloudflare.com](https://dash.cloudflare.com) — sidebar of any Workers & Pages page |
| `CLOUDFLARE_API_TOKEN` | dash.cloudflare.com → My Profile → API Tokens → Create Token → "Workers AI" template |

The backup image generators, video generator, web search, and voice-out
(Edge TTS) need **no key at all** — see `.env.example` for their optional
override variables.

---

## 🚀 Deploy on Render (free tier)

1. **New → Web Service** → connect this repo
2. **Instance Type:** Free
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `python hotbot_bot.py`
5. Go to the **Environment** tab and add the five required secrets above
6. Deploy, then check **Logs** for:
   ```
   health check server listening on 0.0.0.0:xxxx
   Waiting for messages… (/start)
   ```

> Free-tier services sleep after 15 minutes of inactivity — the first
> message after a quiet period may take 30–50 seconds to get a reply.

---

## 💻 Run locally

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="..."
export GROQ_API_KEY="..."
export GEMINI_API_KEY="..."
export CLOUDFLARE_ACCOUNT_ID="..."
export CLOUDFLARE_API_TOKEN="..."
python hotbot_bot.py
```

---

## ⚠️ Known limitations (honest notes)

- **Video generation** (`/video`) talks to a small community "Video Studio"
  Cloudflare Worker whose exact API contract isn't publicly documented —
  the field names in `video_generate()` are inferred from its web UI. If it
  ever breaks after the provider changes something, that one function is
  the only place to fix, and errors include the provider's raw response
  text to make debugging fast.
- **Web search** (`/search`) similarly parses a few common response field
  names defensively; if TeCoxBeta changes its response shape, adjust
  `web_search()`.
- **Voice replies** are sent as a normal audio file (`reply_audio`), not a
  native round Telegram "voice bubble" (`reply_voice`) — a real voice bubble
  requires OGG/Opus audio, which needs an `ffmpeg` conversion step. Skipping
  that keeps the bot dependency-light and reliable on free hosting; the
  audio still plays fine, just with a different-looking bubble.
- **Image editing precision:** the Cloudflare img2img model can restyle a
  whole photo (filters, art styles) or, with a mask, edit a specific region.
  This bot currently only does whole-image img2img — "change only the
  background, keep everything else" isn't pixel-precise the way Gemini's
  Nano Banana was, but it doesn't hit Google's tiny free-tier quota either.

---

## 🔒 Security notes

- No API keys or bot tokens are hardcoded anywhere in this repo
- `bot_memory.json` (created at runtime, holds per-user chat history) is
  git-ignored — don't commit it
- If a token or key is ever accidentally committed, revoke/rotate it
  immediately and remove it from Git history
