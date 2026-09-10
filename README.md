# 🤖 AI Telegram Bot

A Telegram bot for chat, vision, and image generation/editing — built on
**official provider APIs only** (no third-party proxies, no shared keys).

| Feature | Provider |
|---|---|
| 💬 Chat | [Groq](https://console.groq.com) |
| 👁️ Vision (ask about a photo) | [Google Gemini](https://ai.google.dev) |
| 🖼️ Image generate / edit | [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai) (free tier, Stable Diffusion XL — real image-to-image editing) |
| 🆓 Backup image generate | Community Cloudflare-Workers-based service (generation only) |
| 🧠 Memory | Local JSON, per-user, on/off toggle |

---

## ✨ Features

- `/start` — main menu with inline buttons (Chat / Image / Memory / Help)
- `/ask <question>` — quick chat without opening the menu
- `/img <prompt>` — quick image generation
- Send a **photo** → bot asks whether you want it *analyzed* (vision) or
  *edited* (image editing), then does it
- `/memory` — view memory status, `/memory on` / `/memory off`
- `/clear` — wipe your saved conversation history
- `/cancel` — abort whatever the bot is currently waiting for

---

## 🗂️ Project structure

```
.
├── hotbot_bot.py       # the entire bot
├── requirements.txt    # Python dependencies
├── .env.example        # names of the secrets you need (no real values)
└── README.md
```

---

## 🔑 Required secrets

Set these as **environment variables** on your host (never commit real
values to Git):

| Variable | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) on Telegram |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `CLOUDFLARE_ACCOUNT_ID` | [dash.cloudflare.com](https://dash.cloudflare.com) — sidebar of any Workers & Pages page |
| `CLOUDFLARE_API_TOKEN` | dash.cloudflare.com → My Profile → API Tokens → Create Token → "Workers AI" template |

Optional overrides (defaults already sensible):
`CHAT_MODEL`, `GEMINI_VISION_MODEL`, `GEMINI_IMAGE_MODEL`, `CHAT_BASE_URL`,
`GEMINI_BASE_URL`, `MEMORY_MAX_MESSAGES`, `MEMORY_MAX_CHARS`.

---

## 🚀 Deploy on Render (free tier)

1. **New → Web Service** → connect this repo
2. **Instance Type:** Free
3. **Build Command:**
   ```
   pip install -r requirements.txt
   ```
4. **Start Command:**
   ```
   python hotbot_bot.py
   ```
5. Go to the **Environment** tab and add the three secrets above
6. Deploy, then check **Logs** for:
   ```
   health check server listening on 0.0.0.0:xxxx
   Waiting for messages… (/start)
   ```

> Free-tier services sleep after 15 minutes of inactivity — the first
> message after a quiet period may take 30–50 seconds to get a reply.
> The bot includes a tiny built-in HTTP health-check endpoint purely so
> Render recognizes it as a valid Web Service; Telegram updates are
> still handled via long polling, not HTTP.

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

## 🔄 Swapping providers later

All provider settings live in one place at the top of `hotbot_bot.py`:
`CHAT_PROVIDER` and `GEMINI_PROVIDER`. To switch chat to a different
OpenAI-compatible API, just change the `base_url` / `model` / API-key
environment variable name there. Vision/image providers with a different
request shape need a small edit inside `gemini_vision_answer` /
`gemini_image_generate` — the rest of the bot is untouched.

---

## 🔒 Security notes

- No API keys or bot tokens are hardcoded anywhere in this repo
- `bot_memory.json` (created at runtime, holds per-user chat history) is
  git-ignored — don't commit it
- If a token or key is ever accidentally committed, revoke/rotate it
  immediately (BotFather `/revoke` for Telegram, provider dashboard for
  API keys) and remove it from Git history
