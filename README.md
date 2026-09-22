# Polymind Server

FastAPI backend for **Polymind** — a multi-model AI workspace. Streaming chat across OpenAI,
Anthropic and Google models with automatic routing and side-by-side model comparison, media
studios (images, speech, avatar video), per-user usage analytics, and a credit wallet funded
through Stripe and Razorpay.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-336791)
![License](https://img.shields.io/badge/license-proprietary-lightgrey)

The web client lives in [Polymind](https://github.com/AbirHasanPiash/Polymind).

## Features

- **Streaming chat** over a WebSocket with per-turn model choice, adjustable reasoning depth,
  stop, regenerate and edit-and-resend
- **Auto routing** — every message is scored for intent (coding, reasoning, long context,
  writing, quick reply) and sent to the model that suits it; the reason is reported back
- **Arena mode** — two or three models answer the same prompt side by side, each keeping its
  own thread of the conversation
- **Model registry** — one file owns every model's id, provider, pricing and metadata; the
  API, the router, the biller and the client's picker all read from it
- **Current models** — GPT-5.5 / 5.5 Pro / 5.4 family, Claude Fable 5.1 / Opus 5 / Sonnet 5 /
  Haiku 4.5, Gemini 3.1 Pro / 3.8 Flash / 3.5 Flash-Lite (OpenAI via the Responses API,
  Claude with adaptive thinking and server-side refusal fallbacks, Gemini with thinking levels)
- **Conversation management** — auto-generated titles, rename, pin, search across messages,
  per-chat instructions, Markdown/JSON export and public read-only share links
- **Media generation** — GPT Image 2 / 1.5 and Google Nano Banana 2 / Pro for images, Google
  Cloud and OpenAI voices for speech (including "read aloud" for any reply), D-ID avatar video;
  all run by Celery workers with automatic refunds on failure
- **Wallet** with atomic, `Decimal`-exact accounting, affordability checks before every turn,
  Stripe Checkout and Razorpay, and a per-user usage summary
- **Accounts** — email/password and Google sign-in, sliding session refresh, password reset by
  email, password change, preferences (default model, custom instructions, saved prompts),
  self-service account deletion
- **Operations** — Redis rate limiting, security headers, request ids, JSON logs in production,
  health probes, admin analytics (usage by model, spend by category, top users, transactions)

## Tech stack

| Layer | Choice |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Data | PostgreSQL, SQLAlchemy 2 (async), Alembic |
| Cache & queue | Redis, Celery |
| Storage | Cloudflare R2 (S3-compatible) |
| Providers | OpenAI, Anthropic, Google Gemini, Google Cloud TTS, D-ID |

## Architecture

```
app/
├── api/v1/endpoints/   HTTP + WebSocket routes (auth, users, chat, media, payments, admin)
├── core/               config, database, redis, security, rate limiting, logging
├── models/             SQLAlchemy ORM models
├── schemas/            Pydantic request/response models
├── services/
│   ├── llm/            model registry, router, adapters, usage accounting, titles
│   ├── media/          image, speech and avatar-video services
│   ├── chat_service.py history, budgeting, export, system prompt
│   ├── billing.py      wallet debits, credits and refunds
│   ├── usage.py        per-user analytics
│   ├── email.py        transactional email (optional SMTP)
│   └── storage.py      object storage
├── workers/            Celery app and background tasks
└── main.py             application entry point
```

Three rules shape the design:

- **Money never moves in Python.** Every balance change is one conditional SQL statement
  (`UPDATE … WHERE credits >= amount RETURNING credits`), so concurrent requests cannot spend
  the same credits twice. Media endpoints debit before queueing work; the worker refunds if the
  job fails. A chat turn is refused up front when the balance cannot cover it, and the reply
  length is capped by what the wallet can pay for.
- **One registry owns each model.** `services/llm/models.py` holds the public id, provider,
  upstream id, price and display metadata of every model, so a request can never be served by
  one model and billed at another's rate, unknown ids are rejected before they reach a
  provider, and the client renders its picker from `GET /api/v1/models`.
- **Provider quirks stay in adapters.** `services/llm/*_adapter.py` translate one neutral
  message shape into each vendor's API (Responses API, Messages API, Gemini), report which
  model actually served the request, and surface refusals and truncation as data.

## Getting started

**Requirements:** Python 3.11+, PostgreSQL 14+, Redis 7+

```bash
git clone https://github.com/AbirHasanPiash/Polymind-server.git
cd Polymind-server

python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                # then fill in the values
alembic upgrade head
uvicorn app.main:app --reload
```

The API listens on `http://localhost:8000`; interactive docs are at `/docs` (disabled when
`ENVIRONMENT=production`). Media generation additionally needs a worker:

```bash
celery -A app.workers.celery_app worker --loglevel=info
# On macOS the default prefork pool can fail to start under recent Python
# releases; use a single-process pool for local development instead:
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

### Docker

```bash
docker compose up --build                # API, worker and Redis
docker compose --profile local-db up     # …and a local PostgreSQL
```

### Configuration

Settings come from the environment and are validated at startup — see
[`.env.example`](.env.example) for the full list.

| Variable | Notes |
| --- | --- |
| `SECRET_KEY` | Required, 32+ characters |
| `DATABASE_URL` | Required, must use the `postgresql+asyncpg://` driver |
| `REDIS_URL` | Required — cache, rate limits, reset tokens and Celery broker |
| `FRONTEND_URL` | Public origin of the web app (payment redirects, share links, reset emails) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DID_API_KEY` | Optional per provider |
| `STORAGE_*`, `STRIPE_*`, `RAZORPAY_*`, `GOOGLE_CLIENT_ID`, `SMTP_*` | Optional per feature |

Provider credentials are optional by design: a missing key disables that one feature, is
listed in the startup log, and is reported to the client through `GET /api/v1/features` so
the UI hides what is not configured. Without SMTP, password-reset links are logged instead
of emailed.

## API

Base path: `/api/v1`

| Area | Endpoints |
| --- | --- |
| Auth | `POST /auth/signup`, `/auth/login`, `/auth/google`, `/auth/refresh`, `/auth/forgot-password`, `/auth/reset-password` |
| Account | `GET`/`PATCH /users/me`, `POST /users/me/password`, `DELETE /users/me`, `GET /users/me/usage` |
| Chat | `WS /chat/ws`, `GET /chat/list`, `GET /chat/search`, `GET /chat/history/{id}`, `GET`/`PATCH`/`DELETE /chat/{id}`, `GET /chat/{id}/export`, `POST`/`DELETE /chat/{id}/share`, `GET /chat/shared/{token}` (public), `POST /chat/upload` |
| Media | `GET /media/voices`, `GET /media/images/options`, `POST /media/generate` (speech), `/media/generate-image`, `/media/generate-avatar`, `POST`/`GET /media/tts/{message_id}`, `/media/upload`, plus `GET`/`DELETE` listings per asset type |
| Wallet | `GET /packages/`, `GET /payments/history` |
| Payments | `POST /payments/create-checkout-session/{package_id}`, `/payments/create-razorpay-order/{package_id}`, `/payments/verify-razorpay-payment` |
| Admin | `GET /admin/stats/overview`, `GET /admin/stats/transactions`, `GET /admin/users`, `PATCH`/`DELETE /admin/users/{id}`, package CRUD on `/packages/` |
| Meta | `GET /health`, `GET /health/live`, `GET /api/v1/models`, `GET /api/v1/features` |

`GET /health` reports database and Redis connectivity and returns 503 when either is down;
`/health/live` is a dependency-free liveness probe. Media generation is asynchronous — those
endpoints return `202 Accepted` and the worker fills in the result.

### Chat WebSocket

Connect to `/api/v1/chat/ws?chat_id=<optional>` and authenticate with the first frame:

```json
{ "type": "auth", "token": "<jwt>" }
```

Then send turns:

```json
{ "type": "user_message", "content": "Explain async/await", "attachments": [],
  "model": "auto", "effort": "medium" }
```

- `model` is any id from `GET /api/v1/models`, or `auto` to let the router choose.
- `models: ["gpt-5.5", "claude-opus-5"]` instead of `model` starts an **arena** turn; every
  model streams in its own `slot`.
- `effort` is `low`, `medium` or `high` (reasoning depth).
- `edit_message_id` replaces an earlier user message and everything after it.
- `{"type": "regenerate", ...}` re-answers the last prompt; `{"type": "stop"}` ends the
  current generation (what was produced is kept and billed).

The server replies with `turn_start` (the models and the routing reason per slot), `content`
deltas, `message_done` (served model, cost, tokens, duration, finish reason) per slot,
`turn_done` (new balance), and `system` events for `chat_id`, `title`, `truncated` and
`stopped`. Failures arrive as `{"type": "error", "code": ..., "message": ...}`.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                      # unit and API tests, no live services needed
ruff check app tests        # lint
ruff format app tests       # format
```

Database changes:

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Adding a model is one entry in `app/services/llm/models.py` (id, provider, upstream id,
prices, context window, tier and description). The tests in `tests/test_llm_routing.py` and
`tests/test_pricing.py` check that every entry resolves to an adapter and is priceable.

## Deployment

Pushes to `main` run lint and tests, build the image and publish it to GHCR, then deploy over
SSH and verify `/health` before finishing — see
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). [`render.yaml`](render.yaml)
describes an equivalent Render deployment.

Production checklist: set `ENVIRONMENT=production`, use a strong unique `SECRET_KEY`, restrict
`CORS_ORIGINS` to your own domains, set `FRONTEND_URL`, terminate TLS at the proxy, configure
the Stripe webhook signing secret, configure SMTP for password resets, and run
`alembic upgrade head` as part of the release (the deploy workflow does this).

## License

Proprietary. All rights reserved © Polymind.
