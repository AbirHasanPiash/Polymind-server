# MultiAIModel Backend

FastAPI backend for a multi-provider AI platform: streaming chat across OpenAI, Anthropic and
Google models, media generation (image, speech, avatar video), and a credit wallet funded through
Stripe and Razorpay.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-336791)
![License](https://img.shields.io/badge/license-proprietary-lightgrey)

## Features

- **Streaming chat** over WebSockets, with automatic model routing and per-turn billing
- **Multi-provider LLMs** behind one interface — ids, routing and pricing live in a single registry
- **Media generation** — OpenAI images, Google Cloud TTS and D-ID avatar video, run by Celery workers
- **Credit wallet** with atomic, `Decimal`-exact accounting and automatic refunds on failed jobs
- **Payments** via Stripe Checkout and Razorpay, idempotent against replayed webhooks
- **Admin API** for user management, credit adjustments, package CRUD and revenue analytics
- **JWT auth** with email/password (Argon2) and Google OAuth, plus role-based access control

## Tech stack

| Layer | Choice |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Data | PostgreSQL, SQLAlchemy 2 (async), Alembic |
| Cache & queue | Redis, Celery |
| Storage | Cloudflare R2 (S3-compatible) |
| Providers | OpenAI, Anthropic, Google Gemini, Google TTS, D-ID |

## Architecture

```
app/
├── api/v1/endpoints/   HTTP + WebSocket routes
├── core/               config, database, redis, security, logging
├── models/             SQLAlchemy ORM models
├── schemas/            Pydantic request/response models
├── services/
│   ├── llm/            provider adapters, model registry, router
│   ├── media/          image, speech and video generation
│   ├── billing.py      wallet debits, credits and refunds
│   └── storage.py      object storage
├── workers/            Celery app and background tasks
└── main.py             application entry point
```

Two rules shape the design:

- **Money never moves in Python.** Every balance change is one conditional SQL statement
  (`UPDATE … WHERE credits >= amount RETURNING credits`), so concurrent requests cannot spend the
  same credits twice. Media endpoints debit before queueing work; the worker refunds if the job fails.
- **One registry owns each model.** `services/llm/models.py` holds the public id, provider, upstream
  id and price of every model, so a request can never be served by one model and billed at another's
  rate — and unknown ids are rejected before they reach a provider.

## Getting started

**Requirements:** Python 3.11+, PostgreSQL 14+, Redis 7+

```bash
git clone https://github.com/AbirHasanPiash/multimodal-ai-platform.git
cd multimodal-ai-platform

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
| `REDIS_URL` | Required — cache and Celery broker |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DID_API_KEY` | Optional per provider |
| `STORAGE_*`, `STRIPE_*`, `RAZORPAY_*`, `GOOGLE_CLIENT_ID` | Optional per feature |

Provider credentials are optional by design: a missing key disables that one feature and is listed
in the startup log instead of preventing boot.

## API

Base path: `/api/v1`

| Area | Endpoints |
| --- | --- |
| Auth | `POST /auth/signup`, `POST /auth/login`, `POST /auth/google` |
| Chat | `WS /chat/ws`, `GET /chat/list`, `GET /chat/history/{id}`, `POST /chat/upload`, `DELETE /chat/{id}` |
| Media | `POST /media/generate` (speech), `/media/generate-image`, `/media/generate-avatar`, `/media/tts/{message_id}`, `/media/upload`, plus `GET`/`DELETE` listings per asset type |
| Wallet | `GET /users/me`, `GET /packages/`, `GET /payments/history` |
| Payments | `POST /payments/create-checkout-session/{package_id}`, `POST /payments/create-razorpay-order/{package_id}`, `POST /payments/verify-razorpay-payment` |
| Admin | `GET /admin/stats/overview`, `GET /admin/users`, `PATCH /admin/users/{id}`, package CRUD on `/packages/` |
| Meta | `GET /health`, `GET /health/live`, `GET /api/v1/models` |

`GET /health` reports database and Redis connectivity and returns 503 when either is down;
`/health/live` is a dependency-free liveness probe. Media generation is asynchronous — those
endpoints return `202 Accepted` and the worker fills in the result.

### Chat WebSocket

Connect to `/api/v1/chat/ws?token=<jwt>&model=auto&chat_id=<optional>` and send:

```json
{ "type": "user_message", "content": "Explain async/await", "attachments": [] }
```

The server streams `{"type": "content", "delta": "…"}` and emits `system` events for the chat id,
the model the router chose, and the cost of the turn; failures arrive as `{"type": "error"}`.
`GET /api/v1/models` lists the ids a client may request — `auto` lets the router pick from the
prompt's intent (coding, reasoning, long-context or fast).

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

## Deployment

Pushes to `main` run lint and tests, build the image and publish it to GHCR, then deploy over SSH
and verify `/health` before finishing — see [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).
[`render.yaml`](render.yaml) describes an equivalent Render deployment.

Production checklist: set `ENVIRONMENT=production`, use a strong unique `SECRET_KEY`, restrict
`CORS_ORIGINS` to your own domains, terminate TLS at the proxy, configure the Stripe webhook signing
secret, and run `alembic upgrade head` as part of the release.

## License

Proprietary. All rights reserved © MultiAIModel.
