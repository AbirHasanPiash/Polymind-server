# MultiAIModel Backend

FastAPI backend for a multi-provider AI platform: streaming chat across OpenAI, Anthropic and
Gemini, media generation (image, speech, avatar video), and a credit wallet funded through
Stripe and Razorpay.

## Features

- **Streaming chat** over WebSockets, with automatic model routing and per-turn billing
- **Multi-provider LLMs** behind one interface — models, routing and pricing live in a single registry
- **Media generation** — OpenAI images, Google Cloud TTS, D-ID avatar video, processed by Celery workers
- **Credit wallet** with atomic, `Decimal`-exact accounting and automatic refunds on failed jobs
- **Payments** through Stripe Checkout and Razorpay, idempotent against replayed webhooks
- **Admin API** for user management, credit adjustments and revenue analytics
- **JWT auth** with email/password and Google OAuth, plus role-based access control

## Stack

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

Requests never mutate a balance in Python. Every credit movement is one conditional SQL
statement (`UPDATE … WHERE credits >= amount RETURNING credits`), so concurrent requests cannot
spend the same credits twice. Media endpoints debit before queueing work and the worker refunds
if the job fails.

## Quick start

**Requirements:** Python 3.11+, PostgreSQL 14+, Redis 7+

```bash
git clone <repository-url> && cd ai-platform-backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                 # then fill in the values
alembic upgrade head
uvicorn app.main:app --reload
```

The API is served at `http://localhost:8000`, interactive docs at `/docs` (disabled when
`ENVIRONMENT=production`).

Media generation needs a worker as well:

```bash
celery -A app.workers.celery_app worker --loglevel=info
```

### Configuration

All settings come from the environment and are validated at startup — see
[`.env.example`](.env.example) for the full list. Required: `SECRET_KEY` (32+ characters),
`DATABASE_URL` (must use the `postgresql+asyncpg://` driver) and `REDIS_URL`. Provider
credentials are optional; a missing key disables that one feature and is reported in the
startup log instead of preventing boot.

### Docker

```bash
docker compose up --build                # API, worker and Redis
docker compose --profile local-db up     # …and a local PostgreSQL
```

## API

Base path: `/api/v1`

| Area | Endpoints |
| --- | --- |
| Auth | `POST /auth/signup`, `POST /auth/login`, `POST /auth/google` |
| Chat | `WS /chat/ws`, `GET /chat/list`, `GET /chat/history/{id}`, `POST /chat/upload`, `DELETE /chat/{id}` |
| Media | `POST /media/generate`, `/media/generate-image`, `/media/generate-avatar`, `POST /media/tts/{message_id}` |
| Wallet | `GET /users/me`, `GET /packages/`, `POST /payments/create-checkout-session/{package_id}` |
| Admin | `GET /admin/stats/overview`, `GET /admin/users`, `PATCH /admin/users/{id}` |
| Meta | `GET /health`, `GET /health/live`, `GET /api/v1/models` |

`GET /health` reports database and Redis connectivity and returns 503 when either is down;
`/health/live` is a dependency-free liveness probe.

### Chat WebSocket

Connect to `/api/v1/chat/ws?token=<jwt>&model=auto&chat_id=<optional>` and send:

```json
{ "type": "user_message", "content": "Explain async/await", "attachments": [] }
```

The server streams `{"type": "content", "delta": "…"}` and emits `system` events for the chat
id, the model chosen by the router, and the cost of the turn. Model ids are validated against
the registry — `GET /api/v1/models` lists what can be requested; `auto` lets the router decide.

## Development

```bash
pip install -r requirements-dev.txt
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

Pushes to `main` run lint and tests, build the image and publish it to GHCR, then deploy over
SSH and verify `/health` before finishing (`.github/workflows/deploy.yml`).
[`render.yaml`](render.yaml) describes an equivalent Render deployment.

Before going live: set `ENVIRONMENT=production`, use a strong unique `SECRET_KEY`, restrict
`CORS_ORIGINS` to your own domains, terminate TLS at the proxy, configure the Stripe webhook
signing secret, and run `alembic upgrade head` as part of the release.

## License

Proprietary. All rights reserved © MultiAIModel.
