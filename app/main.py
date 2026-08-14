"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.endpoints import (
    admin_stats,
    auth,
    chat,
    manage_user,
    media,
    packages,
    payments,
    users,
)
from app.core.config import settings
from app.core.database import check_db_connection, engine, warmup_db_connection
from app.core.logging import configure_logging
from app.core.redis import check_redis_connection, close_redis
from app.services.llm.base import ProviderNotConfiguredError
from app.services.llm.factory import LLMFactory
from app.services.llm.models import UnknownModelError

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the database on boot and release resources on shutdown."""
    logger.info("Starting %s (%s)", settings.PROJECT_NAME, settings.ENVIRONMENT)

    disabled = settings.disabled_features()
    if disabled:
        logger.warning("Disabled — missing configuration: %s", "; ".join(disabled))

    await warmup_db_connection()
    yield

    logger.info("Shutting down")
    await close_redis()
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
    # Interactive docs are useful in development and an attack surface map in
    # production, where the schema is served to authenticated tooling only.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id and log how long it took."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = request_id

    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Request-ID"] = request_id
    log = logger.warning if duration_ms > 3000 else logger.info
    log(
        "%s %s -> %s in %.0fms [%s]",
        request.method, request.url.path, response.status_code, duration_ms, request_id,
    )
    return response


# Error handlers — the client gets a stable shape, the logs get the detail.


@app.exception_handler(UnknownModelError)
async def unknown_model_handler(_: Request, exc: UnknownModelError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.exception_handler(ProviderNotConfiguredError)
async def provider_not_configured_handler(_: Request, exc: ProviderNotConfiguredError) -> JSONResponse:
    logger.error("Provider unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"{exc.provider} is temporarily unavailable"},
    )


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("Database error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The service is temporarily unavailable. Please try again."},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled error on %s [%s]", request.url.path, request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# Routes

api = settings.API_V1_STR
app.include_router(auth.router, prefix=f"{api}/auth", tags=["auth"])
app.include_router(users.router, prefix=f"{api}/users", tags=["users"])
app.include_router(chat.router, prefix=f"{api}/chat", tags=["chat"])
app.include_router(media.router, prefix=f"{api}/media", tags=["media"])
app.include_router(packages.router, prefix=f"{api}/packages", tags=["packages"])
app.include_router(payments.router, prefix=f"{api}/payments", tags=["payments"])
app.include_router(admin_stats.router, prefix=f"{api}/admin/stats", tags=["admin"])
app.include_router(manage_user.router, prefix=f"{api}/admin/users", tags=["admin"])


@app.get("/", tags=["meta"])
async def read_root() -> dict:
    return {
        "service": settings.PROJECT_NAME,
        "status": "running",
        "docs": None if settings.is_production else "/docs",
    }


@app.get("/health", tags=["meta"])
async def health() -> JSONResponse:
    """Readiness probe: reports the state of every backing service."""
    db_ok = await check_db_connection()
    redis_ok = await check_redis_connection()
    healthy = db_ok and redis_ok

    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if healthy else "degraded",
            "database": "connected" if db_ok else "disconnected",
            "redis": "connected" if redis_ok else "disconnected",
        },
    )


@app.get("/health/live", tags=["meta"])
async def liveness() -> dict:
    """Liveness probe: the process is up. No dependency checks."""
    return {"status": "alive"}


@app.get(f"{settings.API_V1_STR}/models", tags=["chat"])
async def list_models() -> dict:
    """Models the client may request, with their descriptions."""
    return {
        "models": [
            {"id": spec.id, "provider": spec.provider, "description": spec.description}
            for spec in LLMFactory.get_all_models()
        ]
    }
