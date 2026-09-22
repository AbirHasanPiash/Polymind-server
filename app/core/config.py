"""Application settings.

All configuration is read from the environment (or a local ``.env`` file) and
validated once at import time, so a misconfigured deployment fails fast with a
readable error instead of at the first request.

Third-party credentials are optional: a missing key disables that one feature
(and is reported at startup) rather than preventing the process from booting.
"""

from __future__ import annotations

import json
from functools import cached_property
from typing import Annotated, Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # Application
    PROJECT_NAME: str = "Polymind"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Environment = "development"
    LOG_LEVEL: str = "INFO"
    FRONTEND_URL: str = "http://localhost:5173"

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    # Sessions are refreshed by the client while it is in use (POST /auth/refresh),
    # so the hard expiry can stay reasonably short.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=240, gt=0)
    PASSWORD_RESET_TOKEN_MINUTES: int = Field(default=30, gt=0)
    # NoDecode: pydantic-settings would otherwise JSON-decode the variable before
    # the validator below can accept the comma-separated form.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default=[
            "https://polymindai.com",
            "https://www.polymindai.com",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
        ]
    )

    # Database
    DATABASE_URL: str
    DB_USE_NULL_POOL: bool = True  # required in front of PgBouncer / Supabase pooler
    DB_POOL_SIZE: int = Field(default=5, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0)
    DB_POOL_TIMEOUT: int = Field(default=30, gt=0)
    DB_POOL_RECYCLE: int = Field(default=1800, gt=0)
    DB_CONNECT_TIMEOUT: int = Field(default=30, gt=0)
    DB_COMMAND_TIMEOUT: int = Field(default=60, gt=0)
    DB_ECHO: bool = False

    # Redis (cache + Celery broker)
    REDIS_URL: str
    REDIS_MAX_CONNECTIONS: int = Field(default=20, ge=1)
    CACHE_TTL_SECONDS: int = Field(default=3600, gt=0)

    # Chat
    CHAT_HISTORY_LIMIT: int = Field(default=24, ge=1, le=200)
    CHAT_STREAM_TIMEOUT_SECONDS: int = Field(default=300, gt=0)
    CHAT_TITLE_GENERATION: bool = True
    MAX_UPLOAD_SIZE_MB: int = Field(default=10, gt=0)
    MAX_UPLOAD_FILES: int = Field(default=5, gt=0)

    # Auth providers
    GOOGLE_CLIENT_ID: str | None = None

    # LLM providers
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    DID_API_KEY: str | None = None
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None

    # Object storage (Cloudflare R2 / any S3-compatible service)
    STORAGE_ENDPOINT: str | None = None
    STORAGE_ACCESS_KEY: str | None = None
    STORAGE_SECRET_KEY: str | None = None
    STORAGE_BUCKET_NAME: str | None = None
    STORAGE_REGION: str = "auto"
    STORAGE_PUBLIC_URL: str | None = None

    # Payments
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_WEBHOOK_SECRET: str | None = None
    RAZORPAY_KEY_ID: str | None = None
    RAZORPAY_KEY_SECRET: str | None = None

    # Email (password resets). Optional: without SMTP the message is logged.
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_USE_TLS: bool = True
    SMTP_USE_SSL: bool = False
    EMAIL_FROM: str = "Polymind <no-reply@localhost>"

    # Media generation
    DID_POLL_INTERVAL_SECONDS: int = Field(default=5, gt=0)
    DID_POLL_MAX_ATTEMPTS: int = Field(default=100, gt=0)

    # Abuse protection (per window)
    RATE_LIMIT_LOGIN_PER_MINUTE: int = Field(default=10, ge=1)
    RATE_LIMIT_SIGNUP_PER_HOUR: int = Field(default=20, ge=1)
    RATE_LIMIT_MEDIA_PER_HOUR: int = Field(default=60, ge=1)
    RATE_LIMIT_UPLOADS_PER_HOUR: int = Field(default=120, ge=1)

    # Development helpers — must stay off in production (see validator below)
    ALLOW_DEV_TOPUP: bool = False

    @field_validator("SECRET_KEY")
    @classmethod
    def _secret_key_is_strong(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError(
                "SECRET_KEY must be at least 32 characters. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(48))'"
            )
        return value

    @field_validator("DATABASE_URL")
    @classmethod
    def _database_url_is_async(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver, "
                "e.g. postgresql+asyncpg://user:pass@host:5432/dbname"
            )
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def _log_level_is_valid(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"Invalid LOG_LEVEL: {value}")
        return level

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept a JSON array or a comma-separated string in the environment."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                return json.loads(text)
            return [origin.strip().rstrip("/") for origin in text.split(",") if origin.strip()]
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @cached_property
    def storage_enabled(self) -> bool:
        return all(
            (
                self.STORAGE_ENDPOINT,
                self.STORAGE_ACCESS_KEY,
                self.STORAGE_SECRET_KEY,
                self.STORAGE_BUCKET_NAME,
                self.STORAGE_PUBLIC_URL,
            )
        )

    @cached_property
    def stripe_enabled(self) -> bool:
        return bool(self.STRIPE_SECRET_KEY and self.STRIPE_WEBHOOK_SECRET)

    @cached_property
    def razorpay_enabled(self) -> bool:
        return bool(self.RAZORPAY_KEY_ID and self.RAZORPAY_KEY_SECRET)

    @cached_property
    def google_login_enabled(self) -> bool:
        return bool(self.GOOGLE_CLIENT_ID)

    @cached_property
    def email_enabled(self) -> bool:
        return bool(self.SMTP_HOST)

    @cached_property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def disabled_features(self) -> list[str]:
        """Human-readable list of features switched off by missing credentials."""
        checks = {
            "Google login (GOOGLE_CLIENT_ID)": self.google_login_enabled,
            "OpenAI models (OPENAI_API_KEY)": bool(self.OPENAI_API_KEY),
            "Anthropic models (ANTHROPIC_API_KEY)": bool(self.ANTHROPIC_API_KEY),
            "Gemini models (GOOGLE_API_KEY)": bool(self.GOOGLE_API_KEY),
            "Avatar video (DID_API_KEY)": bool(self.DID_API_KEY),
            "Object storage (STORAGE_*)": self.storage_enabled,
            "Stripe payments (STRIPE_*)": self.stripe_enabled,
            "Razorpay payments (RAZORPAY_*)": self.razorpay_enabled,
            "Email delivery (SMTP_*)": self.email_enabled,
        }
        return [name for name, enabled in checks.items() if not enabled]

    def public_features(self) -> dict[str, bool]:
        """Feature flags the client uses to hide what is not configured."""
        return {
            "google_login": self.google_login_enabled,
            "openai": bool(self.OPENAI_API_KEY),
            "anthropic": bool(self.ANTHROPIC_API_KEY),
            "google": bool(self.GOOGLE_API_KEY),
            "avatar_video": bool(self.DID_API_KEY),
            "storage": self.storage_enabled,
            "stripe": self.stripe_enabled,
            "razorpay": self.razorpay_enabled,
            "password_reset": True,
        }

    def model_post_init(self, __context: object) -> None:
        if self.is_production and self.ALLOW_DEV_TOPUP:
            raise ValueError("ALLOW_DEV_TOPUP must be disabled in production")


settings = Settings()
