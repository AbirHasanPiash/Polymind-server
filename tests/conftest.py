"""Test configuration.

Settings are validated at import time, so the environment has to be populated
before anything from ``app`` is imported. Defaults are only filled in when the
variable is missing, which lets a developer run the suite against their own .env.
"""

from __future__ import annotations

import os

_TEST_ENV = {
    "PROJECT_NAME": "Polymind Test",
    "API_V1_STR": "/api/v1",
    "ENVIRONMENT": "development",
    "SECRET_KEY": "test-secret-key-that-is-definitely-long-enough-32",
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "60",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test",
    "REDIS_URL": "redis://localhost:6379/15",
    "ALLOW_DEV_TOPUP": "false",
}

for key, value in _TEST_ENV.items():
    os.environ.setdefault(key, value)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    """HTTP client that surfaces handler errors instead of masking them as 500s."""
    return TestClient(app, raise_server_exceptions=False)
