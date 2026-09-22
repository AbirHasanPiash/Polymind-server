"""Settings parsing that has bitten deployments before."""

from __future__ import annotations

import pytest

from app.core.config import Settings

BASE = {
    "SECRET_KEY": "test-secret-key-that-is-definitely-long-enough-32",
    "DATABASE_URL": "postgresql+asyncpg://t:t@localhost:5432/t",
    "REDIS_URL": "redis://localhost:6379/15",
}


def _settings(**env: str) -> Settings:
    return Settings(_env_file=None, **{**BASE, **env})  # type: ignore[arg-type]


class TestCorsOrigins:
    def test_comma_separated_list_is_accepted(self):
        settings = _settings(CORS_ORIGINS="http://localhost:5173, https://app.example.com/")
        assert settings.CORS_ORIGINS == ["http://localhost:5173", "https://app.example.com"]

    def test_json_array_is_accepted(self):
        settings = _settings(CORS_ORIGINS='["http://a.test", "http://b.test"]')
        assert settings.CORS_ORIGINS == ["http://a.test", "http://b.test"]

    def test_defaults_include_local_development(self):
        assert "http://localhost:5173" in _settings().CORS_ORIGINS


class TestGuards:
    def test_short_secret_is_rejected(self):
        with pytest.raises(ValueError):
            _settings(SECRET_KEY="short")

    def test_sync_database_driver_is_rejected(self):
        with pytest.raises(ValueError):
            _settings(DATABASE_URL="postgresql://t:t@localhost/t")

    def test_dev_topup_cannot_be_enabled_in_production(self):
        with pytest.raises(ValueError):
            _settings(ENVIRONMENT="production", ALLOW_DEV_TOPUP="true")

    def test_public_features_reflect_configuration(self):
        features = _settings(OPENAI_API_KEY="sk-test").public_features()
        assert features["openai"] is True
        assert features["anthropic"] is False
