"""Central logging configuration.

Development gets human-readable lines; production gets one JSON object per line
so log aggregators (CloudWatch, Loki, Datadog…) can parse them without regexes.
"""

from __future__ import annotations

import json
import logging
import logging.config
from typing import Any

from app.core.config import settings

_NOISY_LOGGERS = ("botocore", "boto3", "urllib3", "httpx", "httpcore", "asyncio")


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter — no third-party dependency required."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    formatter = "json" if settings.is_production else "console"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": formatter,
                }
            },
            "root": {"handlers": ["default"], "level": settings.LOG_LEVEL},
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": settings.LOG_LEVEL, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": settings.LOG_LEVEL, "propagate": False},
                "sqlalchemy.engine": {"level": "WARNING"},
                **{name: {"level": "WARNING"} for name in _NOISY_LOGGERS},
            },
        }
    )
