"""Transactional email.

SMTP is optional: without it the message is logged instead of sent, which keeps
development and CI simple and makes a misconfigured mailer visible in the logs
rather than silently swallowing password resets.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from email.message import EmailMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


def _send_sync(to: str, subject: str, text: str, html: str | None) -> None:
    message = EmailMessage()
    message["From"] = settings.EMAIL_FROM
    message["To"] = to
    message["Subject"] = subject
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")

    if settings.SMTP_USE_SSL:
        server: smtplib.SMTP = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20)
    else:
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20)
    with server:
        if not settings.SMTP_USE_SSL and settings.SMTP_USE_TLS:
            server.starttls()
        if settings.SMTP_USERNAME:
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
        server.send_message(message)


async def send_email(to: str, subject: str, text: str, html: str | None = None) -> bool:
    """Send an email. Returns True when it left through SMTP, False when logged."""
    if not settings.email_enabled:
        logger.warning("Email not configured; would send to %s: %s\n%s", to, subject, text)
        return False
    try:
        await asyncio.to_thread(_send_sync, to, subject, text, html)
        return True
    except Exception:
        logger.exception("Failed to send email to %s", to)
        return False


def password_reset_email(reset_url: str) -> tuple[str, str, str]:
    subject = f"Reset your {settings.PROJECT_NAME} password"
    text = (
        f"We received a request to reset the password for your {settings.PROJECT_NAME} account.\n\n"
        f"Open this link to choose a new password (valid for "
        f"{settings.PASSWORD_RESET_TOKEN_MINUTES} minutes):\n{reset_url}\n\n"
        "If you did not ask for this, you can ignore this email."
    )
    html = (
        f"<p>We received a request to reset the password for your {settings.PROJECT_NAME} account.</p>"
        f'<p><a href="{reset_url}">Choose a new password</a> (valid for '
        f"{settings.PASSWORD_RESET_TOKEN_MINUTES} minutes).</p>"
        "<p>If you did not ask for this, you can ignore this email.</p>"
    )
    return subject, text, html
