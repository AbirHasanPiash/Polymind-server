"""Token issuing/verification and password hashing."""

from __future__ import annotations

from datetime import timedelta

from jose import jwt

from app.core.config import settings
from app.core.security import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)


class TestAccessTokens:
    def test_round_trip(self):
        payload = decode_token(create_access_token("user@example.com"))
        assert payload is not None
        assert payload["sub"] == "user@example.com"
        assert payload["type"] == "access"

    def test_expired_token_is_rejected(self):
        token = create_access_token("user@example.com", expires_delta=timedelta(seconds=-1))
        assert decode_token(token) is None

    def test_token_signed_with_another_key_is_rejected(self):
        forged = jwt.encode(
            {"sub": "attacker@example.com", "exp": 9_999_999_999},
            "a-different-secret-key-entirely-0123456789",
            algorithm=settings.ALGORITHM,
        )
        assert decode_token(forged) is None

    def test_garbage_is_rejected(self):
        assert decode_token("not.a.token") is None
        assert decode_token("") is None

    def test_token_without_a_subject_is_rejected(self):
        token = jwt.encode(
            {"exp": 9_999_999_999}, settings.SECRET_KEY, algorithm=settings.ALGORITHM
        )
        assert decode_token(token) is None

    def test_tokens_issued_before_the_type_claim_still_work(self):
        """Existing sessions must survive a deploy that adds new claims."""
        legacy = jwt.encode(
            {"sub": "old@example.com", "exp": 9_999_999_999},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        assert decode_token(legacy)["sub"] == "old@example.com"

    def test_tokens_are_unique_per_issue(self):
        first = create_access_token("user@example.com")
        second = create_access_token("user@example.com")
        assert decode_token(first)["jti"] != decode_token(second)["jti"]


class TestPasswords:
    def test_hash_and_verify(self):
        hashed = get_password_hash("correct horse battery staple")
        assert hashed != "correct horse battery staple"
        assert verify_password("correct horse battery staple", hashed)
        assert not verify_password("wrong password", hashed)

    def test_oauth_accounts_without_a_password_cannot_log_in(self):
        """A null hash must fail closed, not raise or pass."""
        assert not verify_password("", None)
        assert not verify_password("anything", None)

    def test_corrupt_hash_fails_closed(self):
        assert not verify_password("anything", "not-a-real-hash")

    def test_hashes_are_salted(self):
        assert get_password_hash("same") != get_password_hash("same")
