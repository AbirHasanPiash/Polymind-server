"""Access control at the HTTP surface.

These are regression tests for the most serious defect found in the audit: the
admin routers shipped with no authentication dependency at all.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ADMIN_ROUTES = [
    ("GET", "/api/v1/admin/stats/overview"),
    ("GET", "/api/v1/admin/users"),
    ("PATCH", "/api/v1/admin/users/00000000-0000-4000-8000-000000000000"),
    ("DELETE", "/api/v1/admin/users/00000000-0000-4000-8000-000000000000"),
]

USER_ROUTES = [
    ("GET", "/api/v1/users/me"),
    ("PATCH", "/api/v1/users/me"),
    ("POST", "/api/v1/users/me/password"),
    ("DELETE", "/api/v1/users/me"),
    ("GET", "/api/v1/users/me/usage"),
    ("POST", "/api/v1/users/topup?amount=100"),
    ("POST", "/api/v1/auth/refresh"),
    ("GET", "/api/v1/chat/list"),
    ("GET", "/api/v1/chat/search?q=hello"),
    ("GET", "/api/v1/chat/00000000-0000-4000-8000-000000000000/export"),
    ("POST", "/api/v1/chat/00000000-0000-4000-8000-000000000000/share"),
    ("GET", "/api/v1/media/list"),
    ("POST", "/api/v1/media/generate"),
    ("GET", "/api/v1/packages/"),
    ("GET", "/api/v1/payments/history"),
]


class TestAuthenticationRequired:
    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
    def test_admin_routes_reject_anonymous_callers(self, client: TestClient, method, path):
        assert client.request(method, path, json={}).status_code == 401

    @pytest.mark.parametrize(("method", "path"), USER_ROUTES)
    def test_user_routes_reject_anonymous_callers(self, client: TestClient, method, path):
        assert client.request(method, path, json={}).status_code == 401

    @pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES + USER_ROUTES)
    def test_invalid_tokens_are_rejected(self, client: TestClient, method, path):
        response = client.request(
            method, path, json={}, headers={"Authorization": "Bearer forged.token.value"}
        )
        assert response.status_code == 401


class TestPublicEndpoints:
    def test_liveness_needs_no_dependencies(self, client: TestClient):
        assert client.get("/health/live").json() == {"status": "alive"}

    def test_root_reports_the_service(self, client: TestClient):
        assert client.get("/").json()["status"] == "running"

    def test_model_catalogue_is_public(self, client: TestClient):
        payload = client.get("/api/v1/models").json()
        models = payload["models"]
        assert models
        assert {"id", "provider", "display_name", "description", "tier", "pricing"} <= set(
            models[0]
        )
        assert payload["routing"]["default"] in {m["id"] for m in models}
        assert payload["default_effort"] in payload["effort_levels"]

    def test_feature_flags_are_public(self, client: TestClient):
        assert "features" in client.get("/api/v1/features").json()

    def test_voice_and_image_catalogues_are_public(self, client: TestClient):
        assert client.get("/api/v1/media/voices").json()["voices"]
        assert client.get("/api/v1/media/images/options").json()["models"]

    def test_security_headers_are_set(self, client: TestClient):
        response = client.get("/health/live")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Request-ID"]


class TestInputValidation:
    def test_signup_rejects_a_malformed_email(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/signup", json={"email": "not-an-email", "password": "sufficient1"}
        )
        assert response.status_code == 422

    def test_signup_rejects_a_short_password(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/signup", json={"email": "user@example.com", "password": "short"}
        )
        assert response.status_code == 422

    def test_unknown_paths_are_404(self, client: TestClient):
        assert client.get("/api/v1/does-not-exist").status_code == 404
