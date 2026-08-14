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
    ("POST", "/api/v1/users/topup?amount=100"),
    ("GET", "/api/v1/chat/list"),
    ("GET", "/api/v1/media/list"),
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
        models = client.get("/api/v1/models").json()["models"]
        assert models
        assert {"id", "provider", "description"} <= set(models[0])


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
