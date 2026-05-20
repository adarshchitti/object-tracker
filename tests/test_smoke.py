"""Smoke tests: verify the API starts and exposes the expected endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})
    assert "/tasks" in paths, "POST /tasks missing from OpenAPI schema"
    assert "/tasks/{task_id}" in paths, "GET /tasks/{task_id} missing from OpenAPI schema"
    assert "/tasks/{task_id}/result" in paths, "GET /tasks/{task_id}/result missing from OpenAPI schema"
