from fastapi.testclient import TestClient

from pfas_lit_rag.api import app


def test_health_endpoint_reports_service_status() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "qwen2.5:3b"
