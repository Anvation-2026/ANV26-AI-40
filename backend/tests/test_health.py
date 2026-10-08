from starlette.testclient import TestClient
from backend.core.config import settings


def test_health_endpoint(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data
    assert data["mode"] in ("real", "demo")


def test_model_status_real_mode_stub(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "real"
    # With stub, available is False
    assert data["available"] is False
    assert data["inference_ready"] is False
    assert "ML module" in (data.get("message") or "")


def test_model_status_demo_mode(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "demo")
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "demo"
    assert data["available"] is True
    assert data["inference_ready"] is True
    assert "DEMO FIXTURE" in (data.get("model_name") or "")


def test_model_status_handles_ml_import_failure(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(settings, "ML_MODULE_PATH", "non_existent_module_path")
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["inference_ready"] is False
