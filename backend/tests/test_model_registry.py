"""
Unit Tests for Model Registry Routes
MedGuard AI - Clinical Evidence & Triage Support System
"""

import pytest
from starlette.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_all_models(client):
    """Verifies that GET /api/models returns all 4 registered models."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_registered_models"] == 4
    assert "gpu_state" in data
    assert len(data["models"]) == 4

    model_ids = [m["model_id"] for m in data["models"]]
    assert "resnet18_pneumonia" in model_ids
    assert "resnet18_tb" in model_ids
    assert "densenet201_chest14" in model_ids
    assert "convnext_base_fracture" in model_ids


def test_get_single_model_details(client):
    """Verifies GET /api/models/{model_id} returns accurate specs."""
    resp = client.get("/api/models/convnext_base_fracture")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "convnext_base_fracture"
    assert data["architecture"] == "ConvNeXt-Base + 3-Stage GELU MLP Head (1024 -> 512 -> 256 -> 1)"
    assert data["parameter_count"] == "88,222,849"
    assert "wrist" in data["supported_anatomies"]
    assert data["decision_threshold"] == 0.52


def test_get_invalid_model_returns_404(client):
    """Verifies GET /api/models/unknown returns 404."""
    resp = client.get("/api/models/nonexistent_model")
    assert resp.status_code == 404


def test_get_gpu_memory_telemetry(client):
    """Verifies GET /api/models/system/gpu returns memory structure."""
    resp = client.get("/api/models/system/gpu")
    assert resp.status_code == 200
    data = resp.json()
    assert "cuda_available" in data
    assert "device_name" in data
