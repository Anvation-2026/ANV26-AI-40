"""
Unit Tests for Fracture API Routes
MedGuard AI - Bone Fracture Analysis Subsystem
"""

import io
from PIL import Image
import pytest
from starlette.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_fracture_status_endpoint():
    """Verify GET /api/fracture/status returns expected contract."""
    response = client.get("/api/fracture/status")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "model_name" in data
    assert "supported_anatomies" in data
    assert isinstance(data["supported_anatomies"], list)


def test_fracture_model_info_endpoint():
    """Verify GET /api/fracture/model-info returns architecture details."""
    response = client.get("/api/fracture/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "backbone" in data
    assert data["backbone"] == "convnext_base"


def test_fracture_validation_endpoint():
    """Verify GET /api/fracture/validation returns valid status response."""
    response = client.get("/api/fracture/validation")
    assert response.status_code == 200


def test_fracture_predict_invalid_file_rejected():
    """Verify that uploading invalid text content is rejected as invalid input."""
    dummy_text = io.BytesIO(b"Not a valid medical radiograph")
    response = client.post(
        "/api/fracture/predict",
        files={"file": ("test.txt", dummy_text, "text/plain")}
    )
    assert response.status_code in [400, 415, 422]
    data = response.json()
    assert "status" in data or "detail" in data


def test_fracture_predict_graceful_handling_when_untrained():
    """
    Verify that if the model checkpoint is missing or not yet trained,
    the endpoint returns HTTP 503 with model_unavailable status rather than crashing or faking predictions.
    """
    # Create a small valid test PNG image
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/fracture/predict",
        files={"file": ("xray.png", buf, "image/png")}
    )
    # If checkpoint exists, it returns 200; if untrained, it returns 503
    assert response.status_code in [200, 503]
    data = response.json()
    if response.status_code == 503:
        assert data.get("status") == "model_unavailable"
        assert "not yet" in data.get("message", "").lower() or "unavailable" in data.get("message", "").lower()
    else:
        assert "finding" in data
