import pytest
from starlette.testclient import TestClient
from backend.core.config import settings
from backend.schemas.analysis import AnalysisResponse, AnalysisStatus
from backend.services import ml_service
from backend.services.ml_service import MLServiceError, MLTimeoutError


def test_predict_real_mode_with_stub_model_unavailable(client: TestClient, valid_png_bytes, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": False, "inference_ready": False, "message": "ML module not loaded"})
    files = {"file": ("xray.png", valid_png_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 503
    data = response.json()
    parsed = AnalysisResponse.model_validate(data)
    assert parsed.status == AnalysisStatus.MODEL_UNAVAILABLE
    assert parsed.mode == "real"
    assert parsed.finding is None
    assert parsed.probability is None


def test_predict_real_mode_success_pneumonia(client: TestClient, valid_png_bytes, fake_ml_doubles, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": True, "inference_ready": True, "model_name": "TestModel"})
    
    async def mock_inference(img):
        return fake_ml_doubles.pneumonia_success()

    monkeypatch.setattr(ml_service, "run_inference", mock_inference)

    files = {"file": ("chest.png", valid_png_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    parsed = AnalysisResponse.model_validate(data)
    assert parsed.status == AnalysisStatus.SUCCESS
    assert parsed.finding == "pneumonia"
    assert parsed.mode == "real"
    assert parsed.probability == 0.871
    assert parsed.heatmap.available is True


def test_predict_real_mode_poor_quality(client: TestClient, valid_png_bytes, fake_ml_doubles, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": True, "inference_ready": True})
    
    async def mock_inference(img):
        return fake_ml_doubles.poor_quality()

    monkeypatch.setattr(ml_service, "run_inference", mock_inference)

    files = {"file": ("chest.png", valid_png_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    parsed = AnalysisResponse.model_validate(data)
    assert parsed.status == AnalysisStatus.POOR_QUALITY
    assert parsed.finding is None
    assert parsed.heatmap.available is False


def test_predict_real_mode_ood(client: TestClient, valid_png_bytes, fake_ml_doubles, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": True, "inference_ready": True})
    
    async def mock_inference(img):
        return fake_ml_doubles.ood()

    monkeypatch.setattr(ml_service, "run_inference", mock_inference)

    files = {"file": ("chest.png", valid_png_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    parsed = AnalysisResponse.model_validate(data)
    assert parsed.status == AnalysisStatus.OOD
    assert parsed.finding is None
    assert parsed.heatmap.available is False


def test_predict_inference_exception_returns_500_no_traceback(client: TestClient, valid_png_bytes, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": True, "inference_ready": True})

    async def mock_inference(img):
        raise MLServiceError("Internal crash in C++ tensor engine")

    monkeypatch.setattr(ml_service, "run_inference", mock_inference)

    files = {"file": ("chest.png", valid_png_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 500
    data = response.json()
    parsed = AnalysisResponse.model_validate(data)
    assert parsed.status == AnalysisStatus.ERROR
    # Ensure no traceback or internal paths leaked
    assert "C++ tensor" not in parsed.explanation
    assert "Traceback" not in parsed.explanation


def test_predict_inference_timeout(client: TestClient, valid_png_bytes, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": True, "inference_ready": True})

    async def mock_inference(img):
        raise MLTimeoutError("Timed out after 30s")

    monkeypatch.setattr(ml_service, "run_inference", mock_inference)

    files = {"file": ("chest.png", valid_png_bytes, "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 500
    data = response.json()
    parsed = AnalysisResponse.model_validate(data)
    assert parsed.status == AnalysisStatus.ERROR
    assert "timed out" in parsed.explanation.lower()


def test_predict_demo_mode_scenarios(client: TestClient, valid_png_bytes, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "demo")

    scenarios = ["success_pneumonia", "success_normal", "uncertain", "poor_quality", "ood"]
    for sc in scenarios:
        files = {"file": ("chest.png", valid_png_bytes, "image/png")}
        response = client.post("/api/predict", files=files, headers={"X-Demo-Scenario": sc})
        assert response.status_code == 200
        data = response.json()
        parsed = AnalysisResponse.model_validate(data)
        assert parsed.mode == "demo"
        assert "DEMO FIXTURE" in parsed.model.model_name
        assert parsed.heatmap.available is False


def test_repeated_uploads_consecutively(client: TestClient, valid_png_bytes, fake_ml_doubles, monkeypatch):
    monkeypatch.setattr(settings, "MEDGUARD_MODE", "real")
    monkeypatch.setattr(ml_service, "get_status", lambda: {"available": True, "inference_ready": True})

    async def mock_inference(img):
        return fake_ml_doubles.normal_success()

    monkeypatch.setattr(ml_service, "run_inference", mock_inference)

    for i in range(3):
        files = {"file": (f"chest_{i}.png", valid_png_bytes, "image/png")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        parsed = AnalysisResponse.model_validate(data)
        assert parsed.status == AnalysisStatus.SUCCESS
        assert parsed.finding == "normal"
