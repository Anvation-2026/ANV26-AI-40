import json
from pathlib import Path
from starlette.testclient import TestClient
from backend.core.config import settings


def test_validation_report_missing_file(client: TestClient, monkeypatch, tmp_path):
    non_existent = tmp_path / "missing_report.json"
    monkeypatch.setattr(settings, "VALIDATION_REPORT_PATH", str(non_existent))

    response = client.get("/api/validation")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["report"] is None
    assert "pending" in (data.get("message") or "").lower()


def test_validation_report_malformed_json(client: TestClient, monkeypatch, tmp_path):
    bad_json_file = tmp_path / "bad_report.json"
    bad_json_file.write_text("{ this is not valid json :", encoding="utf-8")
    monkeypatch.setattr(settings, "VALIDATION_REPORT_PATH", str(bad_json_file))

    response = client.get("/api/validation")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["report"] is None


def test_validation_report_partial_report_with_nulls(client: TestClient, monkeypatch, tmp_path):
    partial_data = {
        "model_name": "ResNet-18",
        "model_version": "v1",
        "dataset": "PneumoniaMNIST+",
        "evaluated_on": "test",
        "metrics": {
            "accuracy": None,
            "sensitivity": None,
            "specificity": None
        },
        "limitations": ["Small dataset"]
    }
    report_file = tmp_path / "partial_report.json"
    report_file.write_text(json.dumps(partial_data), encoding="utf-8")
    monkeypatch.setattr(settings, "VALIDATION_REPORT_PATH", str(report_file))

    response = client.get("/api/validation")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert data["report"]["metrics"]["accuracy"] is None
    assert data["report"]["metrics"]["sensitivity"] is None


def test_validation_report_full_report_passes_values(client: TestClient, monkeypatch, tmp_path):
    full_data = {
        "model_name": "ResNet-18",
        "model_version": "resnet18-v1",
        "dataset": "PneumoniaMNIST+",
        "evaluated_on": "test",
        "split_counts": {"train": 4708, "validation": 524, "test": 624},
        "metrics": {
            "accuracy": 0.892,
            "sensitivity": 0.915,
            "specificity": 0.865,
            "precision": 0.880,
            "recall": 0.915,
            "f1": 0.897,
            "auroc": 0.942,
            "false_negative_rate": 0.085,
            "ece": 0.041,
            "abstention_coverage": 0.920,
            "rejection_rate": 0.080
        },
        "confusion_matrix": {
            "labels": ["normal", "pneumonia"],
            "matrix": [[202, 32], [20, 370]]
        },
        "limitations": ["Educational evaluation only"],
        "generated_at": "2026-10-08T09:30:00Z"
    }
    report_file = tmp_path / "full_report.json"
    report_file.write_text(json.dumps(full_data), encoding="utf-8")
    monkeypatch.setattr(settings, "VALIDATION_REPORT_PATH", str(report_file))

    response = client.get("/api/validation")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert data["report"]["metrics"]["accuracy"] == 0.892
    assert data["report"]["metrics"]["auroc"] == 0.942
    assert data["report"]["confusion_matrix"]["matrix"] == [[202, 32], [20, 370]]
