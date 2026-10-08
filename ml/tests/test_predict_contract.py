import json
from pathlib import Path
import numpy as np
import pytest

from config import DATASET_PATH
from src.dataset import get_split
from src.predict import (
    MedGuardPredictor,
    ModelUnavailableError,
    get_predictor,
    model_status,
    predict_image,
)

REQUIRED_KEYS = [
    "status",
    "finding",
    "probability",
    "uncertainty",
    "quality",
    "ood",
    "explanation",
    "evidence",
    "heatmap_path",
    "model_version",
    "limitations",
    "details",
]

BANNED_WORDS = ["diagnosed with", "you have", "treatment", "confirmed finding"]


def test_predict_contract_keys_and_json_serializable():
    val_images, _ = get_split("val", DATASET_PATH)
    clean_img = val_images[0]

    result = predict_image(clean_img, generate_heatmap=False)

    for k in REQUIRED_KEYS:
        assert k in result, f"Missing required contract key: {k}"

    # Verify JSON serializability
    json_str = json.dumps(result)
    assert json_str is not None

    # Check banned words
    for banned in BANNED_WORDS:
        assert banned not in result["explanation"].lower(), f"Banned phrase '{banned}' in explanation"


def test_predict_deterministic():
    val_images, _ = get_split("val", DATASET_PATH)
    img = val_images[1]

    r1 = predict_image(img, generate_heatmap=False)
    r2 = predict_image(img, generate_heatmap=False)

    assert r1["status"] == r2["status"]
    assert r1["probability"] == r2["probability"]
    assert r1["details"]["p_pneumonia_calibrated"] == r2["details"]["p_pneumonia_calibrated"]


def test_predict_unsupported_file_garbage_bytes():
    res = predict_image(b"random corrupted bytes not an image")
    assert res["status"] == "unsupported_file"
    assert res["finding"] is None
    assert res["probability"] is None
    assert res["heatmap_path"] is None


def test_model_status_never_raises():
    status = model_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "device" in status
    assert "capabilities" in status


def test_missing_model_raises_model_unavailable_error(tmp_path):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    with pytest.raises(ModelUnavailableError):
        MedGuardPredictor(models_dir=empty_dir)
