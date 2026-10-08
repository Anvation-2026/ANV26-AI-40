# MedGuard AI — API Contract Specification

This document details the HTTP REST API contracts for MedGuard AI.

## Base URL
All API endpoints are mounted at `/api`.

## Endpoints

### 1. `GET /api/health`
Checks backend liveness and basic runtime mode.
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "mode": "real",
  "version": "1.0.0",
  "ml_module_loaded": false,
  "timestamp": "2026-10-08T10:00:00Z"
}
```

### 2. `GET /api/model/status`
Checks the readiness and operational metadata of the ML decision-support module.
- **Response `200 OK`**:
```json
{
  "available": false,
  "model_name": null,
  "model_version": null,
  "supported_image_type": "Chest X-ray (educational)",
  "inference_ready": false,
  "calibration_available": false,
  "ood_available": false,
  "quality_available": false,
  "gradcam_available": false,
  "mode": "real",
  "message": "ML module not yet integrated"
}
```

### 3. `POST /api/predict`
Multipart form upload with field `file`.
- **Headers**:
  - `Content-Type: multipart/form-data`
  - `X-Demo-Scenario` *(optional, demo mode only)*: `success_pneumonia | success_normal | uncertain | poor_quality | ood`
- **Response Status Codes**:
  - `200 OK`: Successful triage resolution (including `success`, `uncertain`, `poor_quality`, `ood`).
  - `400 Bad Request`: `invalid_input` (empty file, corrupt image, min/max dimension violation).
  - `413 Payload Too Large`: Upload exceeds maximum size (10 MB).
  - `415 Unsupported Media Type`: Unsupported or mismatched file signature (not PNG, JPEG, or WEBP).
  - `503 Service Unavailable`: `model_unavailable` when the ML module is not loaded or ready.
  - `500 Internal Server Error`: `error` for internal runtime exceptions.
- **Response Body**: Always conforms to `AnalysisResponse` schema regardless of HTTP status code.

```json
{
  "request_id": "c1f7b889-4bc5-4702-86f3-f5c71d3df85e",
  "status": "success",
  "mode": "real",
  "finding": "pneumonia",
  "abstained": false,
  "raw_score": 0.884,
  "probability": 0.871,
  "probability_of": "pneumonia",
  "uncertainty": {
    "level": "low",
    "value": 0.042,
    "method": "MC-Dropout (10 samples)"
  },
  "quality": {
    "evaluated": true,
    "status": "acceptable",
    "blur": {"value": 240.5, "threshold": 100.0, "passed": true},
    "brightness": {"value": 128.0, "threshold": 40.0, "passed": true},
    "contrast": {"value": 65.2, "threshold": 20.0, "passed": true},
    "reasons": []
  },
  "ood": {
    "evaluated": true,
    "is_ood": false,
    "score": 0.12,
    "method": "Mahalanobis distance",
    "reason": null
  },
  "heatmap": {
    "available": true,
    "data_url": "data:image/png;base64,iVBORw0KGgoAAA...",
    "kind": "overlay",
    "message": null
  },
  "triage": {
    "action": "expert_review_recommended",
    "title": "Educational Review Recommended",
    "message": "The educational model identified patterns consistent with pneumonia.",
    "reasons": ["Pattern match detected for pneumonia class"]
  },
  "explanation": "The model assigned the image to the 'pneumonia' class with calibrated probability 87.1%. Human expert interpretation is required.",
  "evidence": [
    "Image quality passed acceptable criteria (blur=240.5, brightness=128.0, contrast=65.2)",
    "Image determined to be in-distribution (Chest X-ray)",
    "Low model uncertainty (value=0.042)"
  ],
  "limitations": [
    "Educational research prototype: not a medical device and not certified for clinical diagnostic use.",
    "Trained on a limited public dataset (PneumoniaMNIST+) and may not generalize across clinical environments.",
    "Calibrated probability does not guarantee clinical certainty or absence of alternative pathology.",
    "Grad-CAM visual overlays highlight regions influencing model predictions and do not confirm anatomical abnormalities.",
    "Image quality and out-of-distribution detectors are heuristic and imperfect.",
    "Qualified human expert interpretation is strictly required for any clinical context."
  ],
  "model": {
    "available": true,
    "model_name": "ResNet-18 (fine-tuned)",
    "model_version": "resnet18-v1",
    "dataset": "PneumoniaMNIST+",
    "calibration_available": true
  },
  "error": null,
  "disclaimer": "Educational research prototype. Not a medical device. Not for diagnosis or treatment decisions. Use only public or de-identified educational images."
}
```

### 4. `GET /api/validation`
Fetches the current validation metrics report produced by the ML engineering team.
- **Response `200 OK`**:
```json
{
  "status": "available",
  "report": {
    "model_name": "ResNet-18 (fine-tuned)",
    "model_version": "resnet18-v1",
    "dataset": "PneumoniaMNIST+",
    "evaluated_on": "test",
    "split_counts": {
      "train": 4708,
      "validation": 524,
      "test": 624
    },
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
    "limitations": [
      "Evaluated on public PneumoniaMNIST+ test partition only.",
      "Sensitivity may vary across pediatric versus adult demographics."
    ],
    "generated_at": "2026-10-08T09:30:00Z"
  },
  "message": null
}
```
If the report file does not exist, status is `"pending"`. If the file is malformed, status is `"error"`.
