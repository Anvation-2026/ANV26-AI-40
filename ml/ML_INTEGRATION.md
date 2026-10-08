# MedGuard AI — Machine Learning Integration Contract
### Team Vouken | AI/ML Engineering & Backend Handoff

This document defines the interface contract between the `ml/` sub-system and Samuel's FastAPI backend service.

---

## 1. Quick Integration for Samuel (`backend/services/ml_service.py`)

```python
import sys
from pathlib import Path

# Add ml root to Python path
ML_ROOT = Path(__file__).resolve().parent.parent.parent / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.predict import (
    get_predictor,
    predict_image,
    model_status,
    get_validation_report,
    ModelUnavailableError,
)

# 1. Health check & model status
status = model_status()
if not status["available"]:
    # Handle model unavailable
    pass

# 2. Predict on image (accepts bytes, Path, str, PIL.Image, or np.ndarray)
# Thread-safe, non-blocking lock, deterministic, never raises on bad user files
with open("upload.png", "rb") as f:
    raw_bytes = f.read()

result = predict_image(raw_bytes, generate_heatmap=True)

# 3. Read validation report for developer/clinical dashboard
val_report = get_validation_report()
```

---

## 2. Response Schema (Exact Specification)

Every call to `predict_image()` returns a standard, JSON-serializable Python dictionary with these keys guaranteed to be present:

| Key | Type | Description |
|---|---|---|
| `status` | string | One of: `"success"`, `"uncertain"`, `"poor_quality"`, `"ood_rejected"`, `"unsupported_file"` |
| `finding` | string \| null | `"normal"` or `"pneumonia"` when status is `"success"`. `null` otherwise. |
| `probability` | float \| null | Calibrated probability of the finding (in [0, 1]). `null` when status is not `"success"`. |
| `uncertainty` | string \| null | `"low"`, `"medium"`, or `"high"`. High when status is `"uncertain"`. |
| `quality` | string \| null | `"acceptable"` or `"poor"`. `null` only for `"unsupported_file"`. |
| `ood` | bool \| null | `true` if out-of-distribution or non-X-ray; `null` only for `"unsupported_file"`. |
| `explanation` | string | Clinical decision-support narrative grounded in computed metrics. |
| `evidence` | list[string] | Structured list of 4–7 factual statements extracted from pipeline stages. |
| `heatmap_path` | string \| null | Relative path inside `ml/outputs/` (e.g., `"heatmaps/<uuid>.png"`). `null` for rejected images. |
| `model_version` | string | Version identifier (e.g. `resnet18-pneumoniamnist224-v1+9bda2caa`). |
| `limitations` | list[string] | Standard required regulatory & educational disclaimers. |
| `details` | object | Complete diagnostic and telemetry breakdown. |

### Details Object Format
```json
{
  "p_pneumonia_raw": 0.942,
  "p_pneumonia_calibrated": 0.915,
  "confidence": 0.915,
  "uncertainty_score": 0.28,
  "temperature": 1.15,
  "calibrated": true,
  "thresholds": {
    "tau_accept": 0.80,
    "tau_low_uncertainty": 0.95,
    "decision_threshold": 0.5
  },
  "quality": {
    "status": "acceptable",
    "metrics": {
      "blur_laplacian_var": 124.5,
      "brightness_mean": 0.482,
      "contrast_p99_p1": 0.814,
      "noise_sigma": 0.012
    },
    "reasons": []
  },
  "ood": {
    "flagged": false,
    "method": "mahalanobis+modality",
    "score": 14.2,
    "threshold": 32.1,
    "reasons": []
  },
  "rejection_reason": null,
  "inference_ms": 14.8,
  "device": "cuda",
  "capabilities": {
    "calibration": true,
    "uncertainty": true,
    "quality": true,
    "ood": true,
    "gradcam": true
  }
}
```

---

## 3. Serving Heatmaps & Static Assets

Mount `ml/outputs/` in FastAPI as a static mount:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="ml/outputs"), name="static")
```

When `result["heatmap_path"]` is returned (e.g. `"heatmaps/c735d4fa-....png"`), expose the URL as:
```python
heatmap_url = f"/static/{result['heatmap_path']}"
```

---

## 4. Recommended Status to Triage Mapping (For Samuel's Triage Rule Engine)

| ML Status | Uncertainty / Quality / OOD | Recommended Triage Action |
|---|---|---|
| `success` | `uncertainty == "low"` | Standard educational presentation with limitations. |
| `success` | `uncertainty == "medium"` | Expert review recommended. |
| `uncertain` | `uncertainty == "high"` | Abstain from prediction; secondary radiologist review required. |
| `poor_quality` | `quality == "poor"` | Request new radiograph; image fails diagnostic clarity criteria. |
| `ood_rejected` | `ood == True` | Unsupported image / non-chest-radiograph input rejected. |
| `unsupported_file` | File undecodable | Invalid or corrupted image payload. |

---

## 5. Thread Safety & Latency

- `MedGuardPredictor.predict()` uses an internal `threading.Lock()` protecting GPU tensors and hooks.
- Thread-safe for multi-worker FastAPI servers (`uvicorn`).
- Typical inference latency on RTX 3050 GPU: ~12–25 ms per image including full Grad-CAM synthesis.
