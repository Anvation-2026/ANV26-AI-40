"""
ml/interface.py — Adapter conforming to backend/services/ml_service.py contract.
Exposes get_model_info() and predict() for seamless FastAPI backend integration.
"""
import sys
from pathlib import Path
from typing import Any, Dict
from PIL import Image

_ML_ROOT = Path(__file__).resolve().parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from src.predict import (
    get_predictor,
    predict_image,
    model_status,
    ModelUnavailableError,
)


def get_model_info() -> Dict[str, Any]:
    """
    Returns model health and capability status dictionary.
    Safe against exceptions.
    """
    try:
        status = model_status()
        caps = status.get("capabilities", {}) or {}
        return {
            "available": bool(status.get("available", False)),
            "model_name": status.get("model_name"),
            "model_version": status.get("model_version"),
            "supported_modality": status.get("supported_modality", "Chest X-ray (educational)"),
            "calibration_available": bool(caps.get("calibration", False)),
            "ood_available": bool(caps.get("ood", False)),
            "quality_available": bool(caps.get("quality", False)),
            "gradcam_available": bool(caps.get("gradcam", False)),
            "message": status.get("error") if not status.get("available") else None,
        }
    except Exception as exc:
        return {
            "available": False,
            "model_name": None,
            "model_version": None,
            "supported_modality": "Chest X-ray (educational)",
            "calibration_available": False,
            "ood_available": False,
            "quality_available": False,
            "gradcam_available": False,
            "message": str(exc),
        }


def predict(image: Any) -> Dict[str, Any]:
    """
    Executes full ML diagnostic pipeline and adapts output to MLResult schema.
    """
    raw = predict_image(image, generate_heatmap=True, include_base64=True)
    details = raw.get("details", {}) or {}

    # 1. Quality structure adaptation
    quality_dict = None
    q_raw = details.get("quality")
    if q_raw and isinstance(q_raw, dict):
        metrics = q_raw.get("metrics", {}) or {}
        thresholds = q_raw.get("thresholds", {}) or {}
        passed = q_raw.get("passed", {}) or {}

        def _make_check(key_metric: str, key_thresh: str):
            val = metrics.get(key_metric)
            thr = thresholds.get(key_thresh)
            pas = passed.get(key_metric)
            return {
                "value": float(val) if val is not None else None,
                "threshold": float(thr) if thr is not None else None,
                "passed": bool(pas) if pas is not None else None,
            }

        quality_dict = {
            "evaluated": True,
            "status": "poor" if q_raw.get("status") == "poor" else "acceptable",
            "blur": _make_check("blur_laplacian_var", "blur_min"),
            "brightness": _make_check("brightness_mean", "brightness_min"),
            "contrast": _make_check("contrast_p99_p1", "contrast_min"),
            "reasons": q_raw.get("reasons", []) or [],
        }
    else:
        quality_dict = {
            "evaluated": False,
            "status": "not_evaluated",
            "blur": None,
            "brightness": None,
            "contrast": None,
            "reasons": [],
        }

    # 2. OOD structure adaptation
    ood_dict = None
    ood_raw = details.get("ood")
    if ood_raw and isinstance(ood_raw, dict):
        score_val = ood_raw.get("score")
        ood_dict = {
            "evaluated": True,
            "is_ood": bool(ood_raw.get("flagged", False)),
            "score": float(score_val) if score_val is not None else None,
            "method": str(ood_raw.get("method", "mahalanobis+modality")),
            "reason": ", ".join(ood_raw.get("reasons", [])) if ood_raw.get("reasons") else None,
        }
    else:
        ood_dict = {
            "evaluated": False,
            "is_ood": None,
            "score": None,
            "method": None,
            "reason": None,
        }

    # 3. Uncertainty structure adaptation (map medium -> moderate for Pydantic enum)
    unc_level_raw = raw.get("uncertainty")
    if unc_level_raw == "medium":
        unc_level = "moderate"
    elif unc_level_raw in ["low", "moderate", "high"]:
        unc_level = unc_level_raw
    else:
        unc_level = "not_evaluated"

    unc_val = details.get("uncertainty_score")
    uncertainty_dict = {
        "level": unc_level,
        "value": float(unc_val) if unc_val is not None else None,
        "method": "normalized_entropy" if unc_level != "not_evaluated" else None,
    }

    # 4. Grad-CAM Heatmap
    heatmap_b64 = details.get("heatmap_base64")

    # 5. Finding & Abstained logic
    raw_status = raw.get("status", "error")
    is_success = (raw_status == "success")
    finding = raw.get("finding") if is_success else None

    # Calibrated probability & raw score
    p_cal = raw.get("probability")
    if p_cal is None and details.get("p_pneumonia_calibrated") is not None:
        p_pneumonia_cal = details.get("p_pneumonia_calibrated")
        p_cal = (1.0 - p_pneumonia_cal) if finding == "normal" else p_pneumonia_cal
    elif p_cal is None:
        p_cal = details.get("p_pneumonia_calibrated")

    p_raw = details.get("p_pneumonia_raw")
    prob_of = "pneumonia" if finding == "pneumonia" else ("normal" if finding == "normal" else "pneumonia")

    return {
        "finding": finding,
        "abstained": not is_success,
        "raw_score": float(p_raw) if p_raw is not None else None,
        "calibrated_probability": float(p_cal) if p_cal is not None else None,
        "probability_of": prob_of,
        "uncertainty": uncertainty_dict,
        "quality": quality_dict,
        "ood": ood_dict,
        "heatmap_png_base64": heatmap_b64,
        "heatmap_kind": "overlay" if heatmap_b64 else None,
        "model_version": raw.get("model_version"),
        "evidence_notes": raw.get("evidence", []) or [],
    }
