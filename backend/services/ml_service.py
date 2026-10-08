import asyncio
import importlib
from typing import Any, Dict, Optional
from PIL import Image

from backend.core.config import settings
from backend.core.logging_config import logger
from backend.schemas.analysis import MLResult


class MLServiceError(Exception):
    def __init__(self, message: str, code: str = "ML_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class MLUnavailableError(MLServiceError):
    def __init__(self, message: str = "ML decision-support module is currently unavailable."):
        super().__init__(message, code="MODEL_UNAVAILABLE")


class MLTimeoutError(MLServiceError):
    def __init__(self, message: str = "Inference request timed out."):
        super().__init__(message, code="INFERENCE_TIMEOUT")


def _get_ml_module():
    """Lazily load the ML contract module specified in settings."""
    try:
        mod = importlib.import_module(settings.ML_MODULE_PATH)
        return mod
    except Exception as exc:
        logger.error("Failed to import ML module from %s: %s", settings.ML_MODULE_PATH, exc)
        return None


def get_status() -> Dict[str, Any]:
    """
    Query the ML module status cheaply without running inference.
    Catches all exceptions so the server never crashes.
    """
    mod = _get_ml_module()
    if mod is None:
        return {
            "available": False,
            "model_name": None,
            "model_version": None,
            "supported_image_type": "Chest X-ray (educational)",
            "inference_ready": False,
            "calibration_available": False,
            "ood_available": False,
            "quality_available": False,
            "gradcam_available": False,
            "message": "ML module not loaded or not configured",
        }

    try:
        info_fn = getattr(mod, "get_model_info", None)
        if not callable(info_fn):
            logger.warning("ML module does not export callable 'get_model_info'")
            return {
                "available": False,
                "model_name": None,
                "model_version": None,
                "supported_image_type": "Chest X-ray (educational)",
                "inference_ready": False,
                "calibration_available": False,
                "ood_available": False,
                "quality_available": False,
                "gradcam_available": False,
                "message": "ML interface contract not satisfied",
            }

        raw_info = info_fn()
        if not isinstance(raw_info, dict):
            return {
                "available": False,
                "model_name": None,
                "model_version": None,
                "supported_image_type": "Chest X-ray (educational)",
                "inference_ready": False,
                "calibration_available": False,
                "ood_available": False,
                "quality_available": False,
                "gradcam_available": False,
                "message": "Invalid response format from ML module",
            }

        available = bool(raw_info.get("available", False))
        return {
            "available": available,
            "model_name": raw_info.get("model_name"),
            "model_version": raw_info.get("model_version"),
            "supported_image_type": raw_info.get("supported_modality", "Chest X-ray (educational)"),
            "inference_ready": available,
            "calibration_available": bool(raw_info.get("calibration_available", False)),
            "ood_available": bool(raw_info.get("ood_available", False)),
            "quality_available": bool(raw_info.get("quality_available", False)),
            "gradcam_available": bool(raw_info.get("gradcam_available", False)),
            "message": raw_info.get("message") if not available else None,
        }
    except Exception as exc:
        logger.error("Error executing get_model_info(): %s", exc)
        return {
            "available": False,
            "model_name": None,
            "model_version": None,
            "supported_image_type": "Chest X-ray (educational)",
            "inference_ready": False,
            "calibration_available": False,
            "ood_available": False,
            "quality_available": False,
            "gradcam_available": False,
            "message": "ML module check failed",
        }


async def run_inference(image: Image.Image) -> MLResult:
    """
    Run inference via the ML module with a strict timeout.
    Validates output into a tolerant MLResult schema.
    """
    mod = _get_ml_module()
    if mod is None:
        raise MLUnavailableError("ML module is not available.")

    status = get_status()
    if not status["available"]:
        raise MLUnavailableError(status.get("message") or "ML module is currently unavailable.")

    predict_fn = getattr(mod, "predict", None)
    if not callable(predict_fn):
        raise MLUnavailableError("ML module does not export predict function.")

    try:
        raw_result = await asyncio.wait_for(
            asyncio.to_thread(predict_fn, image),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error("Inference timed out after %s seconds", settings.INFERENCE_TIMEOUT_SECONDS)
        raise MLTimeoutError("Inference timed out.")
    except NotImplementedError:
        logger.warning("ML module predict() raised NotImplementedError")
        raise MLUnavailableError("ML inference logic is not yet implemented.")
    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        raise MLServiceError("ML inference failed.")

    if not isinstance(raw_result, dict):
        logger.error("Inference returned non-dict type: %s", type(raw_result))
        raise MLServiceError("ML module returned unexpected output structure.")

    # Validate into Pydantic MLResult
    try:
        return MLResult.model_validate(raw_result)
    except Exception as exc:
        logger.error("Failed to parse MLResult from output: %s", exc)
        raise MLServiceError("Failed to parse ML inference result.")
