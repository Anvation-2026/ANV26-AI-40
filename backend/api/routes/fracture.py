"""
FastAPI Fracture Prediction & Status Endpoints
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from backend.core.config import settings
from backend.services.image_validation import ImageValidationError, validate_and_load_image

logger = logging.getLogger("fracture_router")

router = APIRouter(prefix="/api/fracture", tags=["bone_fracture"])

# Lazy predictor instance
_PREDICTOR = None


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            from ml.tasks.fracture.predict import FracturePredictor
            _PREDICTOR = FracturePredictor()
        except Exception as e:
            logger.error(f"Failed to instantiate FracturePredictor: {e}")
            return None
    return _PREDICTOR


@router.get("/status")
async def get_fracture_status() -> JSONResponse:
    """Returns availability and operational status of the ConvNeXt-Base fracture model."""
    pred = get_predictor()
    if pred is None:
        return JSONResponse(status_code=200, content={
            "available": False,
            "status": "service_uninitialized",
            "message": "Fracture prediction service could not be initialized."
        })

    is_ready = pred.checkpoint_path.exists()
    return JSONResponse(status_code=200, content={
        "available": is_ready,
        "status": "ready" if is_ready else "checkpoint_not_found",
        "model_name": "ConvNeXt-Base Multi-Region Bone Fracture Classifier",
        "backbone": "convnext_base",
        "device": str(pred.device),
        "checkpoint_path": str(pred.checkpoint_path),
        "supported_anatomies": ["wrist", "hand", "leg", "hip", "shoulder", "mixed"],
        "message": "ConvNeXt-Base model is ready for analysis." if is_ready else "Model checkpoint is not yet trained."
    })


@router.get("/model-info")
async def get_model_info() -> JSONResponse:
    """Returns technical metadata, architecture specifications, and training configuration."""
    from ml.tasks.fracture.config import MODEL_CONFIG_JSON, LABEL_MAP
    if MODEL_CONFIG_JSON.exists():
        with open(MODEL_CONFIG_JSON, "r") as f:
            cfg = json.load(f)
        return JSONResponse(status_code=200, content=cfg)
    
    return JSONResponse(status_code=200, content={
        "model_name": "ConvNeXt-Base Multi-Region Bone Fracture Classifier",
        "backbone": "convnext_base",
        "pretrained_weights": "ConvNeXt_Base_Weights.DEFAULT (ImageNet-1K)",
        "feature_dim": 1024,
        "head_architecture": "1024 -> 512 -> 256 -> 1 (GELU, Dropout 0.3)",
        "loss": "BCEWithLogitsLoss",
        "supported_anatomies": ["wrist", "hand", "leg", "hip", "shoulder", "mixed"],
        "status": "training_not_completed"
    })


@router.get("/validation")
async def get_validation_report() -> JSONResponse:
    """Returns verified independent held-out test evaluation metrics and calibration results."""
    from ml.tasks.fracture.config import REPORTS_DIR
    test_metrics_path = REPORTS_DIR / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path, "r") as f:
            metrics = json.load(f)
        return JSONResponse(status_code=200, content=metrics)
    
    return JSONResponse(status_code=200, content={
        "status": "validation_report_pending",
        "message": "Independent test evaluation metrics will be generated upon model training."
    })


@router.post("/predict")
async def predict_fracture(
    file: UploadFile = File(...)
) -> JSONResponse:
    """
    Analyzes an uploaded bone radiograph for fracture-related findings using ConvNeXt-Base.
    Includes quality evaluation and Grad-CAM visual evidence.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Received fracture prediction request {request_id} for file: {file.filename}")

    # 1. Validate image
    try:
        pil_img, _raw = await validate_and_load_image(file)
    except ImageValidationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request_id,
                "status": "invalid_input",
                "code": exc.code,
                "message": exc.message
            }
        )
    except Exception as exc:
        logger.error(f"Image validation exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "status": "error",
                "message": "Failed to validate and process input radiograph."
            }
        )

    # 2. Retrieve predictor
    pred = get_predictor()
    if pred is None:
        return JSONResponse(
            status_code=503,
            content={
                "request_id": request_id,
                "status": "model_unavailable",
                "message": "Fracture prediction service could not be initialized."
            }
        )

    # 3. Check checkpoint availability
    if not pred.checkpoint_path.exists():
        return JSONResponse(
            status_code=503,
            content={
                "request_id": request_id,
                "status": "model_unavailable",
                "message": "ConvNeXt-Base fracture model has not yet completed training. No inference can be performed."
            }
        )

    # 4. Perform genuine inference
    try:
        result = pred.predict_image(pil_img, generate_cam=True)
        return JSONResponse(
            status_code=200,
            content={"request_id": request_id, **result}
        )
    except Exception as exc:
        logger.error(f"Inference execution failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "status": "error",
                "message": f"Inference execution failed: {str(exc)}"
            }
        )
