import uuid
from typing import Optional
from fastapi import APIRouter, File, Header, UploadFile
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.logging_config import logger
from backend.schemas.analysis import (
    AnalysisResponse,
    AnalysisStatus,
    ModelMeta,
)
from backend.services.image_validation import ImageValidationError, validate_and_load_image
from backend.services import ml_service
from backend.services.ml_service import (
    MLServiceError,
    MLTimeoutError,
    MLUnavailableError,
)
from backend.services.triage_service import create_error_response, evaluate_triage

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=AnalysisResponse)
async def predict_image(
    file: UploadFile = File(...),
    x_demo_scenario: Optional[str] = Header(None, alias="X-Demo-Scenario"),
) -> JSONResponse:
    request_id = str(uuid.uuid4())
    logger.info("Handling prediction request %s (mode=%s)", request_id, settings.MEDGUARD_MODE)

    # 1. In DEMO mode: validate image then return clearly marked synthetic fixture
    if settings.MEDGUARD_MODE == "demo":
        from backend.services.demo_adapter import create_demo_response
        try:
            # Still perform real image validation to test the upload pipeline
            _img, _raw = await validate_and_load_image(file)
        except ImageValidationError as exc:
            status_enum = AnalysisStatus.INVALID_INPUT
            err_resp = create_error_response(
                request_id=request_id,
                status=status_enum,
                code=exc.code,
                message=exc.message,
                action="resubmit_better_sample",
                mode="demo",
            )
            return JSONResponse(status_code=exc.status_code, content=err_resp.model_dump())

        demo_resp = create_demo_response(scenario=x_demo_scenario, request_id=request_id)
        return JSONResponse(status_code=200, content=demo_resp.model_dump())

    # 2. In REAL mode: validate image
    try:
        pil_img, _raw = await validate_and_load_image(file)
    except ImageValidationError as exc:
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.INVALID_INPUT,
            code=exc.code,
            message=exc.message,
            action="resubmit_better_sample",
            mode="real",
        )
        return JSONResponse(status_code=exc.status_code, content=err_resp.model_dump())
    except Exception as exc:
        logger.error("Unexpected error during image validation: %s", exc)
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.ERROR,
            code="INTERNAL_VALIDATION_ERROR",
            message="An unexpected error occurred while processing the uploaded image.",
            action="technical_error",
            mode="real",
        )
        return JSONResponse(status_code=500, content=err_resp.model_dump())

    # 3. Check ML model availability
    status_info = ml_service.get_status()
    model_meta = ModelMeta(
        available=status_info.get("available", False),
        model_name=status_info.get("model_name"),
        model_version=status_info.get("model_version"),
        dataset="PneumoniaMNIST+",
        supported_modality=status_info.get("supported_image_type", "Chest X-ray (educational)"),
        calibration_available=status_info.get("calibration_available", False),
        ood_available=status_info.get("ood_available", False),
        quality_available=status_info.get("quality_available", False),
        gradcam_available=status_info.get("gradcam_available", False),
    )

    if not status_info.get("available"):
        logger.warning("ML module unavailable for request %s: %s", request_id, status_info.get("message"))
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.MODEL_UNAVAILABLE,
            code="MODEL_UNAVAILABLE",
            message="The ML decision-support module is currently unavailable. No prediction can be performed.",
            action="technical_error",
            mode="real",
        )
        err_resp.model = model_meta
        return JSONResponse(status_code=503, content=err_resp.model_dump())

    # 4. Run ML inference
    try:
        ml_result = await ml_service.run_inference(pil_img)
    except MLUnavailableError as exc:
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.MODEL_UNAVAILABLE,
            code=exc.code,
            message=exc.message,
            action="technical_error",
            mode="real",
        )
        err_resp.model = model_meta
        return JSONResponse(status_code=503, content=err_resp.model_dump())
    except MLTimeoutError as exc:
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.ERROR,
            code=exc.code,
            message="The inference operation timed out. Please try again later.",
            action="technical_error",
            mode="real",
        )
        err_resp.model = model_meta
        return JSONResponse(status_code=500, content=err_resp.model_dump())
    except MLServiceError as exc:
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.ERROR,
            code=exc.code,
            message="An error occurred during machine learning inference.",
            action="technical_error",
            mode="real",
        )
        err_resp.model = model_meta
        return JSONResponse(status_code=500, content=err_resp.model_dump())
    except Exception as exc:
        logger.error("Unhandled exception during inference: %s", exc)
        err_resp = create_error_response(
            request_id=request_id,
            status=AnalysisStatus.ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected system error occurred during analysis.",
            action="technical_error",
            mode="real",
        )
        err_resp.model = model_meta
        return JSONResponse(status_code=500, content=err_resp.model_dump())

    # 5. Evaluate triage rules
    analysis_resp = evaluate_triage(
        ml_result=ml_result,
        model_meta=model_meta,
        request_id=request_id,
        mode="real",
    )

    return JSONResponse(status_code=200, content=analysis_resp.model_dump())


@router.post("/predict/chestmnist")
async def predict_chestmnist_multilabel(
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Evaluates 14 thoracic findings on frontal chest radiograph via DenseNet-201.
    Preserves ResNet-18 endpoint separately.
    """
    request_id = str(uuid.uuid4())
    logger.info("Handling ChestMNIST DenseNet-201 prediction request %s", request_id)

    try:
        pil_img, _raw = await validate_and_load_image(file)
    except ImageValidationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request_id,
                "status": "invalid_input",
                "error": exc.message,
            },
        )
    except Exception as exc:
        logger.error("Error during image validation: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "status": "error",
                "error": "Failed to validate image input",
            },
        )

    try:
        result = await ml_service.run_densenet_inference(pil_img)
        return JSONResponse(status_code=200, content={"request_id": request_id, **result})
    except MLUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "request_id": request_id,
                "status": "model_unavailable",
                "message": str(exc),
            },
        )
    except Exception as exc:
        logger.error("DenseNet inference error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "status": "error",
                "message": "Error occurred during DenseNet-201 inference.",
            },
        )

