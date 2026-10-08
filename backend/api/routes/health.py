from datetime import datetime, timezone
from fastapi import APIRouter
from backend.core.config import settings
from backend.schemas.analysis import HealthResponse, ModelStatusResponse
from backend.services import ml_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    ml_status = ml_service.get_status() if settings.MEDGUARD_MODE == "real" else {"available": True}
    return HealthResponse(
        status="ok",
        mode=settings.MEDGUARD_MODE,
        version="1.0.0",
        ml_module_loaded=bool(ml_status.get("available", False)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    if settings.MEDGUARD_MODE == "demo":
        from backend.services.demo_adapter import get_demo_model_info
        demo_info = get_demo_model_info()
        return ModelStatusResponse(
            available=True,
            model_name=demo_info["model_name"],
            model_version=demo_info["model_version"],
            supported_image_type=demo_info["supported_modality"],
            inference_ready=True,
            calibration_available=demo_info["calibration_available"],
            ood_available=demo_info["ood_available"],
            quality_available=demo_info["quality_available"],
            gradcam_available=demo_info["gradcam_available"],
            mode="demo",
            message=demo_info["message"],
        )

    # Real mode
    status = ml_service.get_status()
    return ModelStatusResponse(
        available=status.get("available", False),
        model_name=status.get("model_name"),
        model_version=status.get("model_version"),
        supported_image_type=status.get("supported_image_type", "Chest X-ray (educational)"),
        inference_ready=status.get("inference_ready", False),
        calibration_available=status.get("calibration_available", False),
        ood_available=status.get("ood_available", False),
        quality_available=status.get("quality_available", False),
        gradcam_available=status.get("gradcam_available", False),
        mode="real",
        message=status.get("message"),
    )
