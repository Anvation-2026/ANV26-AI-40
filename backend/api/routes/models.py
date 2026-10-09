"""
FastAPI Routes for Unified Model Registry & Hardware Monitoring
MedGuard AI - Clinical Evidence & Triage Support System
Endpoints:
- GET /api/models: Returns full registry of all 4 AI models
- GET /api/models/{model_id}: Returns detailed specifications for a single model
- GET /api/models/system/gpu: Returns live CUDA VRAM allocations
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from backend.services.model_registry import model_registry

router = APIRouter(prefix="/api/models", tags=["model_registry"])


@router.get("")
async def list_registered_models() -> JSONResponse:
    """Lists all registered models in the MedGuard AI system and their availability."""
    models = model_registry.get_all_models()
    gpu_info = model_registry.get_gpu_memory_summary()
    return JSONResponse(status_code=200, content={
        "total_registered_models": len(models),
        "gpu_state": gpu_info,
        "models": models
    })


@router.get("/system/gpu")
async def get_gpu_memory_state() -> JSONResponse:
    """Returns real-time GPU VRAM telemetry."""
    gpu_info = model_registry.get_gpu_memory_summary()
    return JSONResponse(status_code=200, content=gpu_info)


@router.get("/{model_id}")
async def get_model_details(model_id: str) -> JSONResponse:
    """Returns comprehensive technical specifications for a single registered model."""
    entry = model_registry.get_model(model_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")
    return JSONResponse(status_code=200, content=entry)
