from fastapi import APIRouter
from backend.schemas.analysis import ValidationResponse
from backend.services.validation_service import get_validation_report

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/validation", response_model=ValidationResponse)
async def get_validation() -> ValidationResponse:
    return get_validation_report()
