import os
import sys
import uuid
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure repo root and backend directory are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.api.routes.health import router as health_router
from backend.api.routes.prediction import router as prediction_router
from backend.api.routes.validation import router as validation_router
from backend.api.routes.fracture import router as fracture_router
from backend.api.routes.models import router as models_router
from backend.core.config import settings
from backend.core.logging_config import logger
from backend.schemas.analysis import AnalysisStatus
from backend.services.triage_service import create_error_response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedGuard AI API",
        description="Educational research prototype for chest X-ray decision support",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Security Headers
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS (allow local dev ports plus all vercel.app preview and production domains)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_origin_regex=r"^https://.*\.vercel\.app$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Mount routers under /api
    app.include_router(health_router)
    app.include_router(prediction_router)
    app.include_router(validation_router)
    app.include_router(fracture_router)
    app.include_router(models_router)

    # Mount static assets if directory exists
    public_assets = REPO_ROOT / "frontend" / "public" / "assets"
    if public_assets.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=str(public_assets)), name="assets")

    # Catch-all exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception: %s", exc)
        req_id = str(uuid.uuid4())
        # If requesting prediction, return AnalysisResponse shape
        if request.url.path.startswith("/api/predict"):
            err_resp = create_error_response(
                request_id=req_id,
                status=AnalysisStatus.ERROR,
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred.",
                action="technical_error",
                mode=settings.MEDGUARD_MODE,
            )
            return JSONResponse(status_code=500, content=err_resp.model_dump())
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred.", "request_id": req_id},
        )

    return app


app = create_app()
