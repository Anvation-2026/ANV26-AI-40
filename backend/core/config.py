from pathlib import Path
from typing import List, Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from PIL import Image


class Settings(BaseSettings):
    MEDGUARD_MODE: Literal["real", "demo"] = "real"
    MAX_UPLOAD_BYTES: int = 10485760  # 10 MB
    ALLOWED_FORMATS: List[str] = ["PNG", "JPEG", "WEBP"]
    MIN_IMAGE_DIM: int = 64
    MAX_IMAGE_DIM: int = 8000
    MAX_IMAGE_PIXELS: int = 40_000_000
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:3000,http://127.0.0.1:3000"
    VALIDATION_REPORT_PATH: str = "ml/reports/validation_report.json"
    ML_MODULE_PATH: str = "ml.interface"
    INFERENCE_TIMEOUT_SECONDS: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, list):
            return self.CORS_ORIGINS
        return [orig.strip() for orig in self.CORS_ORIGINS.split(",") if orig.strip()]

    def get_validation_report_file(self) -> Path:
        p = Path(self.VALIDATION_REPORT_PATH)
        if p.is_absolute():
            return p
        if p.exists():
            return p.resolve()
        backend_dir = Path(__file__).resolve().parent.parent
        repo_root = backend_dir.parent
        candidate = repo_root / self.VALIDATION_REPORT_PATH
        if candidate.exists():
            return candidate.resolve()
        standard_path = repo_root / "ml" / "reports" / "validation_report.json"
        if standard_path.exists():
            return standard_path.resolve()
        return candidate.resolve()


settings = Settings()
# Configure Pillow decompression bomb guard
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
