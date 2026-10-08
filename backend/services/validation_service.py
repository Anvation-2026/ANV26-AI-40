import json
from pathlib import Path
from backend.core.config import settings
from backend.core.logging_config import logger
from backend.schemas.analysis import ValidationReport, ValidationResponse


def get_validation_report() -> ValidationResponse:
    """
    Reads the ML validation report file without hardcoding or fabricating any metrics.
    Handles missing file (pending) and malformed JSON (error) safely.
    """
    report_file = settings.get_validation_report_file()
    logger.info("Checking validation report at %s", report_file)

    if not report_file.exists():
        logger.info("Validation report file not found; returning pending status")
        return ValidationResponse(
            status="pending",
            report=None,
            message="Evaluation pending — the ML validation report has not been produced yet.",
        )

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Error reading or parsing validation report file: %s", exc)
        return ValidationResponse(
            status="error",
            report=None,
            message="Validation report exists but could not be parsed.",
        )

    try:
        report = ValidationReport.model_validate(data)
        return ValidationResponse(
            status="available",
            report=report,
            message=None,
        )
    except Exception as exc:
        logger.error("Validation report schema validation failed: %s", exc)
        return ValidationResponse(
            status="error",
            report=None,
            message="Validation report structure does not match expected schema.",
        )
