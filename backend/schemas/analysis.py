from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    SUCCESS = "success"
    UNCERTAIN = "uncertain"
    POOR_QUALITY = "poor_quality"
    OOD = "ood"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"


class QualityCheck(BaseModel):
    value: Optional[float] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None


class QualityInfo(BaseModel):
    evaluated: bool = False
    status: Literal["acceptable", "poor", "not_evaluated"] = "not_evaluated"
    blur: Optional[QualityCheck] = None
    brightness: Optional[QualityCheck] = None
    contrast: Optional[QualityCheck] = None
    reasons: List[str] = Field(default_factory=list)


class OODInfo(BaseModel):
    evaluated: bool = False
    is_ood: Optional[bool] = None
    score: Optional[float] = None
    method: Optional[str] = None
    reason: Optional[str] = None


class UncertaintyInfo(BaseModel):
    level: Literal["low", "moderate", "high", "not_evaluated"] = "not_evaluated"
    value: Optional[float] = None
    method: Optional[str] = None


class HeatmapInfo(BaseModel):
    available: bool = False
    data_url: Optional[str] = None
    kind: Optional[Literal["overlay", "heatmap_only"]] = None
    message: Optional[str] = None


class ModelMeta(BaseModel):
    available: bool = False
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset: Optional[str] = None
    supported_modality: Optional[str] = "Chest X-ray (educational)"
    calibration_available: Optional[bool] = False
    ood_available: Optional[bool] = False
    quality_available: Optional[bool] = False
    gradcam_available: Optional[bool] = False


class TriageInfo(BaseModel):
    action: Literal[
        "expert_review_recommended",
        "expert_review_required_abstained",
        "resubmit_better_sample",
        "unsupported_input",
        "technical_error",
    ]
    title: str
    message: str
    reasons: List[str] = Field(default_factory=list)


class ErrorInfo(BaseModel):
    code: str
    message: str


class AnalysisResponse(BaseModel):
    request_id: str
    status: AnalysisStatus
    mode: Literal["real", "demo"]
    finding: Optional[Literal["normal", "pneumonia"]] = None
    abstained: bool = False
    raw_score: Optional[float] = None
    probability: Optional[float] = None  # calibrated probability
    probability_of: Optional[str] = None
    uncertainty: UncertaintyInfo = Field(default_factory=UncertaintyInfo)
    quality: QualityInfo = Field(default_factory=QualityInfo)
    ood: OODInfo = Field(default_factory=OODInfo)
    heatmap: HeatmapInfo = Field(default_factory=HeatmapInfo)
    triage: TriageInfo
    explanation: str
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    model: ModelMeta = Field(default_factory=ModelMeta)
    error: Optional[ErrorInfo] = None
    disclaimer: str = (
        "Educational research prototype. Not a medical device. Not for diagnosis "
        "or treatment decisions. Use only public or de-identified educational images."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    mode: Literal["real", "demo"]
    version: str = "1.0.0"
    ml_module_loaded: bool
    timestamp: str


class ModelStatusResponse(BaseModel):
    available: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    supported_image_type: str = "Chest X-ray (educational)"
    inference_ready: bool
    calibration_available: bool = False
    ood_available: bool = False
    quality_available: bool = False
    gradcam_available: bool = False
    mode: Literal["real", "demo"]
    message: Optional[str] = None


class ValidationMetrics(BaseModel):
    accuracy: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auroc: Optional[float] = None
    false_negative_rate: Optional[float] = None
    ece: Optional[float] = None
    abstention_coverage: Optional[float] = None
    rejection_rate: Optional[float] = None


class ConfusionMatrixData(BaseModel):
    labels: List[str] = Field(default_factory=lambda: ["normal", "pneumonia"])
    matrix: List[List[int]] = Field(default_factory=list)


class ValidationReport(BaseModel):
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset: Optional[str] = None
    evaluated_on: Optional[str] = None
    split_counts: Optional[Dict[str, int]] = None
    metrics: Optional[ValidationMetrics] = None
    confusion_matrix: Optional[ConfusionMatrixData] = None
    limitations: List[str] = Field(default_factory=list)
    generated_at: Optional[str] = None


class ValidationResponse(BaseModel):
    status: Literal["available", "pending", "error"]
    report: Optional[ValidationReport] = None
    message: Optional[str] = None


# Internal model for parsing ML module dictionary safely
class MLResult(BaseModel):
    finding: Optional[Literal["normal", "pneumonia"]] = None
    abstained: bool = False
    raw_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    probability_of: Optional[str] = None
    uncertainty: Optional[UncertaintyInfo] = None
    quality: Optional[QualityInfo] = None
    ood: Optional[OODInfo] = None
    heatmap_png_base64: Optional[str] = None
    heatmap_kind: Optional[Literal["overlay", "heatmap_only"]] = None
    model_version: Optional[str] = None
    evidence_notes: Optional[List[str]] = None
