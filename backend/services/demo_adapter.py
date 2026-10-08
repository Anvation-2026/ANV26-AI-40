"""
Demo Adapter - Synthetic educational fixtures for UI verification ONLY.
Active ONLY when MEDGUARD_MODE=demo.
NEVER imported or executed in MEDGUARD_MODE=real.
"""
from typing import Dict, Any, Optional
from backend.schemas.analysis import (
    AnalysisResponse,
    AnalysisStatus,
    ErrorInfo,
    HeatmapInfo,
    ModelMeta,
    OODInfo,
    QualityCheck,
    QualityInfo,
    TriageInfo,
    UncertaintyInfo,
)


def get_demo_model_info() -> Dict[str, Any]:
    return {
        "available": True,
        "model_name": "DEMO FIXTURE — not a real model",
        "model_version": "demo-v0",
        "dataset": "PneumoniaMNIST+ (Demo Fixtures)",
        "supported_modality": "Chest X-ray (educational)",
        "classes": ["normal", "pneumonia"],
        "calibration_available": True,
        "ood_available": True,
        "quality_available": True,
        "gradcam_available": False,
        "message": "Demo mode — fixtures only, not real model output",
    }


def create_demo_response(
    scenario: Optional[str],
    request_id: str,
) -> AnalysisResponse:
    selected = (scenario or "uncertain").lower().strip()

    demo_model = ModelMeta(
        available=True,
        model_name="DEMO FIXTURE — not a real model",
        model_version="demo-v0",
        dataset="PneumoniaMNIST+ (Demo Fixtures)",
        supported_modality="Chest X-ray (educational)",
        calibration_available=True,
        ood_available=True,
        quality_available=True,
        gradcam_available=False,
    )

    base_limitations = [
        "DEMO FIXTURE: Generated purely for frontend development and layout testing.",
        "Educational research prototype: not a medical device and not certified for clinical diagnostic use.",
        "Trained on a limited public dataset (PneumoniaMNIST+) and may not generalize across clinical environments.",
        "Calibrated probability does not guarantee clinical certainty or absence of alternative pathology.",
        "Grad-CAM visual overlays highlight regions influencing model predictions and do not confirm anatomical abnormalities.",
        "Image quality and out-of-distribution detectors are heuristic and imperfect.",
        "Qualified human expert interpretation is strictly required for any clinical context.",
    ]

    heatmap_info = HeatmapInfo(
        available=False,
        data_url=None,
        kind=None,
        message="No heatmap in demo mode",
    )

    if selected == "success_pneumonia":
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.SUCCESS,
            mode="demo",
            finding="pneumonia",
            abstained=False,
            raw_score=0.884,
            probability=0.870,
            probability_of="pneumonia",
            uncertainty=UncertaintyInfo(
                level="low",
                value=0.042,
                method="MC-Dropout (Demo)",
            ),
            quality=QualityInfo(
                evaluated=True,
                status="acceptable",
                blur=QualityCheck(value=240.5, threshold=100.0, passed=True),
                brightness=QualityCheck(value=128.0, threshold=40.0, passed=True),
                contrast=QualityCheck(value=65.2, threshold=20.0, passed=True),
                reasons=[],
            ),
            ood=OODInfo(
                evaluated=True,
                is_ood=False,
                score=0.12,
                method="Mahalanobis (Demo)",
                reason=None,
            ),
            heatmap=heatmap_info,
            triage=TriageInfo(
                action="expert_review_recommended",
                title="Educational Review Recommended (DEMO)",
                message="Educational recommendation: have a qualified human expert review this image.",
                reasons=["DEMO: Pattern match detected for pneumonia class"],
            ),
            explanation="DEMO FIXTURE — not a model prediction. The demo fixture assigned pneumonia with synthetic probability 87.0%.",
            evidence=[
                "DEMO: Image quality passed acceptable criteria",
                "DEMO: In-distribution sample",
                "DEMO: Low model uncertainty (0.042)",
            ],
            limitations=base_limitations,
            model=demo_model,
            error=None,
        )

    elif selected == "success_normal":
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.SUCCESS,
            mode="demo",
            finding="normal",
            abstained=False,
            raw_score=0.115,
            probability=0.912,
            probability_of="normal",
            uncertainty=UncertaintyInfo(
                level="low",
                value=0.031,
                method="MC-Dropout (Demo)",
            ),
            quality=QualityInfo(
                evaluated=True,
                status="acceptable",
                blur=QualityCheck(value=265.0, threshold=100.0, passed=True),
                brightness=QualityCheck(value=135.0, threshold=40.0, passed=True),
                contrast=QualityCheck(value=70.1, threshold=20.0, passed=True),
                reasons=[],
            ),
            ood=OODInfo(
                evaluated=True,
                is_ood=False,
                score=0.08,
                method="Mahalanobis (Demo)",
                reason=None,
            ),
            heatmap=heatmap_info,
            triage=TriageInfo(
                action="expert_review_recommended",
                title="Educational Review Recommended (DEMO)",
                message="Educational recommendation: have a qualified human expert review this image.",
                reasons=["DEMO: Pattern match detected for normal chest X-ray class"],
            ),
            explanation="DEMO FIXTURE — not a model prediction. The demo fixture assigned normal with synthetic probability 91.2%.",
            evidence=[
                "DEMO: Image quality passed acceptable criteria",
                "DEMO: In-distribution sample",
                "DEMO: Low model uncertainty (0.031)",
            ],
            limitations=base_limitations,
            model=demo_model,
            error=None,
        )

    elif selected == "poor_quality":
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.POOR_QUALITY,
            mode="demo",
            finding=None,
            abstained=True,
            raw_score=None,
            probability=None,
            probability_of=None,
            uncertainty=UncertaintyInfo(
                level="not_evaluated",
                value=None,
                method=None,
            ),
            quality=QualityInfo(
                evaluated=True,
                status="poor",
                blur=QualityCheck(value=42.1, threshold=100.0, passed=False),
                brightness=QualityCheck(value=22.0, threshold=40.0, passed=False),
                contrast=QualityCheck(value=15.0, threshold=20.0, passed=False),
                reasons=[
                    "DEMO: Image blur index (42.1) is below acceptable threshold (100.0).",
                    "DEMO: Image brightness (22.0) is under-exposed.",
                ],
            ),
            ood=OODInfo(
                evaluated=False,
                is_ood=None,
                score=None,
                method=None,
                reason=None,
            ),
            heatmap=heatmap_info,
            triage=TriageInfo(
                action="resubmit_better_sample",
                title="Poor Image Quality Detected (DEMO)",
                message="Educational recommendation: resubmit a clearer, properly exposed educational chest X-ray.",
                reasons=[
                    "DEMO: Image quality is insufficient for reliable analysis.",
                    "DEMO: Severe blur and low contrast detected.",
                ],
            ),
            explanation="DEMO FIXTURE — not a model prediction. Image quality was flagged as poor; model inference was rejected.",
            evidence=[
                "DEMO: Quality check failed: blur index 42.1 < 100.0",
                "DEMO: Quality check failed: brightness 22.0 < 40.0",
            ],
            limitations=base_limitations,
            model=demo_model,
            error=None,
        )

    elif selected == "ood":
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.OOD,
            mode="demo",
            finding=None,
            abstained=True,
            raw_score=None,
            probability=None,
            probability_of=None,
            uncertainty=UncertaintyInfo(
                level="not_evaluated",
                value=None,
                method=None,
            ),
            quality=QualityInfo(
                evaluated=True,
                status="acceptable",
                blur=QualityCheck(value=210.0, threshold=100.0, passed=True),
                brightness=QualityCheck(value=120.0, threshold=40.0, passed=True),
                contrast=QualityCheck(value=55.0, threshold=20.0, passed=True),
                reasons=[],
            ),
            ood=OODInfo(
                evaluated=True,
                is_ood=True,
                score=0.94,
                method="Mahalanobis (Demo)",
                reason="DEMO: Input image distribution does not match chest X-ray training distribution.",
            ),
            heatmap=heatmap_info,
            triage=TriageInfo(
                action="unsupported_input",
                title="Out-of-Distribution Image Detected (DEMO)",
                message="This system supports educational chest X-rays only.",
                reasons=["DEMO: Input does not conform to educational chest X-ray domain"],
            ),
            explanation="DEMO FIXTURE — not a model prediction. The image was identified as out-of-distribution.",
            evidence=[
                "DEMO: OOD detector flagged non-chest X-ray input (score 0.94)",
            ],
            limitations=base_limitations,
            model=demo_model,
            error=None,
        )

    else:  # default "uncertain"
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.UNCERTAIN,
            mode="demo",
            finding=None,
            abstained=True,
            raw_score=0.51,
            probability=None,
            probability_of=None,
            uncertainty=UncertaintyInfo(
                level="high",
                value=0.38,
                method="MC-Dropout (Demo)",
            ),
            quality=QualityInfo(
                evaluated=True,
                status="acceptable",
                blur=QualityCheck(value=180.0, threshold=100.0, passed=True),
                brightness=QualityCheck(value=110.0, threshold=40.0, passed=True),
                contrast=QualityCheck(value=48.0, threshold=20.0, passed=True),
                reasons=[],
            ),
            ood=OODInfo(
                evaluated=True,
                is_ood=False,
                score=0.15,
                method="Mahalanobis (Demo)",
                reason=None,
            ),
            heatmap=heatmap_info,
            triage=TriageInfo(
                action="expert_review_required_abstained",
                title="High Model Uncertainty — Decision Abstained (DEMO)",
                message="Educational recommendation: have a qualified human expert review this image.",
                reasons=["DEMO: High model uncertainty exceeded acceptable decision threshold"],
            ),
            explanation="DEMO FIXTURE — not a model prediction. The model abstained from providing a definitive finding due to high uncertainty.",
            evidence=[
                "DEMO: Uncertainty level: high (value 0.38)",
                "DEMO: Model abstained from classification",
            ],
            limitations=base_limitations,
            model=demo_model,
            error=None,
        )
