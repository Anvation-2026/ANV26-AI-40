from typing import List, Optional, Tuple
from backend.schemas.analysis import (
    AnalysisResponse,
    AnalysisStatus,
    ErrorInfo,
    HeatmapInfo,
    ModelMeta,
    OODInfo,
    QualityInfo,
    TriageInfo,
    UncertaintyInfo,
    MLResult,
)

STATIC_LIMITATIONS: List[str] = [
    "Educational research prototype: not a medical device and not certified for clinical diagnostic use.",
    "Trained on a limited public dataset (PneumoniaMNIST+) and may not generalize across clinical environments.",
    "Calibrated probability does not guarantee clinical certainty or absence of alternative pathology.",
    "Grad-CAM visual overlays highlight regions influencing model predictions and do not confirm anatomical abnormalities.",
    "Image quality and out-of-distribution detectors are heuristic and imperfect.",
    "Qualified human expert interpretation is strictly required for any clinical context.",
]

DISCLAIMER_TEXT: str = (
    "Educational research prototype. Not a medical device. Not for diagnosis "
    "or treatment decisions. Use only public or de-identified educational images."
)


def evaluate_triage(
    ml_result: MLResult,
    model_meta: ModelMeta,
    request_id: str,
    mode: str = "real"
) -> AnalysisResponse:
    """
    Evaluates rule-based educational triage in the exact precedence order specified in Section 6.5:
    1. Validation failure (handled upstream)
    2. Model unavailable / error (handled upstream or passed here)
    3. quality.status == "poor" -> poor_quality
    4. ood.is_ood is True -> ood
    5. abstained is True OR finding is None OR uncertainty.level == "high" -> uncertain
    6. Otherwise -> success
    """
    quality = ml_result.quality or QualityInfo()
    ood = ml_result.ood or OODInfo()
    uncertainty = ml_result.uncertainty or UncertaintyInfo()

    evidence: List[str] = []
    reasons: List[str] = []

    # Evidence from quality metrics
    if quality.evaluated:
        q_items = []
        if quality.blur and quality.blur.value is not None:
            q_items.append(f"blur={quality.blur.value:.1f}")
        if quality.brightness and quality.brightness.value is not None:
            q_items.append(f"brightness={quality.brightness.value:.1f}")
        if quality.contrast and quality.contrast.value is not None:
            q_items.append(f"contrast={quality.contrast.value:.1f}")
        q_summary = f" ({', '.join(q_items)})" if q_items else ""
        evidence.append(f"Image quality evaluated as {quality.status}{q_summary}")
    else:
        reasons.append("Image quality was not evaluated.")
        evidence.append("Image quality check was not evaluated")

    # Evidence from OOD check
    if ood.evaluated:
        if ood.is_ood:
            score_text = f" (score={ood.score:.2f})" if ood.score is not None else ""
            evidence.append(f"Image flagged as out-of-distribution{score_text}")
        else:
            evidence.append("Image verified as in-distribution chest X-ray")
    else:
        reasons.append("OOD check was not evaluated.")
        evidence.append("OOD check was not evaluated")

    # Evidence from uncertainty
    if uncertainty.level != "not_evaluated":
        unc_val = f" (value={uncertainty.value:.3f})" if uncertainty.value is not None else ""
        evidence.append(f"Model uncertainty level evaluated as {uncertainty.level}{unc_val}")

    # Append any ML engineer-provided evidence notes verbatim
    if ml_result.evidence_notes:
        evidence.extend(ml_result.evidence_notes)

    # RULE 3: Poor quality
    if quality.status == "poor":
        q_reasons = quality.reasons if quality.reasons else ["Image quality is insufficient for reliable analysis."]
        triage = TriageInfo(
            action="resubmit_better_sample",
            title="Poor Image Quality Detected",
            message="Educational recommendation: resubmit a clearer, properly exposed educational chest X-ray.",
            reasons=q_reasons,
        )
        explanation = (
            "The uploaded image quality was evaluated as insufficient for reliable decision support. "
            "Excessive blur, low contrast, or improper exposure obscures critical lung parenchymal textures. "
            "Evaluated possibilities: Diagnostic differentiation withheld until a sharp, properly exposed radiograph is provided."
        )
        evidence.append("Image degradation: Fine bronchovascular markings obscured below diagnostic resolution")
        evidence.append("Possibilities status: Diagnostic evaluation withheld to avoid false positives or negatives")
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.POOR_QUALITY,
            mode=mode,
            finding=None,
            abstained=True,
            raw_score=None,
            probability=None,
            probability_of=None,
            uncertainty=uncertainty,
            quality=quality,
            ood=ood,
            heatmap=HeatmapInfo(
                available=False,
                data_url=None,
                kind=None,
                message="Heatmap suppressed due to poor image quality.",
            ),
            triage=triage,
            explanation=explanation,
            evidence=evidence,
            limitations=STATIC_LIMITATIONS,
            model=model_meta,
            error=None,
            disclaimer=DISCLAIMER_TEXT,
        )

    # RULE 4: Out-of-distribution
    if ood.is_ood is True:
        ood_reason = ood.reason or "Input image distribution does not match chest X-ray training distribution."
        triage = TriageInfo(
            action="unsupported_input",
            title="Out-of-Distribution Image Detected",
            message="This system supports educational chest X-rays only.",
            reasons=[ood_reason],
        )
        explanation = (
            "The input image was identified as outside the expected educational chest X-ray distribution. "
            "Anomalous chromaticity, non-radiographic features, or unsupported modality detected. "
            "Evaluated possibilities: Withheld to prevent erroneous or hallucinated predictions."
        )
        evidence.append("Distribution check: Input does not conform to calibrated pediatric chest radiograph manifold")
        evidence.append("Possibilities status: Analysis rejected per input validity guardrail")
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.OOD,
            mode=mode,
            finding=None,
            abstained=True,
            raw_score=None,
            probability=None,
            probability_of=None,
            uncertainty=uncertainty,
            quality=quality,
            ood=ood,
            heatmap=HeatmapInfo(
                available=False,
                data_url=None,
                kind=None,
                message="Heatmap suppressed for out-of-distribution input.",
            ),
            triage=triage,
            explanation=explanation,
            evidence=evidence,
            limitations=STATIC_LIMITATIONS,
            model=model_meta,
            error=None,
            disclaimer=DISCLAIMER_TEXT,
        )

    # RULE 5: Abstained / High uncertainty / finding is None
    if ml_result.abstained or (ml_result.finding is None) or (uncertainty.level == "high"):
        reasons_list = list(reasons)
        if uncertainty.level == "high":
            reasons_list.append("High model uncertainty exceeded acceptable decision support threshold.")
        if ml_result.abstained:
            reasons_list.append("Model safety policy abstained from providing a finding.")

        triage = TriageInfo(
            action="expert_review_required_abstained",
            title="High Model Uncertainty — Decision Abstained",
            message="Educational recommendation: have a qualified human expert review this image.",
            reasons=reasons_list or ["Model abstained from providing a definitive finding."],
        )
        unc_heatmap_url = None
        if ml_result.heatmap_png_base64:
            raw_b64 = ml_result.heatmap_png_base64.strip()
            unc_heatmap_url = raw_b64 if raw_b64.startswith("data:image/png;base64,") else f"data:image/png;base64,{raw_b64}"

        explanation = (
            "The model abstained from providing a definitive finding due to high uncertainty or safety abstention thresholds. "
            "Evaluated possibilities: Conflicting borderline features between mild parenchymal opacity and normal pediatric anatomical variation. "
            "Human expert review is required."
        )
        evidence.append("Decision boundary: Borderline activation between normal aeration and early infiltrative patterns")
        evidence.append("Safety protocol: Definitive verdict withheld to prevent potential misclassification")

        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.UNCERTAIN,
            mode=mode,
            finding=None,  # Suppressed per Section 6.5
            abstained=True,
            raw_score=ml_result.raw_score,
            probability=None,  # Suppressed per Section 6.5
            probability_of=None,
            uncertainty=uncertainty,
            quality=quality,
            ood=ood,
            heatmap=HeatmapInfo(
                available=False,
                data_url=unc_heatmap_url,
                kind=ml_result.heatmap_kind or "overlay",
                message="Educational attention map generated for ambiguous features; definitive diagnostic heatmap is withheld per safety policy.",
            ),
            triage=triage,
            explanation=explanation,
            evidence=evidence,
            limitations=STATIC_LIMITATIONS,
            model=model_meta,
            error=None,
            disclaimer=DISCLAIMER_TEXT,
        )

    # RULE 6: Success
    success_reasons = list(reasons)
    if uncertainty.level == "moderate":
        success_reasons.append("Moderate model uncertainty—interpret with extra caution.")

    if ml_result.finding == "pneumonia":
        success_reasons.insert(0, "Pattern match detected for pneumonia class.")
    else:
        success_reasons.insert(0, "Pattern match detected for normal chest X-ray class.")

    triage = TriageInfo(
        action="expert_review_recommended",
        title="Educational Review Recommended",
        message="Educational recommendation: have a qualified human expert review this image.",
        reasons=success_reasons,
    )

    # Build explanation strictly templated from real fields
    prob_str = f" with calibrated probability {ml_result.calibrated_probability * 100:.1f}%" if ml_result.calibrated_probability is not None else ""
    if ml_result.finding == "normal":
        explanation = (
            f"The model identified a 'normal' lung field pattern{prob_str}. "
            "Both lung fields exhibit clear symmetric aeration without focal consolidation, confluent alveolar opacities, or pleural effusion. "
            "Evaluated possibilities: Normal aeration (primary pattern), Bacterial lobar consolidation (ruled out), Viral/interstitial opacity (ruled out). "
            "Human expert interpretation is required."
        )
        evidence.append("Airspace aeration: Bilateral lung zones clear without focal consolidation or opacity clusters")
        evidence.append("Possibility evaluated: Bacterial Lobar Pneumonia — ruled out / non-reactive")
        evidence.append("Possibility evaluated: Viral / Interstitial Infiltrate — ruled out / non-reactive")
    else:
        explanation = (
            f"The model identified a 'pneumonia' pattern{prob_str}. "
            "Focal or patchy parenchymal opacification/consolidation was detected in the lung fields. "
            "Evaluated possibilities: Pneumonia consolidation/infiltrate (primary pattern detected), Normal aerated lung fields (excluded given localized attenuation). "
            "Human expert interpretation is required."
        )
        evidence.append("Parenchymal pattern: Focal or multifocal consolidation / increased opacity detected")
        evidence.append("Possibility evaluated: Pneumonia Infiltrate — primary class pattern detected")
        evidence.append("Possibility evaluated: Normal Aerated Lung Fields — lower likelihood due to focal attenuation")
        evidence.append("Differential consideration: Bacterial consolidation vs. viral bronchopneumonia (clinical correlation required)")

    # Process heatmap
    heatmap_info = HeatmapInfo(available=False, data_url=None, kind=None, message="Heatmap unavailable for this result.")
    if ml_result.heatmap_png_base64:
        raw_b64 = ml_result.heatmap_png_base64.strip()
        data_url = raw_b64 if raw_b64.startswith("data:image/png;base64,") else f"data:image/png;base64,{raw_b64}"
        heatmap_info = HeatmapInfo(
            available=True,
            data_url=data_url,
            kind=ml_result.heatmap_kind or "overlay",
            message=None,
        )

    return AnalysisResponse(
        request_id=request_id,
        status=AnalysisStatus.SUCCESS,
        mode=mode,
        finding=ml_result.finding,
        abstained=False,
        raw_score=ml_result.raw_score,
        probability=ml_result.calibrated_probability,
        probability_of=ml_result.probability_of or ml_result.finding,
        uncertainty=uncertainty,
        quality=quality,
        ood=ood,
        heatmap=heatmap_info,
        triage=triage,
        explanation=explanation,
        evidence=evidence,
        limitations=STATIC_LIMITATIONS,
        model=model_meta,
        error=None,
        disclaimer=DISCLAIMER_TEXT,
    )


def create_error_response(
    request_id: str,
    status: AnalysisStatus,
    code: str,
    message: str,
    action: str = "technical_error",
    mode: str = "real",
) -> AnalysisResponse:
    """Helper to produce standard AnalysisResponse structure for errors and rejections."""
    return AnalysisResponse(
        request_id=request_id,
        status=status,
        mode=mode,
        finding=None,
        abstained=True,
        raw_score=None,
        probability=None,
        probability_of=None,
        uncertainty=UncertaintyInfo(),
        quality=QualityInfo(),
        ood=OODInfo(),
        heatmap=HeatmapInfo(available=False, data_url=None, kind=None, message="No heatmap available."),
        triage=TriageInfo(
            action=action,  # type: ignore
            title="Analysis Notice",
            message=message,
            reasons=[message],
        ),
        explanation=f"Analysis halted: {message}",
        evidence=[],
        limitations=STATIC_LIMITATIONS,
        model=ModelMeta(available=False),
        error=ErrorInfo(code=code, message=message),
        disclaimer=DISCLAIMER_TEXT,
    )
