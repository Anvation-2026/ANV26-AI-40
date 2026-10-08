from typing import Any, Dict, List, Optional, Tuple

# Path bootstrap
from src import _ML_ROOT  # noqa: F401

from config import STANDARD_LIMITATIONS


def build_explanation_and_evidence(
    status: str,
    finding: Optional[str],
    probability: Optional[float],
    uncertainty: Optional[str],
    quality_status: Optional[str],
    quality_details: Dict[str, Any],
    ood_flagged: Optional[bool],
    ood_details: Dict[str, Any],
    details: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Constructs grounded, educational explanation text and factual bullet points.
    Strictly follows safety guidelines:
    - Never makes a clinical diagnosis or treatment claim.
    - Explicitly states educational prototype status.
    - Explicitly describes heatmaps as sensitivity indicators, not confirmation.
    - Grounds all statements in computed values.
    """
    calibrated = details.get("calibrated", False)
    calib_str = "calibrated using temperature scaling on the validation set" if calibrated else "uncalibrated"
    temp = details.get("temperature", 1.0)
    conf = details.get("confidence")

    evidence: List[str] = []

    if status == "unsupported_file":
        reason = details.get("rejection_reason", "Unable to decode input as a valid image")
        explanation = (
            f"Input rejected: {reason}. The provided input could not be processed as an image. "
            "Please upload a standard image file (PNG, JPEG, or DICOM-derived grayscale). "
            "This is an educational prototype and not a clinical diagnosis."
        )
        evidence = [
            f"Input status: decode failed ({reason})",
            "Image dimensions: invalid or unreadable",
            "Educational prototype: analysis halted before model evaluation",
        ]
        return explanation, evidence

    if status == "poor_quality":
        reasons = quality_details.get("reasons", [])
        reasons_str = ", ".join(reasons) if reasons else "unspecified degradation"
        metrics = quality_details.get("metrics", {})
        blur_val = metrics.get("blur_laplacian_var", 0.0)
        bright_val = metrics.get("brightness_mean", 0.0)
        contrast_val = metrics.get("contrast_p99_p1", 0.0)

        explanation = (
            f"Image quality check failed ({reasons_str}). The image characteristics do not meet "
            "the baseline quality standards established on validation data. To avoid unreliable outputs, "
            "predictions are withheld. Please supply a clearer pediatric chest radiograph. "
            "This is an educational prototype and not a clinical diagnosis."
        )
        evidence = [
            f"Image quality status: poor (failed checks: {reasons_str})",
            f"Sharpness metric: Laplacian variance = {blur_val:.1f}",
            f"Brightness level: {bright_val:.3f}",
            f"Contrast spread (p99 - p1): {contrast_val:.3f}",
            "Automated quality gate triggered: inference suppressed to prevent misleading findings",
        ]
        return explanation, evidence

    if status == "ood_rejected":
        reasons = ood_details.get("reasons", [])
        reasons_str = ", ".join(reasons) if reasons else "distribution anomaly"
        score = ood_details.get("score", 0.0)
        thresh = ood_details.get("threshold", 0.0)
        method = ood_details.get("method", "heuristic")

        explanation = (
            f"Out-of-distribution detection triggered ({reasons_str}). The input differs significantly "
            "from the training distribution of pediatric chest radiographs. Because model behavior "
            "on unsupported modalities or distributions is undefined, predictions are withheld. "
            "This is an educational prototype and not a clinical diagnosis."
        )
        evidence = [
            f"Out-of-distribution status: flagged ({reasons_str})",
            f"Detection method: {method}",
            f"OOD distance score: {score:.2f} (threshold: {thresh:.2f})",
            "Model domain restriction: calibrated exclusively for pediatric frontal chest X-rays",
            "Educational prototype safety guard: input rejected as out-of-distribution",
        ]
        return explanation, evidence

    if status == "uncertain":
        p_pneumonia = details.get("p_pneumonia_calibrated")
        tau_accept = details.get("thresholds", {}).get("tau_accept", 0.80)
        conf_val = conf if conf is not None else 0.50

        explanation = (
            f"Uncertain prediction: model confidence ({conf_val:.2f}) is below the validation-fitted "
            f"acceptance threshold ({tau_accept:.2f}). To maintain reliability, the system abstains from "
            "rendering a definitive classification. Expert manual review is required for this case. "
            "This is an educational prototype and not a clinical diagnosis."
        )
        evidence = [
            f"Model status: uncertain (abstention rule triggered)",
            f"Calibrated confidence = {conf_val:.2f} (acceptance threshold tau = {tau_accept:.2f})",
            f"Borderline calibrated P(pneumonia) = {p_pneumonia:.2f}" if p_pneumonia is not None else "Borderline output distribution",
            f"Probability calibration: {calib_str} (temperature T = {temp:.2f})",
            f"Image quality: {quality_status} (passed quality checks)",
            "Out-of-distribution check: passed (within in-distribution feature bounds)",
            "Heatmap highlights regions influencing the leading class; not confirmation of a medical finding",
        ]
        return explanation, evidence

    if status == "success":
        finding_name = finding if finding is not None else "unknown"
        prob_val = probability if probability is not None else 0.0
        unc_val = uncertainty if uncertainty is not None else "unknown"
        p_pneumonia = details.get("p_pneumonia_calibrated", prob_val)

        explanation = (
            f"Model analysis completed: {finding_name}-class finding identified with calibrated probability {prob_val:.2f} "
            f"({calib_str}). Uncertainty level is evaluated as {unc_val}. "
            "The highlighted regions show where the model's output was most sensitive; this is not confirmation of a medical finding. "
            "This is an educational prototype and not a clinical diagnosis."
        )
        evidence = [
            f"Predicted finding: {finding_name}-class pattern",
            f"Calibrated probability P({finding_name}) = {prob_val:.2f} ({calib_str}, T = {temp:.2f})",
            f"Calibrated P(pneumonia) = {p_pneumonia:.2f}",
            f"Uncertainty rating: {unc_val} (confidence = {conf:.2f})",
            f"Image quality assessment: {quality_status}",
            "Out-of-distribution check: passed (in-distribution pediatric radiograph)",
            "Heatmap highlights regions influencing the output; not confirmation of a medical finding",
        ]
        return explanation, evidence

    # Fallback generic safe explanation
    explanation = (
        "Educational prototype analysis completed. This is an educational prototype and not a clinical diagnosis."
    )
    evidence = [
        f"Status: {status}",
        "Educational prototype: not a clinical diagnosis",
    ]
    return explanation, evidence
