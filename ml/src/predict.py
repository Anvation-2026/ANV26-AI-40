import argparse
import base64
import io
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

from config import (
    CLASS_NAMES,
    HEATMAPS_DIR,
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    STANDARD_LIMITATIONS,
)
from src.evidence import build_explanation_and_evidence
from src.model import load_checkpoint
from src.preprocessing import InvalidImageError, preprocess_image, to_gray_uint8


class ModelUnavailableError(RuntimeError):
    """Raised when the ML model checkpoint cannot be loaded or is unavailable."""
    pass


_PREDICTOR_INSTANCE: Optional["MedGuardPredictor"] = None
_PREDICTOR_LOCK = threading.Lock()


class MedGuardPredictor:
    """
    Thread-safe, robust predictor implementing the MedGuard AI ML contract.
    Loads ResNet-18 checkpoint and whichever optional artifacts are present.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self.lock = threading.Lock()

        # Target device detection
        if device is not None:
            self.device_str = device
        else:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        # Load primary checkpoint
        ckpt_path = self.models_dir / "best_model.pth"
        if not ckpt_path.exists():
            raise ModelUnavailableError(f"Model checkpoint not found at {ckpt_path}")

        try:
            self.model, self.metadata = load_checkpoint(ckpt_path, device=self.device_str)
            self.model_version = self.metadata.get("full_model_version", "resnet18-pneumoniamnist224-v1")
            self.model_name = "ResNet-18 (ImageNet-pretrained, fine-tuned)"
            ds_info = self.metadata.get("dataset", {})
            self.dataset_name = ds_info.get("name", "PneumoniaMNIST+ (224x224)") if isinstance(ds_info, dict) else "PneumoniaMNIST+ (224x224)"
            self.classes = self.metadata.get("class_names", ["normal", "pneumonia"])
            self.supported_modality = "Chest X-ray (educational, pediatric benchmark)"
        except Exception as e:
            raise ModelUnavailableError(f"Failed to load checkpoint at {ckpt_path}: {e}") from e

        # Capabilities registry
        self.capabilities = {
            "calibration": False,
            "uncertainty": False,
            "quality": False,
            "ood": False,
            "gradcam": False,
        }

        # Optional Artifacts
        self.temperature = 1.0
        self.thresholds = {
            "tau_accept": 0.80,
            "tau_low_uncertainty": 0.95,
            "decision_threshold": 0.50,
        }
        self.quality_evaluator = None
        self.ood_evaluator = None
        self.gradcam_engine = None

        # Load Calibration artifact
        calib_path = self.models_dir / "calibration.json"
        if calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    cal_data = json.load(f)
                    self.temperature = float(cal_data.get("temperature", 1.0))
                    self.capabilities["calibration"] = True
            except Exception as e:
                log.warning("Failed loading calibration.json: %s", e)

        # Load Thresholds artifact
        thresh_path = self.models_dir / "thresholds.json"
        if thresh_path.exists():
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    th_data = json.load(f)
                    self.thresholds["tau_accept"] = float(th_data.get("tau_accept", 0.80))
                    self.thresholds["tau_low_uncertainty"] = float(th_data.get("tau_low_uncertainty", 0.95))
                    self.thresholds["decision_threshold"] = float(th_data.get("decision_threshold", 0.50))
                    self.capabilities["uncertainty"] = True
            except Exception as e:
                log.warning("Failed loading thresholds.json: %s", e)

        # Optional Quality module
        quality_path = self.models_dir / "quality_thresholds.json"
        if quality_path.exists():
            try:
                from src.quality import QualityGate
                self.quality_evaluator = QualityGate(quality_path)
                self.capabilities["quality"] = True
            except Exception as e:
                log.warning("Failed to initialize QualityGate: %s", e)

        # Optional OOD module
        ood_path = self.models_dir / "ood_stats.npz"
        if ood_path.exists():
            try:
                from src.ood import OODDetector
                self.ood_evaluator = OODDetector(ood_path, thresh_path)
                self.capabilities["ood"] = True
            except Exception as e:
                log.warning("Failed to initialize OODDetector: %s", e)

        # Optional Grad-CAM module
        try:
            from src.gradcam import GradCAM
            self.gradcam_engine = GradCAM(self.model, target_layer_name="layer4")
            self.capabilities["gradcam"] = True
        except Exception:
            pass  # Grad-CAM not ready yet (e.g. Phase 1 — module not yet built)


    def predict(
        self,
        image: Union[bytes, str, Path, Image.Image, np.ndarray],
        *,
        generate_heatmap: bool = True,
        heatmap_dir: Optional[Union[str, Path]] = None,
        include_base64: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes prediction pipeline on single image input.
        Never raises for bad user inputs.
        Thread-safe execution.
        """
        start_time = time.perf_counter()
        rejection_reason = None

        # 1. Image decoding & preprocessing
        try:
            tensor, gray224 = preprocess_image(image)
        except InvalidImageError as e:
            inference_ms = round((time.perf_counter() - start_time) * 1000, 2)
            rejection_reason = str(e)
            details = {
                "p_pneumonia_raw": None,
                "p_pneumonia_calibrated": None,
                "confidence": None,
                "uncertainty_score": None,
                "temperature": float(self.temperature),
                "calibrated": self.capabilities["calibration"],
                "thresholds": self.thresholds,
                "quality": None,
                "ood": None,
                "rejection_reason": rejection_reason,
                "inference_ms": inference_ms,
                "device": self.device_str,
                "capabilities": self.capabilities,
            }
            exp, ev = build_explanation_and_evidence(
                status="unsupported_file",
                finding=None,
                probability=None,
                uncertainty=None,
                quality_status=None,
                quality_details={},
                ood_flagged=None,
                ood_details={},
                details=details,
            )
            return {
                "status": "unsupported_file",
                "finding": None,
                "probability": None,
                "uncertainty": None,
                "quality": None,
                "ood": None,
                "explanation": exp,
                "evidence": ev,
                "heatmap_path": None,
                "model_version": self.model_version,
                "limitations": STANDARD_LIMITATIONS,
                "details": details,
            }
        except Exception as e:
            inference_ms = round((time.perf_counter() - start_time) * 1000, 2)
            rejection_reason = f"Unexpected decode failure: {e}"
            details = {
                "p_pneumonia_raw": None,
                "p_pneumonia_calibrated": None,
                "confidence": None,
                "uncertainty_score": None,
                "temperature": float(self.temperature),
                "calibrated": self.capabilities["calibration"],
                "thresholds": self.thresholds,
                "quality": None,
                "ood": None,
                "rejection_reason": rejection_reason,
                "inference_ms": inference_ms,
                "device": self.device_str,
                "capabilities": self.capabilities,
            }
            exp, ev = build_explanation_and_evidence(
                status="unsupported_file",
                finding=None,
                probability=None,
                uncertainty=None,
                quality_status=None,
                quality_details={},
                ood_flagged=None,
                ood_details={},
                details=details,
            )
            return {
                "status": "unsupported_file",
                "finding": None,
                "probability": None,
                "uncertainty": None,
                "quality": None,
                "ood": None,
                "explanation": exp,
                "evidence": ev,
                "heatmap_path": None,
                "model_version": self.model_version,
                "limitations": STANDARD_LIMITATIONS,
                "details": details,
            }

        # 2. Quality assessment
        quality_res = {
            "status": "acceptable",
            "metrics": {
                "blur_laplacian_var": 0.0,
                "brightness_mean": 0.0,
                "contrast_p99_p1": 0.0,
                "noise_sigma": 0.0,
            },
            "thresholds": None,
            "passed": None,
            "labels": None,
            "reasons": [],
        }
        if self.quality_evaluator is not None:
            quality_res = self.quality_evaluator.assess(gray224)

        # 3. Modality & OOD checks
        ood_res = {
            "flagged": False,
            "method": "mahalanobis+modality",
            "score": 0.0,
            "threshold": 0.0,
            "reasons": [],
        }
        if self.ood_evaluator is not None:
            ood_res = self.ood_evaluator.assess(image, gray224)

        # 4. Neural Network Forward Pass & Grad-CAM
        tensor = tensor.to(self.device)
        penultimate_features = None

        with self.lock:
            try:
                # Need grad for GradCAM if enabled
                with torch.set_grad_enabled(generate_heatmap and self.gradcam_engine is not None):
                    logits, penultimate_features = self._forward_with_features(tensor)
            except torch.cuda.OutOfMemoryError:
                log.warning("CUDA OOM during inference — falling back to CPU for this request.")
                torch.cuda.empty_cache()
                self.device_str = "cpu"
                self.device = torch.device("cpu")
                self.model.to("cpu")
                tensor = tensor.to("cpu")
                with torch.set_grad_enabled(generate_heatmap and self.gradcam_engine is not None):
                    logits, penultimate_features = self._forward_with_features(tensor)

        # Feature-space OOD score update if evaluator supports it
        if self.ood_evaluator is not None and penultimate_features is not None:
            ood_res = self.ood_evaluator.score_features(penultimate_features.detach().cpu().numpy(), ood_res)

        # Sanitize OOD dict: convert numpy scalars -> Python natives
        ood_res = self._sanitize_json(ood_res)

        # Compute probabilities
        logits_np = logits.detach().cpu().squeeze().numpy()
        exp_raw = np.exp(logits_np - np.max(logits_np))
        p_raw = exp_raw / np.sum(exp_raw)
        p_pneumonia_raw = float(p_raw[1])

        # Temperature calibration
        temp = max(1e-4, float(self.temperature))
        scaled_logits = logits_np / temp
        exp_cal = np.exp(scaled_logits - np.max(scaled_logits))
        p_calibrated = exp_cal / np.sum(exp_cal)
        p_pneumonia_cal = float(p_calibrated[1])

        conf = float(np.max(p_calibrated))
        # Normalized entropy: -sum(p * log2(p))
        eps = 1e-12
        entropy = -float(np.sum(p_calibrated * np.log2(p_calibrated + eps)))
        uncertainty_score = round(entropy, 4)

        # Determine uncertainty level
        tau_accept = self.thresholds.get("tau_accept", 0.80)
        tau_low = self.thresholds.get("tau_low_uncertainty", 0.95)

        if conf < tau_accept:
            uncertainty_level = "high"
        elif conf < tau_low:
            uncertainty_level = "medium"
        else:
            uncertainty_level = "low"

        # Determine Status via Precedence Order
        # 1. Unsupported (already handled above)
        # 2. Modality check fails
        if ood_res.get("modality_failed", False):
            status = "ood_rejected"
            rejection_reason = "unsupported_modality:color_image"
        # 3. Quality gate fails
        elif quality_res["status"] == "poor":
            status = "poor_quality"
            rejection_reason = f"poor_quality: {', '.join(quality_res['reasons'])}"
        # 4. Feature OOD flagged
        elif ood_res["flagged"]:
            status = "ood_rejected"
            rejection_reason = f"out_of_distribution: {', '.join(ood_res['reasons'])}"
        # 5. Borderline confidence below acceptance threshold
        elif conf < tau_accept:
            status = "uncertain"
        # 6. Success
        else:
            status = "success"

        # Findings & Probability assignment based on status
        if status == "success":
            argmax_idx = int(np.argmax(p_calibrated))
            finding = CLASS_NAMES[argmax_idx]
            probability = round(float(p_calibrated[argmax_idx]), 4)
        else:
            finding = None
            probability = None

        # Grad-CAM overlay generation
        heatmap_path = None
        heatmap_base64 = None
        heatmap_target_class: Optional[str] = None
        heatmap_note: Optional[str] = None

        if generate_heatmap and self.gradcam_engine is not None:
            if status in ["success", "uncertain"]:
                try:
                    target_class_idx = int(np.argmax(p_calibrated))
                    heatmap_target_class = CLASS_NAMES[target_class_idx]
                    hm_dir = Path(heatmap_dir) if heatmap_dir else HEATMAPS_DIR
                    hm_dir.mkdir(parents=True, exist_ok=True)
                    unique_name = f"cam_{uuid.uuid4().hex[:12]}.png"
                    full_hm_path = hm_dir / unique_name

                    # Generate and save heatmap overlay
                    overlay_arr = self.gradcam_engine.generate(
                        tensor,
                        gray224,
                        target_class=target_class_idx,
                        save_path=full_hm_path,
                    )

                    # Path relative to OUTPUTS_DIR
                    try:
                        heatmap_path = str(full_hm_path.relative_to(OUTPUTS_DIR)).replace("\\", "/")
                    except ValueError:
                        heatmap_path = str(full_hm_path).replace("\\", "/")

                    if include_base64 and overlay_arr is not None:
                        # Encode to PNG base64 (no data-URI prefix)
                        success_enc, buffer = cv2.imencode(".png", overlay_arr)
                        if success_enc:
                            heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

                    if status == "uncertain":
                        heatmap_note = (
                            "Heatmap generated but confidence is below acceptance threshold; "
                            "visual attribution may be unreliable."
                        )

                    # Cleanup old heatmaps
                    self._cleanup_old_heatmaps(hm_dir, max_files=200)

                except Exception as e:
                    log.warning("Grad-CAM generation failed: %s", e)
                    heatmap_path = None
                    heatmap_target_class = None
            else:
                # Status is rejected — heatmap intentionally suppressed
                heatmap_note = f"Heatmap suppressed: status is '{status}'"

        inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

        details: Dict[str, Any] = {
            "p_pneumonia_raw": round(p_pneumonia_raw, 4),
            "p_pneumonia_calibrated": round(p_pneumonia_cal, 4),
            "confidence": round(conf, 4),
            "uncertainty_score": uncertainty_score,
            "temperature": round(temp, 4),
            "calibrated": self.capabilities["calibration"],
            "thresholds": self.thresholds,
            "quality": quality_res,
            "ood": ood_res,
            "rejection_reason": rejection_reason,
            "inference_ms": inference_ms,
            "device": self.device_str,
            "capabilities": self.capabilities,
            # Task 5: heatmap metadata for UI suppression logic
            "heatmap_target_class": heatmap_target_class,
            "heatmap_note": heatmap_note,
        }
        if include_base64 and heatmap_base64 is not None:
            details["heatmap_base64"] = heatmap_base64

        explanation, evidence = build_explanation_and_evidence(
            status=status,
            finding=finding,
            probability=probability,
            uncertainty=uncertainty_level,
            quality_status=quality_res.get("status"),
            quality_details=quality_res,
            ood_flagged=ood_res.get("flagged"),
            ood_details=ood_res,
            details=details,
        )

        return {
            "status": status,
            "finding": finding,
            "probability": probability,
            "uncertainty": uncertainty_level if status != "unsupported_file" else None,
            "quality": quality_res["status"],
            "ood": bool(ood_res["flagged"]),
            "explanation": explanation,
            "evidence": evidence,
            "heatmap_path": heatmap_path,
            "model_version": self.model_version,
            "limitations": STANDARD_LIMITATIONS,
            "details": details,
        }

    def _forward_with_features(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs model forward pass, extracting penultimate avg-pool features (512-d)
        and final classification logits (2-d).
        """
        x = self.model.conv1(tensor)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        features = self.model.avgpool(x)
        features_flat = torch.flatten(features, 1)
        logits = self.model.fc(features_flat)
        return logits, features_flat

    @staticmethod
    def _sanitize_json(obj: Any) -> Any:
        """
        Recursively converts numpy scalars to native Python types so that
        json.dumps() does not raise TypeError on np.bool_, np.int*, np.float*.
        """
        if isinstance(obj, dict):
            return {k: MedGuardPredictor._sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MedGuardPredictor._sanitize_json(v) for v in obj]
        # numpy scalar types
        if hasattr(obj, "item"):  # np.generic has .item()
            return obj.item()
        return obj

    @staticmethod
    def _cleanup_old_heatmaps(dir_path: Path, max_files: int = 200) -> None:
        """Removes oldest heatmap files if count exceeds max_files."""
        try:
            files = list(dir_path.glob("*.png"))
            if len(files) > max_files:
                files.sort(key=lambda p: p.stat().st_mtime)
                for f in files[: len(files) - max_files]:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        except Exception:
            pass


def get_predictor() -> MedGuardPredictor:
    """Returns thread-safe singleton predictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        with _PREDICTOR_LOCK:
            if _PREDICTOR_INSTANCE is None:
                _PREDICTOR_INSTANCE = MedGuardPredictor()
    return _PREDICTOR_INSTANCE


def predict_image(image: Any, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper function over get_predictor().predict."""
    return get_predictor().predict(image, **kwargs)


def model_status() -> Dict[str, Any]:
    """
    Returns system and model availability diagnostics.
    Never raises an exception.
    """
    try:
        predictor = get_predictor()
        val_report_exists = (REPORTS_DIR / "validation_report.json").exists()
        use_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if use_cuda else "N/A"

        return {
            "available": True,
            "model_name": predictor.model_name,
            "model_version": predictor.model_version,
            "dataset": predictor.dataset_name,
            "classes": predictor.classes,
            "supported_modality": predictor.supported_modality,
            "device": predictor.device_str,
            "gpu_name": gpu_name,
            "capabilities": predictor.capabilities,
            "validation_report_available": val_report_exists,
            "trained_on": "PneumoniaMNIST 224 (pediatric chest X-ray, educational)",
            "error": None,
        }
    except Exception as e:
        use_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if use_cuda else "N/A"
        return {
            "available": False,
            "model_name": None,
            "model_version": "unavailable",
            "dataset": None,
            "classes": ["normal", "pneumonia"],
            "supported_modality": "Chest X-ray (educational, pediatric benchmark)",
            "device": "cuda" if use_cuda else "cpu",
            "gpu_name": gpu_name,
            "capabilities": {
                "calibration": False,
                "uncertainty": False,
                "quality": False,
                "ood": False,
                "gradcam": False,
            },
            "validation_report_available": False,
            "trained_on": "PneumoniaMNIST 224 (pediatric chest X-ray, educational)",
            "error": str(e),
        }


def get_validation_report() -> Dict[str, Any]:
    """
    Reads reports/validation_report.json.
    Raises FileNotFoundError if absent.
    """
    report_path = REPORTS_DIR / "validation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Validation report not found at {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard AI CLI Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap overlay generation")
    args = parser.parse_args()

    result = predict_image(args.image, generate_heatmap=not args.no_heatmap)
    print(json.dumps(result, indent=2))
