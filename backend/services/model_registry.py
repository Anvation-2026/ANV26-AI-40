"""
Unified Medical Model Registry Service
MedGuard AI - Clinical Evidence & Triage Support System
Section 10 Implementation:
Maintains state, metadata, hardware specifications, and capabilities
for all registered models in the MedGuard AI ecosystem:
1. ResNet-18 — Pneumonia Classification
2. ResNet-18 — Tuberculosis Classification
3. DenseNet-201 — Multi-Label Chest Pathology (14 Classes)
4. ConvNeXt-Base — Multi-Region Bone Fracture Analysis
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelRegistry:
    """Central registry tracking all AI diagnostic models and hardware memory state."""

    def __init__(self):
        self.repo_root = REPO_ROOT
        self.models_dir = self.repo_root / "ml" / "models"

    def get_gpu_memory_summary(self) -> Dict[str, Any]:
        """Returns physical GPU memory metrics if CUDA is available."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
            reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
            total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
            return {
                "cuda_available": True,
                "device_name": device_name,
                "total_memory_mb": total_mb,
                "allocated_memory_mb": allocated_mb,
                "reserved_memory_mb": reserved_mb,
                "free_memory_mb": round(total_mb - reserved_mb, 2)
            }
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "total_memory_mb": 0,
            "allocated_memory_mb": 0,
            "reserved_memory_mb": 0,
            "free_memory_mb": 0
        }

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Returns comprehensive metadata for all registered models."""
        entries = [
            self._get_pneumonia_entry(),
            self._get_tb_entry(),
            self._get_chest14_entry(),
            self._get_fracture_entry()
        ]
        return entries

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        for m in self.get_all_models():
            if m["model_id"] == model_id:
                return m
        return None

    def _get_pneumonia_entry(self) -> Dict[str, Any]:
        ckpt = self.models_dir / "best_model.pth"
        exists = ckpt.exists()
        return {
            "model_id": "resnet18_pneumonia",
            "task_name": "Pneumonia Detection",
            "modality": "Chest Radiograph (CXR)",
            "architecture": "ResNet-18 + MLP Classifier",
            "parameter_count": "11,180,098",
            "model_version": "1.0.0",
            "checkpoint_path": str(ckpt),
            "checkpoint_exists": exists,
            "checkpoint_size_mb": round(ckpt.stat().st_size / (1024**2), 2) if exists else 0,
            "validation_status": "validated",
            "inference_status": "available" if exists else "checkpoint_not_found",
            "training_dataset": "Chest X-Ray Pneumonia (Kaggle / Kermany et al.)",
            "labels": ["Normal", "Pneumonia"],
            "input_requirements": {
                "resolution": [224, 224],
                "channels": 3,
                "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "format": "DICOM, PNG, JPEG"
            },
            "capabilities": ["binary_classification", "gradcam", "quality_assessment"],
            "decision_threshold": 0.50
        }

    def _get_tb_entry(self) -> Dict[str, Any]:
        ckpt = self.models_dir / "best_model_tb.pth"
        exists = ckpt.exists()
        return {
            "model_id": "resnet18_tb",
            "task_name": "Tuberculosis Detection",
            "modality": "Chest Radiograph (CXR)",
            "architecture": "ResNet-18 + MLP Classifier",
            "parameter_count": "11,180,098",
            "model_version": "1.0.0",
            "checkpoint_path": str(ckpt),
            "checkpoint_exists": exists,
            "checkpoint_size_mb": round(ckpt.stat().st_size / (1024**2), 2) if exists else 0,
            "validation_status": "validated",
            "inference_status": "available" if exists else "checkpoint_not_found",
            "training_dataset": "TBX11K Benchmark (Active TB / Latent TB / Controls)",
            "labels": ["Normal", "Tuberculosis"],
            "input_requirements": {
                "resolution": [224, 224],
                "channels": 3,
                "normalization": "ImageNet",
                "format": "DICOM, PNG, JPEG"
            },
            "capabilities": ["binary_classification", "gradcam", "quality_assessment"],
            "decision_threshold": 0.50
        }

    def _get_chest14_entry(self) -> Dict[str, Any]:
        ckpt = self.models_dir / "densenet201" / "best_model.pth"
        exists = ckpt.exists()
        return {
            "model_id": "densenet201_chest14",
            "task_name": "Multi-Label Chest Pathology Analysis",
            "modality": "Chest Radiograph (CXR)",
            "architecture": "DenseNet-201 + Multi-Label Head (1920 -> 512 -> 256 -> 14)",
            "parameter_count": "19,252,942",
            "model_version": "2.0.0",
            "checkpoint_path": str(ckpt),
            "checkpoint_exists": exists,
            "checkpoint_size_mb": round(ckpt.stat().st_size / (1024**2), 2) if exists else 0,
            "validation_status": "trained",
            "inference_status": "available" if exists else "checkpoint_not_found",
            "training_dataset": "ChestMNIST 224x224 (NIH ChestX-ray14 Subset)",
            "labels": [
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
            ],
            "input_requirements": {
                "resolution": [224, 224],
                "channels": 3,
                "normalization": "ImageNet",
                "format": "DICOM, PNG, JPEG"
            },
            "capabilities": ["multi_label_classification", "gradcam", "uncertainty_estimation"],
            "decision_threshold": "Per-class calibrated thresholds"
        }

    def _get_fracture_entry(self) -> Dict[str, Any]:
        ckpt = self.models_dir / "fracture_convnext_base" / "best_model.pth"
        exists = ckpt.exists()
        
        # Load calibration if available
        calib_file = self.models_dir / "fracture_convnext_base" / "calibration.json"
        opt_thresh = 0.5200
        if calib_file.exists():
            try:
                with open(calib_file, "r") as f:
                    c = json.load(f)
                    opt_thresh = float(c.get("optimal_threshold_youden", 0.5200))
            except Exception:
                pass

        return {
            "model_id": "convnext_base_fracture",
            "task_name": "Multi-Region Bone Fracture Analysis",
            "modality": "Musculoskeletal Radiograph (MSK X-Ray)",
            "architecture": "ConvNeXt-Base + 3-Stage GELU MLP Head (1024 -> 512 -> 256 -> 1)",
            "parameter_count": "88,222,849",
            "model_version": "1.0.0",
            "checkpoint_path": str(ckpt),
            "checkpoint_exists": exists,
            "checkpoint_size_mb": round(ckpt.stat().st_size / (1024**2), 2) if exists else 0,
            "validation_status": "validated",
            "inference_status": "available" if exists else "checkpoint_not_found",
            "training_dataset": "Graz Pediatric Wrist (15.12 GB) + FracAtlas Benchmark (0.32 GB)",
            "labels": ["Fracture-related findings not detected", "Fracture-related findings present"],
            "supported_anatomies": ["wrist", "hand", "leg", "hip", "shoulder", "mixed"],
            "input_requirements": {
                "resolution": [224, 224],
                "channels": 3,
                "dynamic_range": "16-bit PNG percentile-scaled (P0.5-P99.5) and 8-bit JPEG",
                "normalization": "ImageNet"
            },
            "capabilities": ["binary_classification", "spatial_gradcam", "uncertainty_margin", "quality_assessment"],
            "decision_threshold": opt_thresh,
            "test_performance": {
                "auroc": 0.9388,
                "pr_auc": 0.9583,
                "sensitivity": 0.8588,
                "specificity": 0.9036,
                "f1_score": 0.8971
            },
            "limitations": (
                "Educational triage prototype. High specificity across all regions; "
                "sensitivity is highest on wrist (87.2%) and lower on hand/leg. "
                "Requires human clinical confirmation."
            )
        }


# Singleton instance
model_registry = ModelRegistry()
