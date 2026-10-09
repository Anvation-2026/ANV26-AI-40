"""
Fracture Pipeline Configuration
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"D:\anvation")
ML_ROOT = PROJECT_ROOT / "ml"
DATA_ROOT = ML_ROOT / "data"

DATASET_A_ROOT = DATA_ROOT / "bone fracture"
DATASET_B_ROOT = DATA_ROOT / "FracAtlas"

TASKS_FRACTURE_ROOT = ML_ROOT / "tasks" / "fracture"
MODELS_DIR = ML_ROOT / "models" / "fracture_convnext_base"
REPORTS_DIR = ML_ROOT / "reports" / "fracture"

MANIFEST_TRAIN = TASKS_FRACTURE_ROOT / "manifest_train.csv"
MANIFEST_VAL = TASKS_FRACTURE_ROOT / "manifest_val.csv"
MANIFEST_TEST = TASKS_FRACTURE_ROOT / "manifest_test.csv"
MANIFEST_QUARANTINE = TASKS_FRACTURE_ROOT / "manifest_quarantine.csv"
DATASET_AUDIT_JSON = REPORTS_DIR / "dataset_audit.json"

CHECKPOINT_BEST = MODELS_DIR / "best_model.pth"
CHECKPOINT_LATEST = MODELS_DIR / "latest_checkpoint.pth"
MODEL_CONFIG_JSON = MODELS_DIR / "model_config.json"
LABEL_MAP_JSON = MODELS_DIR / "label_map.json"
CALIBRATION_JSON = MODELS_DIR / "calibration.json"
OOD_CONFIG_JSON = MODELS_DIR / "ood_config.json"

# Architecture
BACKBONE_NAME = "convnext_base"
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3
FEATURE_DIM = 1024
HEAD_DIMS = [1024, 512, 256, 1]
DROPOUT_RATE = 0.3
ACTIVATION = "GELU"

# Normalization constants (Standard ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Label Semantics
LABEL_MAP = {
    0: "Fracture-related findings not detected",
    1: "Fracture-related findings present"
}

# Anatomical Regions
SUPPORTED_ANATOMIES = ["wrist", "hand", "leg", "hip", "shoulder", "mixed"]

# Training Defaults
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR_HEAD = 1e-3
DEFAULT_LR_BACKBONE = 2e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_NUM_WORKERS = 0  # Safe Windows multiprocessing default

# Ensure necessary directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
