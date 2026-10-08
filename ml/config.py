from pathlib import Path

# Base Paths (ml directory is root for this module)
ML_ROOT = Path(__file__).resolve().parent
DATA_DIR = ML_ROOT / "data"
MODELS_DIR = ML_ROOT / "models"
REPORTS_DIR = ML_ROOT / "reports"
OUTPUTS_DIR = ML_ROOT / "outputs"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
DEMO_SAMPLES_DIR = ML_ROOT / "demo_samples"
TESTS_DIR = ML_ROOT / "tests"

# Dataset Constants
DATASET_FILENAME = "pneumoniamnist_224.npz"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_DOWNLOAD_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1"

EXPECTED_SPLIT_SIZES = {
    "train": 4708,
    "val": 524,
    "test": 624,
}
TOTAL_EXPECTED_SAMPLES = 5856

# Class mapping & labels
CLASS_NAMES = ["normal", "pneumonia"]
LABEL_MAPPING = {0: "normal", 1: "pneumonia"}
POSITIVE_CLASS = "pneumonia"
POSITIVE_CLASS_IDX = 1

# Image specifications
IMAGE_SIZE = 224
MIN_RAW_IMAGE_DIM = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reproducibility
SEED = 42

# Training Defaults
DEFAULT_BATCH_SIZE = 64
FALLBACK_BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
NUM_WORKERS = 0

# Model Version String template
MODEL_VERSION_BASE = "resnet18-pneumoniamnist224-v1"

# Abstention & Uncertainty thresholds defaults
DEFAULT_TARGET_ACCEPTED_ACC = 0.95
DEFAULT_MIN_COVERAGE = 0.60
DEFAULT_TARGET_LOW_UNCERTAINTY_ACC = 0.98
DEFAULT_MIN_LOW_UNCERTAINTY_COVERAGE = 0.30
DEFAULT_DECISION_THRESHOLD = 0.50

# OOD & Modality Constants
MODALITY_COLOR_THRESHOLD = 30.0 / 255.0  # Mean absolute inter-channel difference (accommodates subtle compression chroma)
DEFAULT_OOD_PERCENTILE = 99.0

# Rejection Precedence Order
# Can be tested/swapped in Phase 3 precedence experiment
REJECTION_PRECEDENCE = [
    "unsupported_file",
    "modality_check",
    "quality_gate",
    "feature_ood",
    "uncertainty_abstention",
    "success",
]

# Standard Educational Limitations
STANDARD_LIMITATIONS = [
    "Educational research prototype; not a medical device and not a clinical diagnosis.",
    "Trained and evaluated on PneumoniaMNIST (pediatric chest X-rays, downsampled public benchmark); performance on adults, other hospitals, scanners, or populations is unknown.",
    "Calibrated probabilities reflect average reliability on validation data, not a guarantee that any individual prediction is correct.",
    "Grad-CAM heatmaps show where the model's output was sensitive; they do not confirm a medical finding.",
    "Image-quality and out-of-distribution checks are heuristic; some unsuitable or unsupported images may pass, and some valid images may be rejected.",
    "The system cannot guarantee zero false negatives.",
    "No demographic metadata is available, so subgroup fairness could not be evaluated.",
    "Validation and test splits may differ in distribution; reported test performance is the unbiased estimate.",
    "Quality-degradation tests use synthetic degradations created for this project.",
]
