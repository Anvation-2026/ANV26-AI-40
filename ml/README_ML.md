# MedGuard AI — Machine Learning Subsystem
### Team Vouken | AI/ML Engineering

MedGuard AI is an educational decision-support research prototype for pediatric chest radiograph interpretation. It combines a deep convolutional neural network (ResNet-18) with a rigorous five-layer safety pipeline:

1. **Input validation & single-point-of-truth preprocessing** (224x224, grayscale, ImageNet normalization)
2. **Automated image quality gate** (sharpness, brightness, contrast, noise)
3. **Out-of-distribution (OOD) detection** (inter-channel modality pre-check + 512-d penultimate Mahalanobis distance)
4. **Post-hoc probability calibration** (temperature scaling fitted on validation NLL)
5. **Selective classification & uncertainty abstention** (empirical confidence thresholding)
6. **Interpretability** (Grad-CAM heatmaps for leading class predictions)

---

## 1. Quick Start & Exact Reproduction Sequence

All commands are designed to be run from inside the `ml/` directory:

```bash
# 1. Navigate to ml/ directory
cd ml

# 2. Setup virtual environment & dependencies
python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 3. Verify dataset & check SHA-256 + duplicates
python -m src.dataset --check

# 4. Train the model (ResNet-18, class-weighted CrossEntropy, Cosine Annealing, AMP on CUDA)
python -m src.train --epochs 15 --batch-size 64 --lr 1e-4 --seed 42

# 5. Fit temperature scaling and abstention thresholds (validation split only)
python -m src.calibration --fit

# 6. Fit image quality thresholds and evaluate synthetic degradations
python -m src.quality --fit
python -m src.quality --eval

# 7. Fit Mahalanobis OOD detector on training features and evaluate proxy sets
python -m src.ood --fit
python -m src.ood --eval

# 8. Run full evaluation suite (bootstrap 95% CIs, selective prediction, calibration)
python -m src.evaluate --full

# 9. Export live demo samples with complete provenance manifest
python -m src.export_demo_samples

# 10. Run complete unit test suite
pytest -q

# 11. Run CLI prediction on sample images
python -m src.predict --image demo_samples/positive_1.png
python -m src.predict --image demo_samples/negative_1.png
python -m src.predict --image demo_samples/uncertain_1.png
python -m src.predict --image demo_samples/poor_quality_blur.png
python -m src.predict --image demo_samples/ood_noise.png
```

---

## 2. Directory Layout

```
ml/
├── config.py                  # Single source of truth for paths, thresholds, seeds
├── requirements.txt           # Pinned production dependencies
├── ML_INTEGRATION.md          # Integration contract for FastAPI backend (Samuel)
├── README_ML.md               # This reproduction manual
├── data/                      # Dataset storage (PneumoniaMNIST+ 224x224 npz)
├── models/
│   ├── best_model.pth         # Trained PyTorch checkpoint (state_dict + metadata)
│   ├── best_model.prev.pth    # Preserved previous checkpoint
│   ├── model_card.json        # Machine & human readable model card
│   ├── artifacts_manifest.json# SHA-256 integrity manifest for all model artifacts
│   ├── calibration.json       # Temperature scaling metadata & ECE stats
│   ├── thresholds.json        # Acceptance & uncertainty thresholds (tau)
│   ├── quality_thresholds.json# Data-fitted quality percentiles
│   └── ood_stats.npz          # Class means & precision matrix for Mahalanobis OOD
├── outputs/
│   └── heatmaps/              # Generated Grad-CAM overlay PNG files
├── reports/
│   ├── dataset_summary.json   # Split sizes, class balance, duplicate analysis
│   ├── training_log.csv       # Per-epoch training/validation metrics
│   ├── training_curves.png    # Loss and AUROC curves
│   ├── test_metrics_basic.json# Basic test split metrics
│   ├── reliability_*.png      # Reliability diagrams before/after calibration
│   ├── risk_coverage_*.png    # Selective prediction risk-coverage curves
│   ├── quality_eval.json      # Synthetic degradation detection tables
│   ├── ood_eval.json          # OOD proxy detection rates and AUROC
│   ├── false_negatives_grid.png # Visual audit grid of test false negatives
│   ├── validation_report.json # Comprehensive JSON metrics for dashboard
│   └── validation_report.md   # GitHub-formatted full validation report
├── demo_samples/              # Demo images and manifest.json for full-stack integration
├── src/
│   ├── dataset.py             # Data loading, duplicate hash audit, summary
│   ├── preprocessing.py       # Deterministic Pillow/OpenCV image pipeline
│   ├── model.py               # ResNet-18 architecture & checkpoint management
│   ├── train.py               # Reproducible training loop with early stopping
│   ├── calibration.py         # Temperature scaling & abstention threshold fitting
│   ├── quality.py             # Image quality metrics & gate
│   ├── ood.py                 # Color modality pre-check & Mahalanobis OOD detector
│   ├── gradcam.py             # Native Grad-CAM overlay generator with hook safety
│   ├── evidence.py            # Grounded decision-support evidence builder
│   ├── predict.py             # Thread-safe predictor singleton & CLI
│   ├── evaluate.py            # Comprehensive evaluation suite & bootstrap CIs
│   └── export_demo_samples.py # Demo sample exporter & false-negative renderer
└── tests/                     # 20 unit tests covering all safety and contract invariants
```

---

## 3. Real Performance Metrics (Test Split Benchmark)

Evaluated on official **Test split** (N = 624, unaugmented):

| Metric | Test Value (Calibrated) | 95% Bootstrap CI | Validation Value |
|---|---|---|---|
| **AUROC** | **0.9929** | [0.9880, 0.9968] | 0.9990 |
| **Accuracy** | **0.9551** | [0.9375, 0.9712] | 0.9790 |
| **Sensitivity (Recall)** | **0.9897** | [0.9791, 1.0000] | 0.9769 |
| **Specificity** | **0.8974** | [0.8583, 0.9342] | 0.9852 |
| **Precision** | **0.9415** | - | 0.9948 |
| **F1 Score** | **0.9650** | - | 0.9857 |
| **False-Negative Rate** | **0.0103** | - | 0.0231 |

### Confusion Matrix (Test Split)
```
                Predicted Normal    Predicted Pneumonia
Actual Normal        210                 24
Actual Pneumonia       4                386
```

### Calibration & Abstention
- **Learned Temperature $T$:** `1.5773` (fitted on validation split NLL minimization)
- **Test ECE:** `0.0349` $\to$ `0.0278` (improved post-calibration)
- **Acceptance Threshold $\tau_{\text{accept}}$:** `0.5618`
- **Selective Coverage on Test:** `98.56%` (615 accepted, 9 abstained/uncertain)
- **Error Rate on Accepted Samples:** `4.23%` vs **Error Rate on Abstained Samples:** `22.22%`

---

## 4. Key Limitations & Regulatory Disclaimers

1. **Educational prototype:** Not a medical device and not for clinical diagnosis or treatment planning.
2. **Pediatric population constraint:** The ResNet-18 pneumonia model was trained on PneumoniaMNIST (pediatric chest radiographs). Performance on adults or alternate imaging centers is unknown.
3. **Heatmap interpretation:** Grad-CAM overlays indicate regions where the neural network's activations were sensitive; they do not confirm pathology.
4. **Zero false negatives not guaranteed:** The model achieves 98.97% test sensitivity (4 false negatives out of 390 cases).

---

## 5. DenseNet-201 Multi-Label Chest Radiograph Subsystem

### A. Overview & Target Architecture
For comprehensive multi-label thoracic finding analysis across 14 co-existing pathologies, MedGuard AI integrates a specialized DenseNet-201 model:

```
Input: 1x224x224 grayscale radiograph
  │
  ▼
[3-Channel Adaptation + ImageNet Normalization]
  │
  ▼
[DenseNet-201 Backbone (ImageNet Pretrained)] ──► features.denseblock4 (Grad-CAM hook)
  │
  ▼
[Global Average Pooling] ──► 1920-d Penultimate Feature Vector (OOD Detector)
  │
  ▼
[MLP Classifier Head]
  ├─ Linear(1920 -> 512) -> ReLU -> Dropout(0.3)
  ├─ Linear(512 -> 256) -> ReLU -> Dropout(0.3)
  └─ Linear(256 -> 14 logits)
  │
  ├─ Training: BCEWithLogitsLoss with inverse-frequency pos_weights
  └─ Inference: Sigmoid -> 14 Independent Probabilities in [0, 1]
```

### B. Official Dataset (ChestMNIST 224x224)
- **Source**: MedMNIST v2 (NIH-ChestXray14 subset, Record 10519652)
- **Archive**: `ml/data/chestmnist_224.npz` (3,889,293,042 bytes, MD5: `45bd33e6f06c3e8cdb481c74a89152aa`)
- **Partitions**: Train (78,468), Val (11,219), Test (22,433) — Total: 112,120 images
- **14 Official Finding Labels**:
  `atelectasis`, `cardiomegaly`, `effusion`, `infiltration`, `mass`, `nodule`, `pneumonia`, `pneumothorax`, `consolidation`, `edema`, `emphysema`, `fibrosis`, `pleural`, `hernia`.
- *(Note: ChestMNIST does not include a Tuberculosis label; TB screening is decoupled into `train_tb.py`)*.

### C. 4 GB VRAM Optimization
- **Stage 1 (Warmup)**: Backbone frozen, training only the MLP classifier head (peak VRAM: **682.1 MB**).
- **Stage 2 (Fine-Tuning)**: Unfreezes `features.denseblock4` and `features.norm5`.
- **Micro-Batching**: Batch size 4 with 8 gradient accumulation steps ($\rightarrow$ Effective batch size: 32).
- **Mixed Precision**: Automatic FP16 scaling (`torch.amp.autocast('cuda')`).

### D. Training, Evaluation & API Commands
```bash
# 1. Download official 3.89 GB ChestMNIST dataset (resumable)
python ml/download_chestmnist.py

# 2. Run unit & GPU forward/backward tests
python -m pytest ml/tests/test_densenet.py -v

# 3. Train DenseNet-201 (Two-stage memory-conscious pipeline)
python ml/src/train_densenet.py --batch-size 4 --grad-accum 8 --head-epochs 5 --finetune-epochs 5

# 4. Multi-Label Inference API endpoint
# POST http://127.0.0.1:8000/api/predict/chestmnist (multipart/form-data with 'file')
```

---

## 6. Isolated Model Registry

| Model Name | Architecture | Task | Dataset | Checkpoint Path | Status |
|---|---|---|---|---|:---:|
| **MedGuard Pneumonia** | ResNet-18 + Linear | Binary (Normal / Pneumonia) | PneumoniaMNIST 224x224 | `ml/models/best_model.pth` | Active |
| **MedGuard TB** | ResNet-18 + Dropout + Head | Binary (Non-TB / TB) | TBX11K (12,278 imgs) | `ml/models/best_model_tb.pth` | Trained (AUROC 0.9975) |
| **MedGuard Chest-14** | DenseNet-201 + 3-layer MLP | 14-label Multi-Label | ChestMNIST 224x224 | `ml/models/densenet201/best_model.pth` | Active / Training |

