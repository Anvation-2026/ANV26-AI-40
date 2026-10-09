# MedGuard AI — Clinical Evidence & Triage Support System

> **Educational Research Prototype Notice**: MedGuard AI is an experimental decision-support prototype developed for instructional demonstration and educational research. It is **not a medical device**, has not been evaluated by clinical regulatory authorities (FDA/CE), and must **never be used for diagnosis, clinical staging, or patient treatment decisions**. Uploads must consist exclusively of public or de-identified educational images.

---

## 1. Product Summary

MedGuard AI provides transparent, rule-based decision support for educational medical imaging across both chest and musculoskeletal modalities:
- **Chest Radiograph Analysis**:
  - ResNet-18 (Normal vs. Pneumonia)
  - ResNet-18 (Normal vs. Tuberculosis)
  - DenseNet-201 (14 Multi-Label Chest Pathologies)
- **Multi-Region Bone Fracture Analysis**:
  - ImageNet-pretrained **ConvNeXt-Base** (88.2M parameters) with custom 3-stage GELU MLP classification head (`1024 -> 512 -> 256 -> 1`).
  - Supports radiographs across wrist, hand, leg, hip, and shoulder.
  - Generates spatial Grad-CAM activation heatmaps hooked to `features[7]`.
- **Strict Pre-Inference Quality Checks**: Heuristic evaluation of image blur, underexposure/overexposure, and dynamic range thresholds.
- **Out-of-Distribution (OOD) Guardrails**: Domain validation ensuring inputs match expected radiographic distribution.
- **Safety Abstention & Triage**: When model uncertainty exceeds acceptable thresholds (margin < 0.10) or image quality is degraded, the engine suppresses definitive findings and directly routes to expert human clinician review.
- **Unified Model Registry**: Live API tracking hardware footprint, checkpoint state, and parameter counts for all active models.
- **Honest Metrics Dashboard**: Independent validation metrics loaded strictly from benchmark reports without fabrication or hardcoded numbers.

---

## 2. Tech Stack

- **Frontend**: React 18, Vite, TypeScript (strict), Tailwind CSS v3.4, React Router v6, Recharts, Lucide React, native `fetch`.
- **Backend**: Python 3.12, FastAPI, Pydantic v2, Uvicorn, Pillow, python-multipart, pydantic-settings.
- **Machine Learning**: PyTorch 2.11+, torchvision, CUDA 12.8, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).
- **Testing**: pytest (backend: 46 unit & integration tests), vitest + @testing-library/react (frontend).

---

## 3. Quick Start & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm

### Backend Setup

```bash
# 1. Navigate to backend directory and create virtual environment
cd backend
python -m venv .venv

# Activate virtual environment:
# Windows (PowerShell):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Run backend server from repository root (port 8000)
cd ..
uvicorn backend.main:app --reload --port 8000
```

#### Demo Mode (Synthetic Educational Fixtures)
To run the application with synthetic demo fixtures (no model weights required):
```powershell
# Windows PowerShell:
$env:MEDGUARD_MODE="demo"; uvicorn backend.main:app --reload --port 8000
```
```bash
# macOS / Linux:
MEDGUARD_MODE=demo uvicorn backend.main:app --reload --port 8000
```

---

### Frontend Setup

```bash
# 1. Navigate to frontend directory and install dependencies
cd frontend
npm install

# 2. Start Vite development server (port 5173 with proxy to :8000)
npm run dev

# 3. Production build
npm run build
```

The frontend will be accessible at: `http://localhost:5173`.

---

## 4. Running Tests

### Backend Tests (pytest)
From repository root:
```bash
python -m pytest backend/tests -v
```
Covers 46 comprehensive automated test suites:
- Image validation (corrupt, oversized, magic bytes, dimensions, decompression bombs)
- Triage rule precedence (poor quality > OOD > high uncertainty > success)
- Error shielding (503 model unavailable, 500 without leaking stack traces or paths)
- Validation report parsing (missing file -> pending, malformed -> error, exact metric pass-through)
- ConvNeXt-Base fracture endpoints (`/api/fracture/status`, `/api/fracture/predict`, `/api/fracture/model-info`, `/api/fracture/validation`)
- Preprocessing and standalone-vs-API numerical equivalence (`test_fracture_consistency.py`)
- Unified Model Registry and real-time GPU telemetry (`/api/models`, `/api/models/system/gpu`)

### Reproducing Evaluation & Benchmarks
```bash
# Independent test evaluation on 3,448 held-out radiographs:
python -m ml.tasks.fracture.evaluate

# Generate diagnostic curves, comparison charts, and false-negative review sheet:
python -m ml.tasks.fracture.generate_evaluation_artifacts

# Measure cold load, warm latency, Grad-CAM, and VRAM memory benchmarks:
python -m ml.tasks.fracture.benchmark_inference
```

### Frontend Tests (vitest)
From `frontend/` directory:
```bash
cd frontend
npm run test
```
Covers:
- Display-only formatting (`formatProbability(null)` -> `"Not evaluated"`)
- `FindingCard` state suppression for non-success cases
- `HeatmapViewer` fallback states
- `MetricCard` null handling
- `ImageUploader` client pre-checks
- Bone fracture workspace state transitions and Grad-CAM blending controls

---

## 5. Security & Privacy Safeguards

- **In-Memory Image Processing**: Uploaded images are decoded in memory, verified, and discarded immediately after inference. No images or metadata are stored to disk.
- **Decompression-Bomb Protection**: Pillow limits enforced server-side (`MAX_IMAGE_PIXELS = 40,000,000`).
- **Magic-Byte Sniffing**: Files verified via header byte signatures rather than user-supplied extensions or content-types.
- **CORS & Security Headers**: Strict origin policies and `X-Content-Type-Options: nosniff` headers.
- **Information Leak Protection**: Server exceptions return standardized client responses with generic error messages and zero internal file paths or stack traces.