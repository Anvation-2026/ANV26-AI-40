# MedGuard AI — Clinical Evidence & Triage Support System

> **Educational Research Prototype Notice**: MedGuard AI is an experimental decision-support prototype developed for instructional demonstration and educational research. It is **not a medical device**, has not been evaluated by clinical regulatory authorities (FDA/CE), and must **never be used for diagnosis, clinical staging, or patient treatment decisions**. Uploads must consist exclusively of public or de-identified educational images.

---

## 1. Product Summary

MedGuard AI provides transparent, rule-based decision support for educational chest radiographs (Normal vs. Pneumonia). The system prioritizes clinical safety through:
- **Strict Pre-Inference Quality Checks**: Heuristic evaluation of image blur, underexposure/overexposure, and contrast thresholds.
- **Out-of-Distribution (OOD) Guardrails**: Domain validation ensuring inputs match chest X-ray distribution.
- **Safety Abstention**: When model uncertainty exceeds acceptable thresholds or image quality is degraded, the engine suppresses definitive findings and directly routes to expert human review.
- **Explainable Evidence Localization**: Grad-CAM activation heatmaps with original/overlay/side-by-side toggles.
- **Honest Metrics Dashboard**: Independent validation metrics loaded strictly from benchmark reports without fabrication or hardcoded numbers.

---

## 2. Tech Stack

- **Frontend**: React 18, Vite, TypeScript (strict), Tailwind CSS v3.4, React Router v6, Recharts, Lucide React, native `fetch`.
- **Backend**: Python 3.10+, FastAPI, Pydantic v2, Uvicorn, Pillow, python-multipart, pydantic-settings.
- **Testing**: pytest (backend), vitest + @testing-library/react (frontend).

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
python -m pytest backend/tests -q
```
Covers:
- Image validation (corrupt, oversized, magic bytes, dimensions, decompression bombs)
- Triage rule precedence (poor quality > OOD > high uncertainty > success)
- Error shielding (503 model unavailable, 500 without leaking stack traces or paths)
- Validation report parsing (missing file -> pending, malformed -> error, exact metric pass-through)

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

---

## 5. Security & Privacy Safeguards

- **In-Memory Image Processing**: Uploaded images are decoded in memory, verified, and discarded immediately after inference. No images or metadata are stored to disk.
- **Decompression-Bomb Protection**: Pillow limits enforced server-side (`MAX_IMAGE_PIXELS = 40,000,000`).
- **Magic-Byte Sniffing**: Files verified via header byte signatures rather than user-supplied extensions or content-types.
- **CORS & Security Headers**: Strict origin policies and `X-Content-Type-Options: nosniff` headers.
- **Information Leak Protection**: Server exceptions return standardized client responses with generic error messages and zero internal file paths or stack traces.