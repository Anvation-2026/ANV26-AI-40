# MedGuard AI — Integration Assumptions & Notes

## Core Architectural Assumptions

1. **Separation of Concerns**:
   - The ML engineer provides `ml/interface.py` with `get_model_info()` and `predict(image)`.
   - The backend never invokes PyTorch, torchvision, or ML training libraries directly.
   - The backend validates the container, format, magic bytes, dimensions, and decompresses safely via Pillow into an in-memory PIL Image.
   - The ML module performs domain preprocessing, resizing, normalization, and inference.

2. **No Data Fabrication**:
   - In `real` mode, all metrics, predictions, probabilities, and heatmaps originate strictly from the ML module.
   - Missing or non-evaluated fields evaluate to `null` (`None`) and display as "Not evaluated" in the UI.
   - Demo mode (`MEDGUARD_MODE=demo`) is strictly isolated to `services/demo_adapter.py`. It is never active when `MEDGUARD_MODE=real`.

3. **In-Memory & Ephemeral Security**:
   - Uploaded images are never written to disk.
   - Image bytes and patient metadata are never logged or stored.
   - All server errors produce generic, safe client-facing messages with no stack traces or server paths.

4. **Triage Hierarchy**:
   - Rule 1: Technical/input validation error → `invalid_input`
   - Rule 2: Model unavailable / internal exception → `model_unavailable` / `error`
   - Rule 3: Poor image quality → `poor_quality` (finding suppressed, heatmap suppressed)
   - Rule 4: Out-of-distribution (OOD) → `ood` (finding suppressed, heatmap suppressed)
   - Rule 5: Model abstained OR finding is None OR high uncertainty → `uncertain` (finding suppressed, heatmap suppressed)
   - Rule 6: Success → `success` (finding, score, calibrated probability, heatmap displayed)

5. **Paths & Environments**:
   - Validation report path defaults to `ml/reports/validation_report.json` resolved relative to the repository root.
   - Backend runs on `http://localhost:8000` and Vite dev server runs on `http://localhost:5173`.
