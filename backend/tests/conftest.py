import io
import pytest
from PIL import Image
from starlette.testclient import TestClient

from backend.main import create_app
from backend.schemas.analysis import (
    MLResult,
    ModelMeta,
    OODInfo,
    QualityCheck,
    QualityInfo,
    UncertaintyInfo,
)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# In-memory test image generators
@pytest.fixture
def valid_png_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(50, 50, 50))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def huge_dim_png_bytes() -> bytes:
    buf = io.BytesIO()
    # 8001 x 64 pixels - exceeds MAX_IMAGE_DIM 8000
    img = Image.new("L", (8001, 64), color=128)
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def corrupt_png_bytes(valid_png_bytes) -> bytes:
    # Magic bytes of PNG but truncated/corrupt payload
    return valid_png_bytes[:20] + b"CORRUPTED_BYTES_TRUNCATED"


@pytest.fixture
def text_renamed_png_bytes() -> bytes:
    return b"This is a plain text file pretending to be an image file."


@pytest.fixture
def empty_bytes() -> bytes:
    return b""


@pytest.fixture
def oversized_bytes() -> bytes:
    # 10 MB + 1024 bytes
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * (10485760 + 1024)


@pytest.fixture
def gif_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(buf, format="GIF")
    return buf.getvalue()


@pytest.fixture
def bmp_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img.save(buf, format="BMP")
    return buf.getvalue()


# Explicit test doubles for ML integration tests (NEVER used in production)
class FakeMLDoubles:
    @staticmethod
    def pneumonia_success() -> MLResult:
        return MLResult(
            finding="pneumonia",
            abstained=False,
            raw_score=0.884,
            calibrated_probability=0.871,
            probability_of="pneumonia",
            uncertainty=UncertaintyInfo(level="low", value=0.042, method="MC-Dropout (test)"),
            quality=QualityInfo(
                evaluated=True,
                status="acceptable",
                blur=QualityCheck(value=240.5, threshold=100.0, passed=True),
                brightness=QualityCheck(value=128.0, threshold=40.0, passed=True),
                contrast=QualityCheck(value=65.2, threshold=20.0, passed=True),
                reasons=[],
            ),
            ood=OODInfo(evaluated=True, is_ood=False, score=0.12, method="Mahalanobis (test)", reason=None),
            heatmap_png_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            heatmap_kind="overlay",
            model_version="resnet18-test",
            evidence_notes=["Test note: Right lower lobe density pattern"],
        )

    @staticmethod
    def normal_success() -> MLResult:
        return MLResult(
            finding="normal",
            abstained=False,
            raw_score=0.082,
            calibrated_probability=0.915,
            probability_of="normal",
            uncertainty=UncertaintyInfo(level="low", value=0.025, method="MC-Dropout (test)"),
            quality=QualityInfo(
                evaluated=True,
                status="acceptable",
                blur=QualityCheck(value=280.0, threshold=100.0, passed=True),
                brightness=QualityCheck(value=130.0, threshold=40.0, passed=True),
                contrast=QualityCheck(value=70.0, threshold=20.0, passed=True),
                reasons=[],
            ),
            ood=OODInfo(evaluated=True, is_ood=False, score=0.05, method="Mahalanobis (test)", reason=None),
            heatmap_png_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            heatmap_kind="overlay",
            model_version="resnet18-test",
            evidence_notes=["Test note: Clear pulmonary vasculature"],
        )

    @staticmethod
    def poor_quality() -> MLResult:
        return MLResult(
            finding=None,
            abstained=True,
            raw_score=None,
            calibrated_probability=None,
            probability_of=None,
            uncertainty=UncertaintyInfo(level="not_evaluated"),
            quality=QualityInfo(
                evaluated=True,
                status="poor",
                blur=QualityCheck(value=42.0, threshold=100.0, passed=False),
                brightness=QualityCheck(value=20.0, threshold=40.0, passed=False),
                contrast=QualityCheck(value=10.0, threshold=20.0, passed=False),
                reasons=["Test: Severe motion blur", "Test: Underexposed"],
            ),
            ood=OODInfo(evaluated=False),
            heatmap_png_base64=None,
            model_version="resnet18-test",
        )

    @staticmethod
    def ood() -> MLResult:
        return MLResult(
            finding=None,
            abstained=True,
            raw_score=None,
            calibrated_probability=None,
            probability_of=None,
            uncertainty=UncertaintyInfo(level="not_evaluated"),
            quality=QualityInfo(evaluated=True, status="acceptable"),
            ood=OODInfo(
                evaluated=True,
                is_ood=True,
                score=0.96,
                method="Mahalanobis (test)",
                reason="Test: Non-chest anatomy detected",
            ),
            heatmap_png_base64=None,
            model_version="resnet18-test",
        )

    @staticmethod
    def high_uncertainty() -> MLResult:
        return MLResult(
            finding="pneumonia",  # Note: should be suppressed by triage
            abstained=False,
            raw_score=0.55,
            calibrated_probability=0.54,
            probability_of="pneumonia",
            uncertainty=UncertaintyInfo(level="high", value=0.45, method="MC-Dropout (test)"),
            quality=QualityInfo(evaluated=True, status="acceptable"),
            ood=OODInfo(evaluated=True, is_ood=False, score=0.10),
            heatmap_png_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            model_version="resnet18-test",
        )

    @staticmethod
    def abstained_by_model() -> MLResult:
        return MLResult(
            finding=None,
            abstained=True,
            raw_score=0.49,
            calibrated_probability=None,
            probability_of=None,
            uncertainty=UncertaintyInfo(level="moderate", value=0.18, method="MC-Dropout (test)"),
            quality=QualityInfo(evaluated=True, status="acceptable"),
            ood=OODInfo(evaluated=True, is_ood=False, score=0.08),
            heatmap_png_base64=None,
            model_version="resnet18-test",
        )

    @staticmethod
    def success_missing_heatmap() -> MLResult:
        return MLResult(
            finding="normal",
            abstained=False,
            raw_score=0.12,
            calibrated_probability=0.88,
            probability_of="normal",
            uncertainty=UncertaintyInfo(level="low", value=0.03),
            quality=QualityInfo(evaluated=True, status="acceptable"),
            ood=OODInfo(evaluated=True, is_ood=False, score=0.04),
            heatmap_png_base64=None,
            model_version="resnet18-test",
        )

    @staticmethod
    def not_evaluated_checks() -> MLResult:
        return MLResult(
            finding="pneumonia",
            abstained=False,
            raw_score=0.80,
            calibrated_probability=0.82,
            probability_of="pneumonia",
            uncertainty=UncertaintyInfo(level="not_evaluated"),
            quality=QualityInfo(evaluated=False, status="not_evaluated"),
            ood=OODInfo(evaluated=False, is_ood=None),
            heatmap_png_base64=None,
            model_version="resnet18-test",
        )


@pytest.fixture
def fake_ml_doubles():
    return FakeMLDoubles


@pytest.fixture
def fake_model_meta():
    return ModelMeta(
        available=True,
        model_name="Fake Test ResNet-18",
        model_version="resnet18-test",
        dataset="PneumoniaMNIST+",
        calibration_available=True,
        ood_available=True,
        quality_available=True,
        gradcam_available=True,
    )
