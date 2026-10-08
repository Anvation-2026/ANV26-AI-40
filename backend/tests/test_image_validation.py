import asyncio
import io
import pytest
from starlette.datastructures import Headers, UploadFile
from backend.services.image_validation import ImageValidationError, validate_and_load_image


def make_upload_file(data: bytes, filename: str = "test.png", content_type: str = "image/png") -> UploadFile:
    file_obj = io.BytesIO(data)
    headers = Headers({"content-type": content_type})
    return UploadFile(file=file_obj, size=len(data), filename=filename, headers=headers)


def test_valid_png_passes(valid_png_bytes):
    upload = make_upload_file(valid_png_bytes, filename="test.png", content_type="image/png")
    img, raw = asyncio.run(validate_and_load_image(upload))
    assert img.size == (224, 224)
    assert img.mode in ("RGB", "L")
    assert len(raw) == len(valid_png_bytes)


def test_valid_jpeg_passes(valid_jpeg_bytes):
    upload = make_upload_file(valid_jpeg_bytes, filename="chest.jpg", content_type="image/jpeg")
    img, raw = asyncio.run(validate_and_load_image(upload))
    assert img.size == (224, 224)
    assert len(raw) == len(valid_jpeg_bytes)


def test_empty_file_rejected(empty_bytes):
    upload = make_upload_file(empty_bytes, filename="empty.png")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 400
    assert exc.value.code == "EMPTY_FILE"


def test_oversized_file_rejected(oversized_bytes):
    upload = make_upload_file(oversized_bytes, filename="huge.png")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 413
    assert "10 MB" in exc.value.message


def test_magic_mismatch_text_renamed_png_rejected(text_renamed_png_bytes):
    upload = make_upload_file(text_renamed_png_bytes, filename="fake.png")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 415
    assert exc.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_unsupported_gif_rejected(gif_bytes):
    upload = make_upload_file(gif_bytes, filename="anim.gif", content_type="image/gif")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 415


def test_unsupported_bmp_rejected(bmp_bytes):
    upload = make_upload_file(bmp_bytes, filename="sample.bmp", content_type="image/bmp")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 415


def test_corrupt_png_rejected(corrupt_png_bytes):
    upload = make_upload_file(corrupt_png_bytes, filename="corrupt.png")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 400
    assert exc.value.code == "DECODE_ERROR"


def test_tiny_image_rejected(tiny_png_bytes):
    upload = make_upload_file(tiny_png_bytes, filename="tiny.png")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 400
    assert exc.value.code == "IMAGE_TOO_SMALL"


def test_huge_dimension_rejected(huge_dim_png_bytes):
    upload = make_upload_file(huge_dim_png_bytes, filename="huge_dim.png")
    with pytest.raises(ImageValidationError) as exc:
        asyncio.run(validate_and_load_image(upload))
    assert exc.value.status_code == 400
    assert exc.value.code == "IMAGE_TOO_LARGE"
