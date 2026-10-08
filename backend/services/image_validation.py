import io
import re
from typing import Optional, Tuple
from fastapi import UploadFile
from PIL import Image, ImageOps

from backend.core.config import settings
from backend.core.logging_config import logger


class ImageValidationError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _matches_magic_bytes(header: bytes) -> bool:
    """Sniff magic bytes for PNG, JPEG, and WEBP formats."""
    # PNG magic: \x89PNG\r\n\x1a\n (8 bytes)
    if len(header) >= 8 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # JPEG magic: \xff\xd8\xff (3 bytes)
    if len(header) >= 3 and header[:3] == b"\xff\xd8\xff":
        return True
    # WEBP magic: 'RIFF' .... 'WEBP' (12 bytes)
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True
    return False


def sanitize_filename(filename: Optional[str]) -> str:
    """Sanitize filename for safe logging purposes (strip path traversal characters)."""
    if not filename:
        return "unnamed_upload"
    # Take basename only
    base = filename.replace("\\", "/").split("/")[-1]
    # Remove any characters except alphanumeric, dot, underscore, dash
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return clean[:100] or "unnamed_upload"


async def validate_and_load_image(file: UploadFile) -> Tuple[Image.Image, bytes]:
    """
    Validates uploaded file following Section 6.4 rules:
    1. Check presence and non-empty
    2. Check size <= MAX_UPLOAD_BYTES in memory (chunked read)
    3. Sniff magic bytes
    4. Pillow open, verify, and load safely
    5. Check format, dimensions, and pixel count
    6. Exif transpose, mode check (RGB or L)
    7. Return clean PIL image (in memory) and raw bytes (for client round-trip if needed)
    """
    safe_name = sanitize_filename(file.filename)
    logger.info("Starting image validation for upload: %s", safe_name)

    # 1. Check content-length header if provided
    # Note: Client can spoof content-length, so we always enforce during reading
    chunk_size = 64 * 1024  # 64 KB
    total_bytes = bytearray()
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes.extend(chunk)
        if len(total_bytes) > settings.MAX_UPLOAD_BYTES:
            logger.warning("Upload rejected: exceeded max upload size (%d bytes)", len(total_bytes))
            raise ImageValidationError(
                status_code=413,
                code="PAYLOAD_TOO_LARGE",
                message="File exceeds the 10 MB limit."
            )

    raw_bytes = bytes(total_bytes)
    if not raw_bytes:
        logger.warning("Upload rejected: empty file payload")
        raise ImageValidationError(
            status_code=400,
            code="EMPTY_FILE",
            message="No file received or file is empty."
        )

    # 3. Sniff magic bytes
    if not _matches_magic_bytes(raw_bytes[:16]):
        logger.warning("Upload rejected: magic byte mismatch (unsupported format)")
        raise ImageValidationError(
            status_code=415,
            code="UNSUPPORTED_MEDIA_TYPE",
            message="Unsupported or mismatched file type. Allowed: PNG, JPEG, WEBP."
        )

    # 4. Pillow verify
    try:
        verify_img = Image.open(io.BytesIO(raw_bytes))
        detected_format = verify_img.format
        verify_img.verify()
    except Exception as exc:
        logger.warning("Upload rejected: Pillow verify failed: %s", exc)
        raise ImageValidationError(
            status_code=400,
            code="DECODE_ERROR",
            message="Image could not be decoded."
        )

    # 5. Pillow load and decompression bomb check
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Image.DecompressionBombError:
        logger.warning("Upload rejected: decompression bomb detected")
        raise ImageValidationError(
            status_code=400,
            code="DECOMPRESSION_BOMB",
            message="Image pixel dimensions exceed safety limits."
        )
    except Exception as exc:
        logger.warning("Upload rejected: Pillow load failed: %s", exc)
        raise ImageValidationError(
            status_code=400,
            code="DECODE_ERROR",
            message="Image could not be decoded."
        )

    # Format check against allowed formats
    if detected_format not in settings.ALLOWED_FORMATS:
        logger.warning("Upload rejected: format %s not in allowed formats", detected_format)
        raise ImageValidationError(
            status_code=415,
            code="UNSUPPORTED_FORMAT",
            message="Unsupported or mismatched file type. Allowed: PNG, JPEG, WEBP."
        )

    # Dimensions check
    width, height = img.size
    if width < settings.MIN_IMAGE_DIM or height < settings.MIN_IMAGE_DIM:
        logger.warning("Upload rejected: image too small (%dx%d)", width, height)
        raise ImageValidationError(
            status_code=400,
            code="IMAGE_TOO_SMALL",
            message=f"Image dimensions must be at least {settings.MIN_IMAGE_DIM}x{settings.MIN_IMAGE_DIM} pixels."
        )

    if width > settings.MAX_IMAGE_DIM or height > settings.MAX_IMAGE_DIM:
        logger.warning("Upload rejected: image too large (%dx%d)", width, height)
        raise ImageValidationError(
            status_code=400,
            code="IMAGE_TOO_LARGE",
            message=f"Image dimensions exceed maximum allowed {settings.MAX_IMAGE_DIM}x{settings.MAX_IMAGE_DIM} pixels."
        )

    if width * height > settings.MAX_IMAGE_PIXELS:
        logger.warning("Upload rejected: total pixel count exceeds limit (%d)", width * height)
        raise ImageValidationError(
            status_code=400,
            code="DECOMPRESSION_BOMB",
            message="Image pixel count exceeds maximum allowed limits."
        )

    # 6. Apply EXIF transpose and ensure clean RGB or L mode
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    logger.info("Image validation succeeded: %s (%dx%d, mode=%s)", safe_name, img.width, img.height, img.mode)
    return img, raw_bytes
