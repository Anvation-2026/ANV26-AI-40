import io
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_RAW_IMAGE_DIM,
)


class InvalidImageError(ValueError):
    """Raised when an input cannot be decoded or is degenerate."""
    pass


def to_gray_uint8(
    image: Union[bytes, str, Path, Image.Image, np.ndarray],
) -> np.ndarray:
    """
    Accepts bytes, filepath, PIL.Image, or ndarray.
    Decodes/converts to a single 2D grayscale uint8 numpy array (H, W).
    Raises InvalidImageError for undecodable, empty, or degenerate inputs.
    """
    if image is None:
        raise InvalidImageError("Input image is None.")

    pil_img: Optional[Image.Image] = None

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists() or not path.is_file():
            raise InvalidImageError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise InvalidImageError(f"File is empty (0 bytes): {path}")
        try:
            with open(path, "rb") as f:
                content = f.read()
            pil_img = Image.open(io.BytesIO(content))
            pil_img.load()
        except Exception as e:
            raise InvalidImageError(f"Cannot decode image file {path}: {e}") from e

    elif isinstance(image, bytes):
        if len(image) == 0:
            raise InvalidImageError("Image bytes are empty (0 bytes).")
        try:
            pil_img = Image.open(io.BytesIO(image))
            pil_img.load()
        except Exception as e:
            raise InvalidImageError(f"Cannot decode image bytes: {e}") from e

    elif isinstance(image, Image.Image):
        pil_img = image

    elif isinstance(image, np.ndarray):
        arr = image
        if arr.size == 0:
            raise InvalidImageError("Input numpy array has size 0.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise InvalidImageError("Input array contains NaNs or Infs.")

        # Handle floating point or integer arrays
        if arr.dtype in [np.float32, np.float64]:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            pass  # Already (H, W)
        elif arr.ndim == 3:
            # Check shape: (C, H, W) or (H, W, C)
            if arr.shape[0] in [1, 3, 4] and arr.shape[1] > 4 and arr.shape[2] > 4:
                # Channel-first
                if arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                elif arr.shape[0] == 4:
                    arr = np.transpose(arr, (1, 2, 0))
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            elif arr.shape[2] in [1, 3, 4]:
                # Channel-last
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                elif arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                elif arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            else:
                raise InvalidImageError(f"Unsupported 3D array shape: {arr.shape}")
        else:
            raise InvalidImageError(f"Unsupported array dimensions: {arr.ndim}")

        # Check degenerate sizes
        if arr.shape[0] <= 1 or arr.shape[1] <= 1:
            raise InvalidImageError(f"Degenerate image dimensions: {arr.shape}")
        return arr

    else:
        raise InvalidImageError(f"Unsupported image type: {type(image)}")

    # Process PIL image
    if pil_img is not None:
        w, h = pil_img.size
        if w <= 1 or h <= 1:
            raise InvalidImageError(f"Degenerate PIL image dimensions: ({w}, {h})")

        # Convert to Grayscale ('L')
        gray_pil = pil_img.convert("L")
        gray_arr = np.array(gray_pil, dtype=np.uint8)
        if gray_arr.size == 0 or np.isnan(gray_arr).any():
            raise InvalidImageError("Extracted grayscale array is empty or contains NaNs.")
        return gray_arr

    raise InvalidImageError("Could not convert input to grayscale array.")


def resize_224(gray: np.ndarray, preserve_aspect: bool = True) -> np.ndarray:
    """
    If not 224x224, resizes using cv2.INTER_AREA for downscale
    and cv2.INTER_CUBIC for upscale. 224x224 inputs pass through untouched.
    For non-square clinical chest radiographs, preserve_aspect=True symmetrically
    pads the shorter axis to prevent anatomical squashing.
    """
    if gray.shape == (IMAGE_SIZE, IMAGE_SIZE):
        return gray

    h, w = gray.shape[:2]
    if not preserve_aspect or h == w:
        if h >= IMAGE_SIZE and w >= IMAGE_SIZE:
            interpolation = cv2.INTER_AREA
        elif h <= IMAGE_SIZE and w <= IMAGE_SIZE:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)

    # Scale proportionally so longer side fits IMAGE_SIZE
    scale = float(IMAGE_SIZE) / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)

    # Symmetrical padding with black margin (standard medical collimation border)
    pad_top = (IMAGE_SIZE - new_h) // 2
    pad_bottom = IMAGE_SIZE - new_h - pad_top
    pad_left = (IMAGE_SIZE - new_w) // 2
    pad_right = IMAGE_SIZE - new_w - pad_left

    padded = cv2.copyMakeBorder(
        scaled, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    return padded


def to_tensor(
    gray224: np.ndarray,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> torch.Tensor:
    """
    Scales uint8 [0, 255] to float32 [0.0, 1.0], replicates to 3 channels,
    and applies ImageNet normalization. Returns (1, 3, 224, 224) torch.Tensor.
    """
    if gray224.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected input shape ({IMAGE_SIZE}, {IMAGE_SIZE}), got {gray224.shape}")

    mean_vals = mean if mean is not None else IMAGENET_MEAN
    std_vals = std if std is not None else IMAGENET_STD

    # Convert uint8 to float32 [0, 1]
    t = torch.from_numpy(gray224).float() / 255.0  # (224, 224)
    # Shape to (3, 224, 224)
    t = t.unsqueeze(0).repeat(3, 1, 1)

    # Normalize
    mean_t = torch.tensor(mean_vals, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std_vals, dtype=torch.float32).view(3, 1, 1)
    normalized = (t - mean_t) / std_t

    # Add batch dimension -> (1, 3, 224, 224)
    return normalized.unsqueeze(0)


def preprocess_image(
    image: Union[bytes, str, Path, Image.Image, np.ndarray],
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Complete inference pipeline: decodes, resizes, converts to tensor.
    Returns:
        tensor: (1, 3, 224, 224) normalized tensor
        gray224: (224, 224) uint8 array (for quality checks and Grad-CAM)
    """
    gray = to_gray_uint8(image)
    gray224 = resize_224(gray)
    tensor = to_tensor(gray224, mean=mean, std=std)
    return tensor, gray224


# Training & Evaluation PyTorch Transform Callables
class EvalTransform:
    """Evaluation / Test deterministic preprocessing."""
    def __init__(self, mean=None, std=None):
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD

    def __call__(self, img_input: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        gray = to_gray_uint8(img_input)
        gray224 = resize_224(gray)
        # Returns (3, 224, 224) without batch dimension for DataLoader
        return to_tensor(gray224, self.mean, self.std).squeeze(0)


class TrainTransform:
    """
    Training augmentation preprocessing:
    - Random rotation ±10°
    - Random resized crop scale=(0.85, 1.0), ratio=(0.95, 1.05)
    - Brightness and contrast jitter (±15%)
    - NO horizontal flip (anatomical laterality)
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        self.aug = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),  # Converts PIL [0, 255] to Tensor [0.0, 1.0] (1, H, W)
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, img_input: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        gray = to_gray_uint8(img_input)
        gray224 = resize_224(gray)
        return self.aug(gray224)


def get_train_transform(mean=None, std=None) -> TrainTransform:
    return TrainTransform(mean=mean, std=std)


def get_eval_transform(mean=None, std=None) -> EvalTransform:
    return EvalTransform(mean=mean, std=std)
