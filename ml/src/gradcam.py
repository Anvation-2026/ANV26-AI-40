"""
ml/src/gradcam.py -- Grad-CAM Visualization for ResNet-18 and DenseNet-201.
===========================================================================
Generates visual attention maps representing model decision evidence.

Clinical Disclaimer:
  Grad-CAM heatmaps highlight visual patterns that influenced the model's prediction.
  They are explanatory representations of model attention, NOT verified anatomical lesions
  or definitive medical diagnoses.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

# Path bootstrap
try:
    from config import HEATMAPS_DIR, IMAGE_SIZE
except ModuleNotFoundError:
    from ..config import HEATMAPS_DIR, IMAGE_SIZE

GRADCAM_DISCLAIMER = (
    "Grad-CAM heatmaps show where the model focused when calculating disease probability; "
    "they are explanatory model evidence, not medically verified disease locations."
)


def _resolve_target_layer(model: nn.Module, target_layer: Optional[Union[str, nn.Module]] = None) -> nn.Module:
    """
    Safely resolves the appropriate target layer for either ResNet-18 or DenseNet-201.
    """
    if isinstance(target_layer, nn.Module):
        return target_layer

    if target_layer is not None and isinstance(target_layer, str):
        # Allow nested dotted paths like "features.denseblock4"
        curr = model
        for part in target_layer.split("."):
            curr = getattr(curr, part)
        if isinstance(curr, nn.Sequential) or hasattr(curr, "__getitem__"):
            try:
                return curr[-1]
            except Exception:
                pass
        return curr

    # Automatic detection based on architecture
    if hasattr(model, "get_gradcam_target_layer") and callable(model.get_gradcam_target_layer):
        return model.get_gradcam_target_layer()

    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        # DenseNet architecture
        return model.features.denseblock4

    if hasattr(model, "layer4"):
        # ResNet architecture
        return getattr(model, "layer4")[-1]

    raise ValueError(f"Could not automatically determine Grad-CAM target layer for {type(model)}")


class GradCAM:
    """
    Grad-CAM implementation supporting both ResNet-18 and DenseNet-201.
    Uses forward hooks for activations and full backward hooks for gradients.
    Guarantees hook cleanup via context manager.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[str, nn.Module]] = None,
    ):
        self.model = model
        self.target_layer = _resolve_target_layer(model, target_layer)
        self.disclaimer = GRADCAM_DISCLAIMER

    @contextlib.contextmanager
    def _hook_context(self):
        """Context manager that registers hooks and guarantees removal."""
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations["value"] = output

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0]

        fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            yield activations, gradients
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    def compute_cam(
        self,
        tensor: torch.Tensor,
        target_class: int = 0,
    ) -> np.ndarray:
        """
        Computes normalized Grad-CAM map (H', W') in [0, 1].
        Returns a 224x224 float32 array (all-zero handled without NaN).
        """
        self.model.eval()

        with self._hook_context() as (activations, gradients):
            # Enable grad tracking for this forward pass
            tensor = tensor.detach().requires_grad_(True)
            logits = self.model(tensor)

            self.model.zero_grad()
            if logits.shape[1] == 1:
                # Binary single logit (e.g. ResNet pneumonia or TB)
                target_score = logits[0, 0] if target_class == 1 else -logits[0, 0]
            else:
                # Multi-class or multi-label (e.g. ChestMNIST 14)
                target_idx = min(target_class, logits.shape[1] - 1)
                target_score = logits[0, target_idx]

            target_score.backward(retain_graph=False)

            # activations: (1, C, H, W); gradients: (1, C, H, W)
            acts = activations["value"].detach().cpu().squeeze(0).numpy()  # (C, H, W)
            grads = gradients["value"].detach().cpu().squeeze(0).numpy()   # (C, H, W)

        # Global average pooled gradients
        weights = grads.mean(axis=(1, 2))  # (C,)

        # Weighted combination of activation maps + ReLU
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)  # (H, W)
        cam = np.maximum(cam, 0)  # ReLU

        # Resize to input resolution (224, 224)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # Min-max normalize to [0, 1] safely
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            cam = np.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def generate(
        self,
        tensor: torch.Tensor,
        gray224: np.ndarray,
        target_class: int = 0,
        save_path: Optional[Path] = None,
        alpha: float = 0.4,
    ) -> Optional[np.ndarray]:
        """
        Generates Grad-CAM overlay on the grayscale X-ray.
        Saves PNG to save_path if provided.
        Returns the overlay array (H, W, 3) uint8, or None on failure.
        """
        try:
            cam = self.compute_cam(tensor, target_class)

            # Ensure grayscale array is uint8
            if gray224.dtype != np.uint8:
                gray224 = (np.clip(gray224, 0.0, 1.0) * 255).astype(np.uint8)

            # Convert grayscale to 3-channel BGR
            gray_bgr = cv2.cvtColor(gray224, cv2.COLOR_GRAY2BGR)

            # Apply JET colormap to CAM
            cam_uint8 = (cam * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

            # Blend: overlay = (1-alpha)*original + alpha*heatmap
            overlay = cv2.addWeighted(gray_bgr, 1 - alpha, heatmap, alpha, 0)

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), overlay)

            return overlay

        except Exception as e:
            print(f"[GradCAM] Warning: generation failed: {e}")
            return None
