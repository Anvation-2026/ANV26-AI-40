import contextlib
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

from config import IMAGE_SIZE, HEATMAPS_DIR


class GradCAM:
    """
    Grad-CAM implementation for ResNet-18.
    Target layer: model.layer4[-1] (7x7 spatial feature maps for 224 input).
    Uses forward hooks for activations and backward hooks for gradients.
    All hooks are safely removed after each call.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer = getattr(model, target_layer_name)[-1]

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
        target_class: int,
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

            # Compute gradients with respect to the target class logit
            self.model.zero_grad()
            if logits.shape[1] == 1:
                target_score = logits[0, 0] if target_class == 1 else -logits[0, 0]
            else:
                target_score = logits[0, target_class]
            target_score.backward(retain_graph=False)

            # activations: (1, C, H, W); gradients: (1, C, H, W)
            acts = activations["value"].detach().cpu().squeeze(0).numpy()  # (C, H, W)
            grads = gradients["value"].detach().cpu().squeeze(0).numpy()   # (C, H, W)

        # Weights: mean of gradients over spatial dimensions
        weights = grads.mean(axis=(1, 2))  # (C,)

        # CAM: weighted sum of activations + ReLU
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)  # (H, W)
        cam = np.maximum(cam, 0)  # ReLU

        # Resize to 224x224
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # Min-max normalize [0, 1] — handle all-zero maps
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
        target_class: int,
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
