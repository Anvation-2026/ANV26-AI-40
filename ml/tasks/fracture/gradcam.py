"""
Grad-CAM Visual Attribution Module
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation

Hooks the final stage of ConvNeXt-Base (features[7]) to compute spatial feature attributions.
Includes overlay blending and clinical disclaimer.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.tasks.fracture.model import ConvNeXtFractureClassifier


class ConvNeXtGradCAM:
    """
    Gradient-weighted Class Activation Mapping for ConvNeXt-Base.
    Captures spatial activations at features[7] (final stage).
    """

    def __init__(self, model: ConvNeXtFractureClassifier):
        self.model = model
        self.target_layer = model.features[7]
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap in [0.0, 1.0].
        input_tensor shape: (1, 3, 224, 224) on the appropriate device.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logit = self.model(input_tensor)
        
        # Backward pass on the fracture logit
        logit.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check hook registration.")

        # Pool gradients across spatial dimensions: (1, 1024, H, W) -> (1, 1024, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only positive attribution
        cam = F.relu(cam)

        # Resize to input resolution (224, 224)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        # Normalize to [0.0, 1.0]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        """Clean up PyTorch hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def apply_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap onto a PIL radiograph image.
    Preserves original image dimensions.
    """
    import matplotlib.cm as cm

    # Ensure original is RGB
    orig_rgb = original_image.convert("RGB")
    w, h = orig_rgb.size

    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_norm = np.array(heatmap_pil) / 255.0

    # Colorize heatmap
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps[colormap_name]
    except Exception:
        import matplotlib.cm as cm
        cmap = getattr(cm, "colormaps", {}).get(colormap_name, getattr(cm, "get_cmap", lambda x: None)(colormap_name))
    colored_heatmap = (cmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored_heatmap)

    # Blend
    blended = Image.blend(orig_rgb, heatmap_img, alpha=alpha)
    return blended
