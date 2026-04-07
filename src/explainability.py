"""Grad-CAM explainability for the deepfake detector.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights image
regions that most influenced the model's prediction.  This implementation
hooks into the last convolutional layer of the ResNet-18 backbone and
produces a colour heatmap that is blended with the original image.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.model import DeepfakeDetector
from src.preprocess import INPUT_SIZE, denormalize


class GradCAM:
    """Compute Grad-CAM maps for a :class:`DeepfakeDetector` model."""

    def __init__(self, model: DeepfakeDetector) -> None:
        self.model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Hook the last convolutional layer of the ResNet backbone.
        target_layer = model.backbone.layer4[-1].conv2
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activations(self, _module: nn.Module, _input: Tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: Tuple,
        grad_output: Tuple,
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """Return a Grad-CAM heatmap as a uint8 array of shape ``(H, W)``.

        Args:
            tensor: Pre-processed image tensor ``(1, 3, H, W)``.
            class_idx: Class index to explain.  When *None* the predicted class
                       is used.
            device: PyTorch device string.

        Returns:
            Normalised heatmap of shape ``(H, W)`` with values in ``[0, 255]``.
        """
        tensor = tensor.to(device)
        self.model.eval()

        # Forward pass (keep gradients).
        logits = self.model(tensor)                     # (1, 2)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1))

        # Backward pass for the target class.
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions → importance weights.
        gradients = self._gradients                     # (1, C, h, w)
        activations = self._activations                 # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1).squeeze(0)  # (h, w)
        cam = torch.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (INPUT_SIZE, INPUT_SIZE))

        # Normalise to [0, 255].
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return (cam * 255).astype(np.uint8)


def overlay_heatmap(
    original_image_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM *heatmap* on top of the original image.

    Args:
        original_image_path: Path to the source image.
        heatmap: Grayscale heatmap array of shape ``(H, W)`` with values in
                 ``[0, 255]``.
        alpha: Blend factor for the heatmap overlay (0 = original only,
               1 = heatmap only).

    Returns:
        Blended BGR image as a uint8 numpy array of shape ``(H, W, 3)``.
    """
    img = Image.open(original_image_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img_np = np.array(img)[:, :, ::-1].copy()  # RGB → BGR

    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # (H, W, 3) BGR
    blended = cv2.addWeighted(colored, alpha, img_np, 1 - alpha, 0)
    return blended
