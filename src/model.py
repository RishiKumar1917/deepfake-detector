"""Deepfake detection model.

A ResNet-18 backbone is used with its final fully-connected layer replaced by
a two-class head (REAL vs FAKE).  Pre-trained ImageNet weights are used for
the backbone so the model is useful out-of-the-box even before domain-specific
fine-tuning.

If a fine-tuned checkpoint exists at ``models/deepfake_detector.pth`` it is
loaded automatically; otherwise the model runs with ImageNet weights and emits
an informational message.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

# Class labels used by the binary head.
LABELS = {0: "REAL", 1: "FAKE"}

# Default path (relative to project root) for the optional fine-tuned weights.
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "deepfake_detector.pth",
)


class DeepfakeDetector(nn.Module):
    """ResNet-18 backbone with a binary classification head."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace the final FC layer: 512 → 2 (REAL / FAKE).
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) → (B, 2)
        return self.backbone(x)


def load_model(weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = "cpu") -> DeepfakeDetector:
    """Instantiate and return the detector, loading fine-tuned weights if available.

    Args:
        weights_path: Path to a ``state_dict`` checkpoint saved with
                      ``torch.save(model.state_dict(), path)``.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        The model in evaluation mode on *device*.
    """
    model = DeepfakeDetector(pretrained=True)

    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"[model] Loaded fine-tuned weights from: {weights_path}")
    else:
        print(
            "[model] No fine-tuned weights found at '{}'. "
            "Running with ImageNet pre-trained backbone only. "
            "Predictions are illustrative until the model is trained on deepfake data.".format(
                weights_path
            )
        )

    model.to(device)
    model.eval()
    return model


def predict(
    model: DeepfakeDetector,
    tensor: torch.Tensor,
    device: str = "cpu",
) -> Tuple[str, float]:
    """Run inference on a pre-processed image tensor.

    Args:
        model: The detector in evaluation mode.
        tensor: A ``(1, 3, H, W)`` float tensor (already normalised).
        device: PyTorch device string.

    Returns:
        A tuple ``(label, confidence)`` where *label* is ``"REAL"`` or
        ``"FAKE"`` and *confidence* is a float in ``[0, 1]``.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)                          # (1, 2)
        probs = torch.softmax(logits, dim=1)            # (1, 2)
        class_idx = int(torch.argmax(probs, dim=1))
        confidence = float(probs[0, class_idx])

    label = LABELS[class_idx]
    return label, confidence
