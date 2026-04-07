"""Image preprocessing utilities for the deepfake detector."""

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# ImageNet normalisation statistics used by pretrained torchvision models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Input resolution expected by the model backbone (ResNet-18).
INPUT_SIZE = 224


def load_image_pil(image_path: str) -> Image.Image:
    """Load an image from *image_path* and convert to RGB PIL Image."""
    img = Image.open(image_path).convert("RGB")
    return img


def load_image_cv2(image_path: str) -> np.ndarray:
    """Load an image from *image_path* as an OpenCV BGR numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def get_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Return the standard torchvision transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(image_path: str, input_size: int = INPUT_SIZE) -> torch.Tensor:
    """Load and preprocess a single image for model inference.

    Returns a 4-D tensor of shape ``(1, 3, input_size, input_size)``.
    """
    img = load_image_pil(image_path)
    transform = get_transform(input_size)
    tensor = transform(img)        # (3, H, W)
    return tensor.unsqueeze(0)     # (1, 3, H, W)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to a uint8 numpy array (H, W, 3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.squeeze(0).cpu() * std + mean   # (3, H, W)
    img = img.permute(1, 2, 0).numpy()           # (H, W, 3)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
