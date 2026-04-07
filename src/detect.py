#!/usr/bin/env python3
"""Deepfake Detection and Forensic Analysis – CLI entry point.

Usage
-----
    python src/detect.py --input path/to/image.jpg

Options
-------
    --input       Path to the input image (required).
    --output-dir  Directory where visualisation results are saved
                  (default: outputs/).
    --no-viz      Skip saving the Grad-CAM / FFT visualisation.
    --device      PyTorch device, e.g. ``cpu`` or ``cuda`` (default: cpu).
    --weights     Path to fine-tuned model weights
                  (default: models/deepfake_detector.pth).
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")           # non-interactive backend – no display needed
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``import src.*`` works when
# the script is executed from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_image, load_image_pil
from src.model import load_model, predict, DEFAULT_WEIGHTS_PATH
from src.frequency import get_fft_visualization, extract_frequency_features
from src.explainability import GradCAM, overlay_heatmap
from src.logger import get_logger, log_detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deepfake Detection and Forensic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="IMAGE",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs"),
        metavar="DIR",
        help="Directory where visualisation results are saved (default: outputs/).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip saving the Grad-CAM / FFT visualisation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (default: cpu).",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        metavar="PATH",
        help="Path to fine-tuned model weights.",
    )
    return parser.parse_args()


def _save_visualization(
    image_path: str,
    heatmap_bgr: np.ndarray,
    fft_gray: np.ndarray,
    label: str,
    confidence: float,
    output_dir: str,
) -> str:
    """Save a three-panel figure and return the output file path."""
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_analysis.png")

    original = np.array(load_image_pil(image_path))  # H×W×3 RGB

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Prediction: {label}  |  Confidence: {confidence:.2%}",
        fontsize=14,
        fontweight="bold",
        color="red" if label == "FAKE" else "green",
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM overlay is BGR; convert to RGB for matplotlib.
    axes[1].imshow(heatmap_bgr[:, :, ::-1])
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(fft_gray, cmap="inferno")
    axes[2].set_title("FFT Frequency Spectrum")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logger = get_logger()

    # ---- validate input ----------------------------------------------------
    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("Analysing: %s", args.input)

    # ---- load model --------------------------------------------------------
    model = load_model(weights_path=args.weights, device=args.device)

    # ---- preprocess --------------------------------------------------------
    tensor = preprocess_image(args.input)

    # ---- spatial prediction ------------------------------------------------
    label, confidence = predict(model, tensor, device=args.device)

    # ---- frequency features (informational) --------------------------------
    freq_features = extract_frequency_features(args.input)
    logger.info(
        "Frequency feature vector – mean=%.4f  std=%.4f  max=%.4f",
        freq_features.mean(),
        freq_features.std(),
        freq_features.max(),
    )

    # ---- print results to console ------------------------------------------
    separator = "=" * 50
    print(f"\n{separator}")
    print(f"  File       : {os.path.basename(args.input)}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.2%}")
    print(f"{separator}\n")

    # ---- log detection -----------------------------------------------------
    log_detection(args.input, label, confidence, logger)

    # ---- visualisation -----------------------------------------------------
    if not args.no_viz:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(tensor, device=args.device)
        heatmap_bgr = overlay_heatmap(args.input, heatmap)
        fft_viz = get_fft_visualization(args.input)

        out_path = _save_visualization(
            args.input, heatmap_bgr, fft_viz, label, confidence, args.output_dir
        )
        logger.info("Visualisation saved to: %s", out_path)


if __name__ == "__main__":
    main()
