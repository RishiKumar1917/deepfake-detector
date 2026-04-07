"""Logging utilities for the deepfake detector.

Detection results (timestamp, filename, prediction, confidence) are appended
to a plain-text log file inside the ``outputs/`` directory.
"""

import logging
import os
from datetime import datetime
from typing import Optional

# Default log file location (relative to project root).
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)
LOG_FILE = os.path.join(LOG_DIR, "detections.log")


def get_logger(log_file: str = LOG_FILE) -> logging.Logger:
    """Return a named logger that writes to *log_file* and to stdout.

    Calling this function more than once with the same *log_file* returns the
    same logger instance without adding duplicate handlers.
    """
    logger = logging.getLogger("deepfake_detector")

    if logger.handlers:          # already configured – return as-is
        return logger

    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def log_detection(
    image_path: str,
    label: str,
    confidence: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Append a detection result to the log.

    Args:
        image_path: Path (or filename) of the analysed image.
        label: Prediction label (``"REAL"`` or ``"FAKE"``).
        confidence: Confidence score in ``[0, 1]``.
        logger: Logger instance.  If *None* the default logger is used.
    """
    if logger is None:
        logger = get_logger()

    filename = os.path.basename(image_path)
    logger.info(
        "file=%-40s  prediction=%-4s  confidence=%.4f",
        filename,
        label,
        confidence,
    )
