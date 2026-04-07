# Deepfake Detector

A Python-based **Deepfake Detection and Forensic Analysis** system that
classifies images as **REAL** or **FAKE** and produces explainability
visualisations to highlight suspicious regions.

---
## Features

| Feature | Details |
|---|---|
| **CNN backbone** | ResNet-18 pre-trained on ImageNet; swap in your own weights for domain-specific accuracy |
| **Frequency analysis** | FFT-based spectrum computed alongside spatial features to capture deepfake artefacts |
| **Explainability** | Grad-CAM heatmap overlaid on the original image |
| **CLI** | One-command analysis: `python src/detect.py --input image.jpg` |
| **Logging** | Detection results (timestamp, file, label, confidence) appended to `outputs/detections.log` |
| **CPU-friendly** | Runs entirely on CPU; GPU optional |

---

## Project Structure

```
deepfake-detector/
├── data/               # Place input images / videos here
├── models/             # Optional fine-tuned weights (deepfake_detector.pth)
├── outputs/            # Saved visualisations and detection log
├── src/
│   ├── detect.py       # CLI entry point
│   ├── model.py        # ResNet-18 detector wrapper
│   ├── preprocess.py   # Image loading & normalisation
│   ├── frequency.py    # FFT frequency-domain analysis
│   ├── explainability.py  # Grad-CAM heatmap generation
│   └── logger.py       # Detection logging
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/RishiKumar1917/deepfake-detector.git
cd deepfake-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic detection

```bash
python src/detect.py --input data/sample.jpg
```

Sample output:

```
==================================================
  File       : sample.jpg
  Prediction : FAKE
  Confidence : 73.42%
==================================================
```

### All options

```
python src/detect.py --help

  --input       Path to the input image (required)
  --output-dir  Directory for visualisation output  (default: outputs/)
  --no-viz      Skip Grad-CAM / FFT visualisation
  --device      PyTorch device: cpu or cuda         (default: cpu)
  --weights     Path to fine-tuned weights file
```

### Example with custom output directory

```bash
python src/detect.py --input data/face.jpg --output-dir results/
```

---

## Output

For every analysed image a three-panel PNG is saved in `outputs/` (unless
`--no-viz` is used):

1. **Original image**
2. **Grad-CAM heatmap** – warmer colours indicate regions the model focused on
3. **FFT frequency spectrum** – log-magnitude spectrum of the image

Detection results are also appended to `outputs/detections.log`:

```
2024-06-01 12:34:56  INFO      file=face.jpg   prediction=FAKE  confidence=0.7342
```

---

## Using Fine-Tuned Weights

The system ships with ImageNet pre-trained backbone weights.  For production
accuracy you should fine-tune the model on a deepfake dataset (e.g.
FaceForensics++, DFDC) and save the state dict:

```python
torch.save(model.state_dict(), "models/deepfake_detector.pth")
```

The detector will load the file automatically on the next run.

---

## Extending the System

The modular design makes it straightforward to add:

* **Video detection** – iterate over frames and aggregate per-frame predictions
* **Webcam detection** – feed `cv2.VideoCapture(0)` frames to `predict()`
* **Real-time analysis** – wrap the pipeline in a Flask / FastAPI endpoint

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Deep learning inference |
| `torchvision` | ResNet-18 backbone + transforms |
| `opencv-python` | Image I/O and heatmap blending |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation output |
| `Pillow` | PIL image loading |
