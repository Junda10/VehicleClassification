# Vehicle Classification

A PyTorch + Streamlit computer vision project for classifying vehicle images into 10 categories. The repository includes model-training notebooks, saved model weights, and a small Streamlit inference app. The full training dataset is documented but intentionally excluded from the public Git repository to keep the project lightweight.

## Overview

This project uses transfer learning to classify vehicle images into the following classes:

- `SUV`
- `bus`
- `family sedan`
- `fire engine`
- `heavy truck`
- `jeep`
- `minibus`
- `racing car`
- `taxi`
- `truck`

The Streamlit app allows users to upload a vehicle image, choose a trained model, and receive a predicted class with confidence-based handling for unknown or non-vehicle images.

## Demo App

Run the local Streamlit application:

```bash
streamlit run Project.py
```

The app currently supports two ResNet50 checkpoints:

| Model | Checkpoint | Notes |
| --- | --- | --- |
| ResNet50 with frozen layers | `best_model.pth` | Transfer-learning baseline |
| ResNet50 with deeper layers unfrozen | `best_model_unfreeze.pth` | Higher-performing fine-tuned model |

## Dataset

The full dataset used during development is not tracked in this Git repository. Locally, it was organized under `vehicleClass/`:

```text
vehicleClass/
├── train/   # 1,400 labeled images, 10 classes, 140 images per class
├── val/     # 200 labeled images, 10 classes, 20 images per class
└── test/    # 200 unlabeled test images
```

The training and validation folders are class-balanced across all 10 vehicle categories. The Streamlit inference app does **not** require this dataset at runtime; it only needs the model checkpoint files and an uploaded image.

## Model Performance

The notebooks report the following best validation results during experimentation:

| Experiment | Architecture | Best validation accuracy |
| --- | --- | ---: |
| `Project.ipynb` | ResNet50 transfer learning | 93.5% |
| `Project copy_unfreezeDeeperLayer.ipynb` | ResNet50 fine-tuning | 97.5% |
| `Project copy_VGG.ipynb` | VGG19 transfer learning | 97.5% |

> Note: Results come from the existing notebook outputs and should be revalidated before publishing this as a production benchmark.

## Repository Structure

```text
.
├── Project.py                              # Streamlit inference app
├── Project.ipynb                          # ResNet50 training notebook
├── Project copy_unfreezeDeeperLayer.ipynb # ResNet50 fine-tuning notebook
├── Project copy_VGG.ipynb                 # VGG19 experiment notebook
├── best_model.pth                         # ResNet50 baseline checkpoint
├── best_model_unfreeze.pth                # Fine-tuned ResNet50 checkpoint
├── beep*.mp3                              # Audio alerts used by the app
├── requirements.txt                       # App/runtime Python dependencies
├── requirements-training.txt              # Notebook/training dependencies
└── docs/                                  # Deployment, demo, and cleanup notes
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Junda10/VehicleClassification.git
cd VehicleClassification
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run Project.py
```

## Usage

1. Open the Streamlit URL shown in your terminal.
2. Select a model checkpoint from the dropdown.
3. Upload a `.jpg`, `.jpeg`, or `.png` image.
4. Review the predicted vehicle class and confidence score.

The app also plays different alert sounds for selected categories, such as heavy vehicles and emergency vehicles.

## Production Readiness Notes

This repository is being prepared for a more production-ready GitHub presentation. Current improvement opportunities include:

- Move model artifacts to GitHub Releases, Git LFS, or external storage if the repo needs to be even lighter.
- Add a cleaner `src/` package structure for model loading, preprocessing, and prediction logic.
- Add a lightweight demo deployment using Streamlit Community Cloud or Hugging Face Spaces.
- Add screenshots, example predictions, a license, and a model card.

Completed cleanup so far:

- `Project.py` no longer requires scanning `vehicleClass/train/` at runtime to determine class names.
- Inference preprocessing now uses deterministic resize/crop/normalize transforms instead of random training augmentations.
- Runtime dependencies are separated from training/notebook dependencies.
- The full `vehicleClass/` dataset has been removed from Git tracking; local copies remain ignored by `.gitignore`.

## Deployment Options

GitHub Pages is best for static websites and cannot directly host this PyTorch/Streamlit inference app. Better options for an interactive demo are:

- **Streamlit Community Cloud** — easiest path for the current `Project.py` app.
- **Hugging Face Spaces** — good for ML demos and model artifacts.
- **GitHub Pages** — useful for a static project landing page that links to the live app.

See the project docs for next steps:

- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — recommended hosting plan and pre-deployment checklist.
- [`docs/DEMO.md`](docs/DEMO.md) — screenshot/GIF guide for a polished GitHub presentation.
- [`docs/REPOSITORY_CLEANUP.md`](docs/REPOSITORY_CLEANUP.md) — large artifact cleanup options for dataset and model files.

## Tech Stack

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow
- Matplotlib

## Status

Work in progress: the project currently runs as a local Streamlit prototype and is being cleaned up for portfolio-quality GitHub presentation.
