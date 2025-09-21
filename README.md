# PaveMark-Lite

Lightweight lane-marking segmentation pipeline — **PyTorch → ONNX**, built with modern **MLOps practices**.

---

## Project Goals
- Extract lane markings (painted lines) from road images.
- Train a lightweight segmentation model (U-Net with MobileNetV2).
- Export to **ONNX Runtime** for real-time inference (≥15 FPS @ 720p).
- Demonstrate end-to-end **MLOps workflow**:  data → preprocessing → training → evaluation → ONNX export → demo.

---

## Repository Structure
```bash
pavemark-lite/
├── src/                # core python code (training, models, utils)
│   └── __init__.py
├── configs/            # Hydra configs (data, model, train, eval, hpo, export)
├── scripts/            # helper scripts (peek sample, debug, preprocessing, etc.)
│   └── preprocessing.py
├── data/               # local dataset (ignored by git)
│   └── tusimple/       # TuSimple dataset (train_set, test_set, JSON labels)
├── tests/              # unit tests, export parity tests
├── docs/               # reports, diagrams, notes
├── README.md
├── .gitignore
├── LICENSE
```

---

Dataset & Preprocessing
- Dataset: TuSimple Lane Detection
 (≈23GB raw).
- Labels: Provided as polylines in JSON → rasterized into binary masks (0 = background, 1 = lane).
- Preprocessing pipeline (via scripts/preprocessing.py):
- Convert JSON labels → images/ + masks/ (unique filenames).
- Create fixed train/val/test splits.
- (Optional) Resize/letterbox → 1280×720 into preproc/ folder.

Note: Data folders are .gitignored — only scripts/configs are versioned.


## Current Status
- [x] Repo initialized
- [x] Dataset integrated (TuSimple, preprocessing pipeline)
- [] Baseline model training
- [] Evaluation & error analysis
- [] Hyperparameter optimization (Optuna)
- [] ONNX export + parity tests
- [] Inference demo (Gradio / OpenCV)

--- 

## Tech Stack
- PyTorch · segmentation_models.pytorch
- Hydra for configs
- Optuna for hyperparameter search
- ONNX Runtime for deployment
- Metaflow (planned) for pipeline orchestration

---

