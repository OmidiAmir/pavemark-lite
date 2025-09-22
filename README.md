# PaveMark-Lite

Lightweight **lane-marking segmentation** — **PyTorch → ONNX**, showcasing clean **MLOps** practices.

---

## Project Goals
- Extract **lane markings** from driving footage (binary segmentation).
- Train a **lightweight** model (U-Net + MobileNetV2).
- Export to **ONNX Runtime** for real-time inference (≥15 FPS @ 720p).
- End-to-end **MLOps flow**: data → preprocessing → training → evaluation → ONNX → demo.

---

## Repository Structure
```bash
pavemark-lite/
├── src/                # core code (dataset, training, models, utils)
│   ├── __init__.py
│   └── dataset/
│       ├── __init__.py
│       └── tusimple.py
├── configs/            # Hydra configs (data, train, model, export)
│   ├── train.yaml
│   └── data/
│       └── tusimple.yaml
├── scripts/            # helper tools (preprocessing, checks, debug)
│   ├── preprocessing.py
│   └── check_data_config.py
├── data/               # local datasets (ignored by git)
├── tests/              # unit tests (coming next)
├── README.md
├── .gitignore
├── LICENSE
```

---
## Quickstart (what already works)
```bash
# 1) Create TuSimple pairs (images/masks) + splits (e.g., 300 samples)
python scripts/preprocessing.py --src data/tusimple --out data/tusimple ^
  --clean --convert-train --cap 300 --make-splits --resize

# 2) Validate config paths
python scripts/check_data_config.py --fast

# 3) Load a batch via Hydra entrypoint (no training yet)
python -m src.train
# override example:
python -m src.train train.batch_size=4

```
---

Dataset & Preprocessing
- Dataset: TuSimple Lane Detection (≈23GB raw).
- Labels: provided as polylines → converted to binary masks (0=background, 255=lane).
- Pipeline (scripts/preprocessing.py):
  1. Convert JSON → images/ + masks/ (unique filenames)
  2. Write fixed splits/ (train/val/test)
  3. (Optional) letterbox to 1280×720 in preproc/

Data folders are .gitignored. Only scripts/configs are versioned.

--- 
## Status
- [x] Repo initialized & README polished
- [x] Dataset integration + preprocessing pipeline
- [x] Hydra entrypoint + DataLoader smoke-test
- [ ] Baseline training loop (U-Net MobileNetV2)
- [ ] Evaluation & error analysis
- [ ] Optuna HPO
- [ ] ONNX export + parity tests
- [ ] Inference demo (Gradio / OpenCV)

--- 

## Tech Stack
PyTorch · segmentation_models.pytorch · Hydra · Optuna · ONNX Runtime

---

## Config Management
- Dataset paths, splits, and image size are stored in **Hydra configs** (`configs/data/tusimple.yaml`).
- All training scripts read from these configs instead of hardcoding values.
- You can verify the dataset setup with:
  ```bash
  python scripts/check_data_config.py
  ```

  --- 

