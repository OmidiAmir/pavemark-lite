# PaveMark-Lite

Lightweight lane-marking segmentation pipeline — **PyTorch → ONNX**, built with modern **MLOps practices**.

---

## 🔹 Project Goals
- Extract lane markings (painted lines) from road images.
- Train a lightweight segmentation model (U-Net with MobileNetV2).
- Export to **ONNX Runtime** for real-time inference (≥15 FPS @ 720p).
- Demonstrate end-to-end **MLOps workflow**: data → training → evaluation → export → demo.

---

## 📂 Repository Structure
```bash
pavemark-lite/
├── src/            # all python code (training, models, utils)
│   └── __init__.py
├── configs/        # Hydra configs (data, model, train, eval, hpo, export)
├── scripts/        # helper scripts (fetch data, make splits, etc.)
├── data/           # local data folder (ignored in git)
├── tests/          # unit tests, export parity tests
├── docs/           # notes, reports, diagrams
├── README.md
├── .gitignore
├── LICENSE

---

## 🚀 Status
- [x] Repo initialized
- [ ] Dataset integration
- [ ] Baseline training pipeline
- [ ] ONNX export + parity checks
- [ ] Demo app (Gradio / OpenCV)

--- 