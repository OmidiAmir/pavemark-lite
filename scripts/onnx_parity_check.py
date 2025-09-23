# scripts/onnx_parity_check.py
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf
from pavemark_lite.dataset.tusimple import TuSimpleSeg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def normalize(x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W] in [0,1]
    mean = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(1,3,1,1)
    return (x - mean) / std

def iou_bin(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [B,1,H,W] in {0,1}
    inter = (a*b).sum().item()
    union = ((a+b)>0).sum().item()
    return inter/union if union>0 else 1.0

def main():
    # Load YAMLs directly (no Hydra composition needed)
    train_cfg = OmegaConf.load("configs/train.yaml")
    data_cfg  = OmegaConf.load("configs/data/tusimple.yaml").dataset

    # dataset
    ds = TuSimpleSeg(
        images_dir=data_cfg.images,
        masks_dir=data_cfg.masks,
        split_file=data_cfg.val_file,  # use val for parity
        height=data_cfg.height,
        width=data_cfg.width,
    )

    # PyTorch model (+ weights)
    model = smp.Unet(
        encoder_name=train_cfg.model.encoder,
        encoder_weights=None,   # weights from checkpoint
        in_channels=3,
        classes=1,
    )
    ckpt = Path(train_cfg.train.save_dir) / train_cfg.train.save_name
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # ONNX session
    onnx_path = Path("export/best.onnx")
    if not onnx_path.exists():
        raise SystemExit("Run the exporter first to create export/best.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Compare a handful of samples
    n = min(8, len(ds))
    ious = []
    for idx in range(n):
        img_t, _ = ds[idx]                        # [3,H,W] in [0,1]
        img_b = img_t.unsqueeze(0)               # [1,3,H,W]
        img_n = normalize(img_b)                 # [1,3,H,W]

        # Torch
        with torch.no_grad():
            logits_t = model(img_n)              # [1,1,H,W]
            pred_t = (torch.sigmoid(logits_t) > 0.5).float()

        # ONNX
        logits_o = sess.run([out_name], {in_name: img_n.numpy().astype(np.float32)})[0]  # (1,1,H,W)
        pred_o = (1/(1+np.exp(-logits_o)) > 0.5).astype(np.float32)
        pred_o = torch.from_numpy(pred_o)

        iou = iou_bin(pred_t, pred_o)
        ious.append(iou)

    mean_iou = sum(ious)/len(ious)
    print(f"[parity] mean IoU(model vs onnx) = {mean_iou:.4f}")
    if mean_iou < 0.99:
        raise SystemExit("Parity too low. Expected IoU≥0.99. Check normalization or export settings.")
    print("[parity] OK")

if __name__ == "__main__":
    main()
