import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from pavemark_lite.dataset.tusimple import TuSimpleSeg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD, device=x.device).view(1,3,1,1)
    return (x - mean) / std

def make_model(cfg):
    return smp.Unet(
        encoder_name=cfg.model.encoder,
        encoder_weights=None,   # weights come from checkpoint
        in_channels=3,
        classes=1,
    )

def save_overlay(img: torch.Tensor, pred: torch.Tensor, out_path: Path, alpha=0.45):
    """
    img: [3,H,W] in [0,1]
    pred: [1,H,W] binary {0,1}
    """
    img_np = (img.cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
    msk_np = (pred.cpu().numpy().squeeze(0) * 255).astype(np.uint8)

    base = Image.fromarray(img_np).convert("RGB")
    mask = Image.fromarray(msk_np).convert("L")
    red  = Image.new("RGB", base.size, (255,0,0))
    mask_a = mask.point(lambda v: int(v>127) * int(255*alpha))
    over = Image.composite(red, base, mask_a)
    out = Image.blend(base, over, alpha)
    out.save(out_path)

@hydra.main(version_base=None, config_path='../../configs', config_name='train.yaml')
def main(cfg: DictConfig):
    print("[eval] using config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    ds_cfg = cfg.data.dataset
    save_dir = Path("reports/eval"); save_dir.mkdir(parents=True, exist_ok=True)

    # dataset/loader (use test split; switch to val if you prefer)
    ds = TuSimpleSeg(ds_cfg.images, ds_cfg.masks, ds_cfg.test_file, ds_cfg.height, ds_cfg.width)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    # model + load checkpoint
    ckpt_path = Path(cfg.train.save_dir) / cfg.train.save_name
    model = make_model(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    f1 = BinaryF1Score().to(device)
    iou = BinaryJaccardIndex().to(device)

    with torch.no_grad():
        for i, (imgs, msks) in enumerate(dl):
            imgs, msks = imgs.to(device), msks.to(device)
            imgs_n = normalize_batch(imgs)
            logits = model(imgs_n)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            f1.update(preds.squeeze(1), msks.squeeze(1))
            iou.update(preds.squeeze(1), msks.squeeze(1))

            # save a few overlays (first 6 images)
            if i < 3:
                for b in range(imgs.size(0)):
                    out_path = save_dir / f"overlay_{i:03d}_{b}.jpg"
                    save_overlay(imgs[b], preds[b], out_path)

    print(f"[eval] F1  = {f1.compute().item():.3f}")
    print(f"[eval] mIoU= {iou.compute().item():.3f}")
    (save_dir / "metrics.txt").write_text(f"F1={f1.compute().item():.6f}\nmIoU={iou.compute().item():.6f}\n")
    print(f"[eval] wrote overlays & metrics to: {save_dir.resolve()}")

if __name__ == "__main__":
    main()
