import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

# dataset
from pavemark_lite.dataset.tusimple import TuSimpleSeg

# model
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

# --- tiny helpers ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    # x in [0,1], shape [B,3,H,W]
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD, device=x.device).view(1,3,1,1)
    return (x - mean) / std

def make_model(cfg):
    model = smp.Unet(
        encoder_name=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=3,
        classes=cfg.model.classes,
    )
    return model

def freeze_encoder(model: nn.Module, freeze: bool = True):
    for name, param in model.named_parameters():
        if "encoder." in name:
            param.requires_grad = not freeze

def step_metrics(logits, target):
    # logits: [B,1,H,W], target: [B,1,H,W] in {0,1}
    preds = (torch.sigmoid(logits) > 0.5).float()
    return preds

@hydra.main(version_base=None, config_path='../../configs', config_name='train.yaml')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ---- device / amp ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")

    # ---- dataset & loaders ----
    ds_cfg = cfg.data.dataset
    train_ds = TuSimpleSeg(
        images_dir=ds_cfg.images,
        masks_dir=ds_cfg.masks,
        split_file=ds_cfg.train_file,
        height=ds_cfg.height,
        width=ds_cfg.width,
    )
    val_ds = TuSimpleSeg(
        images_dir=ds_cfg.images,
        masks_dir=ds_cfg.masks,
        split_file=ds_cfg.val_file,
        height=ds_cfg.height,
        width=ds_cfg.width,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

    # ---- model / loss / opt ----
    model = make_model(cfg).to(device)
    freeze_encoder(model, freeze=True)   # start frozen
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    f1 = BinaryF1Score().to(device)
    iou = BinaryJaccardIndex().to(device)

    save_dir = Path(cfg.train.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    # ---- train loop ----
    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        if epoch == cfg.train.freeze_epochs + 1:
            # unfreeze encoder for fine-tuning
            freeze_encoder(model, freeze=False)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        running = 0.0
        f1.reset(); iou.reset()

        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
            imgs = normalize_batch(imgs)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)                 # [B,1,H,W]
                loss = bce(logits, msks)             # msks in {0,1}
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # metrics on-the-fly (threshold 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            f1.update(preds.squeeze(1), msks.squeeze(1))
            iou.update(preds.squeeze(1), msks.squeeze(1))

            running += loss.item()
            global_step += 1

        train_loss = running / max(1, len(train_loader))
        train_f1 = f1.compute().item()
        train_iou = iou.compute().item()

        # ---- validation ----
        model.eval()
        f1.reset(); iou.reset()
        val_running = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
                imgs = normalize_batch(imgs)
                logits = model(imgs)
                val_running += bce(logits, msks).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                f1.update(preds.squeeze(1), msks.squeeze(1))
                iou.update(preds.squeeze(1), msks.squeeze(1))

        val_loss = val_running / max(1, len(val_loader))
        val_f1 = f1.compute().item()
        val_iou = iou.compute().item()

        print(f"[epoch {epoch}/{cfg.train.epochs}] "
              f"train_loss={train_loss:.4f} f1={train_f1:.3f} iou={train_iou:.3f} | "
              f"val_loss={val_loss:.4f} f1={val_f1:.3f} iou={val_iou:.3f}")

        # save best by val_loss
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = save_dir / cfg.train.save_name
            torch.save({"model": model.state_dict(),
                        "cfg": OmegaConf.to_container(cfg, resolve=True)},
                       ckpt_path)
            print(f"[save] best checkpoint → {ckpt_path}")

    print("[done] baseline training finished.")

if __name__ == "__main__":
    main()
