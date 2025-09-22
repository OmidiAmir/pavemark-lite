import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pathlib import Path

# deps
import numpy as np
import torch
from PIL import Image

# dataset
from src.dataset.tusimple import TuSimpleSeg

@hydra.main(config_path="../configs", config_name="train.yaml")  # path is relative to this file
def main(cfg: DictConfig):
    # Print the merged config (nice for debugging)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    ds_cfg = cfg.dataset
    tr_cfg = cfg.train

    # Build dataset & dataloader (train split)
    train_ds = TuSimpleSeg(
        images_dir=ds_cfg.images,
        masks_dir=ds_cfg.masks,
        split_file=ds_cfg.train_file,
        height=ds_cfg.height,
        width=ds_cfg.width,
    )
    train_loader = DataLoader(
        train_ds, batch_size=tr_cfg.batch_size, shuffle=True, num_workers=tr_cfg.num_workers
    )

    # Smoke-check one batch
    imgs, msks = next(iter(train_loader))
    print("batch imgs:", imgs.shape, imgs.dtype, imgs.min().item(), imgs.max().item())
    print("batch msks:", msks.shape, msks.dtype, msks.min().item(), msks.max().item())

    # No training yet — just verifying Hydra + data wiring.
    print("[OK] Hydra entrypoint works. Ready to add a real training loop next.")

if __name__ == "__main__":
    main()
