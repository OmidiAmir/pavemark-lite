import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset.tusimple import TuSimpleSeg
import torch

cfg = OmegaConf.load("configs/data/tusimple.yaml").dataset

# Use train split for a quick check
ds = TuSimpleSeg(
    images_dir=cfg.images,
    masks_dir=cfg.masks,
    split_file=cfg.train_file,
    height=cfg.height,
    width=cfg.width,
)

dl = DataLoader(ds, batch_size=2, shuffle=False)

batch = next(iter(dl))
imgs, msks = batch
print("imgs:", imgs.shape, imgs.dtype, imgs.min().item(), imgs.max().item())
print("msks:", msks.shape, msks.dtype, msks.min().item(), msks.max().item())

# sanity: visualize first overlay (optional)
try:
    import torchvision.utils as vutils
    vutils.save_image(imgs[0], "data/tusimple/debug/loader_img0.jpg")
    vutils.save_image(msks[0], "data/tusimple/debug/loader_msk0.jpg")
    print("Saved debug images under data/tusimple/debug/")
except Exception as e:
    print("Optional save failed:", e)
