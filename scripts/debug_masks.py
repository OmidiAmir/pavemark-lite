from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from omegaconf import OmegaConf
from pavemark_lite.dataset.tusimple import TuSimpleSeg

def main():
    # load configs
    cfg = OmegaConf.load("configs/data/tusimple.yaml").dataset

    # dataset (train split, but val/test also fine)
    ds = TuSimpleSeg(
        images_dir=cfg.images,
        masks_dir=cfg.masks,
        split_file=cfg.train_file,
        height=cfg.height,
        width=cfg.width,
    )

    print(f"Dataset size: {len(ds)} samples")

    # check a few samples
    for i in range(5):
        img_t, msk_t = ds[i]   # img: [3,H,W], msk: [1,H,W]
        img = img_t.permute(1, 2, 0).numpy()     # [H,W,3]
        msk = msk_t.squeeze(0).numpy()           # [H,W]

        print(f"Sample {i}: mask unique values = {np.unique(msk)}")

        # plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Image")
        ax[0].axis("off")
        ax[1].imshow(img)
        ax[1].imshow(msk, cmap="Reds", alpha=0.5)  # overlay in red
        ax[1].set_title("Overlay mask")
        ax[1].axis("off")
        plt.show()

if __name__ == "__main__":
    main()
