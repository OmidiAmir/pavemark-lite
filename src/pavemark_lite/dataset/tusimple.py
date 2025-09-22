import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

class TuSimpleSeg(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, split_file: str, height: int = 720, width: int = 1280):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.height, self.width = height, width
        self.stems = [s for s in Path(split_file).read_text().splitlines() if s.strip()]

        if len(self.stems) == 0:
            raise ValueError(f"No items found in split file: {split_file}")

    def __len__(self) -> int:
        return len(self.stems) 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img = Image.open(self.images_dir / f"{stem}.jpg").convert("RGB")
        msk = Image.open(self.masks_dir  / f"{stem}.png").convert("L")  # 0/255

        # Ensure size matches target (your preprocessing already did this; keeping as safety)
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.BILINEAR)
        if msk.size != (self.width, self.height):
            msk = msk.resize((self.width, self.height), Image.NEAREST)

        # to tensors
        img_t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0        # [3,H,W]
        msk_t = torch.from_numpy(np.array(msk)).unsqueeze(0).float() / 255.0          # [1,H,W] in {0,1}

        return img_t, msk_t
