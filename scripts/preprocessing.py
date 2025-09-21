#!/usr/bin/env python
"""
preprocessing.py
TuSimple → segmentation-ready dataset (images, masks, splits, optional 720p preproc)

Usage examples (from repo root, PowerShell):

# 1) Convert a small training subset (e.g., 300), create splits, and resize to 720p
python preprocessing.py --src data/tusimple --out data/tusimple --convert-train --cap 300 --make-splits --resize

# 2) Also include labeled test (if test_label.json exists)
python preprocessing.py --src data/tusimple --out data/tusimple --convert-train --convert-test --make-splits --resize

# 3) Clean (remove images/, masks/, splits/, preproc/) then rebuild
python preprocessing.py --src data/tusimple --out data/tusimple --clean --convert-train --cap 300 --make-splits --resize
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw


class TuSimplePreprocessor:
    def __init__(self, src: Path, out: Path):
        """
        :param src: folder containing train_set/, test_set/, test_label.json
        :param out: dataset root where images/, masks/, splits/, preproc/ live
        """
        self.src = Path(src)
        self.out = Path(out)

        # Input roots
        self.train_root = self.src / "train_set"
        self.test_root = self.src / "test_set"
        self.test_label = self.src / "test_label.json"

        # Output dirs
        self.images_dir = self.out / "images"
        self.masks_dir = self.out / "masks"
        self.splits_dir = self.out / "splits"
        self.preproc_root = self.out / "preproc"
        self.images_720 = self.preproc_root / "images_720"
        self.masks_720 = self.preproc_root / "masks_720"

        # Make base output root
        self.out.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------

    @staticmethod
    def _parse_jsonlines(p: Path):
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)

    @staticmethod
    def _rasterize_mask(size: Tuple[int, int], polylines: List[List[Tuple[int, int]]], thickness: int = 5) -> Image.Image:
        m = Image.new("L", size, 0)
        d = ImageDraw.Draw(m)
        for pts in polylines:
            if len(pts) >= 2:
                d.line(pts, fill=255, width=thickness)
        return m

    @staticmethod
    def _letterbox(im: Image.Image, target_w: int, target_h: int, is_mask: bool) -> Image.Image:
        w, h = im.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        im_resized = im.resize((new_w, new_h), resample=resample)
        mode = "L" if is_mask else "RGB"
        canvas = Image.new(mode, (target_w, target_h), 0)
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        canvas.paste(im_resized, (left, top))
        if is_mask:
            canvas = canvas.point(lambda v: 255 if v >= 128 else 0).convert("L")
        return canvas

    # ---------- core steps ----------

    def clean_outputs(self):
        """Remove images/, masks/, splits/, preproc/."""
        for p in [self.images_dir, self.masks_dir, self.splits_dir, self.preproc_root]:
            if p.exists():
                shutil.rmtree(p)
                print(f"[clean] removed {p}")
        print("[clean] done.")

    def _collect_labeled_from_train(self) -> List[Tuple[Path, List[List[Tuple[int, int]]]]]:
        """Return (img_path, polylines) for all labeled train frames."""
        if not self.train_root.exists():
            raise SystemExit(f"train_set not found at {self.train_root}")
        label_files = sorted(self.train_root.rglob("*label*.json"))
        if not label_files:
            raise SystemExit(f"No *label*.json files in {self.train_root}")

        items = []
        for lf in label_files:
            for jd in self._parse_jsonlines(lf):
                raw_rel = jd.get("raw_file")
                lanes = jd.get("lanes", [])
                hs = jd.get("h_samples", [])
                if not raw_rel or not lanes or not hs:
                    continue
                polylines = []
                for lane in lanes:
                    pts = [(x, y) for x, y in zip(lane, hs) if x != -2]
                    if len(pts) >= 2:
                        polylines.append(pts)
                if polylines:
                    items.append((self.train_root / raw_rel, polylines))
        return items

    def _collect_labeled_from_test(self) -> List[Tuple[Path, List[List[Tuple[int, int]]]]]:
        """Return (img_path, polylines) for labeled test frames (test_label.json)."""
        if not self.test_root.exists() or not self.test_label.exists():
            print("[warn] test_set or test_label.json missing; skipping test labels.")
            return []
        items = []
        for jd in self._parse_jsonlines(self.test_label):
            raw_rel = jd.get("raw_file")
            lanes = jd.get("lanes", [])
            hs = jd.get("h_samples", [])
            if not raw_rel or not lanes or not hs:
                continue
            polylines = []
            for lane in lanes:
                pts = [(x, y) for x, y in zip(lane, hs) if x != -2]
                if len(pts) >= 2:
                    polylines.append(pts)
            if polylines:
                items.append((self.test_root / raw_rel, polylines))
        return items

    def convert_train(self, cap: int = 300, thickness: int = 5) -> List[str]:
        """Create images/ + masks/ from train_set labels; returns list of stems."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        items = self._collect_labeled_from_train()
        if not items:
            raise SystemExit("No labeled train items found.")
        random.seed(1337)
        random.shuffle(items)
        if cap and len(items) > cap:
            items = items[:cap]

        stems = []
        kept = 0
        for img_path, polys in items:
            if not img_path.exists():
                continue
            with Image.open(img_path).convert("RGB") as img:
                w, h = img.size
                mask = self._rasterize_mask((w, h), polys, thickness=thickness)

                # Unique stem from path under train_set (avoid overwrites)
                rel = img_path.relative_to(self.train_root).with_suffix("")
                stem = "_".join(rel.parts).replace("-", "_")
                img.save(self.images_dir / f"{stem}.jpg", quality=92)
                mask.save(self.masks_dir / f"{stem}.png")
                stems.append(stem)
                kept += 1

        print(f"[convert_train] saved {kept} pairs → {self.images_dir} , {self.masks_dir}")
        return stems

    def convert_test(self) -> List[str]:
        """Optionally add labeled test pairs to the same images/ + masks/; returns stems."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        items = self._collect_labeled_from_test()
        stems = []
        kept = 0
        for img_path, polys in items:
            if not img_path.exists():
                continue
            with Image.open(img_path).convert("RGB") as img:
                w, h = img.size
                mask = self._rasterize_mask((w, h), polys, thickness=5)
                rel = img_path.relative_to(self.test_root).with_suffix("")
                stem = "test_" + "_".join(rel.parts).replace("-", "_")
                img.save(self.images_dir / f"{stem}.jpg", quality=92)
                mask.save(self.masks_dir / f"{stem}.png")
                stems.append(stem)
                kept += 1

        print(f"[convert_test] saved {kept} pairs (if any).")
        return stems

    def make_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, use_test_prefix: bool = True):
        """Create train/val/test splits from saved files."""
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        img_stems = {p.stem for p in self.images_dir.glob("*.jpg")}
        msk_stems = {p.stem for p in self.masks_dir.glob("*.png")}
        stems = sorted(img_stems & msk_stems)
        if not stems:
            raise SystemExit("No paired (image, mask) files found; convert first.")

        # If we added test images with a "test_" prefix, we can place them into the test split directly.
        test_stems = sorted([s for s in stems if s.startswith("test_")]) if use_test_prefix else []
        train_pool = [s for s in stems if s not in test_stems]

        random.seed(1337)
        random.shuffle(train_pool)
        n = len(train_pool)
        n_tr = int(train_ratio * n)
        n_val = int(val_ratio * n)
        tr = train_pool[:n_tr]
        val = train_pool[n_tr:n_tr + n_val]
        te = test_stems if test_stems else train_pool[n_tr + n_val:]

        (self.splits_dir / "train.txt").write_text("\n".join(tr))
        (self.splits_dir / "val.txt").write_text("\n".join(val))
        (self.splits_dir / "test.txt").write_text("\n".join(te))
        print(f"[splits] train={len(tr)}, val={len(val)}, test={len(te)} → {self.splits_dir}")

    def resize_to_720p(self, target_w: int = 1280, target_h: int = 720):
        """Letterbox-resize all pairs listed in splits/ to 1280x720 → preproc/images_720, masks_720."""
        self.images_720.mkdir(parents=True, exist_ok=True)
        self.masks_720.mkdir(parents=True, exist_ok=True)

        # Read all stems in splits (train + val + test)
        stems = set()
        for name in ["train.txt", "val.txt", "test.txt"]:
            p = self.splits_dir / name
            if p.exists():
                stems.update(p.read_text().splitlines())

        if not stems:
            raise SystemExit("No stems found in splits; run make_splits first.")

        done = 0
        for stem in sorted(stems):
            ip = self.images_dir / f"{stem}.jpg"
            mp = self.masks_dir / f"{stem}.png"
            if not (ip.exists() and mp.exists()):
                continue
            with Image.open(ip).convert("RGB") as im:
                im720 = self._letterbox(im, target_w, target_h, is_mask=False)
                im720.save(self.images_720 / f"{stem}.jpg", quality=92)
            with Image.open(mp).convert("L") as mk:
                mk720 = self._letterbox(mk, target_w, target_h, is_mask=True)
                mk720.save(self.masks_720 / f"{stem}.png")
            done += 1
            if done % 50 == 0:
                print(f"[resize] processed {done} pairs...")

        print(f"[resize] finished {done} pairs → {self.images_720} , {self.masks_720}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tusimple", help="Folder with train_set/, test_set/, test_label.json")
    ap.add_argument("--out", default="data/tusimple", help="Output dataset root")
    ap.add_argument("--clean", action="store_true", help="Remove images/, masks/, splits/, preproc/ before running")

    # which actions
    ap.add_argument("--convert-train", action="store_true", help="Convert labeled train_set → images/masks")
    ap.add_argument("--convert-test", action="store_true", help="Convert labeled test_set → images/masks (if test_label.json exists)")
    ap.add_argument("--make-splits", action="store_true", help="Write train/val/test splits")
    ap.add_argument("--resize", action="store_true", help="Letterbox to 1280x720 into preproc/")

    # options
    ap.add_argument("--cap", type=int, default=300, help="Max TRAIN samples for conversion (ignored for test)")
    ap.add_argument("--thickness", type=int, default=5, help="Polyline thickness when rasterizing masks")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    proc = TuSimplePreprocessor(Path(args.src), Path(args.out))

    if args.clean:
        proc.clean_outputs()

    stems_train = []
    stems_test = []
    if args.convert_train:
        stems_train = proc.convert_train(cap=args.cap, thickness=args.thickness)
    if args.convert_test:
        stems_test = proc.convert_test()

    if args.make_splits:
        proc.make_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio, use_test_prefix=bool(stems_test))

    if args.resize:
        proc.resize_to_720p(target_w=1280, target_h=720)

    if not any([args.convert_train, args.convert_test, args.make_splits, args.resize, args.clean]):
        print("No action specified. Use --convert-train / --convert-test / --make-splits / --resize / --clean")


if __name__ == "__main__":
    main()
