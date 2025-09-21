#!/usr/bin/env python
# Faster config checker with optional --fast sampling and clearer diagnostics.

import argparse, os
from pathlib import Path
from omegaconf import OmegaConf

CFG_PATH_DEFAULT = "configs/data/tusimple.yaml"

def count_files_scandir(dir_path: Path, ext: str, limit: int | None = None):
    """Fast count & (optionally) sample stems using os.scandir (non-recursive)."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return 0, []
    n = 0
    stems = []
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file():
                name = entry.name
                if name.lower().endswith(ext):
                    n += 1
                    if limit is not None and len(stems) < limit:
                        stems.append(Path(name).stem)
    return n, stems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=CFG_PATH_DEFAULT, help="Path to hydra/omegaconf YAML")
    ap.add_argument("--fast", action="store_true", help="Sample a subset (speeds up on large dirs)")
    ap.add_argument("--sample", type=int, default=200, help="Sample size when --fast is used")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    ds = cfg.dataset

    root   = Path(ds.root)
    imgs   = Path(ds.images)
    masks  = Path(ds.masks)
    splits = Path(ds.splits)

    print("[check] Using config:", Path(args.cfg).resolve())
    print(f"[check] dataset.root : {root.resolve()}")
    print(f"[check] images dir   : {imgs.resolve()}")
    print(f"[check] masks dir    : {masks.resolve()}")
    print(f"[check] splits dir   : {splits.resolve()}")

    problems = []
    for p, label in [(root, "dataset.root"), (imgs, "images"), (masks, "masks"), (splits, "splits")]:
        if not p.exists():
            problems.append(f"- Missing: {label} → {p}")
    for fkey in ["train_file", "val_file", "test_file"]:
        f = Path(getattr(ds, fkey))
        if not f.exists():
            problems.append(f"- Missing: {fkey} → {f}")

    if problems:
        print("[FAIL] Config points to missing paths/files:")
        print("\n".join(problems))
        raise SystemExit(1)

    # Fast counts and optional sampling
    limit = args.sample if args.fast else None
    n_img, img_sample = count_files_scandir(imgs, ".jpg", limit=limit)
    n_msk, msk_sample = count_files_scandir(masks, ".png", limit=limit)

    print(f"[ok] images: {n_img} *.jpg  (sampled {len(img_sample) if limit else 'all'})")
    print(f"[ok] masks : {n_msk} *.png  (sampled {len(msk_sample) if limit else 'all'})")

    # If fast, estimate paired count using the sample; else compute exactly on full sets.
    if args.fast:
        img_set = set(img_sample)
        msk_set = set(msk_sample)
        paired_est = len(img_set & msk_set)
        print(f"[ok] paired (estimated from sample): ~{paired_est}")
    else:
        # full (may be slower)
        img_set = {p.stem for p in imgs.glob("*.jpg")}
        msk_set = {p.stem for p in masks.glob("*.png")}
        paired = len(img_set & msk_set)
        print(f"[ok] paired (exact): {paired}")

    print(f"[ok] target size (HxW): {ds.height}x{ds.width}")
    print("[done] Config check completed.")

if __name__ == "__main__":
    main()


