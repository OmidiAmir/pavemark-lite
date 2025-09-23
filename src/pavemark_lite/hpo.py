import hydra
from omegaconf import DictConfig
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryF1Score

from pavemark_lite.dataset.tusimple import TuSimpleSeg
from pavemark_lite.train import normalize_batch, freeze_encoder
from pathlib import Path
import json
import random

def make_model(cfg):
    return smp.Unet(
        encoder_name=cfg.model.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

def objective(trial, cfg):
    # sample space (log sampling for lr & wd)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    bs = trial.suggest_categorical("batch_size", [2, 4, 8])

    ds = cfg.data.dataset
    # small, reproducible subset for speed
    train_ds = TuSimpleSeg(ds.images, ds.masks, ds.train_file, ds.height, ds.width)
    val_ds   = TuSimpleSeg(ds.images, ds.masks, ds.val_file,   ds.height, ds.width)

    # (optional) downsample dataset for fast search
    random.seed(1337)
    if len(train_ds) > 400:
        train_ds.stems = train_ds.stems[:400]
    if len(val_ds) > 200:
        val_ds.stems = val_ds.stems[:200]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg).to(device)
    freeze_encoder(model, freeze=True)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    f1 = BinaryF1Score().to(device)

    # one short epoch
    model.train()
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        imgs = normalize_batch(imgs)
        opt.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, msks)
        loss.backward()
        opt.step()

    # quick val
    model.eval(); f1.reset()
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            imgs = normalize_batch(imgs)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            f1.update(preds.squeeze(1), msks.squeeze(1))

    score = float(f1.compute().item())
    return score  # maximize F1

@hydra.main(version_base=None, config_path='../../configs', config_name='train.yaml')
def main(cfg: DictConfig):
    n_trials = int(cfg.hpo.n_trials)
    timeout  = int(cfg.hpo.timeout) if "timeout" in cfg.hpo else None

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, cfg), n_trials=n_trials, timeout=timeout)

    print("[HPO] Best val_f1:", study.best_trial.value)
    print("[HPO] Best params:", study.best_trial.params)

    # save best params for easy reuse
    out_dir = Path("reports/hpo"); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_params.json"
    best_cmd  = (
        f'python -m pavemark_lite.train '
        f'train.lr={study.best_trial.params["lr"]} '
        f'train.weight_decay={study.best_trial.params["weight_decay"]} '
        f'train.batch_size={study.best_trial.params["batch_size"]} '
        f'train.epochs=20'
    )
    json.dump(
        {"best_val_f1": study.best_trial.value,
         "params": study.best_trial.params,
         "train_command": best_cmd},
        open(best_path, "w"), indent=2
    )
    print(f"[HPO] Saved best params → {best_path}")
    print("[HPO] Next, run:\n", best_cmd)

if __name__ == "__main__":
    main()
