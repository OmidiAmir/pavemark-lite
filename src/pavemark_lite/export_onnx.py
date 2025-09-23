import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

@hydra.main(version_base=None, config_path='../../configs', config_name='train.yaml')
def main(cfg: DictConfig):
    # --- build the model (same arch as training) ---
    model = smp.Unet(
        encoder_name=cfg.model.encoder,
        encoder_weights=None,   # weights come from checkpoint
        in_channels=3,
        classes=1,
    )
    ckpt = Path(cfg.train.save_dir) / cfg.train.save_name
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # --- dummy input (B=1, 3, H, W) ---
    H, W = cfg.data.dataset.height, cfg.data.dataset.width
    dummy = torch.randn(1, 3, H, W, dtype=torch.float32)

    # --- export path ---
    out_dir = Path("export"); out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "best.onnx"

    # --- export with dynamic axes ---
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"[export] ONNX saved → {onnx_path.resolve()}")

if __name__ == "__main__":
    main()
