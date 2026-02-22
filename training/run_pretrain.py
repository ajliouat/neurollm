"""Pre-training run script — trains MCM on synthetic TUH proxy data
and records convergence numbers.

Usage:
    python -m training.run_pretrain [--epochs 50] [--batch_size 32]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import TUHEEGDataset
from model.transformer import NeuRoLLM
from training.pretrain import MCMPretrainer, train_mcm


DEFAULT_CONFIG = {
    "n_channels": 19,
    "patch_size": 50,
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 512,
    "n_layers": 6,
    "dropout": 0.1,
    "mask_ratio": 0.3,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "n_epochs": 50,
    "batch_size": 32,
}


def run_pretrain(
    config: dict | None = None,
    data_dir: str | Path = "data/raw/tuh",
    checkpoint_dir: str | Path = "checkpoints/pretrain",
    device: str = "cpu",
) -> dict:
    """Execute pre-training and return results dict."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    device_obj = torch.device(device)

    # Dataset (falls back to synthetic)
    dataset = TUHEEGDataset(data_dir)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
    )

    # Model
    model = NeuRoLLM(
        n_channels=cfg["n_channels"],
        patch_size=cfg["patch_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )

    pretrainer = MCMPretrainer(model, mask_ratio=cfg["mask_ratio"])

    print(f"Model parameters: {model.num_parameters:,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches/epoch: {len(loader)}")
    print(f"Device: {device_obj}")

    t0 = time.time()
    result = train_mcm(
        pretrainer=pretrainer,
        dataloader=loader,
        n_epochs=cfg["n_epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        checkpoint_dir=checkpoint_dir,
        device=device_obj,
    )
    elapsed = time.time() - t0

    # Store results
    output = {
        "config": cfg,
        "dataset_size": len(dataset),
        "model_params": model.num_parameters,
        "loss_history": result["loss_history"],
        "best_loss": result["best_loss"],
        "epochs_trained": result["epochs_trained"],
        "first_loss": result["loss_history"][0] if result["loss_history"] else None,
        "final_loss": result["loss_history"][-1] if result["loss_history"] else None,
        "elapsed_seconds": round(elapsed, 2),
    }

    # Save results JSON
    results_dir = Path(checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "pretrain_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nPre-training complete in {elapsed:.1f}s")
    print(f"  First loss: {output['first_loss']:.4f}")
    print(f"  Final loss: {output['final_loss']:.4f}")
    print(f"  Best loss:  {output['best_loss']:.4f}")
    print(f"  Reduction:  {(1 - output['final_loss']/output['first_loss'])*100:.1f}%")

    return output


def main():
    parser = argparse.ArgumentParser(description="NeuRoLLM MCM pre-training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_dir", type=str, default="data/raw/tuh")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain")
    args = parser.parse_args()

    config = {
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    run_pretrain(
        config=config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
