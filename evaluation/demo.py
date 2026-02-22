"""Quick demo — synthetic pre-train → fine-tune → evaluate → visualize.

Usage:
    python -m evaluation.demo
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import BCIIV2aDataset, TUHEEGDataset
from model.transformer import NeuRoLLM
from training.pretrain import MCMPretrainer, train_mcm
from training.finetune import (
    finetune_subject,
    load_pretrained_encoder,
)
from evaluation.metrics import (
    compute_metrics,
    extract_attention_maps,
    attention_to_channel_importance,
    plot_confusion_matrix,
    plot_channel_attention,
    plot_loss_curve,
)
from data.preprocessing import BCI_IV_CHANNELS_22


def run_demo() -> dict:
    """Run a compact end-to-end demo on synthetic data."""
    print("=" * 60)
    print("  NeuRoLLM — Quick Demo (synthetic data)")
    print("=" * 60)

    device = torch.device("cpu")
    tmp = tempfile.mkdtemp(prefix="neurollm_demo_")
    ckpt_dir = Path(tmp) / "checkpoints"
    fig_dir = Path(tmp) / "figures"
    ckpt_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)

    # ── Step 1: Pre-training ──────────────────────────────────────
    print("\n[1/4] Pre-training (MCM, 3 epochs on synthetic TUH)...")
    tuh = TUHEEGDataset("data/raw/tuh")
    loader_pt = DataLoader(tuh, batch_size=32, shuffle=True, drop_last=True)

    model_pt = NeuRoLLM(
        n_channels=19, patch_size=50, d_model=64,
        n_heads=2, d_ff=128, n_layers=1,
    )
    pretrainer = MCMPretrainer(model_pt, mask_ratio=0.3)

    pt_result = train_mcm(
        pretrainer, loader_pt,
        n_epochs=3, lr=1e-3, warmup_epochs=1,
        checkpoint_dir=str(ckpt_dir), device=device,
    )
    print(f"  Loss: {pt_result['loss_history'][0]:.4f} → {pt_result['loss_history'][-1]:.4f}")

    # ── Step 2: Fine-tuning ───────────────────────────────────────
    print("\n[2/4] Fine-tuning (subject 1, 3 epochs)...")
    model_ft = NeuRoLLM(
        n_channels=22, patch_size=50, d_model=64,
        n_heads=2, d_ff=128, n_layers=1, n_classes=4,
    )

    ckpt_path = ckpt_dir / "best_mcm.pt"
    if ckpt_path.exists():
        model_ft = load_pretrained_encoder(model_ft, ckpt_path, device)
        print("  Loaded pre-trained weights (compatible keys)")

    train_ds = BCIIV2aDataset("data/raw/bci_iv_2a", subject=1, session="train")
    test_ds = BCIIV2aDataset("data/raw/bci_iv_2a", subject=1, session="test")

    ft_result = finetune_subject(
        model_ft,
        DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True),
        DataLoader(test_ds, batch_size=32, shuffle=False),
        n_epochs=3, lr=5e-4, patience=2, freeze_epochs=1, device=device,
    )
    print(f"  Best accuracy: {ft_result['best_accuracy']:.3f} (epoch {ft_result['best_epoch']})")

    # ── Step 3: Evaluation ────────────────────────────────────────
    print("\n[3/4] Evaluation...")
    model_ft.eval()
    all_preds, all_labels = [], []
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model_ft(bx, mode="finetune").argmax(1).numpy()
            all_preds.append(preds)
            all_labels.append(by.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred)

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Kappa:    {metrics['kappa']:.3f}")
    print(f"  Per-class: {metrics['per_class_accuracy']}")

    # ── Step 4: Visualization ─────────────────────────────────────
    print("\n[4/4] Generating figures...")

    # Confusion matrix
    cm_path = str(fig_dir / "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)
    print(f"  Saved: {cm_path}")

    # Loss curve
    loss_path = str(fig_dir / "training_loss.png")
    plot_loss_curve(ft_result["train_losses"], title="Fine-tuning Loss", save_path=loss_path)
    print(f"  Saved: {loss_path}")

    # Attention map
    x_sample = torch.randn(1, 22, 250)
    attn_maps = extract_attention_maps(model_ft, x_sample, mode="finetune")
    if attn_maps:
        imp = attention_to_channel_importance(attn_maps[-1], 22, 5)
        attn_path = str(fig_dir / "channel_attention.png")
        plot_channel_attention(imp, BCI_IV_CHANNELS_22, save_path=attn_path)
        print(f"  Saved: {attn_path}")

    print(f"\n{'=' * 60}")
    print(f"  Demo complete. Figures in: {fig_dir}")
    print(f"{'=' * 60}")

    return {
        "pretrain_loss": pt_result["loss_history"],
        "finetune_accuracy": ft_result["best_accuracy"],
        "metrics": metrics,
        "figure_dir": str(fig_dir),
    }


if __name__ == "__main__":
    run_demo()
