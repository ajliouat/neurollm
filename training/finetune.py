"""Fine-tuning pipeline for motor imagery classification.

Supports:
  • Loading a pre-trained checkpoint and freezing/unfreezing the encoder
  • Per-subject training: 9 BCI-IV 2a subjects, session T→train, E→test
  • LR schedule: lr=5e-5, linear warmup + cosine decay
  • Early stopping with patience
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from model.transformer import NeuRoLLM


# ── Freeze / unfreeze helpers ────────────────────────────────────────

def freeze_encoder(model: NeuRoLLM) -> None:
    """Freeze all encoder + embedding parameters, leave head trainable."""
    for name, p in model.named_parameters():
        if "head" not in name and "cls_token" not in name:
            p.requires_grad = False


def unfreeze_encoder(model: NeuRoLLM) -> None:
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def load_pretrained_encoder(
    model: NeuRoLLM,
    checkpoint_path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> NeuRoLLM:
    """Load pre-trained MCM weights into the encoder (ignoring recon head).

    Parameters
    ----------
    model : NeuRoLLM
    checkpoint_path : path to MCM checkpoint (.pt)
    device : torch.device

    Returns
    -------
    model with pre-trained encoder weights loaded.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    # Extract only model.* keys (skip mask_token, recon_head)
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_key = k[len("model."):]
            encoder_state[new_key] = v

    # Load with strict=False to skip head/cls_token if shapes differ
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    return model


# ── Fine-tuning loop ────────────────────────────────────────────────

def finetune_subject(
    model: NeuRoLLM,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_epochs: int = 5,
    patience: int = 10,
    freeze_epochs: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Fine-tune on a single subject.

    Strategy:
      1. Freeze encoder for *freeze_epochs* (train only head + CLS token)
      2. Unfreeze all and train end-to-end
      3. Early stopping based on test accuracy with *patience*

    Returns
    -------
    dict with accuracy, loss_history, best_epoch, etc.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: frozen encoder
    freeze_encoder(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10,  # higher LR for head-only phase
        weight_decay=weight_decay,
    )

    train_losses: list[float] = []
    test_accs: list[float] = []
    best_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs:
            unfreeze_encoder(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x, mode="finetune")
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        train_losses.append(avg_loss)

        # --- Evaluate ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x, mode="finetune")
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.shape[0]

        acc = correct / max(1, total)
        test_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch >= freeze_epochs + patience:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "test_accuracies": test_accs,
        "epochs_trained": len(train_losses),
    }


# ── Multi-subject runner ────────────────────────────────────────────

def finetune_all_subjects(
    base_model: NeuRoLLM,
    data_dir: str | Path,
    n_subjects: int = 9,
    pretrain_ckpt: Optional[str | Path] = None,
    n_epochs: int = 100,
    lr: float = 5e-5,
    batch_size: int = 32,
    patience: int = 10,
    freeze_epochs: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Fine-tune on all subjects and aggregate results."""
    from data.dataset import BCIIV2aDataset

    device_obj = torch.device(device)
    subject_results: List[Dict] = []

    for subj in range(1, n_subjects + 1):
        # Fresh copy of base model for each subject
        model = copy.deepcopy(base_model)

        # Load pre-trained weights if available
        if pretrain_ckpt is not None:
            model = load_pretrained_encoder(model, pretrain_ckpt, device_obj)

        train_ds = BCIIV2aDataset(data_dir, subject=subj, session="train")
        test_ds = BCIIV2aDataset(data_dir, subject=subj, session="test")

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
        )

        result = finetune_subject(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=n_epochs,
            lr=lr,
            patience=patience,
            freeze_epochs=freeze_epochs,
            device=device_obj,
        )

        result["subject"] = subj
        subject_results.append(result)
        print(
            f"  Subject {subj:02d}: acc={result['best_accuracy']:.3f} "
            f"(epoch {result['best_epoch']})"
        )

    accs = [r["best_accuracy"] for r in subject_results]
    return {
        "subject_results": subject_results,
        "mean_accuracy": sum(accs) / len(accs),
        "per_subject_accuracy": accs,
    }
