"""Masked Channel Modeling (MCM) pre-training.

Strategy:
  • 30% of channel-time patch tokens are randomly masked.
  • Model predicts the original patch values via MSE loss.
  • Optimizer: AdamW, lr=1e-4, cosine decay.
  • Checkpoint save/resume.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from model.transformer import NeuRoLLM, PatchEmbedding


# ── Masking ──────────────────────────────────────────────────────────

def create_patch_mask(
    n_tokens: int,
    mask_ratio: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generate a random boolean mask for token positions.

    Parameters
    ----------
    n_tokens : int
        Total number of tokens (C * n_patches).
    mask_ratio : float
        Fraction of tokens to mask.
    device : torch.device

    Returns
    -------
    mask : bool Tensor, shape (n_tokens,)
        True = masked (to be predicted).
    """
    n_mask = max(1, int(n_tokens * mask_ratio))
    perm = torch.randperm(n_tokens, device=device)
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    mask[perm[:n_mask]] = True
    return mask


def apply_mask(
    tokens: Tensor,
    mask: Tensor,
    mask_token: Tensor,
) -> Tensor:
    """Replace masked positions with a learnable mask token.

    Parameters
    ----------
    tokens : Tensor, shape (B, N, d)
    mask : bool Tensor, shape (N,)
    mask_token : Tensor, shape (d,) or (1, 1, d)

    Returns
    -------
    masked_tokens : Tensor, shape (B, N, d)
    """
    mask_token = mask_token.view(1, 1, -1)  # broadcast-ready
    out = tokens.clone()
    out[:, mask] = mask_token.expand(tokens.shape[0], mask.sum(), -1)
    return out


# ── Reconstruction head ─────────────────────────────────────────────

class ReconstructionHead(nn.Module):
    """Linear projection: d_model → patch_size (reconstruct raw patch)."""

    def __init__(self, d_model: int = 256, patch_size: int = 50) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# ── Pre-training module ─────────────────────────────────────────────

class MCMPretrainer(nn.Module):
    """Masked Channel Modeling wrapper around NeuRoLLM.

    Adds a learnable [MASK] token and a reconstruction head.
    """

    def __init__(
        self,
        model: NeuRoLLM,
        mask_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        d = model.d_model
        p = model.patch_size

        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.recon_head = ReconstructionHead(d, p)

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, T)

        Returns
        -------
        dict with keys:
            ``"loss"``   — scalar MSE on masked patches
            ``"pred"``   — (B, n_masked, P) predicted patches
            ``"target"`` — (B, n_masked, P) original patches
            ``"mask"``   — bool (N,) mask
        """
        B, C, T = x.shape
        P = self.model.patch_size
        n_patches = T // P

        # Get raw patches (before embedding) for targets
        truncated = x[:, :, : n_patches * P]
        raw_patches = truncated.reshape(B, C, n_patches, P)
        raw_patches = raw_patches.reshape(B, C * n_patches, P)  # (B, N, P)

        # Embed
        tokens = self.model.patch_embed(x)  # (B, N, d)
        N = tokens.shape[1]

        # Add positional encoding
        pe = self.model.pos_enc(n_patches, x.device)
        tokens = tokens + pe

        # Mask
        mask = create_patch_mask(N, self.mask_ratio, x.device)
        masked_tokens = apply_mask(tokens, mask, self.mask_token)

        # Encode
        encoded = self.model.encoder(masked_tokens)  # (B, N, d)

        # Reconstruct only masked positions
        masked_encoded = encoded[:, mask]  # (B, n_masked, d)
        pred = self.recon_head(masked_encoded)  # (B, n_masked, P)
        target = raw_patches[:, mask]  # (B, n_masked, P)

        loss = F.mse_loss(pred, target)

        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "mask": mask,
        }


# ── Training loop ───────────────────────────────────────────────────

def train_mcm(
    pretrainer: MCMPretrainer,
    dataloader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 5,
    checkpoint_dir: Optional[str | Path] = None,
    resume_from: Optional[str | Path] = None,
    device: torch.device = torch.device("cpu"),
    log_every: int = 10,
) -> Dict[str, Any]:
    """Train MCM pre-training loop.

    Parameters
    ----------
    pretrainer : MCMPretrainer
    dataloader : DataLoader yielding (B, C, T) tensors.
    n_epochs : int
    lr : float
    weight_decay : float
    warmup_epochs : int
    checkpoint_dir : optional path for saving checkpoints.
    resume_from : optional checkpoint path to resume.
    device : torch.device
    log_every : int — log every N steps.

    Returns
    -------
    dict:
        ``"loss_history"`` — list of per-epoch average losses
        ``"best_loss"`` — best validation loss seen
        ``"epochs_trained"`` — total epochs completed
    """
    pretrainer = pretrainer.to(device)
    pretrainer.train()

    optimizer = torch.optim.AdamW(
        pretrainer.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Cosine annealing with warmup
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    loss_history: list[float] = []
    best_loss = float("inf")

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        pretrainer.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        loss_history = ckpt.get("loss_history", [])
        best_loss = ckpt.get("best_loss", float("inf"))

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # some datasets return (data, labels)
            batch = batch.to(device)

            result = pretrainer(batch)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": pretrainer.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "loss_history": loss_history,
                        "best_loss": best_loss,
                    },
                    checkpoint_dir / "best_mcm.pt",
                )

        if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": pretrainer.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss_history": loss_history,
                    "best_loss": best_loss,
                },
                checkpoint_dir / f"mcm_epoch{epoch:04d}.pt",
            )

    return {
        "loss_history": loss_history,
        "best_loss": best_loss,
        "epochs_trained": len(loss_history),
    }
