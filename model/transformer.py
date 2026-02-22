"""EEG Transformer — patch embedding, positional encoding, encoder stack.

Architecture (from PROJECT_SPEC):
  • Patch embedding: channel-wise temporal patches, P=50 → d_model=256
  • Positional encoding: learnable spatial (channel) + temporal (patch pos)
  • Transformer encoder: 6 layers, 4 heads, d_model=256, d_ff=512
  • CLS token → MLP classification head (4 classes)
  • ~10 M parameters
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Patch Embedding ──────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Convert raw EEG to channel-time patch tokens.

    Input:  (B, C, T)  — C channels, T time samples
    Output: (B, N, d_model), where N = C * (T // P)

    Each token corresponds to one channel at one temporal patch.
    """

    def __init__(
        self,
        n_channels: int = 22,
        patch_size: int = 50,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.d_model = d_model

        # Linear projection: patch_size → d_model
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, T)

        Returns
        -------
        tokens : Tensor, shape (B, N, d_model)
            N = C * n_patches, where n_patches = T // patch_size
        """
        B, C, T = x.shape
        n_patches = T // self.patch_size
        # Reshape to (B, C, n_patches, P)
        x = x[:, :, : n_patches * self.patch_size]
        x = x.reshape(B, C, n_patches, self.patch_size)
        # Merge channel and patch dims → (B, C * n_patches, P)
        x = x.reshape(B, C * n_patches, self.patch_size)
        # Project
        tokens = self.proj(x)  # (B, N, d_model)
        return tokens


# ── Positional Encoding ─────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable spatial (channel) + temporal (patch position) embeddings.

    For a token at channel *c* and patch position *p*:
        PE[c, p] = spatial_emb[c] + temporal_emb[p]
    """

    def __init__(
        self,
        n_channels: int = 22,
        max_patches: int = 64,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.spatial = nn.Embedding(n_channels, d_model)
        self.temporal = nn.Embedding(max_patches, d_model)

    def forward(self, n_patches: int, device: torch.device) -> Tensor:
        """Return positional embeddings for C * n_patches tokens.

        Returns
        -------
        pe : Tensor, shape (1, C * n_patches, d_model)
        """
        C = self.n_channels
        ch_ids = torch.arange(C, device=device)          # (C,)
        pat_ids = torch.arange(n_patches, device=device)  # (P,)

        s = self.spatial(ch_ids)   # (C, d)
        t = self.temporal(pat_ids) # (P, d)

        # Broadcast: (C, 1, d) + (1, P, d) → (C, P, d)
        pe = s.unsqueeze(1) + t.unsqueeze(0)
        pe = pe.reshape(1, C * n_patches, -1)  # (1, N, d)
        return pe


# ── Transformer Encoder ─────────────────────────────────────────────

class EEGTransformerEncoder(nn.Module):
    """Standard Transformer encoder with pre-norm (LayerNorm → Attention → FFN).

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    n_layers : int
        Number of encoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, N, d_model)
        src_key_padding_mask : optional bool Tensor, shape (B, N)

        Returns
        -------
        encoded : Tensor, shape (B, N, d_model)
        """
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(out)


# ── Classification Head ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """MLP head: CLS token → hidden → n_classes."""

    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, cls_token: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cls_token : Tensor, shape (B, d_model)

        Returns
        -------
        logits : Tensor, shape (B, n_classes)
        """
        return self.mlp(cls_token)


# ── Full Model ───────────────────────────────────────────────────────

class NeuRoLLM(nn.Module):
    """Foundation-model EEG Transformer.

    Modes:
      • ``mode="pretrain"``  — returns encoded tokens (no CLS, no head)
      • ``mode="finetune"``  — prepends CLS token, returns logits

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    patch_size : int
        Temporal patch size in samples.
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads.
    d_ff : int
        Feed-forward dimension.
    n_layers : int
        Encoder layers.
    n_classes : int
        Number of output classes (fine-tune mode).
    max_patches : int
        Maximum number of temporal patches per channel.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_channels: int = 22,
        patch_size: int = 50,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 6,
        n_classes: int = 4,
        max_patches: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.pos_enc = LearnablePositionalEncoding(
            n_channels, max_patches, d_model
        )
        self.encoder = EEGTransformerEncoder(
            d_model, n_heads, d_ff, n_layers, dropout
        )

        # CLS token (used only in finetune mode)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = ClassificationHead(d_model, n_classes, dropout)

    def forward(
        self,
        x: Tensor,
        mode: str = "finetune",
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, T)
        mode : str
            ``"pretrain"`` → (B, N, d_model)  encoded tokens
            ``"finetune"``  → (B, n_classes)    class logits

        Returns
        -------
        output : Tensor
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)  # (B, N, d)
        N = tokens.shape[1]
        n_patches = N // self.n_channels

        # Add positional encoding
        pe = self.pos_enc(n_patches, x.device)  # (1, N, d)
        tokens = tokens + pe

        if mode == "pretrain":
            encoded = self.encoder(tokens)
            return encoded  # (B, N, d)

        # Finetune: prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, d)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]  # (B, d)
        logits = self.head(cls_out)  # (B, n_classes)
        return logits

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
