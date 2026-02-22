"""Frequency-Band Attention — PyTorch reference + optional Triton kernel.

Decomposes each patch into δ θ α β γ bands via FFT, applies
band-specific learnable attention biases, then recombines.

Provides:
  • freq_band_decompose() — FFT-based band power extraction
  • FrequencyBandAttention (nn.Module) — drop-in attention layer
  • freq_band_attention_naive() — pure PyTorch reference impl
  • freq_band_attention_triton() — Triton kernel (when available)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.preprocessing import FREQ_BANDS

# Try importing Triton — gracefully degrade on CPU-only systems
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ── FFT Band Decomposition ──────────────────────────────────────────

def freq_band_decompose(
    patches: Tensor,
    fs: float = 250.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tensor:
    """Decompose temporal patches into frequency-band power features.

    Parameters
    ----------
    patches : Tensor, shape (B, N, P)
        Temporal patches (P = patch_size).
    fs : float
        Sampling rate.
    bands : dict
        Frequency band definitions {name: (lo, hi)}.
        Defaults to FREQ_BANDS (δ θ α β γ).

    Returns
    -------
    band_powers : Tensor, shape (B, N, n_bands)
        Log-power in each frequency band for each patch.
    """
    if bands is None:
        bands = FREQ_BANDS

    B, N, P = patches.shape
    # FFT
    freqs = torch.fft.rfftfreq(P, d=1.0 / fs).to(patches.device)  # (P//2+1,)
    fft_mag = torch.abs(torch.fft.rfft(patches, dim=-1))  # (B, N, P//2+1)
    power = fft_mag ** 2

    band_powers = []
    for name in sorted(bands.keys()):
        lo, hi = bands[name]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            # No FFT bins fall in this band — use total power as fallback
            bp = power.mean(dim=-1, keepdim=True)
        else:
            bp = power[:, :, mask].mean(dim=-1, keepdim=True)  # (B, N, 1)
        band_powers.append(bp)

    band_powers = torch.cat(band_powers, dim=-1)  # (B, N, n_bands)
    # Log-scale for stability
    return torch.log1p(band_powers)


# ── Naive PyTorch Reference ─────────────────────────────────────────

def freq_band_attention_naive(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    band_bias: Tensor,
    patches: Tensor,
    fs: float = 250.0,
) -> Tensor:
    """Frequency-band attention — pure PyTorch reference.

    Parameters
    ----------
    Q, K, V : Tensor, shape (B, H, N, d_k)
    band_bias : Tensor, shape (H, n_bands)
        Per-head learnable bias for each frequency band.
    patches : Tensor, shape (B, N, P)
        Raw temporal patches (for FFT decomposition).
    fs : float

    Returns
    -------
    out : Tensor, shape (B, H, N, d_k)
    """
    B, H, N, d_k = Q.shape
    n_bands = band_bias.shape[1]

    # Standard scaled dot-product attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, N, N)

    # Frequency-band bias
    band_powers = freq_band_decompose(patches, fs=fs)  # (B, N, n_bands)
    # Attention bias per query-key pair: sum of band_bias * (query_power + key_power)
    # Shape: (B, 1, N, n_bands) x (H, n_bands) → (B, H, N, 1) added to scores
    bp_q = band_powers.unsqueeze(1)  # (B, 1, N, n_bands)
    bp_k = band_powers.unsqueeze(1)  # (B, 1, N, n_bands)

    # Per-query bias: sum over bands
    bias_q = (bp_q * band_bias.unsqueeze(0).unsqueeze(2)).sum(-1)  # (B, H, N)
    bias_k = (bp_k * band_bias.unsqueeze(0).unsqueeze(2)).sum(-1)  # (B, H, N)

    # Add as row + column bias
    scores = scores + bias_q.unsqueeze(-1) + bias_k.unsqueeze(-2)

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out


# ── Triton Kernel (when available) ──────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def _fba_kernel(
        scores_ptr, band_q_ptr, band_k_ptr, out_ptr,
        N: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Triton kernel: add band biases to attention scores in-place."""
        pid = tl.program_id(0)
        row = pid // N
        col = pid % N

        # Load score
        idx = row * N + col
        score = tl.load(scores_ptr + idx)

        # Load biases
        bq = tl.load(band_q_ptr + row)
        bk = tl.load(band_k_ptr + col)

        # Write modified score
        tl.store(out_ptr + idx, score + bq + bk)


def freq_band_attention_triton(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    band_bias: Tensor,
    patches: Tensor,
    fs: float = 250.0,
) -> Tensor:
    """Frequency-band attention — Triton implementation.

    Falls back to naive PyTorch when Triton is not available.
    """
    if not _HAS_TRITON or not Q.is_cuda:
        return freq_band_attention_naive(Q, K, V, band_bias, patches, fs)

    B, H, N, d_k = Q.shape

    # Standard scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Band decomposition
    band_powers = freq_band_decompose(patches, fs=fs)
    bp = band_powers.unsqueeze(1)  # (B, 1, N, n_bands)
    bias_q = (bp * band_bias.unsqueeze(0).unsqueeze(2)).sum(-1)  # (B, H, N)
    bias_k = (bp * band_bias.unsqueeze(0).unsqueeze(2)).sum(-1)

    # Apply via Triton kernel for each (b, h) slice
    out_scores = scores.clone()
    for b in range(B):
        for h in range(H):
            grid = (N * N,)
            _fba_kernel[grid](
                scores[b, h].data_ptr(),
                bias_q[b, h].data_ptr(),
                bias_k[b, h].data_ptr(),
                out_scores[b, h].data_ptr(),
                N=N, BLOCK=1,
            )

    attn = F.softmax(out_scores, dim=-1)
    return torch.matmul(attn, V)


# ── nn.Module wrapper ───────────────────────────────────────────────

class FrequencyBandAttention(nn.Module):
    """Multi-head attention with frequency-band biases.

    Parameters
    ----------
    d_model : int
    n_heads : int
    n_bands : int
        Number of frequency bands (default 5: δ θ α β γ).
    patch_size : int
        For FFT decomposition.
    fs : float
        Sampling rate.
    use_triton : bool
        If True and Triton is available and on CUDA, use Triton kernel.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_bands: int = 5,
        patch_size: int = 50,
        fs: float = 250.0,
        use_triton: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.patch_size = patch_size
        self.fs = fs
        self.use_triton = use_triton

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable band biases: one weight per head per band
        self.band_bias = nn.Parameter(torch.zeros(n_heads, n_bands))

    def forward(
        self,
        x: Tensor,
        patches: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, N, d_model) — encoded tokens
        patches : Tensor (B, N, P) — raw temporal patches for FFT

        Returns
        -------
        out : Tensor (B, N, d_model)
        """
        B, N, _ = x.shape
        H, d_k = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        K = self.W_k(x).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, d_k).transpose(1, 2)

        if self.use_triton and _HAS_TRITON and x.is_cuda:
            attn_out = freq_band_attention_triton(
                Q, K, V, self.band_bias, patches, self.fs
            )
        else:
            attn_out = freq_band_attention_naive(
                Q, K, V, self.band_bias, patches, self.fs
            )

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(attn_out)
