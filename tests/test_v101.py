"""Tests for v1.0.1 — EEG Transformer Architecture.

Covers:
  • PatchEmbedding: output shape, token count
  • LearnablePositionalEncoding: shape, broadcasting
  • EEGTransformerEncoder: shape, gradient flow
  • ClassificationHead: shape
  • NeuRoLLM full model: pretrain mode, finetune mode, parameter count,
    gradient flow, batch independence
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.transformer import (
    PatchEmbedding,
    LearnablePositionalEncoding,
    EEGTransformerEncoder,
    ClassificationHead,
    NeuRoLLM,
)


# ── Defaults ──────────────────────────────────────────────────────────
B, C, T = 4, 22, 1000  # batch, channels, time samples
P = 50                  # patch size
D = 256                 # d_model
N_PATCHES = T // P      # 20
N_TOKENS = C * N_PATCHES  # 440


# =====================================================================
#  PatchEmbedding
# =====================================================================

class TestPatchEmbedding:

    def test_output_shape(self):
        pe = PatchEmbedding(n_channels=C, patch_size=P, d_model=D)
        x = torch.randn(B, C, T)
        out = pe(x)
        assert out.shape == (B, N_TOKENS, D)

    def test_different_lengths(self):
        """Handles signals that aren't exact multiples of P."""
        pe = PatchEmbedding(n_channels=C, patch_size=P, d_model=D)
        x = torch.randn(B, C, 1023)  # 1023 // 50 = 20 patches
        out = pe(x)
        assert out.shape == (B, C * 20, D)

    def test_single_channel(self):
        pe = PatchEmbedding(n_channels=1, patch_size=P, d_model=D)
        x = torch.randn(2, 1, 500)
        out = pe(x)
        assert out.shape == (2, 10, D)

    def test_gradient_flows(self):
        pe = PatchEmbedding(n_channels=C, patch_size=P, d_model=D)
        x = torch.randn(B, C, T, requires_grad=True)
        out = pe(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# =====================================================================
#  Positional Encoding
# =====================================================================

class TestPositionalEncoding:

    def test_shape(self):
        enc = LearnablePositionalEncoding(n_channels=C, max_patches=64, d_model=D)
        pe = enc(n_patches=N_PATCHES, device=torch.device("cpu"))
        assert pe.shape == (1, N_TOKENS, D)

    def test_different_patch_count(self):
        enc = LearnablePositionalEncoding(n_channels=C, max_patches=64, d_model=D)
        pe = enc(n_patches=10, device=torch.device("cpu"))
        assert pe.shape == (1, C * 10, D)

    def test_learnable(self):
        enc = LearnablePositionalEncoding(n_channels=C, max_patches=64, d_model=D)
        assert enc.spatial.weight.requires_grad
        assert enc.temporal.weight.requires_grad

    def test_spatial_temporal_differ(self):
        """Spatial and temporal embeddings should be distinct tensors."""
        enc = LearnablePositionalEncoding(n_channels=C, max_patches=64, d_model=D)
        assert not torch.equal(enc.spatial.weight, enc.temporal.weight[:C])


# =====================================================================
#  Transformer Encoder
# =====================================================================

class TestTransformerEncoder:

    def test_shape(self):
        enc = EEGTransformerEncoder(d_model=D, n_heads=4, d_ff=512, n_layers=6)
        x = torch.randn(B, N_TOKENS, D)
        out = enc(x)
        assert out.shape == (B, N_TOKENS, D)

    def test_with_cls_token(self):
        """Shape correct when CLS token is prepended."""
        enc = EEGTransformerEncoder(d_model=D, n_heads=4, d_ff=512, n_layers=2)
        x = torch.randn(B, N_TOKENS + 1, D)  # +1 for CLS
        out = enc(x)
        assert out.shape == (B, N_TOKENS + 1, D)

    def test_gradient_flow(self):
        enc = EEGTransformerEncoder(d_model=D, n_heads=4, d_ff=512, n_layers=2)
        x = torch.randn(B, 20, D, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


# =====================================================================
#  Classification Head
# =====================================================================

class TestClassificationHead:

    def test_shape(self):
        head = ClassificationHead(d_model=D, n_classes=4)
        x = torch.randn(B, D)
        logits = head(x)
        assert logits.shape == (B, 4)

    def test_gradient_flow(self):
        head = ClassificationHead(d_model=D, n_classes=4)
        x = torch.randn(B, D, requires_grad=True)
        logits = head(x)
        logits.sum().backward()
        assert x.grad is not None


# =====================================================================
#  Full NeuRoLLM Model
# =====================================================================

class TestNeuRoLLM:

    @pytest.fixture
    def model(self):
        return NeuRoLLM(
            n_channels=C,
            patch_size=P,
            d_model=D,
            n_heads=4,
            d_ff=512,
            n_layers=6,
            n_classes=4,
            max_patches=64,
            dropout=0.0,  # deterministic for testing
        )

    def test_pretrain_mode(self, model):
        x = torch.randn(B, C, T)
        out = model(x, mode="pretrain")
        assert out.shape == (B, N_TOKENS, D)

    def test_finetune_mode(self, model):
        x = torch.randn(B, C, T)
        logits = model(x, mode="finetune")
        assert logits.shape == (B, 4)

    def test_parameter_count(self, model):
        """Model should be roughly ~10M params."""
        n = model.num_parameters
        assert 1_000_000 < n < 50_000_000, f"Got {n:,} params"

    def test_forward_backward_finetune(self, model):
        x = torch.randn(B, C, T)
        logits = model(x, mode="finetune")
        loss = logits.sum()
        loss.backward()
        # All parameters should have gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_forward_backward_pretrain(self, model):
        x = torch.randn(B, C, T)
        out = model(x, mode="pretrain")
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and "head" not in name and "cls_token" not in name:
                assert p.grad is not None, f"No grad for {name}"

    def test_batch_independence(self, model):
        """Different batch elements should produce different outputs."""
        model.eval()
        x = torch.randn(2, C, T)
        with torch.no_grad():
            logits = model(x, mode="finetune")
        assert not torch.allclose(logits[0], logits[1])

    def test_different_signal_length(self, model):
        """Works with BCI-IV length (875 samples)."""
        x = torch.randn(2, C, 875)
        logits = model(x, mode="finetune")
        n_patches = 875 // P  # 17
        assert logits.shape == (2, 4)

    def test_bci_iv_channels(self):
        """Works with standard BCI-IV 22-channel input."""
        model = NeuRoLLM(n_channels=22, patch_size=50, n_classes=4)
        x = torch.randn(2, 22, 875)
        logits = model(x, mode="finetune")
        assert logits.shape == (2, 4)

    def test_tuh_channels(self):
        """Works with TUH 19-channel input."""
        model = NeuRoLLM(n_channels=19, patch_size=50, n_classes=4)
        x = torch.randn(2, 19, 1000)
        out = model(x, mode="pretrain")
        assert out.shape == (2, 19 * 20, D)

    def test_reproducibility(self, model):
        """Same input produces same output in eval mode."""
        model.eval()
        x = torch.randn(2, C, T)
        with torch.no_grad():
            y1 = model(x, mode="finetune")
            y2 = model(x, mode="finetune")
        torch.testing.assert_close(y1, y2)
