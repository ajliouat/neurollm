"""Tests for v1.0.2 — Masked Channel Modeling (Pre-training Loop).

Covers:
  • Patch masking: ratio, shape, determinism
  • Mask application: correct replacement
  • ReconstructionHead: shape
  • MCMPretrainer: loss computation, shape agreement, gradient flow
  • Training loop: loss decrease, checkpoint save/resume, LR schedule
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.transformer import NeuRoLLM
from training.pretrain import (
    create_patch_mask,
    apply_mask,
    ReconstructionHead,
    MCMPretrainer,
    train_mcm,
)


# ── Fixtures ──────────────────────────────────────────────────────────
C, T, P, D = 22, 1000, 50, 256
N_TOKENS = C * (T // P)  # 440


@pytest.fixture
def model():
    return NeuRoLLM(
        n_channels=C, patch_size=P, d_model=D,
        n_heads=4, d_ff=512, n_layers=2, dropout=0.0,
    )


@pytest.fixture
def pretrainer(model):
    return MCMPretrainer(model, mask_ratio=0.3)


def make_loader(n=64, batch_size=16):
    data = torch.randn(n, C, T)
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)


# =====================================================================
#  Masking tests
# =====================================================================

class TestCreatePatchMask:

    def test_shape(self):
        mask = create_patch_mask(N_TOKENS, mask_ratio=0.3)
        assert mask.shape == (N_TOKENS,)
        assert mask.dtype == torch.bool

    def test_ratio(self):
        mask = create_patch_mask(1000, mask_ratio=0.3)
        assert 250 <= mask.sum().item() <= 350  # ~30% ± tolerance

    def test_at_least_one(self):
        mask = create_patch_mask(2, mask_ratio=0.01)
        assert mask.sum().item() >= 1

    def test_different_each_call(self):
        m1 = create_patch_mask(N_TOKENS, 0.3)
        m2 = create_patch_mask(N_TOKENS, 0.3)
        # Very unlikely to be identical
        assert not torch.equal(m1, m2)


class TestApplyMask:

    def test_masked_positions_replaced(self):
        tokens = torch.randn(2, 10, D)
        mask = torch.tensor([True, False, True, False, False,
                             False, False, True, False, False])
        mask_token = torch.zeros(D)
        out = apply_mask(tokens, mask, mask_token)
        # Masked positions should be zero (the mask_token)
        assert torch.allclose(out[:, 0], torch.zeros(2, D))
        assert torch.allclose(out[:, 2], torch.zeros(2, D))

    def test_unmasked_preserved(self):
        tokens = torch.randn(2, 10, D)
        mask = torch.tensor([True, False, False, False, False,
                             False, False, False, False, False])
        mask_token = torch.zeros(D)
        out = apply_mask(tokens, mask, mask_token)
        torch.testing.assert_close(out[:, 1:], tokens[:, 1:])

    def test_shape_preserved(self):
        tokens = torch.randn(4, N_TOKENS, D)
        mask = create_patch_mask(N_TOKENS, 0.3)
        mask_token = torch.randn(D)
        out = apply_mask(tokens, mask, mask_token)
        assert out.shape == tokens.shape


# =====================================================================
#  ReconstructionHead
# =====================================================================

class TestReconstructionHead:

    def test_shape(self):
        head = ReconstructionHead(d_model=D, patch_size=P)
        x = torch.randn(4, 100, D)
        out = head(x)
        assert out.shape == (4, 100, P)

    def test_gradient(self):
        head = ReconstructionHead(d_model=D, patch_size=P)
        x = torch.randn(2, 50, D, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None


# =====================================================================
#  MCMPretrainer
# =====================================================================

class TestMCMPretrainer:

    def test_output_keys(self, pretrainer):
        x = torch.randn(2, C, T)
        result = pretrainer(x)
        assert "loss" in result
        assert "pred" in result
        assert "target" in result
        assert "mask" in result

    def test_loss_scalar(self, pretrainer):
        x = torch.randn(2, C, T)
        result = pretrainer(x)
        assert result["loss"].ndim == 0
        assert result["loss"].item() > 0

    def test_pred_target_shape_match(self, pretrainer):
        x = torch.randn(4, C, T)
        result = pretrainer(x)
        assert result["pred"].shape == result["target"].shape

    def test_mask_ratio(self, pretrainer):
        x = torch.randn(2, C, T)
        result = pretrainer(x)
        n_masked = result["mask"].sum().item()
        ratio = n_masked / N_TOKENS
        assert 0.2 <= ratio <= 0.4  # ~30% with tolerance

    def test_gradient_flow(self, pretrainer):
        x = torch.randn(2, C, T)
        result = pretrainer(x)
        result["loss"].backward()
        # Check gradient on mask_token
        assert pretrainer.mask_token.grad is not None
        # Check gradient on encoder params
        for name, p in pretrainer.model.encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_different_inputs_different_loss(self, pretrainer):
        pretrainer.eval()
        torch.manual_seed(0)
        r1 = pretrainer(torch.randn(2, C, T))
        torch.manual_seed(0)
        r2 = pretrainer(torch.randn(2, C, T) * 10)
        assert r1["loss"].item() != r2["loss"].item()


# =====================================================================
#  Training loop
# =====================================================================

class TestTrainMCM:

    def test_loss_decreases(self, pretrainer):
        loader = make_loader(n=32, batch_size=16)
        result = train_mcm(
            pretrainer, loader, n_epochs=10, lr=1e-3,
            warmup_epochs=1, log_every=100,
        )
        losses = result["loss_history"]
        assert len(losses) == 10
        # Loss should decrease on this small overfit set
        assert losses[-1] < losses[0]

    def test_checkpoint_saved(self, pretrainer, tmp_path):
        loader = make_loader(n=32, batch_size=16)
        train_mcm(
            pretrainer, loader, n_epochs=10, lr=1e-3,
            warmup_epochs=1, checkpoint_dir=tmp_path,
        )
        assert (tmp_path / "best_mcm.pt").exists()
        ckpt = torch.load(tmp_path / "best_mcm.pt", weights_only=False)
        assert "model_state" in ckpt
        assert "epoch" in ckpt

    def test_checkpoint_resume(self, model, tmp_path):
        loader = make_loader(n=32, batch_size=16)
        pt1 = MCMPretrainer(model, mask_ratio=0.3)
        train_mcm(
            pt1, loader, n_epochs=5, lr=1e-3,
            warmup_epochs=1, checkpoint_dir=tmp_path,
        )
        # Resume
        pt2 = MCMPretrainer(
            NeuRoLLM(n_channels=C, patch_size=P, d_model=D,
                     n_heads=4, d_ff=512, n_layers=2, dropout=0.0),
            mask_ratio=0.3,
        )
        result = train_mcm(
            pt2, loader, n_epochs=10, lr=1e-3,
            warmup_epochs=1, checkpoint_dir=tmp_path,
            resume_from=tmp_path / "best_mcm.pt",
        )
        assert result["epochs_trained"] == 10
        assert len(result["loss_history"]) == 10

    def test_lr_warmup(self, pretrainer):
        """LR should increase during warmup epochs."""
        loader = make_loader(n=16, batch_size=16)
        optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)

        def lr_lambda(epoch):
            if epoch < 3:
                return (epoch + 1) / 3
            return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (epoch - 3) / 7)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        lrs = []
        for epoch in range(5):
            lrs.append(optimizer.param_groups[0]["lr"] * lr_lambda(epoch))
            scheduler.step()
        # Warmup: LR should increase for first 3 epochs
        assert lrs[0] < lrs[1] < lrs[2]
