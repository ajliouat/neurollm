"""Tests for v1.0.4 — Fine-tuning Pipeline.

Covers:
  • freeze/unfreeze encoder
  • load_pretrained_encoder
  • finetune_subject: single-subject training loop, early stopping
  • finetune_all_subjects: multi-subject aggregation
"""

from __future__ import annotations

import copy
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import SyntheticEEGDataset, BCIIV2aDataset
from model.transformer import NeuRoLLM
from training.pretrain import MCMPretrainer, train_mcm
from training.finetune import (
    freeze_encoder,
    unfreeze_encoder,
    load_pretrained_encoder,
    finetune_subject,
    finetune_all_subjects,
)


C, T_BCI, P, D = 22, 875, 50, 64  # small d_model for speed


@pytest.fixture
def small_model():
    return NeuRoLLM(
        n_channels=C, patch_size=P, d_model=D,
        n_heads=2, d_ff=128, n_layers=2,
        n_classes=4, dropout=0.0,
    )


@pytest.fixture
def train_loader():
    ds = SyntheticEEGDataset(n_trials=64, n_channels=C, n_samples=T_BCI, n_classes=4)
    return DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)


@pytest.fixture
def test_loader():
    ds = SyntheticEEGDataset(n_trials=32, n_channels=C, n_samples=T_BCI, n_classes=4, seed=99)
    return DataLoader(ds, batch_size=16, shuffle=False)


# =====================================================================
#  Freeze / Unfreeze
# =====================================================================

class TestFreezeUnfreeze:

    def test_freeze_encoder(self, small_model):
        freeze_encoder(small_model)
        frozen = [n for n, p in small_model.named_parameters() if not p.requires_grad]
        trainable = [n for n, p in small_model.named_parameters() if p.requires_grad]
        assert len(frozen) > 0
        # Head and cls_token should remain trainable
        assert any("head" in n for n in trainable)
        assert any("cls_token" in n for n in trainable)

    def test_unfreeze_all(self, small_model):
        freeze_encoder(small_model)
        unfreeze_encoder(small_model)
        for n, p in small_model.named_parameters():
            assert p.requires_grad, f"{n} still frozen"

    def test_frozen_no_grad_update(self, small_model):
        freeze_encoder(small_model)
        old_weight = small_model.patch_embed.proj.weight.clone()
        x = torch.randn(2, C, T_BCI)
        logits = small_model(x, mode="finetune")
        logits.sum().backward()
        # Frozen parameters shouldn't have gradients
        assert small_model.patch_embed.proj.weight.grad is None


# =====================================================================
#  Load pretrained encoder
# =====================================================================

class TestLoadPretrained:

    def test_loads_without_error(self, small_model, tmp_path):
        # Create a quick pretrain checkpoint
        pt_model = NeuRoLLM(
            n_channels=19, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        )
        pt = MCMPretrainer(pt_model, mask_ratio=0.3)
        ds = SyntheticEEGDataset(n_trials=32, n_channels=19, n_samples=1000)
        loader = DataLoader(ds, batch_size=16)
        train_mcm(pt, loader, n_epochs=2, lr=1e-3, warmup_epochs=1,
                   checkpoint_dir=tmp_path)

        # Load into a fresh model (same architecture, different n_channels for finetune)
        ft_model = NeuRoLLM(
            n_channels=19, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        )
        loaded = load_pretrained_encoder(ft_model, tmp_path / "best_mcm.pt")
        assert loaded is ft_model  # returns same object

    def test_weights_differ_from_random(self, tmp_path):
        pt_model = NeuRoLLM(
            n_channels=19, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        )
        pt = MCMPretrainer(pt_model, mask_ratio=0.3)
        ds = SyntheticEEGDataset(n_trials=32, n_channels=19, n_samples=1000)
        loader = DataLoader(ds, batch_size=16)
        train_mcm(pt, loader, n_epochs=3, lr=1e-3, warmup_epochs=1,
                   checkpoint_dir=tmp_path)

        random_model = NeuRoLLM(
            n_channels=19, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        )
        loaded_model = NeuRoLLM(
            n_channels=19, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        )
        load_pretrained_encoder(loaded_model, tmp_path / "best_mcm.pt")

        # Compare encoder weights
        for (n1, p1), (n2, p2) in zip(
            random_model.encoder.named_parameters(),
            loaded_model.encoder.named_parameters(),
        ):
            if not torch.equal(p1, p2):
                return  # found a difference — good!
        pytest.fail("All encoder weights identical to random init")


# =====================================================================
#  Single-subject fine-tuning
# =====================================================================

class TestFinetuneSubject:

    def test_returns_dict(self, small_model, train_loader, test_loader):
        result = finetune_subject(
            small_model, train_loader, test_loader,
            n_epochs=5, lr=1e-3, patience=50, freeze_epochs=1,
        )
        assert "best_accuracy" in result
        assert "train_losses" in result
        assert "test_accuracies" in result
        assert "best_epoch" in result

    def test_accuracy_above_chance(self, small_model, train_loader, test_loader):
        """After training, should beat random chance (25% for 4 classes)."""
        result = finetune_subject(
            small_model, train_loader, test_loader,
            n_epochs=30, lr=1e-3, patience=50, freeze_epochs=2,
        )
        assert result["best_accuracy"] > 0.25, (
            f"Accuracy {result['best_accuracy']:.3f} not above chance"
        )

    def test_loss_decreases(self, small_model, train_loader, test_loader):
        result = finetune_subject(
            small_model, train_loader, test_loader,
            n_epochs=20, lr=1e-3, patience=50, freeze_epochs=2,
        )
        losses = result["train_losses"]
        assert losses[-1] < losses[0]

    def test_early_stopping(self, small_model, train_loader, test_loader):
        """With patience=3, training should stop early on a small set."""
        result = finetune_subject(
            small_model, train_loader, test_loader,
            n_epochs=200, lr=1e-3, patience=3, freeze_epochs=2,
        )
        # Should stop well before 200 epochs
        assert result["epochs_trained"] < 200


# =====================================================================
#  Multi-subject fine-tuning
# =====================================================================

class TestFinetuneAllSubjects:

    def test_runs_all_subjects(self, tmp_path):
        model = NeuRoLLM(
            n_channels=C, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2,
            n_classes=4, dropout=0.0,
        )
        result = finetune_all_subjects(
            base_model=model,
            data_dir=tmp_path / "fake_bci",
            n_subjects=3,  # fewer for speed
            n_epochs=5,
            lr=1e-3,
            batch_size=16,
            patience=50,
            freeze_epochs=1,
        )
        assert len(result["per_subject_accuracy"]) == 3
        assert "mean_accuracy" in result
        assert result["mean_accuracy"] > 0.0

    def test_subjects_independent(self, tmp_path):
        """Each subject gets an independent model copy and trains separately."""
        model = NeuRoLLM(
            n_channels=C, patch_size=P, d_model=D,
            n_heads=2, d_ff=128, n_layers=2,
            n_classes=4, dropout=0.0,
        )
        result = finetune_all_subjects(
            base_model=model,
            data_dir=tmp_path / "fake_bci",
            n_subjects=3,
            n_epochs=10,
            lr=1e-3,
            batch_size=16,
            patience=50,
            freeze_epochs=2,
        )
        accs = result["per_subject_accuracy"]
        # All subjects should achieve meaningful accuracy
        assert all(a > 0.25 for a in accs)
        assert len(result["subject_results"]) == 3
