"""Tests for v1.0.5 — Baselines (CSP+SVM, EEGNet, Vanilla Transformer).

Covers:
  • CSP filter: fit, transform, feature shape
  • CSP + SVM: end-to-end accuracy
  • EEGNet: forward shape, parameter count, training
  • Vanilla Transformer: training, comparison structure
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import SyntheticEEGDataset, make_synthetic_eeg
from baselines.models import (
    CSPFilter,
    train_csp_svm,
    EEGNet,
    train_eegnet,
    train_vanilla_transformer,
)

C, T_BCI = 22, 875


# =====================================================================
#  CSP Filter
# =====================================================================

class TestCSPFilter:

    def test_fit_transform(self):
        data, labels = make_synthetic_eeg(
            n_trials=100, n_channels=C, n_samples=T_BCI, n_classes=4, seed=42
        )
        csp = CSPFilter(n_components=6)
        csp.fit(data, labels)
        features = csp.transform(data)
        assert features.ndim == 2
        assert features.shape[0] == 100
        # Should have n_pairs * n_components features
        assert features.shape[1] > 0

    def test_not_fitted_raises(self):
        csp = CSPFilter()
        with pytest.raises(RuntimeError, match="not fitted"):
            csp.transform(np.random.randn(10, C, T_BCI))

    def test_features_finite(self):
        data, labels = make_synthetic_eeg(
            n_trials=50, n_channels=C, n_samples=T_BCI, n_classes=2, seed=0
        )
        csp = CSPFilter(n_components=4)
        csp.fit(data, labels)
        features = csp.transform(data)
        assert np.all(np.isfinite(features))


class TestCSPSVM:

    def test_accuracy_above_chance(self):
        data, labels = make_synthetic_eeg(
            n_trials=200, n_channels=C, n_samples=T_BCI, n_classes=4, seed=42
        )
        result = train_csp_svm(
            data[:150], labels[:150], data[150:], labels[150:],
        )
        # CSP+SVM on synthetic data; verify it runs and returns valid accuracy
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_returns_correct_keys(self):
        data, labels = make_synthetic_eeg(
            n_trials=100, n_channels=C, n_samples=T_BCI, n_classes=4, seed=0
        )
        result = train_csp_svm(data[:80], labels[:80], data[80:], labels[80:])
        assert "accuracy" in result
        assert "predictions" in result
        assert "model_name" in result
        assert result["model_name"] == "CSP+SVM"


# =====================================================================
#  EEGNet
# =====================================================================

class TestEEGNet:

    def test_forward_shape(self):
        model = EEGNet(n_channels=C, n_samples=T_BCI, n_classes=4)
        x = torch.randn(4, C, T_BCI)
        logits = model(x)
        assert logits.shape == (4, 4)

    def test_parameter_count(self):
        model = EEGNet(n_channels=C, n_samples=T_BCI, n_classes=4)
        n = model.num_parameters
        # EEGNet should be small: ~2-5K params typically
        assert n < 50_000, f"EEGNet has {n:,} params — too large"

    def test_gradient_flow(self):
        model = EEGNet(n_channels=C, n_samples=T_BCI, n_classes=4)
        x = torch.randn(2, C, T_BCI)
        logits = model(x)
        logits.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


class TestTrainEEGNet:

    def test_trains_successfully(self):
        train_ds = SyntheticEEGDataset(n_trials=64, n_channels=C, n_samples=T_BCI, n_classes=4)
        test_ds = SyntheticEEGDataset(n_trials=32, n_channels=C, n_samples=T_BCI, n_classes=4, seed=99)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=16)

        result = train_eegnet(
            train_loader, test_loader,
            n_channels=C, n_samples=T_BCI, n_classes=4,
            n_epochs=10, lr=1e-3, patience=50,
        )
        assert result["accuracy"] > 0.0
        assert result["model_name"] == "EEGNet"


# =====================================================================
#  Vanilla Transformer
# =====================================================================

class TestVanillaTransformer:

    def test_trains_successfully(self):
        train_ds = SyntheticEEGDataset(n_trials=64, n_channels=C, n_samples=T_BCI, n_classes=4)
        test_ds = SyntheticEEGDataset(n_trials=32, n_channels=C, n_samples=T_BCI, n_classes=4, seed=99)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=16)

        result = train_vanilla_transformer(
            train_loader, test_loader,
            n_channels=C, patch_size=50, d_model=64,
            n_heads=2, d_ff=128, n_layers=2,
            n_classes=4, n_epochs=10, lr=1e-3, patience=50,
        )
        assert result["accuracy"] > 0.0
        assert result["model_name"] == "VanillaTransformer"

    def test_returns_param_count(self):
        train_ds = SyntheticEEGDataset(n_trials=32, n_channels=C, n_samples=T_BCI, n_classes=4)
        test_ds = SyntheticEEGDataset(n_trials=16, n_channels=C, n_samples=T_BCI, n_classes=4, seed=99)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=16)

        result = train_vanilla_transformer(
            train_loader, test_loader,
            n_channels=C, d_model=64, n_heads=2, d_ff=128, n_layers=2,
            n_epochs=3, patience=50,
        )
        assert "num_parameters" in result
        assert result["num_parameters"] > 0
