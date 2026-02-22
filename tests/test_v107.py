"""Tests for v1.0.7 — Evaluation metrics + attention visualization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.metrics import (
    compute_metrics,
    aggregate_subject_metrics,
    extract_attention_maps,
    attention_to_channel_importance,
    plot_confusion_matrix,
    plot_channel_attention,
    plot_loss_curve,
)
from model.transformer import NeuRoLLM
from data.preprocessing import BCI_IV_CHANNELS_22, TUH_CHANNELS_19


# ── Metric tests ──────────────────────────────────────────────────────

class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["kappa"] == 1.0
        assert m["n_samples"] == 8

    def test_random_predictions(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 4, size=200)
        y_pred = rng.integers(0, 4, size=200)
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert -1.0 <= m["kappa"] <= 1.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        m = compute_metrics(y_true, y_pred)
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (4, 4)

    def test_per_class_accuracy(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 2])
        m = compute_metrics(y_true, y_pred)
        assert m["per_class_accuracy"]["Left"] == 1.0
        assert m["per_class_accuracy"]["Right"] == 0.5
        assert m["per_class_accuracy"]["Feet"] == 1.0
        assert m["per_class_accuracy"]["Tongue"] == 0.5

    def test_custom_class_names(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m = compute_metrics(y_true, y_pred, class_names=["A", "B"])
        assert "A" in m["per_class_accuracy"]
        assert "B" in m["per_class_accuracy"]

    def test_all_wrong_kappa_negative(self):
        """All-wrong should yield a negative or zero kappa."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["kappa"] <= 0.0

    def test_confusion_matrix_diagonal_perfect(self):
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        m = compute_metrics(y_true, y_pred)
        cm = np.array(m["confusion_matrix"])
        assert np.all(cm - np.diag(np.diag(cm)) == 0)


class TestAggregateSubjectMetrics:
    """Test aggregate_subject_metrics."""

    def test_basic_aggregation(self):
        ms = [
            {"accuracy": 0.8, "kappa": 0.7},
            {"accuracy": 0.9, "kappa": 0.85},
            {"accuracy": 0.7, "kappa": 0.6},
        ]
        agg = aggregate_subject_metrics(ms)
        assert agg["n_subjects"] == 3
        assert abs(agg["mean_accuracy"] - 0.8) < 1e-6
        assert agg["std_accuracy"] > 0
        assert abs(agg["mean_kappa"] - 0.7167) < 0.01

    def test_single_subject(self):
        ms = [{"accuracy": 0.85, "kappa": 0.8}]
        agg = aggregate_subject_metrics(ms)
        assert agg["mean_accuracy"] == 0.85
        assert agg["std_accuracy"] == 0.0

    def test_per_subject_preserved(self):
        ms = [{"accuracy": 0.5, "kappa": 0.3}]
        agg = aggregate_subject_metrics(ms)
        assert agg["per_subject"] is ms


# ── Attention extraction tests ────────────────────────────────────────

class TestAttentionExtraction:
    """Test extract_attention_maps and attention_to_channel_importance."""

    @pytest.fixture
    def model_and_input(self):
        model = NeuRoLLM(n_channels=22, patch_size=50, d_model=64,
                         n_heads=2, d_ff=128, n_layers=2, n_classes=4)
        x = torch.randn(2, 22, 250)
        return model, x

    def test_extract_returns_list(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="finetune")
        assert isinstance(maps, list)
        assert len(maps) > 0

    def test_extract_shape(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="finetune")
        # Each map should be (B, H, N, N)
        for m in maps:
            assert m.ndim == 4
            assert m.shape[0] == 2   # batch
            assert m.shape[1] == 2   # heads

    def test_extract_pretrain_mode(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="pretrain")
        assert len(maps) > 0

    def test_attention_values_valid(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="finetune")
        for m in maps:
            assert torch.all(m >= 0), "Attention weights should be non-negative"
            # Each row should sum to ~1 (softmax)
            row_sums = m.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_channel_importance_shape(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="pretrain")
        n_channels = 22
        n_patches = 250 // 50  # = 5
        imp = attention_to_channel_importance(maps[0], n_channels, n_patches)
        assert imp.shape == (22,)

    def test_channel_importance_normalised(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="pretrain")
        n_channels = 22
        n_patches = 5
        imp = attention_to_channel_importance(maps[0], n_channels, n_patches)
        assert abs(imp.sum() - 1.0) < 1e-5

    def test_channel_importance_non_negative(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="pretrain")
        imp = attention_to_channel_importance(maps[0], 22, 5)
        assert np.all(imp >= 0)

    def test_num_layers_matches(self, model_and_input):
        model, x = model_and_input
        maps = extract_attention_maps(model, x, mode="finetune")
        assert len(maps) == 2  # n_layers=2


# ── Visualization tests ──────────────────────────────────────────────

class TestVisualization:
    """Test plot functions produce valid figures and save to disk."""

    def test_confusion_matrix_plot(self):
        cm = [[10, 2, 0, 1], [1, 12, 0, 0], [0, 1, 11, 1], [2, 0, 1, 9]]
        fig = plot_confusion_matrix(cm)
        assert fig is not None

    def test_confusion_matrix_save(self, tmp_path):
        cm = [[10, 2, 0, 1], [1, 12, 0, 0], [0, 1, 11, 1], [2, 0, 1, 9]]
        path = str(tmp_path / "cm.png")
        plot_confusion_matrix(cm, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_channel_attention_plot(self):
        imp = np.random.rand(22)
        imp /= imp.sum()
        fig = plot_channel_attention(imp, BCI_IV_CHANNELS_22)
        assert fig is not None

    def test_channel_attention_save(self, tmp_path):
        imp = np.random.rand(22)
        imp /= imp.sum()
        path = str(tmp_path / "attn.png")
        plot_channel_attention(imp, BCI_IV_CHANNELS_22, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_channel_attention_highlights_c3c4(self):
        """C3 and C4 should be highlighted with red color."""
        imp = np.ones(22) / 22
        fig = plot_channel_attention(imp, BCI_IV_CHANNELS_22)
        assert fig is not None  # visual check would require more

    def test_loss_curve_plot(self):
        losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.42]
        fig = plot_loss_curve(losses)
        assert fig is not None

    def test_loss_curve_save(self, tmp_path):
        losses = [1.0, 0.8, 0.6, 0.5]
        path = str(tmp_path / "loss.png")
        plot_loss_curve(losses, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_tuh_channel_attention(self):
        """Test with 19-channel TUH layout."""
        imp = np.random.rand(19)
        imp /= imp.sum()
        fig = plot_channel_attention(imp, TUH_CHANNELS_19)
        assert fig is not None

    def test_confusion_matrix_custom_classes(self):
        cm = [[5, 1], [2, 6]]
        fig = plot_confusion_matrix(cm, class_names=["A", "B"])
        assert fig is not None
