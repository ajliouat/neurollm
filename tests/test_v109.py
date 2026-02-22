"""Tests for v1.0.9 — Integration, demo, polish."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from model.transformer import NeuRoLLM
from data.preprocessing import BCI_IV_CHANNELS_22, TUH_CHANNELS_19
from data.dataset import SyntheticEEGDataset, BCIIV2aDataset, TUHEEGDataset
from evaluation.metrics import (
    compute_metrics,
    extract_attention_maps,
    attention_to_channel_importance,
)


# ── Integration tests ─────────────────────────────────────────────────

class TestEndToEndSmoke:
    """Smoke tests: build model, forward, extract attention, compute metrics."""

    def test_model_forward_pretrain(self):
        model = NeuRoLLM(n_channels=22, patch_size=50, d_model=64,
                         n_heads=2, d_ff=128, n_layers=2)
        x = torch.randn(2, 22, 250)
        out = model(x, mode="pretrain")
        assert out.shape == (2, 22 * 5, 64)

    def test_model_forward_finetune(self):
        model = NeuRoLLM(n_channels=22, patch_size=50, d_model=64,
                         n_heads=2, d_ff=128, n_layers=2, n_classes=4)
        x = torch.randn(2, 22, 250)
        out = model(x, mode="finetune")
        assert out.shape == (2, 4)

    def test_attention_extraction_and_importance(self):
        model = NeuRoLLM(n_channels=22, patch_size=50, d_model=64,
                         n_heads=2, d_ff=128, n_layers=2, n_classes=4)
        x = torch.randn(1, 22, 250)
        maps = extract_attention_maps(model, x, mode="finetune")
        assert len(maps) == 2
        imp = attention_to_channel_importance(maps[-1], 22, 5)
        assert imp.shape == (22,)
        assert abs(imp.sum() - 1.0) < 1e-5

    def test_metrics_from_model_output(self):
        model = NeuRoLLM(n_channels=22, patch_size=50, d_model=64,
                         n_heads=2, d_ff=128, n_layers=2, n_classes=4)
        ds = BCIIV2aDataset("data/raw/bci_iv_2a", subject=1, session="test")
        loader = torch.utils.data.DataLoader(ds, batch_size=32)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for bx, by in loader:
                p = model(bx, mode="finetune").argmax(1).numpy()
                preds.append(p)
                labels.append(by.numpy())

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(labels)
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert "kappa" in m
        assert "confusion_matrix" in m


# ── Module import tests ───────────────────────────────────────────────

class TestModuleImports:
    """Verify all package modules import cleanly."""

    @pytest.mark.parametrize("module", [
        "data.preprocessing",
        "data.dataset",
        "data.download",
        "model.transformer",
        "kernels.freq_band_attention",
        "training.pretrain",
        "training.run_pretrain",
        "training.finetune",
        "evaluation.metrics",
        "evaluation.benchmark",
        "evaluation.demo",
        "baselines.models",
    ])
    def test_import(self, module):
        importlib.import_module(module)


# ── Demo test ─────────────────────────────────────────────────────────

class TestDemo:
    """Test the demo script runs end-to-end."""

    @pytest.mark.timeout(180)
    def test_demo_runs_and_produces_figures(self):
        from evaluation.demo import run_demo
        result = run_demo()
        assert "pretrain_loss" in result
        assert "finetune_accuracy" in result
        assert "metrics" in result
        assert 0.0 <= result["finetune_accuracy"] <= 1.0
        fig_dir = Path(result["figure_dir"])
        assert (fig_dir / "confusion_matrix.png").exists()
        assert (fig_dir / "training_loss.png").exists()


# ── README and docs existence ─────────────────────────────────────────

class TestDocsExist:
    """Verify documentation files exist and are non-empty."""

    @pytest.mark.parametrize("filename", [
        "README.md",
        "ROADMAP.md",
        "DEVELOPMENT_LOG.md",
        "LICENSE",
        "pyproject.toml",
        "Dockerfile",
    ])
    def test_file_exists(self, filename):
        path = Path(__file__).parent.parent / filename
        assert path.exists(), f"{filename} missing"
        assert path.stat().st_size > 0, f"{filename} is empty"
