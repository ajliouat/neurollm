"""Tests for v1.0.8 — Full benchmark suite."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from evaluation.benchmark import (
    PRETRAIN_CONFIG,
    FINETUNE_CONFIG,
    run_pretrain_step,
    evaluate_neurollm,
    evaluate_baselines,
    run_full_benchmark,
    _write_summary_csv,
)
from data.dataset import BCIIV2aDataset, TUHEEGDataset
from model.transformer import NeuRoLLM


# ── Tiny configs for fast testing ─────────────────────────────────────

TINY_PRETRAIN = {
    "n_channels": 19,
    "patch_size": 50,
    "d_model": 32,
    "n_heads": 2,
    "d_ff": 64,
    "n_layers": 1,
    "dropout": 0.1,
    "mask_ratio": 0.3,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "n_epochs": 2,
    "batch_size": 8,
}

TINY_FINETUNE = {
    "n_channels": 22,
    "patch_size": 50,
    "d_model": 32,
    "n_heads": 2,
    "d_ff": 64,
    "n_layers": 1,
    "n_classes": 4,
    "n_epochs": 3,
    "lr": 5e-4,
    "batch_size": 16,
    "patience": 2,
    "freeze_epochs": 1,
    "dropout": 0.1,
}


class TestPretainStep:
    """Test pre-training step of benchmark."""

    def test_pretrain_returns_dict(self, tmp_path):
        result = run_pretrain_step(
            config=TINY_PRETRAIN,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        assert isinstance(result, dict)
        assert "loss_history" in result
        assert "best_loss" in result
        assert "elapsed_seconds" in result

    def test_pretrain_produces_checkpoint(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        run_pretrain_step(
            config=TINY_PRETRAIN,
            checkpoint_dir=ckpt_dir,
            device="cpu",
        )
        assert (Path(ckpt_dir) / "best_mcm.pt").exists()

    def test_pretrain_loss_history_length(self, tmp_path):
        result = run_pretrain_step(
            config=TINY_PRETRAIN,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        assert len(result["loss_history"]) == TINY_PRETRAIN["n_epochs"]


class TestEvaluateNeuRoLLM:
    """Test NeuRoLLM fine-tuning evaluation."""

    def test_evaluate_single_subject(self, tmp_path):
        # First pre-train to get checkpoint
        ckpt_dir = str(tmp_path / "ckpt")
        run_pretrain_step(
            config=TINY_PRETRAIN,
            checkpoint_dir=ckpt_dir,
            device="cpu",
        )
        ckpt_path = str(tmp_path / "ckpt" / "best_mcm.pt")

        result = evaluate_neurollm(
            subjects=[1],
            pretrain_ckpt=ckpt_path,
            config=TINY_FINETUNE,
            device="cpu",
        )
        assert "mean_accuracy" in result
        assert "subject_results" in result
        assert len(result["subject_results"]) == 1
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_evaluate_without_pretrain(self):
        result = evaluate_neurollm(
            subjects=[1],
            pretrain_ckpt=None,
            config=TINY_FINETUNE,
            device="cpu",
        )
        assert "mean_accuracy" in result
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_multi_subject(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        run_pretrain_step(
            config=TINY_PRETRAIN,
            checkpoint_dir=ckpt_dir,
            device="cpu",
        )
        result = evaluate_neurollm(
            subjects=[1, 2],
            pretrain_ckpt=str(tmp_path / "ckpt" / "best_mcm.pt"),
            config=TINY_FINETUNE,
            device="cpu",
        )
        assert result["n_subjects"] == 2
        assert len(result["subject_results"]) == 2


class TestEvaluateBaselines:
    """Test baseline evaluation."""

    def test_all_baselines_returned(self):
        result = evaluate_baselines(
            subjects=[1],
            config=TINY_FINETUNE,
            device="cpu",
        )
        assert "CSP+SVM" in result
        assert "EEGNet" in result
        assert "VanillaTransformer" in result

    def test_baseline_accuracies_valid(self):
        result = evaluate_baselines(
            subjects=[1],
            config=TINY_FINETUNE,
            device="cpu",
        )
        for name, res in result.items():
            assert 0.0 <= res["mean_accuracy"] <= 1.0, f"{name} accuracy out of range"

    def test_baseline_per_subject(self):
        result = evaluate_baselines(
            subjects=[1, 2],
            config=TINY_FINETUNE,
            device="cpu",
        )
        for name, res in result.items():
            assert len(res["per_subject"]) == 2, f"{name} should have 2 subjects"


class TestCSVOutput:
    """Test CSV summary writing."""

    def test_csv_write(self, tmp_path):
        neurollm_result = {
            "mean_accuracy": 0.75,
            "subject_results": [
                {"accuracy": 0.8, "subject": 1},
                {"accuracy": 0.7, "subject": 2},
            ],
        }
        baseline_results = {
            "CSP+SVM": {
                "mean_accuracy": 0.5,
                "per_subject": [
                    {"accuracy": 0.55, "subject": 1},
                    {"accuracy": 0.45, "subject": 2},
                ],
            },
        }
        csv_path = tmp_path / "summary.csv"
        _write_summary_csv(csv_path, neurollm_result, baseline_results, [1, 2])

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["Subject", "NeuRoLLM", "CSP+SVM"]
        assert len(rows) == 4  # header + 2 subjects + mean

    def test_csv_mean_row(self, tmp_path):
        neurollm_result = {
            "mean_accuracy": 0.75,
            "subject_results": [{"accuracy": 0.75, "subject": 1}],
        }
        baseline_results = {
            "EEGNet": {
                "mean_accuracy": 0.6,
                "per_subject": [{"accuracy": 0.6, "subject": 1}],
            },
        }
        csv_path = tmp_path / "summary.csv"
        _write_summary_csv(csv_path, neurollm_result, baseline_results, [1])

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[-1][0] == "Mean"


@pytest.mark.timeout(300)
class TestFullBenchmark:
    """Integration test for full benchmark pipeline."""

    def test_full_benchmark_runs(self, tmp_path):
        result = run_full_benchmark(
            subjects=[1],
            pretrain_epochs=2,
            finetune_epochs=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            output_dir=str(tmp_path / "results"),
            device="cpu",
        )
        assert "pretrain" in result
        assert "neurollm" in result
        assert "baselines" in result

    def test_full_benchmark_saves_json(self, tmp_path):
        run_full_benchmark(
            subjects=[1],
            pretrain_epochs=2,
            finetune_epochs=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            output_dir=str(tmp_path / "results"),
            device="cpu",
        )
        json_path = tmp_path / "results" / "benchmark_results.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "neurollm" in data

    def test_full_benchmark_saves_csv(self, tmp_path):
        run_full_benchmark(
            subjects=[1],
            pretrain_epochs=2,
            finetune_epochs=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            output_dir=str(tmp_path / "results"),
            device="cpu",
        )
        csv_path = tmp_path / "results" / "benchmark_summary.csv"
        assert csv_path.exists()
