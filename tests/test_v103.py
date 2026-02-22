"""Tests for v1.0.3 — Pre-training Run + First Real Numbers.

Covers:
  • run_pretrain function: convergence, results dict, JSON output
  • Loss reduction: trained model vs random init
  • Checkpoint persistence and reuse
"""

from __future__ import annotations

import json
import pytest
import torch

from model.transformer import NeuRoLLM
from training.pretrain import MCMPretrainer
from training.run_pretrain import run_pretrain, DEFAULT_CONFIG


# ── Quick config for tests ────────────────────────────────────────────
FAST_CFG = {
    "n_channels": 19,
    "patch_size": 50,
    "d_model": 64,      # small for speed
    "n_heads": 2,
    "d_ff": 128,
    "n_layers": 2,
    "dropout": 0.0,
    "mask_ratio": 0.3,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "n_epochs": 15,
    "batch_size": 16,
}


class TestRunPretrain:

    def test_returns_dict(self, tmp_path):
        result = run_pretrain(
            config=FAST_CFG,
            data_dir=tmp_path / "fake_tuh",  # triggers synthetic fallback
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert isinstance(result, dict)
        assert "loss_history" in result
        assert "best_loss" in result
        assert "first_loss" in result
        assert "final_loss" in result

    def test_loss_converges(self, tmp_path):
        cfg = {**FAST_CFG, "n_epochs": 30, "lr": 5e-3}
        result = run_pretrain(
            config=cfg,
            data_dir=tmp_path / "fake_tuh",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert result["final_loss"] < result["first_loss"]
        # Meaningful reduction on synthetic noisy data
        reduction = 1 - result["final_loss"] / result["first_loss"]
        assert reduction > 0.02, f"Only {reduction*100:.1f}% reduction"

    def test_json_saved(self, tmp_path):
        run_pretrain(
            config=FAST_CFG,
            data_dir=tmp_path / "fake_tuh",
            checkpoint_dir=tmp_path / "ckpt",
        )
        json_path = tmp_path / "ckpt" / "pretrain_results.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "loss_history" in data
        assert len(data["loss_history"]) == FAST_CFG["n_epochs"]

    def test_checkpoint_exists(self, tmp_path):
        run_pretrain(
            config=FAST_CFG,
            data_dir=tmp_path / "fake_tuh",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert (tmp_path / "ckpt" / "best_mcm.pt").exists()

    def test_trained_vs_random(self, tmp_path):
        """Pre-trained model should have lower reconstruction loss than random."""
        cfg = {**FAST_CFG, "n_epochs": 20}
        run_pretrain(
            config=cfg,
            data_dir=tmp_path / "fake_tuh",
            checkpoint_dir=tmp_path / "ckpt",
        )

        # Load trained checkpoint
        ckpt = torch.load(tmp_path / "ckpt" / "best_mcm.pt", weights_only=False)

        trained_model = NeuRoLLM(
            n_channels=cfg["n_channels"],
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            n_layers=cfg["n_layers"],
            dropout=0.0,
        )
        trained_pt = MCMPretrainer(trained_model, mask_ratio=cfg["mask_ratio"])
        trained_pt.load_state_dict(ckpt["model_state"])
        trained_pt.eval()

        # Random init model
        random_model = NeuRoLLM(
            n_channels=cfg["n_channels"],
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            n_layers=cfg["n_layers"],
            dropout=0.0,
        )
        random_pt = MCMPretrainer(random_model, mask_ratio=cfg["mask_ratio"])
        random_pt.eval()

        # Compare on same data
        torch.manual_seed(42)
        x = torch.randn(16, cfg["n_channels"], 1000)

        with torch.no_grad():
            torch.manual_seed(99)
            trained_loss = trained_pt(x)["loss"].item()
            torch.manual_seed(99)
            random_loss = random_pt(x)["loss"].item()

        assert trained_loss < random_loss, (
            f"Trained {trained_loss:.4f} >= Random {random_loss:.4f}"
        )

    def test_epochs_count(self, tmp_path):
        result = run_pretrain(
            config=FAST_CFG,
            data_dir=tmp_path / "fake_tuh",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert result["epochs_trained"] == FAST_CFG["n_epochs"]
