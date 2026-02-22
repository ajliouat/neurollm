"""Tests for v1.0.0 — Scaffold + EEG Data Pipeline.

Covers:
  • Preprocessing: bandpass, notch, z-score, epoching, artifact rejection
  • Datasets: synthetic generator, BCIIV2aDataset, TUHEEGDataset
  • Pipeline composers: preprocess_tuh, preprocess_bci_iv
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.preprocessing import (
    TUH_CHANNELS_19,
    BCI_IV_CHANNELS_22,
    FREQ_BANDS,
    bandpass_filter,
    notch_filter,
    zscore_per_channel,
    epoch_signal,
    reject_epochs,
    preprocess_tuh,
    preprocess_bci_iv,
)
from data.dataset import (
    make_synthetic_eeg,
    SyntheticEEGDataset,
    BCIIV2aDataset,
    TUHEEGDataset,
)


# ── Constants ─────────────────────────────────────────────────────────

FS = 250.0
N_TUH_CH = 19
N_BCI_CH = 22


# =====================================================================
#  Preprocessing tests
# =====================================================================

class TestBandpassFilter:
    """Bandpass filter shape and energy checks."""

    def test_shape_preserved(self):
        x = np.random.randn(N_BCI_CH, 1000).astype(np.float32)
        y = bandpass_filter(x, low=4.0, high=38.0, fs=FS)
        assert y.shape == x.shape

    def test_dtype_preserved(self):
        x = np.random.randn(5, 500).astype(np.float32)
        y = bandpass_filter(x, low=1.0, high=40.0, fs=FS)
        assert y.dtype == np.float32

    def test_removes_dc(self):
        """A constant signal (DC) should be attenuated."""
        x = np.ones((1, 1000), dtype=np.float64) * 5.0
        y = bandpass_filter(x, low=0.5, high=45.0, fs=FS)
        assert np.abs(y.mean()) < 0.5  # most DC removed

    def test_passband_energy(self):
        """A 10 Hz sine should be preserved by a 4–38 Hz bandpass."""
        t = np.arange(1000) / FS
        x = np.sin(2 * np.pi * 10 * t).reshape(1, -1).astype(np.float64)
        y = bandpass_filter(x, low=4.0, high=38.0, fs=FS)
        # Energy should be at least 80 % of original
        assert np.sum(y ** 2) > 0.8 * np.sum(x ** 2)

    def test_stopband_attenuation(self):
        """A 1 Hz sine should be heavily attenuated by a 4–38 Hz bandpass."""
        t = np.arange(2000) / FS
        x = np.sin(2 * np.pi * 1.0 * t).reshape(1, -1).astype(np.float64)
        y = bandpass_filter(x, low=4.0, high=38.0, fs=FS)
        assert np.sum(y ** 2) < 0.1 * np.sum(x ** 2)

    def test_batch_shape(self):
        """Works with (B, C, T) input."""
        x = np.random.randn(4, 22, 1000).astype(np.float32)
        y = bandpass_filter(x, low=4.0, high=38.0, fs=FS)
        assert y.shape == (4, 22, 1000)


class TestNotchFilter:
    """Notch filter tests."""

    def test_shape_preserved(self):
        x = np.random.randn(N_BCI_CH, 1000).astype(np.float32)
        y = notch_filter(x, freq=50.0, fs=FS)
        assert y.shape == x.shape

    def test_attenuates_target(self):
        """50 Hz component should be attenuated."""
        t = np.arange(2000) / FS
        x = np.sin(2 * np.pi * 50 * t).reshape(1, -1).astype(np.float64)
        y = notch_filter(x, freq=50.0, fs=FS)
        assert np.sum(y ** 2) < 0.2 * np.sum(x ** 2)

    def test_preserves_other_freqs(self):
        """10 Hz component should be preserved."""
        t = np.arange(2000) / FS
        x = np.sin(2 * np.pi * 10 * t).reshape(1, -1).astype(np.float64)
        y = notch_filter(x, freq=50.0, fs=FS)
        assert np.sum(y ** 2) > 0.9 * np.sum(x ** 2)


class TestZscore:
    """Z-score normalisation tests."""

    def test_zero_mean(self):
        x = np.random.randn(5, 500).astype(np.float64) * 10 + 3
        y = zscore_per_channel(x)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)

    def test_unit_var(self):
        x = np.random.randn(5, 500).astype(np.float64) * 10 + 3
        y = zscore_per_channel(x)
        assert np.allclose(y.std(axis=-1), 1.0, atol=0.05)

    def test_shape(self):
        x = np.random.randn(4, 22, 1000).astype(np.float32)
        y = zscore_per_channel(x)
        assert y.shape == x.shape

    def test_dtype(self):
        x = np.random.randn(3, 100).astype(np.float32)
        y = zscore_per_channel(x)
        assert y.dtype == np.float32


class TestEpoching:
    """Signal epoching tests."""

    def test_basic_epoching(self):
        C, T = 19, 5000
        x = np.random.randn(C, T).astype(np.float32)
        epochs = epoch_signal(x, window_samples=1000, overlap=0.5)
        # step = 500, so (5000-1000)/500 + 1 = 9 epochs
        assert epochs.shape == (9, C, 1000)

    def test_no_overlap(self):
        C, T = 5, 3000
        x = np.random.randn(C, T).astype(np.float32)
        epochs = epoch_signal(x, window_samples=1000, overlap=0.0)
        assert epochs.shape == (3, C, 1000)

    def test_short_signal_raises(self):
        x = np.random.randn(5, 100).astype(np.float32)
        with pytest.raises(ValueError, match="too short"):
            epoch_signal(x, window_samples=1000)

    def test_content_correctness(self):
        """First epoch should match original signal slice."""
        x = np.arange(20).reshape(1, 20).astype(np.float64)
        epochs = epoch_signal(x, window_samples=10, overlap=0.0)
        np.testing.assert_array_equal(epochs[0, 0], np.arange(10))
        np.testing.assert_array_equal(epochs[1, 0], np.arange(10, 20))


class TestArtifactRejection:
    """Epoch artifact rejection tests."""

    def test_clean_epochs_kept(self):
        epochs = np.random.randn(10, 5, 100).astype(np.float64) * 10
        clean = reject_epochs(epochs, threshold_uv=200.0)
        assert clean.shape[0] == 10  # all should survive

    def test_bad_epochs_removed(self):
        epochs = np.random.randn(10, 5, 100).astype(np.float64) * 10
        # Inject a huge artifact in epoch 3
        epochs[3, 0, 50] = 500.0
        clean = reject_epochs(epochs, threshold_uv=100.0)
        assert clean.shape[0] < 10

    def test_all_bad(self):
        epochs = np.ones((5, 2, 100), dtype=np.float64) * 1000.0
        epochs[:, :, 0] = -1000.0
        clean = reject_epochs(epochs, threshold_uv=100.0)
        assert clean.shape[0] == 0


# =====================================================================
#  Pipeline composer tests
# =====================================================================

class TestPreprocessTUH:
    """End-to-end TUH pipeline."""

    def test_output_shape(self):
        raw = np.random.randn(N_TUH_CH, 5000).astype(np.float64) * 20
        epochs = preprocess_tuh(raw, fs=FS)
        assert epochs.ndim == 3
        assert epochs.shape[1] == N_TUH_CH
        assert epochs.shape[2] == 1000

    def test_returns_float(self):
        raw = np.random.randn(N_TUH_CH, 5000).astype(np.float64) * 20
        epochs = preprocess_tuh(raw, fs=FS)
        assert epochs.dtype in (np.float32, np.float64)


class TestPreprocessBCIIV:
    """End-to-end BCI-IV pipeline."""

    def test_output_shape(self):
        # Trial with cue onset at t=0, need [0.5s, 4.0s] = samples [125, 1000)
        trial = np.random.randn(N_BCI_CH, 1200).astype(np.float64) * 20
        out = preprocess_bci_iv(trial, fs=FS)
        assert out.shape == (N_BCI_CH, 875)

    def test_normalised(self):
        trial = np.random.randn(N_BCI_CH, 1200).astype(np.float64) * 50
        out = preprocess_bci_iv(trial, fs=FS)
        # Should be approximately zero-mean per channel
        assert np.allclose(out.mean(axis=-1), 0.0, atol=0.1)


# =====================================================================
#  Dataset tests
# =====================================================================

class TestSyntheticGenerator:
    """Synthetic EEG generator tests."""

    def test_shapes(self):
        data, labels = make_synthetic_eeg(
            n_trials=50, n_channels=22, n_samples=1000, n_classes=4
        )
        assert data.shape == (50, 22, 1000)
        assert labels.shape == (50,)

    def test_dtypes(self):
        data, labels = make_synthetic_eeg()
        assert data.dtype == np.float32
        assert labels.dtype == np.int64

    def test_label_range(self):
        _, labels = make_synthetic_eeg(n_classes=4)
        assert labels.min() >= 0
        assert labels.max() <= 3

    def test_reproducibility(self):
        d1, l1 = make_synthetic_eeg(seed=123)
        d2, l2 = make_synthetic_eeg(seed=123)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds(self):
        d1, _ = make_synthetic_eeg(seed=0)
        d2, _ = make_synthetic_eeg(seed=1)
        assert not np.allclose(d1, d2)


class TestSyntheticDataset:
    """SyntheticEEGDataset torch wrapper tests."""

    def test_len(self):
        ds = SyntheticEEGDataset(n_trials=100)
        assert len(ds) == 100

    def test_getitem(self):
        ds = SyntheticEEGDataset(n_trials=50, n_channels=22, n_samples=875)
        x, y = ds[0]
        assert x.shape == (22, 875)
        assert y.shape == ()
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_dataloader(self):
        ds = SyntheticEEGDataset(n_trials=32)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (8, 22, 1000)
        assert batch_y.shape == (8,)


class TestBCIIV2aDataset:
    """BCI-IV Dataset 2a — falls back to synthetic."""

    def test_synthetic_fallback(self, tmp_path):
        ds = BCIIV2aDataset(data_dir=tmp_path, subject=1, session="train")
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (22, 875)
        assert y.dtype == torch.int64

    def test_all_subjects(self, tmp_path):
        for subj in range(1, 10):
            ds = BCIIV2aDataset(data_dir=tmp_path, subject=subj, session="train")
            assert len(ds) > 0

    def test_dataloader(self, tmp_path):
        ds = BCIIV2aDataset(data_dir=tmp_path, subject=1, session="train")
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.ndim == 3
        assert batch_y.ndim == 1


class TestTUHEEGDataset:
    """TUH EEG Dataset — falls back to synthetic."""

    def test_synthetic_fallback(self, tmp_path):
        ds = TUHEEGDataset(data_dir=tmp_path)
        assert len(ds) > 0
        x = ds[0]
        assert x.shape == (19, 1000)
        assert x.dtype == torch.float32

    def test_dataloader(self, tmp_path):
        ds = TUHEEGDataset(data_dir=tmp_path)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        assert batch.shape[0] == 32
        assert batch.shape[1] == 19
        assert batch.shape[2] == 1000


# =====================================================================
#  Channel / constant sanity checks
# =====================================================================

class TestConstants:
    """Verify channel lists and frequency band definitions."""

    def test_tuh_channels(self):
        assert len(TUH_CHANNELS_19) == 19
        assert "C3" in TUH_CHANNELS_19
        assert "C4" in TUH_CHANNELS_19

    def test_bci_channels(self):
        assert len(BCI_IV_CHANNELS_22) == 22
        assert "C3" in BCI_IV_CHANNELS_22
        assert "C4" in BCI_IV_CHANNELS_22

    def test_freq_bands(self):
        assert "delta" in FREQ_BANDS
        assert "alpha" in FREQ_BANDS
        assert "gamma" in FREQ_BANDS
        for name, (lo, hi) in FREQ_BANDS.items():
            assert hi > lo > 0
