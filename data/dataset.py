"""PyTorch Dataset classes and synthetic EEG generators.

Provides:
  • SyntheticEEGDataset — configurable synthetic EEG for unit testing.
  • BCIIV2aDataset      — BCI Competition IV Dataset 2a (22-ch, 4-class MI).
  • TUHEEGDataset       — Temple University Hospital EEG Corpus (19-ch MCM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from data.preprocessing import (
    TUH_CHANNELS_19,
    BCI_IV_CHANNELS_22,
    preprocess_bci_iv,
    preprocess_tuh,
)


# ── Synthetic generators ─────────────────────────────────────────────

def make_synthetic_eeg(
    n_trials: int = 200,
    n_channels: int = 22,
    n_samples: int = 1000,
    n_classes: int = 4,
    fs: float = 250.0,
    seed: int = 42,
) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Generate synthetic multi-channel EEG with class-dependent spectral profiles.

    Each class emphasises a different frequency band so that a good model
    can learn to separate them.

    Parameters
    ----------
    n_trials : int
        Number of trials to generate.
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Temporal length per trial (samples).
    n_classes : int
        Number of classes.
    fs : float
        Sampling rate in Hz.
    seed : int
        Random seed.

    Returns
    -------
    data : float32 array, shape (n_trials, n_channels, n_samples)
    labels : int64 array, shape (n_trials,)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs  # time vector

    # Class-specific dominant frequencies (Hz)
    class_freqs = [8.0, 12.0, 20.0, 30.0]  # mu, alpha, beta, gamma

    data = np.zeros((n_trials, n_channels, n_samples), dtype=np.float32)
    labels = rng.randint(0, n_classes, size=n_trials).astype(np.int64)

    for i in range(n_trials):
        cls = labels[i]
        freq = class_freqs[cls % len(class_freqs)]
        for ch in range(n_channels):
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = 1.0 + 0.3 * rng.randn()
            # Dominant oscillation + broadband noise
            sig = amplitude * np.sin(2 * np.pi * freq * t + phase)
            sig += 0.5 * rng.randn(n_samples)
            # Add channel-specific slight freq shift
            sig += 0.2 * np.sin(2 * np.pi * (freq + ch * 0.1) * t)
            data[i, ch] = sig.astype(np.float32)

    return data, labels


class SyntheticEEGDataset(Dataset):
    """PyTorch Dataset wrapping :func:`make_synthetic_eeg`.

    Parameters
    ----------
    n_trials, n_channels, n_samples, n_classes, fs, seed :
        Forwarded to :func:`make_synthetic_eeg`.
    """

    def __init__(
        self,
        n_trials: int = 200,
        n_channels: int = 22,
        n_samples: int = 1000,
        n_classes: int = 4,
        fs: float = 250.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        data, labels = make_synthetic_eeg(
            n_trials=n_trials,
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            fs=fs,
            seed=seed,
        )
        self.data = torch.from_numpy(data)        # (N, C, T)
        self.labels = torch.from_numpy(labels)     # (N,)
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


# ── BCI Competition IV Dataset 2a ────────────────────────────────────

class BCIIV2aDataset(Dataset):
    """BCI Competition IV Dataset 2a (motor imagery, 4-class).

    Expects pre-processed ``.npz`` files on disk with keys:
      • ``data``   — float32 (N, 22, 875)
      • ``labels`` — int64   (N,)

    If *data_dir* does not exist or the file is missing, falls back to
    synthetic data so that tests always work.

    Parameters
    ----------
    data_dir : path-like
        Root of pre-processed data.
    subject : int
        Subject number 1–9.
    session : str
        ``"train"`` (session T) or ``"test"`` (session E).
    """

    N_CHANNELS = 22
    N_SAMPLES = 875
    N_CLASSES = 4
    FS = 250.0

    def __init__(
        self,
        data_dir: str | Path,
        subject: int = 1,
        session: str = "train",
    ) -> None:
        super().__init__()
        path = Path(data_dir) / f"A{subject:02d}{session[0].upper()}.npz"
        if path.exists():
            blob = np.load(path)
            self.data = torch.from_numpy(blob["data"].astype(np.float32))
            self.labels = torch.from_numpy(blob["labels"].astype(np.int64))
        else:
            # Synthetic fallback
            data, labels = make_synthetic_eeg(
                n_trials=144 if session == "train" else 144,
                n_channels=self.N_CHANNELS,
                n_samples=self.N_SAMPLES,
                n_classes=self.N_CLASSES,
                fs=self.FS,
                seed=subject,
            )
            self.data = torch.from_numpy(data)
            self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


# ── TUH EEG Corpus ──────────────────────────────────────────────────

class TUHEEGDataset(Dataset):
    """Temple University Hospital EEG Corpus for pre-training (MCM).

    Expects a directory of ``.npz`` files, each containing:
      • ``epochs`` — float32 (N_epochs, 19, 1000)

    Falls back to synthetic data if the directory is empty or missing.

    Parameters
    ----------
    data_dir : path-like
        Root of pre-processed TUH epochs.
    max_files : int or None
        Cap on number of files to load (for dev iterations).
    """

    N_CHANNELS = 19
    WINDOW_SAMPLES = 1000  # 4 s × 250 Hz

    def __init__(
        self,
        data_dir: str | Path,
        max_files: Optional[int] = None,
    ) -> None:
        super().__init__()
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.npz")) if data_dir.exists() else []
        if max_files is not None:
            files = files[:max_files]

        if files:
            arrays = []
            for f in files:
                blob = np.load(f)
                arrays.append(blob["epochs"].astype(np.float32))
            all_data = np.concatenate(arrays, axis=0)
            self.data = torch.from_numpy(all_data)
        else:
            # Synthetic fallback — unlabelled epochs for pre-training
            data, _ = make_synthetic_eeg(
                n_trials=500,
                n_channels=self.N_CHANNELS,
                n_samples=self.WINDOW_SAMPLES,
                n_classes=1,
                seed=0,
            )
            self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single epoch — no label (self-supervised)."""
        return self.data[idx]
