"""EEG preprocessing utilities.

Supports two pipelines:
  • TUH EEG Corpus — 19-channel 10-20 montage, resample 250 Hz,
    bandpass 0.5–45 Hz, z-score per channel/session, 4 s windows
    50 % overlap, artifact rejection ±100 μV.
  • BCI Competition IV 2a — 22 EEG channels (3 EOG dropped),
    bandpass 4–38 Hz, extract [0.5 s, 4.0 s] post-cue (875 samples),
    z-score per channel/trial.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


# ── 10-20 standard channel names ──────────────────────────────────────
TUH_CHANNELS_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

BCI_IV_CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

# ── Frequency bands (Hz) ─────────────────────────────────────────────
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ── Filter helpers ────────────────────────────────────────────────────

def bandpass_filter(
    signal: NDArray[np.floating],
    low: float,
    high: float,
    fs: float = 250.0,
    order: int = 4,
) -> NDArray[np.floating]:
    """Zero-phase Butterworth bandpass.

    Parameters
    ----------
    signal : array, shape (..., T)
        Last axis is time.
    low, high : float
        Cut-off frequencies in Hz.
    fs : float
        Sampling rate.
    order : int
        Filter order.

    Returns
    -------
    filtered : same shape as *signal*.
    """
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(signal.dtype)


def notch_filter(
    signal: NDArray[np.floating],
    freq: float = 50.0,
    fs: float = 250.0,
    quality: float = 30.0,
) -> NDArray[np.floating]:
    """Apply a notch (band-stop) filter to remove power-line noise.

    Parameters
    ----------
    signal : array, shape (..., T)
    freq : float
        Frequency to notch out (50 or 60 Hz).
    fs : float
        Sampling rate.
    quality : float
        Quality factor of notch filter.

    Returns
    -------
    filtered : same shape as *signal*.
    """
    b, a = iirnotch(freq, quality, fs=fs)
    return filtfilt(b, a, signal, axis=-1).astype(signal.dtype)


# ── Normalisation ────────────────────────────────────────────────────

def zscore_per_channel(
    signal: NDArray[np.floating],
    eps: float = 1e-8,
) -> NDArray[np.floating]:
    """Z-score normalise each channel independently.

    Parameters
    ----------
    signal : array, shape (C, T) or (B, C, T)
    eps : float
        Numerical stability constant.

    Returns
    -------
    normalised : same shape as *signal*.
    """
    axis = -1  # time axis
    mean = signal.mean(axis=axis, keepdims=True)
    std = signal.std(axis=axis, keepdims=True) + eps
    return ((signal - mean) / std).astype(signal.dtype)


# ── Epoching ─────────────────────────────────────────────────────────

def epoch_signal(
    signal: NDArray[np.floating],
    window_samples: int = 1000,
    overlap: float = 0.5,
) -> NDArray[np.floating]:
    """Slice a continuous recording into fixed-length epochs.

    Parameters
    ----------
    signal : array, shape (C, T)
        Multi-channel continuous EEG.
    window_samples : int
        Epoch length in samples (default 1000 = 4 s at 250 Hz).
    overlap : float
        Fraction of overlap between consecutive windows.

    Returns
    -------
    epochs : array, shape (N, C, W)
    """
    C, T = signal.shape
    step = int(window_samples * (1.0 - overlap))
    if step < 1:
        raise ValueError("Overlap too large; step size < 1.")
    starts = list(range(0, T - window_samples + 1, step))
    if not starts:
        raise ValueError(
            f"Signal length {T} too short for window {window_samples}."
        )
    epochs = np.stack([signal[:, s : s + window_samples] for s in starts])
    return epochs


# ── Artifact rejection ───────────────────────────────────────────────

def reject_epochs(
    epochs: NDArray[np.floating],
    threshold_uv: float = 100.0,
) -> NDArray[np.floating]:
    """Reject epochs where any channel exceeds *threshold_uv* μV.

    Parameters
    ----------
    epochs : array, shape (N, C, T)
    threshold_uv : float
        Peak-to-peak threshold in μV.

    Returns
    -------
    clean_epochs : array, shape (N', C, T)  where N' ≤ N.
    """
    peak_to_peak = epochs.max(axis=-1) - epochs.min(axis=-1)  # (N, C)
    mask = (peak_to_peak <= threshold_uv).all(axis=1)          # (N,)
    return epochs[mask]


# ── Pipeline composers ───────────────────────────────────────────────

def preprocess_tuh(
    raw: NDArray[np.floating],
    fs: float = 250.0,
) -> NDArray[np.floating]:
    """Full TUH pre-processing pipeline.

    Parameters
    ----------
    raw : array, shape (19, T)  — assumed already resampled to *fs*.

    Returns
    -------
    epochs : array, shape (N, 19, 1000)
    """
    x = bandpass_filter(raw, low=0.5, high=45.0, fs=fs, order=4)
    x = notch_filter(x, freq=50.0, fs=fs)
    x = zscore_per_channel(x)
    epochs = epoch_signal(x, window_samples=int(4.0 * fs), overlap=0.5)
    epochs = reject_epochs(epochs, threshold_uv=100.0)
    return epochs


def preprocess_bci_iv(
    trial: NDArray[np.floating],
    fs: float = 250.0,
    cue_offset_s: float = 0.5,
    trial_end_s: float = 4.0,
) -> NDArray[np.floating]:
    """BCI Competition IV 2a single-trial pre-processing.

    Parameters
    ----------
    trial : array, shape (22, T)  — full trial including pre-cue.
    fs : float
        Sampling rate.
    cue_offset_s : float
        Time after cue onset to start (s).
    trial_end_s : float
        Time after cue onset to stop (s).

    Returns
    -------
    processed : array, shape (22, 875)
    """
    start = int(cue_offset_s * fs)
    end = int(trial_end_s * fs)
    x = trial[:, start:end]  # (22, 875)
    x = bandpass_filter(x, low=4.0, high=38.0, fs=fs, order=4)
    x = zscore_per_channel(x)
    return x
