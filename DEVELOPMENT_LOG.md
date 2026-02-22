# NeuroLLM — Development Log

> Real-time build diary. Updated as work happens.

---

## Status: IN PROGRESS

---

## v1.0.0 — Scaffold + EEG Data Pipeline

**Date:** 2025-07-22

### What was built
- Project scaffold: `pyproject.toml`, `Dockerfile`, `.gitignore`, `.github/workflows/ci.yml`, `LICENSE`
- `ROADMAP.md` with all 10 planned releases
- Package structure: `data/`, `model/`, `kernels/`, `training/`, `evaluation/`, `baselines/`, `tests/`
- **`data/preprocessing.py`** — Full EEG preprocessing toolkit:
  - Bandpass filter (4th order Butterworth, zero-phase via `sosfiltfilt`)
  - Notch filter (50/60 Hz power-line removal)
  - Z-score normalisation per channel
  - Epoching (fixed-length windows with configurable overlap)
  - Artifact rejection (peak-to-peak ±100 μV threshold)
  - Pipeline composers: `preprocess_tuh()` and `preprocess_bci_iv()`
- **`data/dataset.py`** — PyTorch Dataset classes:
  - `make_synthetic_eeg()` — class-dependent spectral profiles for testing
  - `SyntheticEEGDataset` — torch wrapper with DataLoader compatibility
  - `BCIIV2aDataset` — 22-ch, 4-class motor imagery (synthetic fallback)
  - `TUHEEGDataset` — 19-ch unlabelled epochs for MCM pre-training (synthetic fallback)
- **`data/download.py`** — Download instructions for TUH and BCI-IV

### Test results
```
40 passed in 26.94s
```
- Bandpass: shape, dtype, DC removal, passband energy, stopband attenuation, batch
- Notch: shape, target attenuation, off-target preservation
- Z-score: zero mean, unit variance, shape, dtype
- Epoching: window count, no-overlap, short signal error, content correctness
- Artifact rejection: clean kept, bad removed, all-bad case
- Pipeline composers: TUH output shape/dtype, BCI-IV shape/normalisation
- Synthetic generator: shapes, dtypes, label range, reproducibility, seed variation
- Datasets: len, getitem, DataLoader batching, all 9 subjects, TUH fallback
- Constants: channel counts, frequency band definitions

### Decisions
- Synthetic EEG uses class-dependent dominant frequencies (8/12/20/30 Hz) so models can learn separation
- Both dataset classes fall back to synthetic data when real files are missing — tests never require data downloads
- TUH pipeline: 0.5–45 Hz bandpass + 50 Hz notch + z-score + 4 s windows (50 % overlap) + ±100 μV rejection
- BCI-IV pipeline: 4–38 Hz bandpass + z-score on [0.5 s, 4.0 s] post-cue window (875 samples)
