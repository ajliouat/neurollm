# NeuRoLLM -- Neural Signal Decoding via Transformer Pre-training

**Brain-Computer Interface x Foundation Model x GPU Compute**

> A foundation-model approach to EEG decoding: pre-train a small
> transformer on large-scale EEG data, then fine-tune for motor-imagery
> classification with a custom frequency-band attention kernel.

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Tests](https://img.shields.io/badge/tests-192%20passed-brightgreen.svg)
![Version](https://img.shields.io/badge/version-1.0.9-informational.svg)

---

## Overview

Brain-computer interfaces (BCIs) traditionally rely on hand-crafted features
(CSP, band-power) fed into classical classifiers. Recent work -- LaBraM,
BrainBERT, EEGFormer -- shows that transformer-based models pre-trained on
large EEG corpora learn general neural-signal representations that transfer
across subjects and tasks.

NeuRoLLM implements this approach at a reproducible scale:

1. **Pre-train** a ~10 M-parameter transformer on the Temple University
   Hospital (TUH) EEG Corpus via Masked Channel Modeling (MCM).
2. **Fine-tune** for 4-class motor-imagery classification on BCI
   Competition IV Dataset 2a.
3. **Custom CUDA / Triton kernel** for frequency-band temporal attention.

## Architecture

```
EEG Signal  (C channels x T samples)
        |
        v
+---------------------+
|  Patch Embedding    |   Channel-wise temporal patches (P=50 -> d=256)
|  (channel x time)   |   N = C x (T // P) tokens
+---------------------+
        |
        v
+---------------------+
|  Positional         |   Learnable spatial (channel) + temporal (patch)
|  Encoding           |   embeddings capturing electrode topology
+---------------------+
        |
        v
+---------------------+
|  Transformer        |   6 layers, 4 heads, d_model=256, d_ff=512
|  Encoder            |   Pre-norm (LayerNorm -> MHSA -> FFN)
|  (Pre-trained MCM)  |   + Frequency-band attention kernel
+---------------------+
        |
        v
+---------------------+
|  Classification     |   [CLS] token -> MLP -> 4 classes
|  Head               |   (left hand, right hand, feet, tongue)
+---------------------+
```

**Model specs:** 6 layers | 4 heads | d_model = 256 | d_ff = 512 |
patch_size = 50 | ~10 M parameters.

## Quick Start

```bash
git clone https://github.com/ajliouat/neurollm.git
cd neurollm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run full test suite
pytest tests/ -v --timeout=60

# Demo: synthetic pretrain -> finetune -> evaluate -> visualise
python -m evaluation.demo
```

## Datasets

### Pre-training -- Temple University Hospital EEG Corpus (TUH)

| Property | Value |
|----------|-------|
| Sessions | ~25 000 clinical EEG |
| Hours | ~15 000 |
| Channels | 19-ch 10-20 montage |
| Use | Self-supervised MCM |

### Fine-tuning -- BCI Competition IV Dataset 2a

| Property | Value |
|----------|-------|
| Subjects | 9 |
| Sessions | 2 per subject (T = train, E = test) |
| Trials | 288 per session |
| Classes | 4 (left hand, right hand, feet, tongue) |
| Channels | 22 EEG + 3 EOG (EOG dropped) |
| Sampling rate | 250 Hz |

> Both datasets fall back to synthetic data when real files are unavailable,
> so the full test suite runs anywhere.

## Project Structure

```
neurollm/
  README.md / ROADMAP.md / DEVELOPMENT_LOG.md / LICENSE
  pyproject.toml / Dockerfile
  data/
    preprocessing.py       # Bandpass, notch, z-score, epoching
    dataset.py             # SyntheticEEG, BCIIV2a, TUH datasets
    download.py            # Download instructions
  model/
    transformer.py         # PatchEmbed, PosEnc, Encoder, NeuRoLLM
  kernels/
    freq_band_attention.py       # FFT band decompose + Triton kernel
    freq_band_attention_cuda.cu  # CUDA C++ extension stub
  training/
    pretrain.py            # MCMPretrainer, train_mcm
    run_pretrain.py        # CLI pre-training runner
    finetune.py            # Freeze / unfreeze, per-subject fine-tuning
  evaluation/
    metrics.py             # Accuracy, kappa, confusion matrix, viz
    benchmark.py           # Full benchmark pipeline
    demo.py                # Quick demo script
  baselines/
    models.py              # CSP+SVM, EEGNet, vanilla transformer
  tests/                   # 10 test suites, 192 tests total
```

## Benchmark (synthetic data)

| Method | Mean Accuracy | Params | Notes |
|--------|:------------:|-------:|-------|
| CSP + SVM | ~25 % | -- | Classical baseline |
| EEGNet | ~25 % | 2.6 K | Compact CNN (Lawhern 2018) |
| Vanilla Transformer | ~25 % | ~10 M | Same arch, random init |
| **NeuRoLLM (pre-trained)** | **~25 %** | **~10 M** | **MCM pre-trained** |

> On synthetic random data all methods converge to chance (25 %).
> With real TUH pre-training + BCI-IV fine-tuning, literature reports 75-85 %.

## Test Results

| Suite | Tests | Scope |
|-------|------:|-------|
| v1.0.0 | 40 | Preprocessing, datasets, download stubs |
| v1.0.1 | 23 | Transformer architecture, forward pass |
| v1.0.2 | 19 | MCM masking, reconstruction loss |
| v1.0.3 | 6 | Pre-training convergence, checkpoints |
| v1.0.4 | 11 | Fine-tuning pipeline, freeze / unfreeze |
| v1.0.5 | 11 | Baseline models (CSP+SVM, EEGNet, VanillaTransformer) |
| v1.0.6 | 18 | Frequency-band attention kernel |
| v1.0.7 | 27 | Evaluation metrics, attention visualisation |
| v1.0.8 | 14 | Full benchmark pipeline |
| v1.0.9 | 23 | Integration smoke tests, demo, docs |
| **Total** | **192** | |

## Hardware

| Component | Recommended | Minimum |
|-----------|------------|---------|
| GPU | NVIDIA A100 (40 GB) | Any CUDA-capable GPU |
| RAM | 32 GB | 16 GB |
| Storage | 100 GB (TUH) | 2 GB (synthetic only) |
| Python | 3.11+ | 3.10+ |

## References

- LaBraM -- Large Brain Model (Jiang et al., 2024)
- BrainBERT (Wang et al., 2023)
- EEGNet (Lawhern et al., 2018)
- BCI Competition IV Dataset 2a
- Attention Is All You Need (Vaswani et al., 2017)

## License

Apache 2.0 -- see [LICENSE](LICENSE).
