# NeuroLLM — Neural Signal Decoding via Transformer Pre-training

**Brain-Computer Interface × LLM × GPU Compute**

> A foundation model approach to EEG decoding: pre-train a small transformer on large-scale EEG data, then fine-tune for motor imagery classification with a custom frequency-band attention kernel.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

Brain-computer interfaces (BCIs) traditionally rely on hand-crafted features (CSP, band-power) fed into classical classifiers. Recent work (LaBraM, BrainBERT, EEGFormer) shows that transformer-based models pre-trained on large EEG corpora can learn general neural signal representations that transfer across subjects and tasks.

NeuroLLM implements this approach at a reproducible scale:
1. **Pre-train** a small transformer (6-12 layers, ~10M params) on the Temple University Hospital EEG Corpus
2. **Fine-tune** for 4-class motor imagery classification on BCI Competition IV Dataset 2a
3. **Custom CUDA kernel** for frequency-band temporal attention (attends across channels within frequency bands)

## Architecture

```
EEG Signal (C channels × T samples)
        │
        ▼
┌─────────────────────┐
│  Patch Embedding    │   Split each channel into temporal patches
│  (channel × time)   │   Flatten: C patches × P samples → tokens
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Positional         │   Learnable position + channel embedding
│  Encoding           │   (captures electrode topology)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Transformer        │   6–12 layers, 4–8 heads
│  Encoder            │   Frequency-band attention mask
│  (Pre-trained)      │   Custom CUDA attention kernel
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Classification     │   [CLS] token → MLP → 4 classes
│  Head               │   (left hand, right hand, feet, tongue)
└─────────────────────┘
```

## Datasets

### Pre-training: Temple University Hospital EEG Corpus (TUH)
- **Size:** ~25,000 clinical EEG sessions, ~15,000 hours
- **Use:** Self-supervised pre-training (masked channel modeling)
- **Download:** [https://isip.piconepress.com/projects/tuh_eeg/](https://isip.piconepress.com/projects/tuh_eeg/)
- **License:** Requires registration (free for research)

### Fine-tuning: BCI Competition IV Dataset 2a
- **Size:** 9 subjects, 2 sessions each, 288 trials per session
- **Task:** 4-class motor imagery (left hand, right hand, feet, tongue)
- **Channels:** 22 EEG + 3 EOG
- **Sampling rate:** 250 Hz
- **Download:** [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/)

## Project Structure

```
neurollm/
├── README.md
├── PROJECT_SPEC.md
├── DEVELOPMENT_LOG.md
├── LICENSE
├── pyproject.toml
├── Dockerfile
├── data/
│   ├── download_tuh.sh          # TUH corpus download script
│   ├── download_bci.sh          # BCI Competition data download
│   ├── preprocessing.py         # Filtering, epoching, artifact rejection
│   └── dataset.py               # PyTorch Dataset/DataLoader
├── model/
│   ├── eeg_transformer.py       # Transformer architecture
│   ├── patch_embedding.py       # EEG → token embedding
│   ├── positional_encoding.py   # Position + channel embedding
│   ├── freq_attention.py        # Frequency-band attention (Python ref)
│   └── classification_head.py   # Fine-tuning head
├── kernels/
│   ├── freq_band_attention.cu   # Custom CUDA kernel
│   ├── freq_band_attention.py   # Triton equivalent
│   └── bindings.cpp             # PyTorch C++ extension
├── training/
│   ├── pretrain.py              # Self-supervised pre-training (masked)
│   ├── finetune.py              # Supervised fine-tuning on BCI-IV
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── evaluation/
│   ├── evaluate.py              # Per-subject accuracy, confusion matrix
│   ├── attention_viz.py         # Attention weight visualization
│   ├── topographic_map.py       # Electrode-space attention heatmap
│   └── results/
│       └── .gitkeep
├── baselines/
│   ├── csp_svm.py               # CSP + SVM baseline
│   ├── eegnet.py                # EEGNet baseline (Lawhern et al.)
│   └── standard_transformer.py  # Vanilla transformer (no pre-training)
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_kernel.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── pretraining_analysis.ipynb
│   └── results_visualization.ipynb
└── .github/
    └── workflows/
        └── ci.yml
```

## Pre-training Strategy

**Objective:** Masked Channel Modeling (MCM)
- Randomly mask 30% of channel-time patches
- Predict masked patches from context (MSE loss)
- Similar to masked image modeling (MAE, BEiT) adapted for EEG

**Model scale:**
- 6 layers, 4 heads, d_model=256, d_ff=512
- ~10M parameters (fits easily on T4)
- Patch size: 50 samples (200ms at 250Hz)

**Training:**
- Pre-train on TUH corpus subset (~2000 sessions, ~2000 hours)
- 100 epochs, batch size 64
- AdamW, cosine schedule, lr=1e-4
- Estimated: ~8 hours on T4

## Benchmarks

_To be populated with real results:_

| Method | Subject Avg Accuracy | Kappa | Params |
|--------|---------------------|-------|--------|
| CSP + SVM | ~65-70% (literature) | — | N/A |
| EEGNet | ~70-75% (literature) | — | 2.6K |
| Vanilla Transformer (no pretrain) | —% | — | ~10M |
| **NeuroLLM (ours, pre-trained)** | **—%** | **—** | **~10M** |

_Per-subject results reported as mean ± std across 9 subjects. 10-fold cross-validation._

## Hardware

| Task | Hardware | Estimated Time |
|------|----------|---------------|
| Data preprocessing | Mac (CPU) | 2-3 hours |
| Pre-training (TUH) | T4 16GB | ~8 hours |
| Fine-tuning (BCI-IV) | T4 16GB | ~30 min |
| CUDA kernel benchmarking | T4 16GB | ~1 hour |
| Attention visualization | Mac (CPU) | Minutes |

## References

- [LaBraM: Large Brain Model for Learning Generic Representations (Jiang et al., 2024)](https://arxiv.org/abs/2405.12220)
- [BrainBERT: Self-supervised representation learning for intracranial recordings (Wang et al., 2023)](https://arxiv.org/abs/2302.14367)
- [EEGNet: A Compact CNN for EEG-Based BCIs (Lawhern et al., 2018)](https://arxiv.org/abs/1611.08024)
- [BCI Competition IV](https://www.bbci.de/competition/iv/)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

## License

Apache 2.0
