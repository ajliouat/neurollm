# NeuRoLLM Roadmap

## v1.0.0 — Scaffold + EEG Data Pipeline
- Project scaffold: pyproject.toml, Dockerfile, CI, .gitignore
- EEG preprocessing: bandpass, notch, z‑score, epoching, artifact rejection
- PyTorch Dataset classes for BCI‑IV 2a and TUH EEG Corpus
- Synthetic EEG generator for unit tests
- Data download helpers with credential gating

## v1.0.1 — EEG Transformer Architecture
- Patch embedding: channel‑wise temporal patches (P=50 → d_model=256)
- Learnable spatial + temporal positional embeddings
- Transformer encoder (6 layers, 4 heads, d_model=256, d_ff=512)
- CLS token and classification head (MLP → 4 classes)
- Full forward‑pass tests with random and synthetic input

## v1.0.2 — Masked Channel Modeling (Pre‑training Loop)
- 30 % channel‑time patch masking strategy
- MSE reconstruction loss on masked patches
- Pre‑training loop with optimizer (AdamW, lr=1e‑4, cosine decay)
- Checkpoint save / resume
- Loss‑curve logging

## v1.0.3 — Pre‑training Run + First Real Numbers
- Pre‑train on synthetic corpus (TUH proxy)
- Track loss convergence over ≥50 epochs on small synthetic set
- Compare reconstruction error: trained vs random init
- Persist best checkpoint
- Document first numeric results in DEVELOPMENT_LOG

## v1.0.4 — Fine‑tuning Pipeline
- Freeze/unfreeze strategy for pre‑trained encoder
- Classification head fine‑tuning on BCI‑IV 2a (synthetic proxy)
- Per‑subject training (9 subjects, session 1→train, session 2→test)
- Learning rate schedule (lr=5e‑5, warmup + cosine)
- Early stopping with patience

## v1.0.5 — Baselines
- CSP + SVM baseline (MNE + scikit‑learn)
- EEGNet baseline (~2.6 K params, Lawhern 2018)
- Vanilla transformer (same architecture, no pre‑training)
- Unified evaluation harness for all models
- Per‑subject accuracy comparison

## v1.0.6 — CUDA / Triton Frequency‑Band Attention Kernel
- FFT‑based band decomposition: δ θ α β γ
- Band‑specific learnable attention biases
- CUDA C++ extension with autograd wrapper
- Triton kernel for portability
- Numerical correctness tests: max |err| < 1e‑3 vs naive PyTorch

## v1.0.7 — Evaluation + Attention Visualisation
- Per‑subject accuracy and Cohen's κ
- Confusion matrices (4‑class)
- Attention‑map extraction over EEG montage
- Topographic attention plots (highlight C3 / C4 for motor imagery)
- Publication‑quality figures

## v1.0.8 — Full Benchmark Suite
- End‑to‑end benchmark: pre‑train → fine‑tune → evaluate all 9 subjects
- Compare NeuRoLLM vs CSP + SVM, EEGNet, vanilla transformer
- Statistical significance tests (paired t‑test across subjects)
- Throughput and latency profiling (CPU + CUDA)
- Results written to JSON / CSV

## v1.0.9 — Polish & Ship
- README badges, architecture diagram final pass
- Blog project page update with real numbers
- Demo GIF / notebook
- DEVELOPMENT_LOG final entry
- Tag v1.0.9 and push

---

## Future Evolution

> These iterations are aspirational — they represent natural next steps for
> the project if development resumes. Not currently scheduled.

### v1.1.0 — Real TUH Pre-training

**Goal:** Pre-train on real Temple University Hospital EEG Corpus instead of synthetic data.

- Download and preprocess TUH EEG Corpus (~15,000 hours)
- Multi-GPU pre-training on A100 cluster (DDP)
- Track MCM reconstruction loss convergence over 100 epochs
- Compare pre-trained vs random-init representations via linear probing
- Checkpoint size optimization: FP16 weights, sharded checkpoints

### v1.2.0 — Real BCI-IV Fine-tuning & Results

**Goal:** Fine-tune on real BCI Competition IV Dataset 2a and report per-subject accuracy.

- Download and preprocess BCI-IV 2a (9 subjects, 22 channels, 250 Hz)
- Per-subject training with cross-session evaluation
- Report: per-subject accuracy, mean ± std, Cohen's κ
- Target: 75–85% mean accuracy (matching LaBraM literature)
- Statistical comparison vs baselines (paired t-test, 9 subjects)

### v1.3.0 — Cross-Dataset Transfer

**Goal:** Evaluate pre-trained representations on multiple downstream BCI tasks.

- BCI Competition III Dataset IVa (2-class motor imagery)
- PhysioNet EEG Motor Movement/Imagery Dataset (109 subjects)
- Few-shot evaluation: 10%, 25%, 50% training data
- Transfer learning curves: accuracy vs fine-tuning data fraction
- Compare NeuRoLLM transfer vs train-from-scratch at each data fraction

### v1.4.0 — Attention Interpretability Study

**Goal:** Comprehensive analysis of what the transformer learns about neural signals.

- Layer-by-layer attention analysis (early vs late layers)
- Band-specific attention patterns: verify mu/beta focus for motor imagery
- Topographic attention maps aligned with 10-20 electrode positions
- Comparison with known neurophysiology (C3/C4 for hand, Cz for feet)
- Publication-quality figures for all 9 subjects

### v1.5.0 — Online BCI Simulation

**Goal:** Simulate real-time BCI decoding with streaming EEG data.

- Sliding window inference: 4-second epochs, 0.5-second stride
- Latency profiling: end-to-end inference time (target <50ms on CPU)
- Information Transfer Rate (ITR) calculation
- Continuous prediction with confidence thresholding
- Simulated cursor control using predicted motor imagery classes

### v1.6.0 — Multi-Task Pre-training

**Goal:** Extend MCM pre-training with additional self-supervised objectives.

- Contrastive learning: augmented views of same EEG segment
- Temporal order prediction: shuffled patch sequence classification
- Joint MCM + contrastive + temporal loss
- Evaluate multi-task vs MCM-only transfer on BCI-IV 2a
- Ablation: contribution of each pre-training objective

---

*v1.0.0–v1.0.9 delivered. Future iterations begin when development resumes.*
