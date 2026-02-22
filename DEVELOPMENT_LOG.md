# NeuRoLLM -- Development Log

**Status: COMPLETE (v1.0.9)**

---

## v1.0.0 -- Scaffold, Data Ingestion and Preprocessing (40 tests)

- Project scaffold: pyproject.toml, Dockerfile, CI, LICENSE.
- data/preprocessing.py: bandpass (0.5-45 Hz), notch (50 Hz), z-score,
  epoch extraction (0-4 s motor-imagery windows).
- data/dataset.py: SyntheticEEG, BCIIV2aDataset, TUHDataset with
  automatic synthetic fallback.
- data/download.py: download stubs with instructions.
- 40 tests covering shapes, dtypes, edge cases, I/O.

## v1.0.1 -- Transformer Architecture (23 tests)

- model/transformer.py: PatchEmbedding (channel-wise 1-D conv,
  patch_size=50), PositionalEncoding (learnable spatial + temporal),
  TransformerEncoder (6 pre-norm layers), NeuRoLLM (CLS head).
- Output shape (B, n_classes), ~10 M params.
- 23 tests: forward pass, masking, param count, device transfer.

## v1.0.2 -- Masked Channel Modeling (19 tests)

- training/pretrain.py: MCMPretrainer -- random 30 % channel masking,
  MSE reconstruction loss with gradient scaling.
- Verified loss decreases over 5 synthetic epochs.
- 19 tests: mask shapes, loss computation, reconstruction.

## v1.0.3 -- Pre-training Loop (6 tests)

- training/run_pretrain.py: CLI runner with cosine-decay LR, AdamW,
  checkpoint save (best_mcm.pt).
- train_mcm() end-to-end function.
- 6 tests: checkpoint creation, loss curve, reproducibility.

## v1.0.4 -- Fine-tuning Pipeline (11 tests)

- training/finetune.py: load_pretrained_encoder (shape-safe state-dict
  loading), progressive unfreeze schedule, per-subject train/eval.
- Handles 19-ch TUH -> 22-ch BCI-IV mismatch by filtering incompatible keys.
- 11 tests: freeze/unfreeze, accuracy on synthetic data, session split.

## v1.0.5 -- Baselines (11 tests)

- baselines/models.py: CSP+SVM, EEGNet (~2.6 K params), VanillaTransformer
  (same arch, random init).
- Unified train_and_evaluate interface returning accuracy dict.
- 11 tests: output shapes, training convergence, baseline comparison.

## v1.0.6 -- CUDA / Triton Frequency-Band Attention Kernel (18 tests)

- kernels/freq_band_attention.py: fft_band_decompose (delta, theta,
  alpha, beta, gamma), FreqBandAttention (learnable per-head band biases),
  Triton freq_band_attn_kernel (fused softmax, optional).
- 5 canonical frequency bands: delta (0.5-4), theta (4-8), alpha (8-13),
  beta (13-30), gamma (30-45).
- 18 tests: FFT decomposition, band energy, kernel numerics, gradient flow.

## v1.0.7 -- Evaluation Metrics and Attention Visualisation (27 tests)

- evaluation/metrics.py: compute_metrics (accuracy, Cohen kappa,
  confusion matrix, per-class accuracy), aggregate_subject_metrics,
  extract_attention_maps (manual encoder-layer iteration to get
  per-head attention weights), attention_to_channel_importance,
  plot_confusion_matrix, plot_channel_attention (C3/C4 highlights),
  plot_loss_curve.
- Workaround: PyTorch TransformerEncoderLayer hardcodes need_weights=False;
  solved by manually iterating layers and calling self_attn() directly.
- 27 tests.

## v1.0.8 -- Full Benchmark Suite (14 tests)

- evaluation/benchmark.py: run_pretrain_step, evaluate_neurollm,
  evaluate_baselines, run_full_benchmark, _write_summary_csv.
- End-to-end pipeline: pretrain -> finetune -> baselines -> CSV report.
- Fixed checkpoint filename (best_mcm.pt) and channel-mismatch loading.
- 14 tests.

## v1.0.9 -- Polish and Ship (23 tests)

- evaluation/demo.py: compact all-in-one demo (pretrain -> finetune ->
  evaluate -> generate plots) on synthetic data.
- Updated README.md: architecture diagram, badges, benchmarks, test table.
- Updated DEVELOPMENT_LOG.md: all 10 releases.
- 23 tests: 4 end-to-end smoke, 12 import checks, 1 demo run,
  6 documentation existence checks.

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Patch-based tokenisation (P=50) | Preserves temporal structure at 250 Hz |
| Pre-norm transformer | Stable training for small models |
| MCM (channel masking) | Learns cross-channel dependencies |
| Shape-safe checkpoint loading | Handles 19-ch -> 22-ch mismatch |
| Manual attention extraction | Bypasses PyTorch hardcoded need_weights=False |
| Synthetic fallback everywhere | Full test suite runs without real data |
| Triton kernel (optional) | Fast path on GPU, pure-PyTorch fallback on CPU |

## Test Summary

- **192 tests** across 10 suites
- All passing on Python 3.11 / PyTorch 2.x / macOS (CPU)
- Average suite runtime: under 30 s
