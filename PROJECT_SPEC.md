# NeuroLLM — Technical Specification

## 1. Problem Statement

Motor imagery BCI systems decode imagined movements from EEG signals. The standard pipeline is: bandpass filter → CSP (Common Spatial Patterns) → SVM/LDA. This works but:
- Requires per-subject calibration (no transfer learning)
- Ignores temporal dynamics beyond fixed windows
- Doesn't scale to richer tasks

Foundation-model approaches (pre-train a transformer on large EEG data, then fine-tune) promise subject-independent features and better generalization, but existing work (LaBraM, BrainBERT) uses massive compute. We show a small model (~10M params) pre-trained on a subset of TUH can still improve over baselines.

## 2. Data Pipeline

### 2.1 TUH EEG Corpus (Pre-training)

**Raw format:** EDF files, variable channels (19-128), variable sampling rates (250-512 Hz)

**Preprocessing:**
1. Select common 19-channel 10-20 montage (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz)
2. Resample to 250 Hz
3. Bandpass filter: 0.5-45 Hz (4th order Butterworth)
4. Z-score normalization per channel per session
5. Segment into 4-second windows (1000 samples), 50% overlap
6. Artifact rejection: reject windows where any channel exceeds ±100 μV

**Expected yield:** ~2M windows from 2000 sessions

### 2.2 BCI Competition IV Dataset 2a (Fine-tuning)

**Preprocessing:**
1. Bandpass filter: 4-38 Hz (motor imagery band)
2. Extract [0.5s, 4.0s] post-cue window (875 samples at 250 Hz)
3. Z-score normalization per channel per trial
4. 22 EEG channels (drop 3 EOG)

**Split:** Session 1 = train, Session 2 = test (standard evaluation protocol)

## 3. Model Architecture

### 3.1 Patch Embedding

EEG signal: [B, C, T] where C=channels, T=time samples

1. Reshape into patches: [B, C, T/P, P] where P=50 (200ms patches)
2. Linear projection: [B, C × num_patches, d_model]
3. Each token = one channel at one time patch

### 3.2 Positional + Channel Encoding

- **Temporal position:** Learnable embedding [num_patches, d_model]
- **Channel position:** Learnable embedding [num_channels, d_model] — captures electrode topology
- Added to patch embeddings

### 3.3 Frequency-Band Attention

Standard attention attends uniformly across all tokens. For EEG, different frequency bands carry different information:
- **Delta (0.5-4 Hz):** Sleep, drowsiness
- **Theta (4-8 Hz):** Working memory
- **Alpha (8-13 Hz):** Relaxation, motor cortex idling
- **Beta (13-30 Hz):** Motor planning and execution
- **Gamma (30-45 Hz):** High-level processing

**Custom attention mask:** Decompose each patch into frequency bands via FFT. Apply band-specific attention weights:
```
Attention(Q, K, V) = softmax(Q·K^T / √d + M_band) · V
```
where M_band is a learned bias per frequency band that modulates which channels attend to which.

**CUDA kernel:** Implements this masked attention with frequency decomposition fused into the attention computation (no separate FFT pass → saves one HBM round-trip).

### 3.4 Pre-training Objective: Masked Channel Modeling

1. Randomly select 30% of tokens (channel-time patches)
2. Replace with learnable [MASK] embedding
3. Predict original values via MSE loss:
   ```
   L = MSE(model(x_masked), x_original)[masked_positions]
   ```
4. Encoder learns to reconstruct masked patches from surrounding spatial-temporal context

### 3.5 Fine-tuning

1. Prepend learnable [CLS] token
2. Pass through pre-trained encoder (all layers trainable)
3. [CLS] representation → MLP [d_model → 128 → 4] → softmax
4. Cross-entropy loss
5. Learning rate: 5e-5 (10× lower than pre-training), 50 epochs, early stopping on validation loss

## 4. Baselines

### CSP + SVM
- Band-pass filter per frequency band (alpha, beta)
- Compute CSP spatial filters (6 components per band)
- Log-variance features → SVM with RBF kernel
- Implementation: MNE-Python + scikit-learn

### EEGNet
- Standard EEGNet architecture (Lawhern et al., 2018)
- Temporal convolution → depthwise spatial convolution → separable convolution
- ~2.6K parameters, trained from scratch on BCI-IV

### Vanilla Transformer
- Same architecture as NeuroLLM but no pre-training (random init)
- Shows the value of pre-training independently of architecture

## 5. Evaluation

### Metrics
- **Accuracy:** Per-subject classification accuracy (4-class)
- **Cohen's Kappa:** Chance-corrected accuracy (κ = 0 = random, κ = 1 = perfect)
- **Confusion matrix:** Per-subject, reveals which classes are confused
- **Attention visualization:** Topographic maps of attention weights over electrode positions

### Protocol
- Per-subject: train on Session 1, test on Session 2 (standard)
- Report: mean ± std across 9 subjects
- Cross-subject: leave-one-subject-out (transfer learning evaluation)

## 6. Success Criteria

| Metric | Threshold |
|--------|-----------|
| Pre-trained model > vanilla transformer | > 3% accuracy improvement |
| Beat CSP+SVM on subject average | > 70% (vs ~65-70% CSP baseline) |
| Attention maps show motor cortex focus | C3/C4 electrodes highlighted |
| Custom CUDA kernel correctness | Max error < 1e-3 vs PyTorch reference |
| Per-subject results with confidence intervals | ✓ |

## 7. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Data download (TUH registration + BCI-IV). Preprocessing pipeline. |
| 2 | EEG Dataset class. Verify data loading and shapes. |
| 3-4 | Implement transformer architecture. Pre-train on TUH subset. |
| 5 | Fine-tune on BCI-IV. Initial accuracy numbers. |
| 6 | Implement baselines (CSP+SVM, EEGNet, vanilla transformer). |
| 7 | CUDA frequency-band attention kernel. |
| 8 | Attention visualization. Topographic maps. |
| 9 | Full benchmark. README with real results. Blog post. |
