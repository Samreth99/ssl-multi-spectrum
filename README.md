# Beyond Visible Spectrum: AI for Agriculture 2026 Task 2

**ICPR 2026 Competition** | Task 2: Boosting Automatic Crop Diseases Classification using Sentinel Satellite Data and Self-Supervised Learning (SSL)

## Overview

This solution tackles the challenge of crop disease classification from Sentinel-2 satellite imagery under **limited labeled data** conditions. The approach is a two-phase pipeline:

1. **Continual SSL Pretraining** — Adapt an existing SatMAE ViT-Large checkpoint (pretrained on fMoW-Sentinel) to the agricultural domain using unlabeled S2A data via Masked Autoencoder (MAE) self-supervised learning.
2. **Supervised Fine-Tuning** — Fine-tune the adapted encoder for 4-class crop disease classification (Aphid, Blast, RPH, Rust) using only 720 labeled samples, with focal loss to handle severe class imbalance.

**Current Leaderboard Score: 87.5%**

---

## Competition Details

| Field       | Details                                                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Competition | [Beyond Visible Spectrum: AI for Agriculture 2026](https://kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026p2) |
| Task        | Task 2 — Self-Supervised Learning for Crop Disease Classification                                                                     |

---

## Method

### Pretrained Weights — SatMAE

Starting point: **SatMAE ViT-Large** pretrained on the fMoW-Sentinel dataset for 199 epochs.

- Paper: [SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery (NeurIPS 2022)](https://arxiv.org/abs/2207.08051)
- Original repository: [sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE)
- Pretrained checkpoint: [zenodo.org/records/7338613](https://zenodo.org/records/7338613) — `pretrain-vit-large-e199.pth`

---

## Phase 1: Self-Supervised Continual Pretraining

### Model Architecture — `MaskedAutoencoderGroupChannelViT`

The SatMAE variant of MAE processes multispectral bands in **spectral groups** rather than as a single flat tensor. Each group has its own independent patch embedding layer, allowing the model to learn intra-group spectral relationships separately from spatial attention in the shared transformer.

**Encoder (ViT-Large):**

- 24 Transformer blocks, embedding dimension 1024, MLP expansion 4096, GELU activations
- Total parameters: 329,212,928

**Grouped Patch Embeddings** (`Conv2d`, kernel/stride = 8×8 → 96/8 = 12 patches per side):

| Group   | Bands           | Channels | Description           |
| ------- | --------------- | -------- | --------------------- |
| Group 0 | B2, B3, B4, B8  | 4        | Visible + NIR         |
| Group 1 | B5, B6, B7, B8A | 4        | Red Edge + Narrow NIR |
| Group 2 | B11, B12        | 2        | SWIR                  |

**Decoder:**

- `Linear(1024 → 512)` embed projection
- 8 Transformer decoder blocks (dim 512, MLP 2048)
- Per-group reconstruction heads: `Linear(512 → 256)` × 2, `Linear(512 → 128)` × 1

---

### SSL Dataset — Sentinel-2A Unlabeled Imagery

| Property      | Value                                                |
| ------------- | ---------------------------------------------------- |
| Source        | Sentinel-2A (S2A) satellite time-series              |
| Archive size  | 124 GB (tar.gz) / 93.78 GB extracted                 |
| Locations     | 24,964 geographic locations                          |
| Total samples | 127,727 (complete acquisitions)                      |
| Bands used    | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (10 bands) |
| Image size    | 96 × 96 pixels                                       |

---

### Data Preprocessing — `SentinelNormalize`

Custom per-band normalization clipping to ±2σ, matching the original SatMAE preprocessing:

```
min_value = mean - 2 * std
max_value = mean + 2 * std
normalized = clip((x - min_value) / (max_value - min_value) * 255, 0, 255)
```

Band statistics computed from the target S2A agricultural dataset:

| Band | Mean    | Std     |
| ---- | ------- | ------- |
| B2   | 1451.57 | 2155.95 |
| B3   | 1714.90 | 2041.80 |
| B4   | 1955.16 | 2098.95 |
| B5   | 2295.86 | 2068.67 |
| B6   | 2822.04 | 1882.61 |
| B7   | 3054.89 | 1830.56 |
| B8   | 3181.44 | 1878.31 |
| B8A  | 3217.62 | 1745.67 |
| B11  | 2697.60 | 1225.58 |
| B12  | 2123.05 | 1120.58 |

---

### SSL Data Augmentation

**Training:**

- `RandomResizedCrop(96, scale=(0.2, 1.0), BICUBIC)`
- `RandomHorizontalFlip()`
- `RandomVerticalFlip()`

**Validation:**

- `Resize(110, BICUBIC)` → `CenterCrop(96)`

---

### Selective Layer Freezing Strategy

To preserve low-level features learned on fMoW while adapting higher-level representations to the agricultural domain, a **partial freeze** is applied:

| Component                       | Status    | Parameters              |
| ------------------------------- | --------- | ----------------------- |
| Patch embeddings                | Frozen    | —                       |
| Encoder blocks 0–5 (bottom 25%) | Frozen    | 76,348,928              |
| Encoder blocks 6–23 (top 75%)   | Trainable | —                       |
| Final LayerNorm                 | Trainable | —                       |
| Full Decoder                    | Trainable | —                       |
| Mask token                      | Trainable | —                       |
| **Trainable total**             |           | **252,864,000 (76.8%)** |

---

### SSL Training Hyperparameters

| Hyperparameter              | Value                       |
| --------------------------- | --------------------------- |
| Epochs                      | 80                          |
| Warmup epochs               | 5                           |
| Physical batch size         | 32                          |
| Gradient accumulation steps | 8                           |
| Effective batch size        | 256                         |
| Base learning rate          | 1.5e-4                      |
| Min learning rate           | 1e-6                        |
| Weight decay                | 0.05                        |
| Gradient clip norm          | 1.0                         |
| Mask ratio                  | 0.75                        |
| Spatial mask                | False (per-channel masking) |
| Norm pixel loss             | False                       |
| Seed                        | 42                          |

---

### Optimizer & Scheduler (SSL)

**Optimizer:** AdamW (`β₁=0.9, β₂=0.95`) with **Layer-wise Learning Rate Decay (LLRD)**

- Decay factor `layer_decay = 0.80` per layer
- Layer scale = `0.80^(num_layers - layer_id)` where `num_layers = 25`
- Lower layers receive exponentially smaller learning rates, preventing catastrophic forgetting
- No weight decay on biases, LayerNorm parameters, and special tokens

**Scheduler:** Cosine decay with linear warmup (applied per iteration):

```
if epoch < warmup_epochs:
    lr = base_lr * epoch / warmup_epochs          # linear warmup
else:
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Mixed Precision:** `torch.cuda.amp.autocast` + `NativeScalerWithGradNormCount`

---

### SSL Loss Function

Standard MAE pixel-level **Mean Squared Error** between predicted and original pixel values on the masked patches only. Raw pixel values used (`norm_pix_loss=False`).

---

### SSL Results

| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Training epochs          | 80                                     |
| Best reconstruction loss | 0.0009 (epoch 71)                      |
| Checkpoint               | `ssl_ckpt_ep0079.pth`                  |
| Published                | HuggingFace: `Samreth/ssl-checkpoints` |

---

### Monitoring — Layer Drift Analysis

A custom diagnostic tracks **weight drift** (%) of each encoder block's `attn.qkv.weight` relative to the original fMoW checkpoint, every 10 epochs. This validates that frozen bottom layers remain stable while top layers adapt:

- **Bottom (blocks 0–5)**: Near-zero drift (frozen)
- **Middle (blocks 6–17)**: Moderate adaptation
- **Top (blocks 18–23)**: Highest adaptation to agricultural domain

---

## Phase 2: Supervised Fine-Tuning

### Dataset

| Property              | Value                                         |
| --------------------- | --------------------------------------------- |
| Total labeled samples | 900                                           |
| Train / Val split     | 720 / 180 (80/20 stratified)                  |
| Classes               | Aphid (232), Blast (60), RPH (396), Rust (32) |

The dataset is **severely imbalanced**: Rust has ~12× fewer samples than RPH.

---

### Model Architecture — `SatMAE_Classifier`

1. Load SSL-adapted encoder from `ssl_ckpt_ep0079.pth`
2. **Delete decoder** (all decoder weights removed)
3. **Freeze entire encoder** initially
4. **Unfreeze last 4 encoder blocks** (blocks 20–23) + final LayerNorm
5. Add **linear classification head**: `Linear(1024 → 4)` (Xavier uniform init)

**Forward pass**: run encoder with `mask_ratio=0.0` (no masking) → extract **CLS token** → linear head

| Parameter Group                 | Count              |
| ------------------------------- | ------------------ |
| Total parameters                | 303,143,300        |
| Trainable (top 4 blocks + head) | 50,391,044 (16.6%) |

---

### Class Imbalance — Focal Loss with Inverse-Frequency Weighting

**Focal Loss** with class-frequency-based alpha weights:

```
alpha = inverse_class_frequencies / sum(inverse_class_frequencies)
     = [0.0787, 0.3044, 0.0461, 0.5708]  # Aphid, Blast, RPH, Rust

focal_weight = (1 - p_t)^γ,   γ = 2.0
loss = mean(alpha_t * focal_weight * cross_entropy)
```

Rust (rarest class) receives weight **0.5708** vs RPH (most common) at **0.0461**.

---

### Downstream Augmentation

**Training:**

- `RandomResizedCrop(96, scale=(0.2, 1.0), BICUBIC)`
- `RandomHorizontalFlip()`
- `RandomVerticalFlip()`

**Validation:**

- `Resize(110, BICUBIC)` → `CenterCrop(96)`

---

### Downstream Training Hyperparameters

| Hyperparameter    | Value                        |
| ----------------- | ---------------------------- |
| Epochs            | 25                           |
| Batch size        | 16                           |
| Learning rate     | 1e-4                         |
| Weight decay      | 0.05                         |
| Focal loss γ      | 2.0                          |
| Optimizer         | AdamW                        |
| Scheduler         | CosineAnnealingLR (T_max=25) |
| Evaluation metric | Macro F1-Score               |

---

### Downstream Results

| Epoch | Train Loss | Train F1 | Val Loss | Val F1           |
| ----- | ---------- | -------- | -------- | ---------------- |
| 1     | 0.0653     | 35.2%    | 0.0460   | 19.1%            |
| 4     | 0.0170     | 56.2%    | 0.0243   | 53.4%            |
| 8     | 0.0132     | 63.2%    | 0.0316   | 58.2%            |
| 10    | 0.0104     | 71.1%    | 0.0255   | 61.3%            |
| 15    | 0.0069     | 82.9%    | 0.0302   | **65.9%** ← best |
| 25    | 0.0046     | 86.5%    | 0.0356   | 61.6%            |

---

### Test-Time Augmentation (TTA)

4-view TTA applied at inference by averaging softmax probabilities:

| View | Transform                  |
| ---- | -------------------------- |
| 1    | Original                   |
| 2    | Horizontal flip            |
| 3    | Vertical flip              |
| 4    | Horizontal + Vertical flip |

---

## Key Techniques & Innovations

| #   | Technique                                    | Description                                                                                                                                              |
| --- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Grouped Channel Patch Embedding**          | Three independent `Conv2d` patch embedders for spectrally-related band groups instead of one 10-channel embedder. Learns intra-group spectral structure. |
| 2   | **Continual SSL Pretraining**                | Domain adaptation from fMoW → agricultural S2A, not from scratch. Leverages 199 epochs of existing pretraining.                                          |
| 3   | **Selective Layer Freezing (SSL)**           | Bottom 6 encoder blocks frozen during SSL adaptation, preserving general low-level features while adapting high-level representations.                   |
| 4   | **Layer-wise LR Decay (LLRD)**               | Exponential decay factor 0.80 per layer ensures lower/older layers update more slowly, preventing catastrophic forgetting.                               |
| 5   | **Layer Drift Monitoring**                   | Custom diagnostic tracking % weight change vs. original fMoW checkpoint for every block across 80 SSL epochs.                                            |
| 6   | **Domain-Adapted Band Statistics**           | Per-band mean/std computed from target S2A agricultural dataset (not ImageNet), used in `SentinelNormalize` clipping to ±2σ.                             |
| 7   | **Focal Loss + Inverse Frequency Weighting** | Combines focal loss (γ=2.0) with class-frequency alpha weights to handle severe class imbalance (Rust: 32 vs RPH: 396 samples).                          |
| 8   | **CLS Token Classification**                 | Uses the global CLS token from `forward_encoder(x, mask_ratio=0.0)` as the image representation passed to the classification head.                       |
| 9   | **4-view Test-Time Augmentation**            | Averages predictions across original + H-flip + V-flip + both-flip views to improve inference robustness.                                                |
| 10  | **Gradient Accumulation**                    | `accum_iter=8` with batch size 32 gives effective batch size 256, matching the original MAE training scale without requiring extra GPU memory.           |

---

## Hardware & Environment

| Component    | Details                                 |
| ------------ | --------------------------------------- |
| GPU          | NVIDIA A100-SXM4-80GB (85.1 GB VRAM)    |
| Framework    | PyTorch 2.10.0 + CUDA 12.8              |
| timm version | 0.3.2 (pinned for SatMAE compatibility) |
| Libraries    | wandb, einops, rasterio, scikit-learn   |

---

## File Structure

```
ssl-multi-spectrum/
├── Beyond_Visible_Spectrum_AI_for_Agriculture_2026.ipynb   # Full pipeline notebook
└── best_model_submission.csv                               # Final test predictions
```
