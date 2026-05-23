# Non-blind-Self-Supervised-Network-for-Cherenkov-Video-Denoising

Official implementation of:

**From Noise Synthesis to Noise Removal: Non-blind Self-Supervised Cherenkov Video Denoising for Quantitative Verification of Radiotherapy**

---

## Overview

This repository provides the official PyTorch implementation of **a non-blind self-supervised Cherenkov video denoising framework** for quantitative radiotherapy verification.

The proposed framework introduces:

- **Cycle-degradation adversarial learning**
- **AOC-guided noise prior conditioning**
- **Self-supervised video restoration**
- **Interpretability analysis using causal effect maps**

It enables robust denoising under extremely low-SNR Cherenkov acquisition conditions for:

- **Dose distribution verification**
- **Patient positioning verification**
- **Beam-edge alignment analysis**

---

## Framework

The framework consists of:

### 1. Denoising Module (`F_den`)
- FastDVDnet-style burst denoising backbone
- 5-frame temporal aggregation
- AdaIN-based noise-prior conditioning
![Framework](figures/framework.tif)
### 2. Degradation Module (`F_deg`)
- Pix2Pix-based controllable degradation generator
- Noise-level conditioned image translation
- CBAM-enhanced decoder

### 3. Adversarial Constraints
- Distribution-level realism supervision
- Cycle-consistent degradation regression

---

## Repository Structure

```bash
.
├── data/                   # Training / validation data
├── models/                 # Network definitions
│   ├── denoising/
│   ├── degradation/
│   └── discriminator/
├── train/
│   ├── train_deg.py
│   ├── train_den.py
│   └── finetune.py
├── test/
│   └── inference.py
├── utils/
├── checkpoints/
├── results/
└── README.md
```

---

## Installation

Create environment:

```bash
conda create -n cherenkov python=3.9
conda activate cherenkov
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main requirements:

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8
- torchvision
- numpy
- opencv-python
- scipy
- visdom
- lpips

---

## Dataset

### Phantom Dataset

Place phantom data under:

```bash
data/phantom/
```

Expected structure:

```bash
phantom/
    clean/
    noisy/
    AOC/
```

### Clinical Dataset

Due to patient privacy restrictions, clinical datasets are not publicly released.

---

## Training

### Step 1: Pretrain degradation module

```bash
python train/train_deg.py
```

### Step 2: Train denoising module

```bash
python train/train_den.py
```

### Step 3: Alternating finetuning

```bash
python train/finetune.py
```

---

## Testing

Inference on Cherenkov video:

```bash
python test/inference.py \
    --input ./data/test \
    --checkpoint ./checkpoints/model.pth
```

---

## Evaluation Metrics

We report:

- **PSNR**
- **SSIM**
- **LPIPS**
- **Gamma Passing Rate (3%/3 mm)**
- **Dice Similarity Coefficient**
- **Mean Distance to Conformity (MDC)**

---

## Results

The proposed method achieves:

| Metric | Phantom | Clinical |
|--------|---------|----------|
| Gamma Passing Rate | **92.99%** | **90.09%** |
| Position Sensitivity Improvement | **150.75%** | — |
| Memory Consumption | **1.754 GB** | — |
| Inference Time | **120 ms/frame** | Real-time compatible |
