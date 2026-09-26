# PartialSpoof: Partially Spoofed Audio Detection & Localization

A comprehensive deep learning framework for detecting and temporally localizing partially spoofed (manipulated / synthetic) segments in speech audio.

This repository contains two complete implementations:
1. **`ps_baseline`**: Standard frame-level linear classifier benchmark utilizing Wav2Vec2 representations.
2. **`ps_1`**: An advanced architecture incorporating multi-layer feature aggregation, temporal BiLSTM sequence modeling, multi-task learning, focal loss, and temporal post-processing.

---

## 📊 Benchmark & Performance Comparison

Evaluated on the **PartialSpoof Development Set** (`dev` split):

| Metric | Baseline (`ps_baseline`) | Improved Model (`ps_1`) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Frame F1** | 77.65% | **94.07%** | **+16.42%** |
| **Frame Precision** | 72.04% | **90.89%** | **+18.85%** |
| **Frame Recall** | 84.20% | **97.48%** | **+13.28%** |
| **Frame AUC-ROC** | 0.8307 (83.07%) | **0.9887 (98.87%)** | **+15.80%** |
| **Average Precision (AP)** | 0.8506 (85.06%) | **0.9902 (99.02%)** | **+13.96%** |
| **Segment Mean IoU** | 56.44% | **84.63%** | **+28.19%** |
| **Mean Onset Error** | 195 ms (0.195 s) | **54 ms (0.054 s)** | **-141 ms (3.6× more precise)** |
| **Mean Offset Error** | 159 ms (0.159 s) | **54 ms (0.054 s)** | **-105 ms (3.0× more precise)** |

> Detailed evaluation records are preserved in [`ps_baseline/results/eval_dev.json`](ps_baseline/results/eval_dev.json) and [`ps_1/results/eval_dev.json`](ps_1/results/eval_dev.json).

---

## 🧠 Architectural Overview

| Component | Baseline (`ps_baseline`) | Improved Architecture (`ps_1`) |
| :--- | :--- | :--- |
| **Backbone** | Pretrained `facebook/wav2vec2-base` (Frozen) | Pretrained `facebook/wav2vec2-base` (Frozen) |
| **Feature Extraction** | Final layer hidden state only (Layer 12) | **Multi-Layer Aggregation**: Learnable softmax weighting over all 13 transformer layers |
| **Temporal Modeling** | None (Independent frame-by-frame Linear projection) | **2-Layer Bidirectional LSTM** (128-d per direction $\rightarrow$ 256-d output) |
| **Multi-Task Heads** | Single frame classification head | **Dual Heads**: Frame-level fake localization head + Utterance-level spoof detection head |
| **Loss Function** | Standard Binary Cross Entropy (BCE) | **Multi-Task Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{frame}} + \lambda_{\text{utt}} \cdot \mathcal{L}_{\text{utt}}$ with **Focal Loss** for frame boundary imbalance |
| **Post-Processing** | Raw thresholding ($th = 0.5$) | **Temporal Filtering**: 1D Median filtering (140 ms window) + Min-duration pruning (160 ms threshold) |

---

## 📁 Repository Structure

```text
Partial-Spoof/
├── dataset/                     # PartialSpoof dataset directory (git-ignored)
│   └── database/
│       ├── train/               # Training audio (con_wav/)
│       ├── dev/                 # Dev audio (con_wav/)
│       ├── eval/                # Evaluation audio (con_wav/)
│       ├── protocols/           # Protocol label files
│       └── segment_labels/      # Frame-level segment annotations (*.seg_label.npy)
│
├── ps_baseline/                 # 1. Baseline Model Implementation
│   ├── checkpoints/             # Saved model weights (*.pt git-ignored)
│   ├── results/                 # Evaluation outputs (eval_dev.json, training_history.json)
│   ├── src/
│   │   ├── config.py            # Central configuration & dynamic path resolution
│   │   ├── dataset.py           # Dataset and collate implementations
│   │   ├── model.py             # Wav2Vec2 + Linear classification head
│   │   ├── train.py             # Baseline training loop
│   │   ├── evaluate.py          # Frame and segment evaluation
│   │   ├── predict.py           # Single-file inference
│   │   ├── extract_features.py  # Offline feature extraction pipeline
│   │   ├── metrics.py           # Precision, Recall, F1, and IoU functions
│   │   └── visualize.py         # Visualizing predictions vs ground-truth
│   ├── sanity_check.py          # Baseline sanity check
│   └── requirements.txt         # Baseline dependencies
│
├── ps_1/                        # 2. Improved Model Implementation
│   ├── checkpoints/             # Model weights (*.pt git-ignored)
│   ├── results/                 # Evaluation outputs (eval_dev.json, training_history.json)
│   ├── src/
│   │   ├── config.py            # Hyperparameters, paths, and post-processing configs
│   │   ├── dataset.py           # Multi-task dataset with utterance & frame targets
│   │   ├── losses.py            # Binary Focal Loss & MultiTaskLoss
│   │   ├── metrics.py           # Metrics, median filtering, and interval post-processing
│   │   ├── model.py             # Wav2Vec2 + Multi-Layer Weights + BiLSTM + Dual Heads
│   │   ├── train.py             # Training loop with scheduler, checkpoint resume & per-epoch history
│   │   ├── evaluate.py          # Evaluation with median filtering & threshold sweeping
│   │   ├── predict.py           # Single file inference & fake interval extraction
│   │   └── visualize.py         # 5-panel visualization (Waveform, GT, Raw, Smoothed, Pruned)
│   ├── sanity_check.py          # ps_1 sanity check
│   └── requirements.txt         # ps_1 dependencies
│
├── README.md                    # Project documentation
└── .gitignore                   # Ignores dataset, virtual environments, and *.pt weights
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/goutham-siva-10/Partial-Spoof.git
   cd Partial-Spoof
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/macOS
   # or: .\venv\Scripts\activate   # On Windows
   
   pip install -r ps_1/requirements.txt
   ```

3. **Dataset Placement**:
   Place the [PartialSpoof database](https://github.com/PartialSpoof/PartialSpoof) in `dataset/database/` at the workspace root:
   ```text
   dataset/database/
   ├── protocols/
   ├── segment_labels/
   ├── train/
   ├── dev/
   └── eval/
   ```
   *(Note: The `dataset/` directory and large `*.pt` model weights are excluded by `.gitignore` to keep the repository lightweight).*

---

## 🚀 Running `ps_1` (Improved Architecture)

### 1. Verify Pipeline (Sanity Check)
Run an end-to-end check on 8 samples (forward pass, multi-task loss, backward pass, and post-processing):
```bash
python ps_1/sanity_check.py
```

### 2. Train Model
Train with Multi-Task Focal Loss and Cosine Annealing learning rate schedule:
```bash
python ps_1/src/train.py --epochs 10 --batch_size 4 --lr 5e-4 --loss_type focal
```

**Resume Interrupted Training**:
If a training run was interrupted or you wish to extend it, simply pass `--resume`:
```bash
python ps_1/src/train.py --resume
```
*Automatically restores model weights, optimizer state, learning rate scheduler, dev benchmark, and history.*

### 3. Evaluate Model
Evaluate on the development set (`dev`) or evaluation set (`eval`) with raw and post-processed metrics:
```bash
# Standard evaluation on dev set
python ps_1/src/evaluate.py --split dev

# With automatic threshold sweep to find optimal F1 operating point
python ps_1/src/evaluate.py --split dev --sweep_thresholds
```

### 4. Run Single-File Inference
Detect exact fake/spoofed interval timestamps on any audio file:
```bash
# Using a dataset file ID
python ps_1/src/predict.py --file_id CON_D_0000009

# Or using any custom audio file
python ps_1/src/predict.py --wav_path path/to/sample.wav
```
**Example Output**:
```text
=================================================================
Prediction for: CON_D_0000009 (2.35 s)
=================================================================
Utterance Spoof Probability : 100.00%  (SPOOF)
Ground-Truth Fake Intervals : [(0.48, 1.92)]
Raw Model Intervals (th=0.5): [(0.36, 2.02)]
Post-Processed Intervals    : [(0.36, 2.02)]
=================================================================
```

### 5. Generate Visualizations
Generate 5-panel plots illustrating raw waveform, ground truth, raw frame probabilities, median-filtered probabilities, and pruned fake intervals:
```bash
# Plot 4 random samples from dev set
python ps_1/src/visualize.py --split dev --num_examples 4

# Or plot a specific file
python ps_1/src/visualize.py --file_id CON_D_0000009
```
*Generated plots are saved into `ps_1/results/plots/`.*

---

## 🔬 Running `ps_baseline` (Baseline Reference)

For comparison and benchmarking against the standard baseline:

```bash
# 1. Run Baseline Sanity Check
python ps_baseline/sanity_check.py

# 2. Train Baseline Model
python ps_baseline/src/train.py --epochs 10 --batch_size 8 --lr 1e-4

# 3. Evaluate Baseline
python ps_baseline/src/evaluate.py --split dev

# 4. Predict on a Single File
python ps_baseline/src/predict.py --file_id CON_D_0000009
```

---

## 📜 Citations & References

- **PartialSpoof Dataset & Protocols**:
  - Lin Zhang, Xin Wang, Erica Cooper, Nicholas Evans, Junichi Yamagishi. *"The PartialSpoof Database and Countermeasures for the Detection of Partially Spoofed Audio"*, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
- **Wav2Vec 2.0**:
  - Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli. *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"*, NeurIPS 2020.
