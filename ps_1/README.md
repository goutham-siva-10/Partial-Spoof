# PartialSpoof ps_1: Improved Architecture

An upgraded deepfake speech localization architecture building upon the baseline.

## Improvements Implemented

1. **Temporal Sequence Modeling (BiLSTM)**:
   - Replaced the single frame-by-frame linear layer with a 2-layer Bidirectional LSTM (`hidden_dim=128`, bidirectional=256-d output).
   - Enables the network to detect temporal transitions, pitch jumps, and boundary phase inconsistencies across consecutive frames.

2. **Multi-Layer Feature Aggregation**:
   - Rather than relying solely on the final layer (which is specialized for ASR phonetics), `ps_1` extracts all 13 hidden states from Wav2Vec2 and computes a learnable softmax-weighted combination.

3. **Multi-Task Learning**:
   - Dual-head objective: Frame-level segmentation + Utterance-level spoof detection.
   - $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{frame}} + \lambda_{\text{utt}} \cdot \mathcal{L}_{\text{utt}}$.

4. **Focal Loss for Boundary Imbalance**:
   - `BinaryFocalLossWithLogits` handles class imbalance and down-weights easy frames, focusing gradients on hard boundary frames.

5. **Temporal Post-Processing**:
   - 1D Median filtering (kernel size = 7 frames / ~140ms) removes isolated single-frame false alarms.
   - Minimum segment duration pruning (discards detected fake intervals shorter than 160ms).
   - Threshold sweep option during evaluation.

## Directory Structure

```text
ps_1/
├── checkpoints/             # Trained models (*.pt ignored by git)
├── results/                 # Evaluation outputs and JSON history
├── src/
│   ├── config.py            # Central parameters & dynamic path resolution
│   ├── dataset.py           # Dataset & Collate supporting utterance targets
│   ├── losses.py            # Focal Loss & MultiTaskLoss
│   ├── metrics.py           # Frame/segment metrics & median filtering
│   ├── model.py             # PartialSpoofBiLSTM
│   ├── train.py             # Training loop with scheduler & focal loss
│   ├── evaluate.py          # Evaluation with post-processing & threshold sweep
│   ├── predict.py           # Single file inference & ground-truth comparison
│   └── visualize.py         # 5-panel visualization
├── sanity_check.py          # Quick 8-sample pipeline verification
└── requirements.txt
```

## Quick Start

1. **Run Sanity Check**:
   ```bash
   python ps_1/sanity_check.py
   ```

2. **Train Model**:
   ```bash
   python ps_1/src/train.py --epochs 10 --batch_size 8 --lr 5e-4 --loss_type focal
   ```

3. **Evaluate with Post-Processing**:
   ```bash
   python ps_1/src/evaluate.py --split dev --sweep_thresholds
   ```

4. **Predict Single File**:
   ```bash
   python ps_1/src/predict.py --file_id CON_D_0000009
   ```
