# Partial Spoof Detection Baseline

Baseline model and evaluation framework for PartialSpoof (detecting partially spoofed/deepfake audio).

## Project Structure

```text
Partial Spoof/
├── dataset/                     # PartialSpoof dataset (git-ignored)
│   └── database/
│       ├── dev/
│       ├── train/
│       ├── protocols/
│       ├── segment_labels/
│       └── features/
├── ps_baseline/
│   ├── checkpoints/             # Trained model checkpoints (*.pt git-ignored)
│   ├── results/                 # Evaluation outputs and metrics
│   ├── src/
│   │   ├── config.py            # Central configuration & path resolution
│   │   ├── dataset.py           # PyTorch Dataset and DataLoader
│   │   ├── model.py             # Frozen Wav2Vec2 + trainable linear classifier
│   │   ├── train.py             # Training loop
│   │   ├── evaluate.py          # Evaluation script (frame- and segment-level metrics)
│   │   ├── predict.py           # Single file inference & comparison
│   │   ├── extract_features.py  # Feature extraction pipeline
│   │   ├── metrics.py           # Precision, recall, and F1 calculations
│   │   └── visualize.py         # Visualizing predictions vs ground-truth
│   ├── sanity_check.py          # Quick sanity test on sample data
│   └── requirements.txt         # Dependencies
└── .gitignore
```

## Setup & Usage

1. **Environment**:
   ```bash
   pip install -r ps_baseline/requirements.txt
   ```

2. **Dataset Placement**:
   Place the PartialSpoof dataset in `dataset/database/` at the root of this project:
   - `dataset/database/segment_labels`
   - `dataset/database/protocols`
   - `dataset/database/train`
   - `dataset/database/dev`

3. **Run Sanity Check**:
   ```bash
   python ps_baseline/sanity_check.py
   ```

4. **Train Model**:
   ```bash
   python ps_baseline/src/train.py --epochs 10 --batch_size 8
   ```

5. **Evaluate Model**:
   ```bash
   python ps_baseline/src/evaluate.py --split dev
   ```
