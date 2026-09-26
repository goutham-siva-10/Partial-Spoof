"""
config.py — Central configuration for PartialSpoof Improved System (ps_1)
========================================================================
Features:
- BiLSTM temporal modeling
- Multi-layer Wav2Vec2 feature aggregation
- Multi-task learning (Frame-level + Utterance-level)
- Temporal post-processing (Median filter + Min-duration pruning)
"""

import os

# ─── Project and Workspace paths ──────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__)) # ps_1/src
PROJECT_ROOT   = os.path.dirname(_HERE)                     # ps_1/
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)              # Partial Spoof/

# ─── Dataset paths ────────────────────────────────────────────────────────────
# Dataset is located in the main Partial Spoof folder: Partial Spoof/dataset/database
DATASET_DIR   = os.environ.get(
    "PARTIALSPOOF_DATASET_DIR",
    os.path.join(WORKSPACE_ROOT, "dataset")
)
DATASET_ROOT  = os.environ.get(
    "PARTIALSPOOF_DATASET_ROOT",
    os.path.join(DATASET_DIR, "database")
)
SEG_LABEL_DIR = os.path.join(DATASET_ROOT, "segment_labels")
PROTOCOL_DIR  = os.path.join(
    DATASET_ROOT, "protocols", "PartialSpoof_LA_cm_protocols"
)
FEATURES_DIR  = os.path.join(DATASET_ROOT, "features")

# ─── Project output paths ─────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000   # Hz -- PartialSpoof is 16 kHz

# ─── Label alignment ──────────────────────────────────────────────────────────
# PartialSpoof v1.1 segment labels: 160 ms per frame, no overlap
ANN_FRAME_DUR = 0.16    # seconds per annotation frame

# Wav2Vec2-base CNN feature extractor: effective stride = 320 samples at 16 kHz
WAV2VEC_STRIDE_SAMPLES = 320
WAV2VEC_STRIDE_S       = WAV2VEC_STRIDE_SAMPLES / SAMPLE_RATE   # 0.020 s = 20 ms

# ─── Model architecture ───────────────────────────────────────────────────────
WAV2VEC_MODEL     = "facebook/wav2vec2-base"
HIDDEN_DIM        = 768     # Wav2Vec2-base hidden state dimension
NUM_HIDDEN_LAYERS = 13      # 1 embedding + 12 transformer encoder layers
LSTM_HIDDEN       = 128     # BiLSTM hidden dim per direction (total = 256)
LSTM_LAYERS       = 2       # BiLSTM number of recurrent layers
LSTM_DROPOUT      = 0.2

# ─── Multi-task & Training defaults ───────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 4      # batch size (works well on CPU or GPU)
LR           = 5e-4   # AdamW learning rate for BiLSTM + Classifier
WEIGHT_DECAY = 1e-4
EPOCHS       = 10
THRESHOLD    = 0.5    # base sigmoid threshold for binary frame prediction

# Multi-task loss weight: Total Loss = Frame_Loss + LAMBDA_UTT * Utt_Loss
LAMBDA_UTT   = 0.5

# Focal loss parameters for frame prediction
FOCAL_GAMMA  = 2.0
FOCAL_ALPHA  = 0.65   # balance weight for fake class

# ─── Post-processing defaults ─────────────────────────────────────────────────
MEDIAN_FILTER_KERNEL   = 7      # 7 frames * 20ms = 140ms filter window
MIN_SEGMENT_DURATION_S = 0.16   # discard detected fake regions shorter than 160ms
