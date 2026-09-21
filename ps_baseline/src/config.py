"""
config.py — Central configuration for PartialSpoof Baseline
============================================================
Edit the path constants below to match your system if needed.
"""

import os

# ─── Project and Workspace paths ──────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__)) # ps_baseline/src
PROJECT_ROOT   = os.path.dirname(_HERE)                     # ps_baseline/
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

# ─── Model ────────────────────────────────────────────────────────────────────
WAV2VEC_MODEL = "facebook/wav2vec2-base"
HIDDEN_DIM    = 768     # Wav2Vec2-base last_hidden_state dimension

# ─── Training defaults (overridable via argparse) ─────────────────────────────
SEED       = 42
BATCH_SIZE = 4      # keep small for CPU; increase to 16-32 on GPU
LR         = 1e-3   # Adam LR for the linear classifier only
EPOCHS     = 10
THRESHOLD  = 0.5    # sigmoid threshold for binary frame prediction
