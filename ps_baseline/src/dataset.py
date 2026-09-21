"""
dataset.py -- PartialSpoof Dataset and DataLoader utilities
============================================================

Label encoding (PartialSpoof v1.1 convention):
    ann '0' (spoof/fake)     --> target 1.0  (P(fake) = high)
    ann '1' (bonafide/real)  --> target 0.0  (P(fake) = low)

Label alignment:
    PartialSpoof annotation frame: 160 ms
    Wav2Vec2-base output stride:    20 ms
    --> 1 annotation frame covers 8 consecutive Wav2Vec2 frames.

    For Wav2Vec2 frame t:
        t_sec   = t * 0.020          (start time of this frame)
        ann_idx = int(t_sec / 0.16)  (which annotation frame)
        target  = 1.0 - float(ann[ann_idx])
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_ROOT,
    SEG_LABEL_DIR,
    PROTOCOL_DIR,
    FEATURES_DIR,
    SAMPLE_RATE,
    ANN_FRAME_DUR,
    WAV2VEC_STRIDE_S,
    WAV2VEC_STRIDE_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_T(n_samples: int, stride: int = WAV2VEC_STRIDE_SAMPLES) -> int:
    """Estimate number of Wav2Vec2 output frames for a given input length.

    Uses the same formula the HuggingFace Wav2Vec2 feature extractor uses
    internally. Actual T from the model may differ by 0 or 1 -- always
    reconcile using min(T_actual, T_est) at loss-computation time.
    """
    return (n_samples - 1) // stride + 1


def align_labels_to_frames(
    ann: np.ndarray,
    T: int,
    ann_frame_dur: float = ANN_FRAME_DUR,
    wav2vec_stride_s: float = WAV2VEC_STRIDE_S,
) -> torch.Tensor:
    """Map PartialSpoof annotation frames to Wav2Vec2 output frames.

    Args:
        ann             : 1-D numpy array of strings ('0' or '1').
                          '0' = fake/spoof, '1' = bonafide/real.
        T               : number of Wav2Vec2 output frames.
        ann_frame_dur   : seconds per annotation frame (default 0.16).
        wav2vec_stride_s: seconds per Wav2Vec2 frame   (default 0.020).

    Returns:
        Float32 tensor of shape (T,).
        Values: 1.0 = fake, 0.0 = real.
    """
    labels = np.empty(T, dtype=np.float32)
    for t in range(T):
        t_sec   = t * wav2vec_stride_s
        ann_idx = int(t_sec / ann_frame_dur)
        ann_idx = min(ann_idx, len(ann) - 1)          # clamp at end
        labels[t] = 1.0 - float(ann[ann_idx])         # '0'->1.0, '1'->0.0
    return torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PartialSpoofDataset(Dataset):
    """PyTorch Dataset for the PartialSpoof database.

    Each item is a dict:
        waveform   : (n_samples,)  float32 -- 16 kHz mono audio
        labels     : (T_est,)      float32 -- 1.0=fake / 0.0=real per Wav2Vec2 frame
        file_id    : str
        utt_label  : str  ('spoof' or 'bonafide') -- utterance-level
        n_samples  : int
        T_est      : int  -- estimated Wav2Vec2 output frames

    Args:
        split       : 'train', 'dev', or 'eval'.
        max_samples : if set, truncate dataset to first N files.
                      Useful for quick sanity checks.
    """

    _PROTOCOL_FILES = {
        'train': 'PartialSpoof.LA.cm.train.trl.txt',
        'dev':   'PartialSpoof.LA.cm.dev.trl.txt',
        'eval':  'PartialSpoof.LA.cm.eval.trl.txt',
    }
    _SEG_LABEL_FILES = {
        'train': 'train_seglab_0.16.npy',
        'dev':   'dev_seglab_0.16.npy',
        'eval':  'eval_seglab_0.16.npy',
    }

    def __init__(self, split: str, max_samples: int = None, use_features: bool = False):
        assert split in ('train', 'dev', 'eval'), f"Unknown split: {split!r}"
        self.split   = split
        self.use_features = use_features
        self.wav_dir = os.path.join(DATASET_ROOT, split, 'con_wav')

        # Guard: audio directory must exist
        if not os.path.isdir(self.wav_dir):
            raise FileNotFoundError(
                f"Audio directory not found: {self.wav_dir}\n"
                f"Please extract database_{split}.tar.gz:\n"
                f"  tar -xzf dataset/database_{split}.tar.gz -C dataset/"
            )

        # Guard: segment label file must exist
        seg_label_path = os.path.join(SEG_LABEL_DIR, self._SEG_LABEL_FILES[split])
        if not os.path.isfile(seg_label_path):
            raise FileNotFoundError(
                f"Segment labels not found: {seg_label_path}\n"
                f"Download database_segment_labels.tar.gz from Zenodo record 5112031."
            )

        # Load segment label dict: {file_id -> np.array(['0','1',...], dtype='<U21')}
        print(f"[Dataset] Loading segment labels for {split} ...")
        self.seg_labels: dict = np.load(
            seg_label_path, allow_pickle=True
        ).item()

        # Load protocol file to get file IDs and utterance-level labels
        protocol_path = os.path.join(
            PROTOCOL_DIR, self._PROTOCOL_FILES[split]
        )
        self.samples = []   # list of (file_id, utt_label)
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id   = parts[1]    # e.g. CON_T_0000000 or LA_T_1138215
                utt_label = parts[4]    # 'spoof' or 'bonafide'
                wav_path  = os.path.join(self.wav_dir, file_id + '.wav')
                # Only include files that are actually on disk
                if os.path.isfile(wav_path):
                    self.samples.append((file_id, utt_label))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        n_spoof  = sum(1 for _, l in self.samples if l == 'spoof')
        n_bonaf  = len(self.samples) - n_spoof
        print(
            f"[Dataset] {split}: {len(self.samples)} files "
            f"(spoof={n_spoof}, bonafide={n_bonaf})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        file_id, utt_label = self.samples[idx]
        wav_path = os.path.join(self.wav_dir, file_id + '.wav')

        # -- Load data --------------------------------------------------
        if self.use_features:
            feat_path = os.path.join(FEATURES_DIR, file_id + '.pt')
            features = torch.load(feat_path, weights_only=False)
            T_est = features.shape[0]
            n_samples = (T_est - 1) * WAV2VEC_STRIDE_SAMPLES + 1
            waveform = torch.zeros(0)
        else:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:                          # multi-channel -> mono
                waveform = waveform.mean(0, keepdim=True)
            if sr != SAMPLE_RATE:                              # resample if needed
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            waveform  = waveform.squeeze(0)                    # (n_samples,)
            n_samples = waveform.shape[0]
            features = None

        # -- Get annotation ----------------------------------------------
        if file_id in self.seg_labels:
            ann = self.seg_labels[file_id]   # np.array of '0'/'1' strings
        else:
            # Fallback: assign uniform label from utterance-level key
            # (should not happen if v1.1 labels are loaded correctly)
            ann_val = '1' if utt_label == 'bonafide' else '0'
            T_fb    = estimate_T(n_samples)
            ann     = np.full(T_fb, ann_val, dtype='<U21')

        # -- Align labels to Wav2Vec2 frames -----------------------------
        T_est  = estimate_T(n_samples)
        labels = align_labels_to_frames(ann, T_est)

        return {
            'waveform' : waveform,     # (n_samples,) float32 or dummy
            'features' : features,     # (T_est, 768) float32 or None
            'labels'   : labels,       # (T_est,)     float32
            'file_id'  : file_id,
            'utt_label': utt_label,
            'n_samples': n_samples,
            'T_est'    : T_est,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> dict:
    """Pad variable-length sequences to the longest item in the batch.

    Waveforms are right-padded with zeros.
    Labels are right-padded with 0.0 (= 'real'), so padding frames never
    contribute positively to the fake-frame loss.
    A boolean loss_mask is included so callers can exclude padded frames.

    Returns:
        waveform        : (B, max_n)    float32
        attention_mask  : (B, max_n)    int64   1=real audio, 0=padding
        labels          : (B, max_T)    float32
        T_ests          : (B,)          int64   actual T_est per sample
        file_ids        : list[str]
        utt_labels      : list[str]
    """
    max_n = max(item['n_samples'] for item in batch)
    max_T = max(item['T_est']     for item in batch)

    waveforms  = []
    attn_masks = []
    features_out = []
    labels_out = []
    T_ests     = []

    for item in batch:
        n = item['n_samples']
        T = item['T_est']

        # Pad waveform to max_n
        if item['waveform'].shape[0] > 0:
            waveforms.append(F.pad(item['waveform'], (0, max_n - n)))
            mask = torch.zeros(max_n, dtype=torch.long)
            mask[:n] = 1
            attn_masks.append(mask)
            
        if item['features'] is not None:
            feat = item['features']
            if max_T - T > 0:
                feat = F.pad(feat, (0, 0, 0, max_T - T))
            features_out.append(feat)

        # Pad label vector to max_T with 0.0 (= 'real')
        labels_out.append(F.pad(item['labels'], (0, max_T - T), value=0.0))

        T_ests.append(T)

    return {
        'waveform'      : torch.stack(waveforms) if waveforms else None,
        'attention_mask': torch.stack(attn_masks) if attn_masks else None,
        'features'      : torch.stack(features_out) if features_out else None,
        'labels'        : torch.stack(labels_out),             # (B, max_T)
        'T_ests'        : torch.tensor(T_ests, dtype=torch.long),  # (B,)
        'file_ids'      : [item['file_id']   for item in batch],
        'utt_labels'    : [item['utt_label'] for item in batch],
    }
