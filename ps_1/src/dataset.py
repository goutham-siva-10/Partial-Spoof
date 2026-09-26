"""
dataset.py — PartialSpoof Dataset and DataLoader utilities for ps_1
===================================================================

Label encoding:
    Frame-level:     ann '0' (spoof/fake)     --> 1.0  (P(fake) = high)
                     ann '1' (bonafide/real)  --> 0.0  (P(fake) = low)
    Utterance-level: 'spoof'                  --> 1.0
                     'bonafide'               --> 0.0
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

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


def estimate_T(n_samples: int, stride: int = WAV2VEC_STRIDE_SAMPLES) -> int:
    """Estimate number of Wav2Vec2 output frames for a given input length."""
    return (n_samples - 1) // stride + 1


def align_labels_to_frames(
    ann: np.ndarray,
    T: int,
    ann_frame_dur: float = ANN_FRAME_DUR,
    wav2vec_stride_s: float = WAV2VEC_STRIDE_S,
) -> torch.Tensor:
    """Map PartialSpoof annotation frames (160ms) to Wav2Vec2 frames (20ms)."""
    labels = np.empty(T, dtype=np.float32)
    for t in range(T):
        t_sec   = t * wav2vec_stride_s
        ann_idx = int(t_sec / ann_frame_dur)
        ann_idx = min(ann_idx, len(ann) - 1)
        labels[t] = 1.0 - float(ann[ann_idx])         # '0'->1.0 (fake), '1'->0.0 (real)
    return torch.from_numpy(labels)


def load_audio(wav_path: str, target_sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    """Load audio with soundfile to avoid torchcodec dependency issues on Windows."""
    try:
        data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.squeeze(0), target_sr
    except Exception:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.squeeze(0), target_sr


class PartialSpoofDataset(Dataset):
    """PyTorch Dataset for the PartialSpoof database."""

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
        self.split = split
        self.use_features = use_features
        self.wav_dir = os.path.join(DATASET_ROOT, split, 'con_wav')

        if not os.path.isdir(self.wav_dir):
            raise FileNotFoundError(
                f"Audio directory not found: {self.wav_dir}\n"
                f"Expected directory under {DATASET_ROOT}/{split}/con_wav"
            )

        seg_label_path = os.path.join(SEG_LABEL_DIR, self._SEG_LABEL_FILES[split])
        if not os.path.isfile(seg_label_path):
            raise FileNotFoundError(
                f"Segment labels not found: {seg_label_path}\n"
                f"Expected under {SEG_LABEL_DIR}"
            )

        print(f"[Dataset] Loading segment labels for {split} ...")
        self.seg_labels: dict = np.load(seg_label_path, allow_pickle=True).item()

        protocol_path = os.path.join(PROTOCOL_DIR, self._PROTOCOL_FILES[split])
        self.samples = []
        with open(protocol_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id   = parts[1]
                utt_label = parts[4]    # 'spoof' or 'bonafide'
                wav_path  = os.path.join(self.wav_dir, file_id + '.wav')
                if os.path.isfile(wav_path):
                    self.samples.append((file_id, utt_label))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        n_spoof = sum(1 for _, l in self.samples if l == 'spoof')
        n_bonaf = len(self.samples) - n_spoof
        print(
            f"[Dataset] {split}: {len(self.samples)} files "
            f"(spoof={n_spoof}, bonafide={n_bonaf})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        file_id, utt_label = self.samples[idx]
        wav_path = os.path.join(self.wav_dir, file_id + '.wav')

        if self.use_features:
            feat_path = os.path.join(FEATURES_DIR, file_id + '.pt')
            features = torch.load(feat_path, weights_only=False)
            T_est = features.shape[0]
            n_samples = (T_est - 1) * WAV2VEC_STRIDE_SAMPLES + 1
            waveform = torch.zeros(0)
        else:
            waveform, _ = load_audio(wav_path, target_sr=SAMPLE_RATE)
            n_samples = waveform.shape[0]
            features = None

        if file_id in self.seg_labels:
            ann = self.seg_labels[file_id]
        else:
            ann_val = '1' if utt_label == 'bonafide' else '0'
            T_fb    = estimate_T(n_samples)
            ann     = np.full(T_fb, ann_val, dtype='<U21')

        T_est  = estimate_T(n_samples)
        labels = align_labels_to_frames(ann, T_est)

        # Utterance binary target: 1.0 = spoof, 0.0 = bonafide
        utt_target = 1.0 if utt_label == 'spoof' else 0.0

        return {
            'waveform'  : waveform,
            'features'  : features,
            'labels'    : labels,
            'utt_target': utt_target,
            'file_id'   : file_id,
            'utt_label' : utt_label,
            'n_samples' : n_samples,
            'T_est'     : T_est,
        }


def collate_fn(batch: list) -> dict:
    """Pad variable-length waveforms, labels, and collate batch elements."""
    max_n = max(item['n_samples'] for item in batch)
    max_T = max(item['T_est']     for item in batch)

    waveforms, attn_masks, features_out, labels_out, T_ests, utt_targets = [], [], [], [], [], []

    for item in batch:
        n = item['n_samples']
        T = item['T_est']

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

        labels_out.append(F.pad(item['labels'], (0, max_T - T), value=0.0))
        T_ests.append(T)
        utt_targets.append(item['utt_target'])

    return {
        'waveform'      : torch.stack(waveforms) if waveforms else None,
        'attention_mask': torch.stack(attn_masks) if attn_masks else None,
        'features'      : torch.stack(features_out) if features_out else None,
        'labels'        : torch.stack(labels_out),
        'utt_targets'   : torch.tensor(utt_targets, dtype=torch.float32),
        'T_ests'        : torch.tensor(T_ests, dtype=torch.long),
        'file_ids'      : [item['file_id'] for item in batch],
        'utt_labels'    : [item['utt_label'] for item in batch],
    }
