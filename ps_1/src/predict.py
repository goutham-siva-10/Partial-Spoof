"""
predict.py — Predict spoof segments for an audio file using ps_1 model
======================================================================

Usage:
    python ps_1/src/predict.py
    python ps_1/src/predict.py --file_id CON_D_0000009
    python ps_1/src/predict.py --wav_path path/to/any_audio.wav
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_ROOT,
    SEG_LABEL_DIR,
    CHECKPOINT_DIR,
    SAMPLE_RATE,
    THRESHOLD,
    MEDIAN_FILTER_KERNEL,
    MIN_SEGMENT_DURATION_S,
    WAV2VEC_STRIDE_S,
)
from model import PartialSpoofBiLSTM
from metrics import post_process_predictions, frames_to_intervals


def predict(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Resolve Audio Path
    if args.wav_path:
        wav_path = args.wav_path
        file_id = os.path.splitext(os.path.basename(wav_path))[0]
    elif args.file_id:
        file_id = args.file_id
        wav_path = os.path.join(DATASET_ROOT, args.split, 'con_wav', f"{file_id}.wav")
    else:
        file_id = "CON_D_0000009"
        wav_path = os.path.join(DATASET_ROOT, "dev", "con_wav", f"{file_id}.wav")

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # 2. Load Model
    model = PartialSpoofBiLSTM().to(device)
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first using: python ps_1/src/train.py"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    # 3. Load & Preprocess Audio
    data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T).mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = AF.resample(waveform, sr, SAMPLE_RATE)

    # 4. Model Inference
    with torch.no_grad():
        frame_logits, utt_logit = model(input_values=waveform.to(device))
        probs = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()
        utt_prob = torch.sigmoid(utt_logit).item()

    # Raw predictions
    raw_binary = (probs >= args.threshold).astype(int)
    raw_intervals = frames_to_intervals(raw_binary, frame_stride_s=WAV2VEC_STRIDE_S)

    # Smoothed predictions (Median filter + Min duration pruning)
    _, smooth_intervals = post_process_predictions(
        probs,
        threshold=args.threshold,
        kernel_size=args.kernel_size,
        min_duration_s=args.min_duration,
        frame_stride_s=WAV2VEC_STRIDE_S,
    )

    # 5. Load Ground Truth if available
    gt_intervals = []
    seg_file = os.path.join(SEG_LABEL_DIR, f"{args.split}_seglab_0.16.npy")
    if os.path.isfile(seg_file):
        try:
            seg_labels = np.load(seg_file, allow_pickle=True).item()
            if file_id in seg_labels:
                ann = seg_labels[file_id]
                in_fake = False
                st = 0.0
                for i, lab in enumerate(ann):
                    is_fake = (lab == '0')
                    if is_fake and not in_fake:
                        st = i * 0.16
                        in_fake = True
                    elif not is_fake and in_fake:
                        gt_intervals.append((round(st, 3), round(i * 0.16, 3)))
                        in_fake = False
                if in_fake:
                    gt_intervals.append((round(st, 3), round(len(ann) * 0.16, 3)))
        except Exception:
            pass

    # 6. Output Comparison
    duration_s = waveform.shape[1] / SAMPLE_RATE
    print(f"\n{'='*65}")
    print(f"Prediction for: {file_id} ({duration_s:.2f} s)")
    print(f"{'='*65}")
    print(f"Utterance Spoof Probability : {utt_prob*100:.2f}%  ({'SPOOF' if utt_prob >= 0.5 else 'BONAFIDE'})")
    if gt_intervals:
        print(f"Ground-Truth Fake Intervals : {gt_intervals}")
    print(f"Raw Model Intervals (th={args.threshold}): {raw_intervals}")
    print(f"Post-Processed Intervals    : {smooth_intervals}")
    print(f"{'='*65}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Predict spoof intervals for an audio file")
    p.add_argument('--file_id',      type=str,   default=None)
    p.add_argument('--wav_path',     type=str,   default=None)
    p.add_argument('--split',        type=str,   default='dev', choices=['train', 'dev', 'eval'])
    p.add_argument('--checkpoint',   type=str,   default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--threshold',    type=float, default=THRESHOLD)
    p.add_argument('--kernel_size',  type=int,   default=MEDIAN_FILTER_KERNEL)
    p.add_argument('--min_duration', type=float, default=MIN_SEGMENT_DURATION_S)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
