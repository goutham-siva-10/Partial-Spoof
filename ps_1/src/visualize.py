"""
visualize.py — Visualize ps_1 model predictions with post-processing
====================================================================

Generates a 5-panel figure for each file:
    Panel 1: Raw audio waveform
    Panel 2: Ground-truth fake intervals (binary bar)
    Panel 3: Raw frame probabilities P(fake)
    Panel 4: Smoothed probabilities (Median filtered) with threshold line
    Panel 5: Post-processed predicted fake intervals (after duration pruning)

Saves PNGs to ps_1/results/plots/
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import soundfile as sf
import torchaudio.functional as AF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_ROOT,
    SEG_LABEL_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
    THRESHOLD,
    WAV2VEC_STRIDE_S,
    MEDIAN_FILTER_KERNEL,
    MIN_SEGMENT_DURATION_S,
)
from model import PartialSpoofBiLSTM
from metrics import post_process_predictions, frames_to_intervals


def plot_example(
    model: PartialSpoofBiLSTM,
    file_id: str,
    wav_path: str,
    ann: np.ndarray,
    device: torch.device,
    threshold: float,
    kernel_size: int,
    min_duration: float,
    save_path: str,
):
    # 1. Load audio
    data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T).mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = AF.resample(waveform, sr, SAMPLE_RATE)
    duration_s = waveform.shape[1] / SAMPLE_RATE
    time_audio = np.linspace(0, duration_s, waveform.shape[1])

    # 2. Forward pass
    with torch.no_grad():
        frame_logits, utt_logit = model(input_values=waveform.to(device))
        probs = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()
        utt_prob = torch.sigmoid(utt_logit).item()

    smooth_probs, pred_intervals = post_process_predictions(
        probs,
        threshold=threshold,
        kernel_size=kernel_size,
        min_duration_s=min_duration,
        frame_stride_s=WAV2VEC_STRIDE_S,
    )
    time_frames = np.arange(len(probs)) * WAV2VEC_STRIDE_S

    # 3. Ground Truth intervals
    gt_intervals = []
    if ann is not None:
        in_fake = False
        st = 0.0
        for i, val in enumerate(ann):
            is_fake = (val == '0')
            if is_fake and not in_fake:
                st = i * 0.16
                in_fake = True
            elif not is_fake and in_fake:
                gt_intervals.append((st, i * 0.16))
                in_fake = False
        if in_fake:
            gt_intervals.append((st, len(ann) * 0.16))

    # 4. Plot 5 Panels
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 2, 2, 1]})

    fig.suptitle(
        f"PartialSpoof ps_1 Analysis: {file_id}\n"
        f"Utterance Spoof Probability: {utt_prob*100:.1f}%",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: Waveform
    ax0 = axes[0]
    ax0.plot(time_audio, waveform.squeeze(0).numpy(), color='#336699', lw=0.6)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Audio Waveform", loc='left', fontsize=10)
    ax0.grid(True, alpha=0.3)

    # Panel 2: Ground Truth Fake Bar
    ax1 = axes[1]
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_ylabel("GT Fake", fontsize=10)
    for iv in gt_intervals:
        ax1.axvspan(iv[0], iv[1], color='#cc0000', alpha=0.7)
    ax1.grid(True, alpha=0.3)

    # Panel 3: Raw P(fake)
    ax2 = axes[2]
    ax2.plot(time_frames, probs, color='#888888', lw=1.0, label='Raw P(fake)')
    ax2.axhline(threshold, color='black', ls='--', lw=0.8, alpha=0.6, label=f'Threshold={threshold}')
    ax2.set_ylabel("P(fake)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Raw Frame Probabilities (BiLSTM)", loc='left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # Panel 4: Smoothed P(fake)
    ax3 = axes[3]
    ax3.plot(time_frames, smooth_probs, color='#e65100', lw=1.5, label=f'Smoothed (Kernel={kernel_size})')
    ax3.axhline(threshold, color='black', ls='--', lw=0.8, alpha=0.6, label=f'Threshold={threshold}')
    ax3.set_ylabel("P(fake)")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Post-Processed Probabilities (Median Filter)", loc='left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)

    # Panel 5: Predicted Intervals Bar
    ax4 = axes[4]
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_ylabel("Predicted", fontsize=10)
    ax4.set_xlabel("Time (seconds)")
    for iv in pred_intervals:
        ax4.axvspan(iv[0], iv[1], color='#00897b', alpha=0.7)
    ax4.grid(True, alpha=0.3)

    plt.xlim(0, duration_s)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def visualize(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PartialSpoofBiLSTM().to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    seg_file = os.path.join(SEG_LABEL_DIR, f"{args.split}_seglab_0.16.npy")
    seg_labels = np.load(seg_file, allow_pickle=True).item() if os.path.isfile(seg_file) else {}

    wav_dir = os.path.join(DATASET_ROOT, args.split, 'con_wav')
    if args.file_id:
        fids = [args.file_id]
    else:
        # Pick N files with mixed real + fake regions
        candidates = [
            fid for fid, ann in seg_labels.items()
            if len(set(ann.tolist())) > 1 and os.path.isfile(os.path.join(wav_dir, f"{fid}.wav"))
        ]
        random.shuffle(candidates)
        fids = candidates[:args.n]

    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    for fid in fids:
        wav_path = os.path.join(wav_dir, f"{fid}.wav")
        if not os.path.isfile(wav_path):
            continue
        save_path = os.path.join(plots_dir, f"{fid}_{args.split}.png")
        plot_example(
            model=model,
            file_id=fid,
            wav_path=wav_path,
            ann=seg_labels.get(fid),
            device=device,
            threshold=args.threshold,
            kernel_size=args.kernel_size,
            min_duration=args.min_duration,
            save_path=save_path,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Visualize ps_1 predictions")
    p.add_argument('--checkpoint',   type=str,   default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--split',        type=str,   default='dev', choices=['train', 'dev', 'eval'])
    p.add_argument('--n', '--num_examples', dest='n', type=int, default=4, help="Number of random samples to visualize")
    p.add_argument('--file_id',      type=str,   default=None)
    p.add_argument('--threshold',    type=float, default=THRESHOLD)
    p.add_argument('--kernel_size',  type=int,   default=MEDIAN_FILTER_KERNEL)
    p.add_argument('--min_duration', type=float, default=MIN_SEGMENT_DURATION_S)
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    visualize(args)
