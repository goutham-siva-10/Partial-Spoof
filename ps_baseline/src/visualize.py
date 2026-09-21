"""
visualize.py -- Visualize frame-level predictions on example files
==================================================================

For each example, produces a 4-panel figure:
    Panel 1 : Raw waveform
    Panel 2 : Ground-truth fake regions (binary bar)
    Panel 3 : Predicted P(fake) curve with threshold line
    Panel 4 : Predicted fake regions (binary bar at threshold)

Usage:
    python src/visualize.py                          # 6 random dev examples
    python src/visualize.py --n 4 --split dev
    python src/visualize.py --file_id CON_T_0000000  # specific file
    python src/visualize.py --checkpoint checkpoints/best_model.pt

Saves PNGs to results/plots/
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')   # headless -- no GUI required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_ROOT, SEG_LABEL_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    SAMPLE_RATE, ANN_FRAME_DUR, WAV2VEC_STRIDE_S,
    THRESHOLD,
)
from dataset import align_labels_to_frames, estimate_T
from model import PartialSpoofBaseline
from evaluate import frames_to_intervals


# ─── Plot one file ────────────────────────────────────────────────────────────

def plot_example(
    model: PartialSpoofBaseline,
    file_id: str,
    wav_path: str,
    ann: np.ndarray,
    device: torch.device,
    threshold: float,
    save_path: str,
):
    """Produce and save a 4-panel figure for one file.

    Args:
        model     : loaded PartialSpoofBaseline (eval mode).
        file_id   : e.g. 'CON_T_0000000'.
        wav_path  : full path to the .wav file.
        ann       : raw annotation array (['0','1',...] strings).
        device    : torch device.
        threshold : P(fake) threshold.
        save_path : output PNG path.
    """
    # ── Load audio ────────────────────────────────────────────────────────
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.squeeze(0)   # (n_samples,)
    n_samples = waveform.shape[0]
    dur_s     = n_samples / SAMPLE_RATE

    # ── Run model ─────────────────────────────────────────────────────────
    with torch.no_grad():
        inp    = waveform.unsqueeze(0).to(device)   # (1, n_samples)
        logits = model(inp)                          # (1, T)
        proba  = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # (T,)

    T = len(proba)

    # ── Align ground-truth labels ─────────────────────────────────────────
    gt_labels = align_labels_to_frames(ann, T).numpy()   # (T,) 1.0=fake

    # ── Time axes ─────────────────────────────────────────────────────────
    t_audio  = np.linspace(0, dur_s, n_samples)
    t_frames = np.arange(T) * WAV2VEC_STRIDE_S

    # ── Compute intervals ─────────────────────────────────────────────────
    gt_intervals   = frames_to_intervals((gt_labels >= 0.5).astype(int))
    pred_intervals = frames_to_intervals((proba     >= threshold).astype(int))
    pred_binary    = (proba >= threshold).astype(float)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 9),
        gridspec_kw={'height_ratios': [2, 0.8, 2, 0.8]},
        sharex=True,
    )
    fig.suptitle(
        f'{file_id}   ({dur_s:.2f}s, {n_samples} samples, {T} Wav2Vec2 frames)',
        fontsize=12, fontweight='bold',
    )

    # Panel 1: Waveform
    ax = axes[0]
    ax.plot(t_audio, waveform.numpy(), color='#4C8BF5', linewidth=0.4, alpha=0.9)
    for s, e in gt_intervals:
        ax.axvspan(s, e, color='#FF4444', alpha=0.15, label='_fake GT')
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_title('Waveform (red shading = GT fake region)', fontsize=9, loc='left')
    ax.set_xlim(0, dur_s)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: Ground-truth label bar
    ax = axes[1]
    ax.fill_between(t_frames, gt_labels, step='post',
                    color='#FF4444', alpha=0.8, linewidth=0)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['real', 'fake'], fontsize=8)
    ax.set_title('Ground-truth labels  (1=fake)', fontsize=9, loc='left')
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 3: P(fake) curve
    ax = axes[2]
    ax.fill_between(t_frames, proba, color='#FF7700', alpha=0.4, linewidth=0)
    ax.plot(t_frames, proba, color='#FF7700', linewidth=1.0)
    ax.axhline(threshold, color='black', linewidth=0.8, linestyle='--',
               label=f'threshold={threshold}')
    # Shade GT fake regions for reference
    for s, e in gt_intervals:
        ax.axvspan(s, e, color='#FF4444', alpha=0.08)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('P(fake)', fontsize=9)
    ax.set_title('Predicted P(fake) per Wav2Vec2 frame', fontsize=9, loc='left')
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 4: Predicted binary bar
    ax = axes[3]
    ax.fill_between(t_frames, pred_binary, step='post',
                    color='#FF7700', alpha=0.8, linewidth=0)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['real', 'fake'], fontsize=8)
    ax.set_title(f'Predicted labels at threshold={threshold}', fontsize=9, loc='left')
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # Legend patches
    legend_handles = [
        mpatches.Patch(color='#FF4444', alpha=0.7, label='GT fake'),
        mpatches.Patch(color='#FF7700', alpha=0.7, label='Predicted fake'),
    ]
    fig.legend(handles=legend_handles, loc='lower right',
               fontsize=8, ncol=2, frameon=False)

    # Interval annotation
    info_lines = []
    info_lines.append(f"GT fake intervals: {gt_intervals}")
    info_lines.append(f"Pred fake intervals: {pred_intervals}")
    fig.text(0.02, 0.01, '\n'.join(info_lines), fontsize=7,
             color='gray', verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def visualize(args: argparse.Namespace):
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PartialSpoof Baseline -- Visualize")
    print(f"{'='*60}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"N examples : {args.n}")
    print(f"Threshold  : {args.threshold}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model = PartialSpoofBaseline().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')}  "
          f"(dev_F1={ckpt.get('dev_f1', float('nan')):.4f})\n")

    # ── Load segment labels ────────────────────────────────────────────────
    seg_label_file = os.path.join(SEG_LABEL_DIR, f'{args.split}_seglab_0.16.npy')
    seg_labels = np.load(seg_label_file, allow_pickle=True).item()

    # ── Pick files ────────────────────────────────────────────────────────
    wav_dir = os.path.join(DATASET_ROOT, args.split, 'con_wav')

    if args.file_id:
        file_ids = [args.file_id]
    else:
        # Pick N random CON_ files that have mixed (partial) fake labels
        all_ids = [
            fid for fid, ann in seg_labels.items()
            if len(set(ann.tolist())) > 1   # mixed real + fake
        ]
        # Filter to files on disk
        all_ids = [
            fid for fid in all_ids
            if os.path.isfile(os.path.join(wav_dir, fid + '.wav'))
        ]
        random.shuffle(all_ids)
        file_ids = all_ids[:args.n]

    print(f"Visualising {len(file_ids)} file(s):\n")

    for file_id in file_ids:
        wav_path = os.path.join(wav_dir, file_id + '.wav')
        if not os.path.isfile(wav_path):
            print(f"  [SKIP] {file_id} -- .wav not found on disk")
            continue

        ann = seg_labels.get(file_id, None)
        if ann is None:
            print(f"  [SKIP] {file_id} -- no annotation found")
            continue

        save_path = os.path.join(
            RESULTS_DIR, 'plots', f'{file_id}_{args.split}.png'
        )
        plot_example(
            model, file_id, wav_path, ann, device,
            threshold=args.threshold, save_path=save_path,
        )

    print(f"\nAll plots saved to {os.path.join(RESULTS_DIR, 'plots')}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Visualize PartialSpoof Baseline')
    p.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--split',      default='dev', choices=['train', 'dev', 'eval'])
    p.add_argument('--n',          type=int,   default=6,
                   help='Number of random examples to visualise')
    p.add_argument('--file_id',    default=None,
                   help='Visualise a specific file (overrides --n)')
    p.add_argument('--threshold',  type=float, default=THRESHOLD)
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    visualize(args)
