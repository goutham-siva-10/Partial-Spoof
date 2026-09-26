"""
evaluate.py — Evaluate the trained PartialSpoof BiLSTM model
============================================================

Usage:
    python ps_1/src/evaluate.py
    python ps_1/src/evaluate.py --split dev --checkpoint ps_1/checkpoints/best_model.pt
    python ps_1/src/evaluate.py --threshold 0.5 --kernel_size 7 --min_duration 0.16
    python ps_1/src/evaluate.py --sweep_thresholds

Reports:
    Frame-level   : Precision, Recall, F1, AUC-ROC, Average Precision
    Segment-level : Mean IoU, Mean Onset Error, Mean Offset Error (seconds)
    Comparison    : Raw thresholding vs Post-processed (Median filter + Duration pruning)

Results saved to ps_1/results/eval_{split}.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    THRESHOLD,
    WAV2VEC_STRIDE_S,
    BATCH_SIZE,
    MEDIAN_FILTER_KERNEL,
    MIN_SEGMENT_DURATION_S,
)
from dataset import PartialSpoofDataset, collate_fn, align_labels_to_frames
from model import PartialSpoofBiLSTM
from metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    frames_to_intervals,
    segment_metrics,
    post_process_predictions,
)


def evaluate(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"PartialSpoof ps_1 Evaluation")
    print(f"{'='*65}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Split        : {args.split}")
    print(f"Device       : {device}")
    print(f"Threshold    : {args.threshold}")
    print(f"Post-process : Median Kernel={args.kernel_size}, Min Duration={args.min_duration}s\n")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 1. Dataset & Loader
    dataset = PartialSpoofDataset(args.split, max_samples=args.max_samples, use_features=args.use_features)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # 2. Load Model
    model = PartialSpoofBiLSTM().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    # 3. Inference
    all_raw_probs    = []
    all_smooth_probs = []
    all_targets      = []
    per_file_results = []

    print(f"Evaluating {len(dataset)} files ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            waveforms  = batch['waveform'].to(device) if batch['waveform'] is not None else None
            attn_mask  = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
            features   = batch['features'].to(device) if batch['features'] is not None else None
            labels_pad = batch['labels'].numpy()
            file_ids   = batch['file_ids']
            T_ests     = batch['T_ests'].numpy()

            frame_logits, _ = model(
                input_values=waveforms,
                attention_mask=attn_mask,
                features=features,
            )
            probs_batch = torch.sigmoid(frame_logits).cpu().numpy()

            for i, fid in enumerate(file_ids):
                T = min(probs_batch.shape[1], int(T_ests[i]), labels_pad.shape[1])
                raw_p = probs_batch[i, :T]
                gt_y  = labels_pad[i, :T]

                # Post-processing
                smooth_p, pred_intervals = post_process_predictions(
                    raw_p,
                    threshold=args.threshold,
                    kernel_size=args.kernel_size,
                    min_duration_s=args.min_duration,
                    frame_stride_s=WAV2VEC_STRIDE_S,
                )

                all_raw_probs.extend(raw_p.tolist())
                all_smooth_probs.extend(smooth_p.tolist())
                all_targets.extend(gt_y.tolist())

                # Segment-level evaluation
                gt_binary = (gt_y >= 0.5).astype(np.int32)
                gt_intervals = frames_to_intervals(gt_binary, frame_stride_s=WAV2VEC_STRIDE_S)
                seg_res = segment_metrics(gt_intervals, pred_intervals)

                per_file_results.append({
                    'file_id'       : fid,
                    'gt_intervals'  : gt_intervals,
                    'pred_intervals': pred_intervals,
                    'mean_iou'      : seg_res['mean_iou'],
                    'onset_err'     : seg_res['onset_err'],
                    'offset_err'    : seg_res['offset_err'],
                })

    all_raw_probs    = np.array(all_raw_probs,    dtype=np.float32)
    all_smooth_probs = np.array(all_smooth_probs, dtype=np.float32)
    all_targets      = np.array(all_targets,      dtype=np.float32)

    # 4. Compute Metrics
    # Raw frame metrics
    r_prec = precision_score(all_targets, all_raw_probs, threshold=args.threshold)
    r_rec  = recall_score(   all_targets, all_raw_probs, threshold=args.threshold)
    r_f1   = f1_score(       all_targets, all_raw_probs, threshold=args.threshold)
    auc    = roc_auc_score(  all_targets, all_raw_probs)
    ap     = average_precision_score(all_targets, all_raw_probs)

    # Smoothed frame metrics
    s_prec = precision_score(all_targets, all_smooth_probs, threshold=args.threshold)
    s_rec  = recall_score(   all_targets, all_smooth_probs, threshold=args.threshold)
    s_f1   = f1_score(       all_targets, all_smooth_probs, threshold=args.threshold)

    # Segment metrics
    ious        = [f['mean_iou']   for f in per_file_results if not np.isnan(f['mean_iou'])]
    onset_errs  = [f['onset_err']  for f in per_file_results if not np.isnan(f['onset_err'])]
    offset_errs = [f['offset_err'] for f in per_file_results if not np.isnan(f['offset_err'])]

    mean_iou    = float(np.mean(ious))        if ious else 0.0
    mean_onset  = float(np.mean(onset_errs))  if onset_errs else float('nan')
    mean_offset = float(np.mean(offset_errs)) if offset_errs else float('nan')

    print(f"\n{'-'*65}")
    print(f"Results Summary ({args.split} split, threshold={args.threshold:.2f})")
    print(f"{'-'*65}")
    print(f"Raw Frame Metrics:")
    print(f"  Precision: {r_prec*100:.2f}%  Recall: {r_rec*100:.2f}%  F1: {r_f1*100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}   Average Precision: {ap:.4f}\n")
    print(f"Smoothed Frame Metrics (Median Kernel={args.kernel_size}):")
    print(f"  Precision: {s_prec*100:.2f}%  Recall: {s_rec*100:.2f}%  F1: {s_f1*100:.2f}%\n")
    print(f"Segment Metrics (with duration pruning >= {args.min_duration}s):")
    print(f"  Mean IoU         : {mean_iou:.4f}")
    print(f"  Mean Onset Error : {mean_onset:.4f} s")
    print(f"  Mean Offset Error: {mean_offset:.4f} s")
    print(f"{'-'*65}\n")

    # Optional Threshold Sweep
    if args.sweep_thresholds:
        print("Threshold Sweep on Smoothed Probabilities:")
        print(f"  {'Threshold':<10} {'Precision':<12} {'Recall':<12} {'F1':<10}")
        best_t, best_f1 = args.threshold, s_f1
        for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            p = precision_score(all_targets, all_smooth_probs, threshold=th)
            r = recall_score(all_targets, all_smooth_probs, threshold=th)
            f = f1_score(all_targets, all_smooth_probs, threshold=th)
            print(f"  {th:<10.2f} {p*100:<11.2f}% {r*100:<11.2f}% {f*100:<9.2f}%")
            if f > best_f1:
                best_t, best_f1 = th, f
        print(f"  --> Optimal threshold for F1: {best_t:.2f} (F1 = {best_f1*100:.2f}%)\n")

    # Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"eval_{args.split}.json")
    out_data = {
        'split'        : args.split,
        'threshold'    : args.threshold,
        'checkpoint'   : args.checkpoint,
        'kernel_size'  : args.kernel_size,
        'min_duration' : args.min_duration,
        'frame_raw'    : {'precision': r_prec, 'recall': r_rec, 'f1': r_f1, 'auc': auc, 'ap': ap},
        'frame_smooth' : {'precision': s_prec, 'recall': s_rec, 'f1': s_f1},
        'segment'      : {'mean_iou': mean_iou, 'mean_onset_err': mean_onset, 'mean_offset_err': mean_offset},
    }
    with open(out_file, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved evaluation results to {out_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PartialSpoof ps_1")
    p.add_argument('--checkpoint',   type=str,   default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--split',        type=str,   default='dev', choices=['train', 'dev', 'eval'])
    p.add_argument('--threshold',    type=float, default=THRESHOLD)
    p.add_argument('--kernel_size',  type=int,   default=MEDIAN_FILTER_KERNEL)
    p.add_argument('--min_duration', type=float, default=MIN_SEGMENT_DURATION_S)
    p.add_argument('--batch_size',   type=int,   default=BATCH_SIZE)
    p.add_argument('--max_samples',  type=int,   default=None)
    p.add_argument('--use_features', action='store_true')
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--sweep_thresholds', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
