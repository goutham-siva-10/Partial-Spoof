"""
evaluate.py -- Evaluate a trained PartialSpoof baseline checkpoint
===================================================================

Usage:
    python src/evaluate.py                          # uses best_model.pt on dev
    python src/evaluate.py --split dev
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
    python src/evaluate.py --threshold 0.4

Reports:
    Frame-level : precision, recall, F1 at threshold, AUC-ROC
    Segment-level: mean IoU between predicted and ground-truth fake intervals
                   mean onset error, mean offset error (in seconds)

Results saved to results/eval_{split}.json
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR, RESULTS_DIR, THRESHOLD, WAV2VEC_STRIDE_S, BATCH_SIZE,
)
from dataset import PartialSpoofDataset, collate_fn, align_labels_to_frames
from model import PartialSpoofBaseline


# ─── Segment-level utilities ──────────────────────────────────────────────────

def frames_to_intervals(
    binary: np.ndarray,
    frame_stride_s: float = WAV2VEC_STRIDE_S,
) -> list:
    """Convert binary frame array to list of (start_s, end_s) intervals.

    Args:
        binary        : 1-D bool/int array; 1 = fake.
        frame_stride_s: seconds between consecutive frames.

    Returns:
        List of (start_s, end_s) tuples.
    """
    intervals = []
    in_fake   = False
    for t, v in enumerate(binary):
        if v and not in_fake:
            start_s = t * frame_stride_s
            in_fake = True
        elif not v and in_fake:
            intervals.append((start_s, t * frame_stride_s))
            in_fake = False
    if in_fake:
        intervals.append((start_s, len(binary) * frame_stride_s))
    return intervals


def interval_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (start, end) intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def segment_metrics(
    gt_intervals: list,
    pred_intervals: list,
    iou_threshold: float = 0.3,
) -> dict:
    """Compute segment-level metrics for one utterance.

    Greedy matching: for each GT interval, find the predicted interval with
    highest IoU. Unmatched GT intervals count as misses; unmatched predicted
    intervals count as false alarms.

    Returns dict with keys:
        mean_iou      : mean IoU of matched pairs (0 if no GT)
        onset_err     : mean |pred_onset - gt_onset| in seconds (matched)
        offset_err    : mean |pred_offset - gt_offset| in seconds (matched)
        n_gt          : number of GT fake intervals
        n_pred        : number of predicted fake intervals
        n_matched     : number of matched pairs
    """
    if not gt_intervals:
        return {
            'mean_iou':   float('nan'),
            'onset_err':  float('nan'),
            'offset_err': float('nan'),
            'n_gt':       0,
            'n_pred':     len(pred_intervals),
            'n_matched':  0,
        }

    if not pred_intervals:
        return {
            'mean_iou':   0.0,
            'onset_err':  float('nan'),
            'offset_err': float('nan'),
            'n_gt':       len(gt_intervals),
            'n_pred':     0,
            'n_matched':  0,
        }

    used_pred = set()
    ious, onset_errs, offset_errs = [], [], []

    for gt in gt_intervals:
        best_iou  = -1.0
        best_pidx = -1
        for pidx, pred in enumerate(pred_intervals):
            if pidx in used_pred:
                continue
            iou = interval_iou(gt, pred)
            if iou > best_iou:
                best_iou  = iou
                best_pidx = pidx
        if best_iou >= iou_threshold and best_pidx >= 0:
            used_pred.add(best_pidx)
            pred = pred_intervals[best_pidx]
            ious.append(best_iou)
            onset_errs.append(abs(pred[0] - gt[0]))
            offset_errs.append(abs(pred[1] - gt[1]))

    return {
        'mean_iou':   float(np.mean(ious))        if ious else 0.0,
        'onset_err':  float(np.mean(onset_errs))  if onset_errs else float('nan'),
        'offset_err': float(np.mean(offset_errs)) if offset_errs else float('nan'),
        'n_gt':       len(gt_intervals),
        'n_pred':     len(pred_intervals),
        'n_matched':  len(ious),
    }


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PartialSpoof Baseline -- Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"Threshold  : {args.threshold}")
    print(f"Device     : {device}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model = PartialSpoofBaseline().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(dev_F1={ckpt.get('dev_f1', float('nan')):.4f})\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    ds     = PartialSpoofDataset(args.split, max_samples=args.max_samples, use_features=args.use_features)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds   = []
    all_targets = []
    seg_results = []   # per-file segment metrics

    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', dynamic_ncols=True):
            waveforms  = batch['waveform'].to(device) if batch['waveform'] is not None else None
            attn_mask  = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
            features   = batch['features'].to(device) if batch.get('features') is not None else None
            labels_pad = batch['labels'].to(device)
            T_ests     = batch['T_ests']

            logits = model(input_values=waveforms, attention_mask=attn_mask, features=features)

            T_actual = logits.shape[1]
            T_label  = labels_pad.shape[1]
            T        = min(T_actual, T_label)
            logits   = logits[:, :T]
            labels   = labels_pad[:, :T]

            proba = torch.sigmoid(logits).cpu().numpy()
            tgts  = labels.cpu().numpy()

            for i, t_len in enumerate(T_ests):
                t_real   = min(int(t_len.item()), T)
                p_i      = proba[i, :t_real]
                t_i      = tgts[i,  :t_real]

                all_preds.extend(p_i.tolist())
                all_targets.extend(t_i.tolist())

                gt_bin   = (t_i >= 0.5).astype(int)
                pred_bin = (p_i >= args.threshold).astype(int)
                gt_segs   = frames_to_intervals(gt_bin)
                pred_segs = frames_to_intervals(pred_bin)
                smx = segment_metrics(gt_segs, pred_segs)
                smx['file_id'] = batch['file_ids'][i]
                smx['gt_intervals'] = gt_segs
                smx['pred_intervals'] = pred_segs
                seg_results.append(smx)

    # ── Frame-level metrics ───────────────────────────────────────────────
    all_preds   = np.array(all_preds,   dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    binary_pred = (all_preds >= args.threshold).astype(int)
    binary_tgt  =  all_targets.astype(int)

    prec   = float(precision_score(binary_tgt, all_preds, threshold=args.threshold))
    rec    = float(recall_score(   binary_tgt, all_preds, threshold=args.threshold))
    f1     = float(f1_score(       binary_tgt, all_preds, threshold=args.threshold))
    auc    = float(roc_auc_score(  binary_tgt, all_preds))
    ap     = float(average_precision_score(binary_tgt, all_preds))

    # ── Segment-level aggregate ───────────────────────────────────────────
    valid_ious  = [s['mean_iou']   for s in seg_results
                   if not np.isnan(s['mean_iou'])]
    valid_onset = [s['onset_err']  for s in seg_results
                   if not np.isnan(s.get('onset_err', float('nan')))]
    valid_off   = [s['offset_err'] for s in seg_results
                   if not np.isnan(s.get('offset_err', float('nan')))]

    mean_iou       = float(np.mean(valid_ious))  if valid_ious  else float('nan')
    mean_onset_err = float(np.mean(valid_onset)) if valid_onset else float('nan')
    mean_off_err   = float(np.mean(valid_off))   if valid_off   else float('nan')

    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FRAME-LEVEL METRICS  (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1                : {f1:.4f}")
    print(f"  AUC-ROC           : {auc:.4f}")
    print(f"  Average Precision : {ap:.4f}")
    print(f"  Total frames      : {len(all_targets):,}")
    print(f"  Fake frames (GT)  : {binary_tgt.sum():,}  ({100*binary_tgt.mean():.1f}%)")
    print(f"  Fake frames (pred): {binary_pred.sum():,}  ({100*binary_pred.mean():.1f}%)")

    print(f"\n{'='*60}")
    print(f"SEGMENT-LEVEL METRICS")
    print(f"{'='*60}")
    print(f"  Mean IoU          : {mean_iou:.4f}")
    print(f"  Mean onset error  : {mean_onset_err:.4f} s")
    print(f"  Mean offset error : {mean_off_err:.4f} s")
    print(f"  Files evaluated   : {len(seg_results)}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'split':     args.split,
        'threshold': args.threshold,
        'checkpoint': args.checkpoint,
        'frame': {
            'precision': prec, 'recall': rec, 'f1': f1,
            'auc_roc': auc, 'avg_precision': ap,
            'n_frames': len(all_targets),
            'n_fake_gt':   int(binary_tgt.sum()),
            'n_fake_pred': int(binary_pred.sum()),
        },
        'segment': {
            'mean_iou':        mean_iou,
            'mean_onset_err':  mean_onset_err,
            'mean_offset_err': mean_off_err,
            'n_files':         len(seg_results),
        },
        'per_file': seg_results,
    }
    out_path = os.path.join(RESULTS_DIR, f'eval_{args.split}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate PartialSpoof Baseline')
    p.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--split',      default='dev', choices=['train', 'dev', 'eval'])
    p.add_argument('--threshold',  type=float, default=THRESHOLD)
    p.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    p.add_argument('--max_samples',type=int,   default=None)
    p.add_argument('--use_features', action='store_true')
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
