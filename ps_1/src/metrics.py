"""
metrics.py — Metric calculations and temporal post-processing for PartialSpoof
==============================================================================
Includes:
- Pure numpy frame metrics (Precision, Recall, F1, ROC-AUC, Average Precision)
- Segment metrics (Mean IoU, Onset error, Offset error)
- Temporal smoothing: 1D Median filtering and minimum segment duration pruning
"""

import numpy as np


# ─── Frame-level metrics ──────────────────────────────────────────────────────

def _to_binary(arr, threshold=0.5):
    return (np.asarray(arr, dtype=np.float32) >= threshold).astype(np.int32)


def precision_score(targets, preds, threshold=0.5, zero_division=0.0):
    """Precision = TP / (TP + FP)."""
    t = np.asarray(targets, dtype=np.int32)
    p = _to_binary(preds, threshold)
    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    denom = tp + fp
    return tp / denom if denom > 0 else float(zero_division)


def recall_score(targets, preds, threshold=0.5, zero_division=0.0):
    """Recall = TP / (TP + FN)."""
    t = np.asarray(targets, dtype=np.int32)
    p = _to_binary(preds, threshold)
    tp = int(np.sum((p == 1) & (t == 1)))
    fn = int(np.sum((p == 0) & (t == 1)))
    denom = tp + fn
    return tp / denom if denom > 0 else float(zero_division)


def f1_score(targets, preds, threshold=0.5, zero_division=0.0):
    """F1 = 2 * P * R / (P + R)."""
    prec = precision_score(targets, preds, threshold, zero_division)
    rec  = recall_score(targets, preds, threshold, zero_division)
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else float(zero_division)


def roc_auc_score(targets, scores):
    """Area Under the ROC Curve (trapezoidal rule)."""
    t = np.asarray(targets, dtype=np.int32)
    s = np.asarray(scores,  dtype=np.float32)

    n_pos = int(t.sum())
    n_neg = len(t) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(-s)
    t_sorted = t[order]

    tps = np.cumsum(t_sorted)
    fps = np.arange(1, len(t_sorted) + 1) - tps

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trapz_fn(tpr, fpr))


def average_precision_score(targets, scores):
    """Area under the Precision-Recall curve."""
    t = np.asarray(targets, dtype=np.int32)
    s = np.asarray(scores,  dtype=np.float32)

    n_pos = int(t.sum())
    if n_pos == 0:
        return float('nan')

    order   = np.argsort(-s)
    t_sorted = t[order]

    tps     = np.cumsum(t_sorted)
    counts  = np.arange(1, len(t_sorted) + 1)
    prec    = tps / counts
    rec     = tps / n_pos

    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])

    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    ap = float(trapz_fn(prec, rec))
    return abs(ap)


# ─── Post-processing: Median Filter & Pruning ─────────────────────────────────

def median_filter_1d(signal: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply a 1D median filter to remove high-frequency spurious spikes.

    Args:
        signal     : 1-D numpy float array of probabilities (T,)
        kernel_size: odd integer window length (default: 7)
    """
    if kernel_size <= 1 or len(signal) < kernel_size:
        return signal.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    pad_signal = np.pad(signal, (radius, radius), mode='edge')
    output = np.empty_like(signal)

    for i in range(len(signal)):
        output[i] = np.median(pad_signal[i : i + kernel_size])

    return output


def prune_short_intervals(intervals: list[tuple[float, float]], min_duration_s: float = 0.16) -> list[tuple[float, float]]:
    """Remove intervals whose duration is less than min_duration_s."""
    return [iv for iv in intervals if (iv[1] - iv[0]) >= min_duration_s]


# ─── Segment-level utilities ──────────────────────────────────────────────────

def frames_to_intervals(
    binary: np.ndarray,
    frame_stride_s: float = 0.020,
) -> list[tuple[float, float]]:
    """Convert binary frame array to a list of (start_s, end_s) intervals.

    Args:
        binary        : 1-D boolean or int array (1 = fake, 0 = real)
        frame_stride_s: seconds between consecutive frames (default: 0.02s)
    """
    intervals = []
    in_fake = False
    start_s = 0.0

    for t, v in enumerate(binary):
        if v and not in_fake:
            start_s = t * frame_stride_s
            in_fake = True
        elif not v and in_fake:
            intervals.append((round(start_s, 4), round(t * frame_stride_s, 4)))
            in_fake = False

    if in_fake:
        intervals.append((round(start_s, 4), round(len(binary) * frame_stride_s, 4)))

    return intervals


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Compute Intersection-over-Union between two (start, end) time intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def segment_metrics(
    gt_intervals: list[tuple[float, float]],
    pred_intervals: list[tuple[float, float]],
    iou_threshold: float = 0.3,
) -> dict:
    """Compute segment-level metrics (Mean IoU, onset error, offset error)."""
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


def post_process_predictions(
    probs: np.ndarray,
    threshold: float = 0.5,
    kernel_size: int = 7,
    min_duration_s: float = 0.16,
    frame_stride_s: float = 0.020,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Apply median filter smoothing and duration pruning to raw probabilities.

    Returns:
        smoothed_probs: 1-D array of smoothed probabilities
        intervals     : list of (start_s, end_s) detected fake intervals
    """
    if kernel_size > 1:
        smoothed_probs = median_filter_1d(probs, kernel_size)
    else:
        smoothed_probs = probs.copy()

    binary = (smoothed_probs >= threshold).astype(np.int32)
    intervals = frames_to_intervals(binary, frame_stride_s=frame_stride_s)

    if min_duration_s > 0.0:
        intervals = prune_short_intervals(intervals, min_duration_s=min_duration_s)

    return smoothed_probs, intervals
