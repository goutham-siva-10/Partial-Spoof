"""
metrics.py -- Pure-numpy metric implementations
================================================
Replaces scikit-learn to avoid the sklearn -> pandas -> pyarrow -> numpy
binary-incompatibility chain that plagues global Python environments.

All functions accept numpy arrays or lists of floats.
Positive class = 1  (fake frames).
"""

import numpy as np


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
    rec  = recall_score(   targets, preds, threshold, zero_division)
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else float(zero_division)


def roc_auc_score(targets, scores):
    """Area Under the ROC Curve (trapezoidal rule).

    Works for any threshold range by sorting scores descending and
    computing the piecewise TPR/FPR curve.
    """
    t = np.asarray(targets, dtype=np.int32)
    s = np.asarray(scores,  dtype=np.float32)

    n_pos = int(t.sum())
    n_neg = len(t) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    # Sort by score descending
    order = np.argsort(-s)
    t_sorted = t[order]

    tps = np.cumsum(t_sorted)
    fps = np.arange(1, len(t_sorted) + 1) - tps

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration
    auc = float(np.trapz(tpr, fpr))
    return auc


def average_precision_score(targets, scores):
    """Area under the Precision-Recall curve (interpolated, trapezoidal)."""
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

    # Prepend (recall=0, precision=1)
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])

    ap = float(np.trapz(prec, rec))
    # trapz over decreasing x gives a negative value -- take abs
    return abs(ap)
