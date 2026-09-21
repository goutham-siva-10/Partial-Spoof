"""
train.py -- Train the PartialSpoof baseline
============================================

Usage:
    python src/train.py
    python src/train.py --epochs 5 --batch_size 8 --lr 1e-3
    python src/train.py --max_samples 200   # quick sanity-check run

Saves:
    checkpoints/best_model.pt    -- highest dev F1
    checkpoints/latest_model.pt  -- end of each epoch
    results/training_history.json
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR, RESULTS_DIR,
    SEED, BATCH_SIZE, LR, EPOCHS, THRESHOLD,
)
from dataset import PartialSpoofDataset, collate_fn
from model import PartialSpoofBaseline


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─── Metrics ──────────────────────────────────────────────────────────────────

def frame_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    threshold: float = THRESHOLD,
) -> tuple:
    """Compute frame-level precision, recall, F1 at a fixed threshold.

    Target convention: 1.0 = fake, 0.0 = real.
    'Positive' class = fake.
    """
    prec = precision_score(all_targets, all_preds, threshold=threshold)
    rec  = recall_score(all_targets, all_preds, threshold=threshold)
    f1   = f1_score(all_targets, all_preds, threshold=threshold)
    return float(prec), float(rec), float(f1)


# ─── Single epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    model: PartialSpoofBaseline,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
) -> tuple:
    """Run one full pass over the data.

    Returns: (avg_loss, precision, recall, f1)
    """
    model.classifier.train(is_train)   # only classifier has trainable params
    # wav2vec2 stays in eval mode always (frozen)

    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_targets = []

    desc = "train" if is_train else "eval "
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        waveforms   = batch['waveform'].to(device) if batch['waveform'] is not None else None
        attn_mask   = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
        features    = batch['features'].to(device) if batch['features'] is not None else None
        labels_pad  = batch['labels'].to(device)          # (B, max_T)
        T_ests      = batch['T_ests']                     # (B,) -- on CPU

        if is_train:
            optimizer.zero_grad()

        # ── Forward ───────────────────────────────────────────────────────
        logits = model(input_values=waveforms, attention_mask=attn_mask, features=features)

        # Reconcile T: model may return T_actual = T_est or T_est - 1
        T_actual = logits.shape[1]
        T_label  = labels_pad.shape[1]
        T        = min(T_actual, T_label)
        logits   = logits[:, :T]        # (B, T)
        labels   = labels_pad[:, :T]    # (B, T)

        # ── Loss mask: exclude padded frames ──────────────────────────────
        B = logits.shape[0]
        loss_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, t_len in enumerate(T_ests):
            t_real = min(int(t_len.item()), T)
            loss_mask[i, :t_real] = True

        # ── Loss on real frames only ───────────────────────────────────────
        loss = criterion(logits[loss_mask], labels[loss_mask])

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        # ── Collect predictions for metrics ───────────────────────────────
        with torch.no_grad():
            proba = torch.sigmoid(logits).cpu().numpy()
            tgts  = labels.cpu().numpy()
        for i, t_len in enumerate(T_ests):
            t_real = min(int(t_len.item()), T)
            all_preds.extend(proba[i, :t_real].tolist())
            all_targets.extend(tgts[i,  :t_real].tolist())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(n_batches, 1)
    all_preds   = np.array(all_preds,   dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    prec, rec, f1 = frame_metrics(all_preds, all_targets)
    return avg_loss, prec, rec, f1


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    set_seed(args.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PartialSpoof Baseline -- Training")
    print(f"{'='*60}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR        : {args.lr}")
    print(f"Seed      : {args.seed}")
    print(f"Features  : {'Pre-extracted' if args.use_features else 'Raw audio'}")
    print(f"Workers   : {args.num_workers}")
    if args.max_samples:
        print(f"Max samples (debug): {args.max_samples}")
    print()

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = PartialSpoofDataset('train', max_samples=args.max_samples, use_features=args.use_features)
    dev_ds   = PartialSpoofDataset('dev',   max_samples=args.max_samples, use_features=args.use_features)
    print()

    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kwargs
    )
    dev_loader = DataLoader(
        dev_ds,   batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Dev   batches : {len(dev_loader)}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────
    model     = PartialSpoofBaseline().to(device)
    print()

    # Only the linear classifier has learnable parameters
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────
    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)

        tr_loss, tr_prec, tr_rec, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        print(
            f"  TRAIN  loss={tr_loss:.4f}  "
            f"P={tr_prec:.4f}  R={tr_rec:.4f}  F1={tr_f1:.4f}"
        )

        with torch.no_grad():
            dev_loss, dev_prec, dev_rec, dev_f1 = run_epoch(
                model, dev_loader, criterion, optimizer, device, is_train=False
            )
        print(
            f"  DEV    loss={dev_loss:.4f}  "
            f"P={dev_prec:.4f}  R={dev_rec:.4f}  F1={dev_f1:.4f}"
        )

        rec = {
            'epoch':    epoch,
            'tr_loss':  tr_loss,   'tr_prec':   tr_prec,
            'tr_rec':   tr_rec,    'tr_f1':     tr_f1,
            'dev_loss': dev_loss,  'dev_prec':  dev_prec,
            'dev_rec':  dev_rec,   'dev_f1':    dev_f1,
        }
        history.append(rec)

        # ── Save best checkpoint ───────────────────────────────────────────
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            ckpt = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_f1':               dev_f1,
                'dev_loss':             dev_loss,
                'args':                 vars(args),
            }
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.save(ckpt, best_path)
            print(f"  --> Saved best model  dev_F1={dev_f1:.4f}  -> {best_path}")

        # ── Save latest checkpoint ────────────────────────────────────────
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history':              history,
            'args':                 vars(args),
        }, os.path.join(CHECKPOINT_DIR, 'latest_model.pt'))

    # ── Save history ──────────────────────────────────────────────────────
    hist_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.  Best dev F1 = {best_f1:.4f}")
    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print(f"History     : {hist_path}")
    print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train PartialSpoof Baseline')
    p.add_argument('--epochs',      type=int,   default=EPOCHS,
                   help=f'Number of training epochs (default {EPOCHS})')
    p.add_argument('--batch_size',  type=int,   default=BATCH_SIZE,
                   help=f'Batch size (default {BATCH_SIZE})')
    p.add_argument('--lr',          type=float, default=LR,
                   help=f'Adam learning rate (default {LR})')
    p.add_argument('--seed',        type=int,   default=SEED,
                   help=f'Random seed (default {SEED})')
    p.add_argument('--max_samples', type=int,   default=None,
                   help='Limit number of files per split (for quick testing)')
    p.add_argument('--use_features', action='store_true',
                   help='Use pre-extracted Wav2Vec2 features for much faster training')
    p.add_argument('--num_workers', type=int,   default=0,
                   help='Number of DataLoader workers (e.g. 4 for faster loading)')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
