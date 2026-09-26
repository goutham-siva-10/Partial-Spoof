"""
train.py — Train the PartialSpoof BiLSTM model with Multi-Task Loss
===================================================================

Usage:
    python ps_1/src/train.py
    python ps_1/src/train.py --resume            # resumes from ps_1/checkpoints/latest_model.pt
    python ps_1/src/train.py --epochs 10 --batch_size 4 --lr 5e-4 --loss_type focal
    python ps_1/src/train.py --max_samples 200   # quick debug run

Saves:
    ps_1/checkpoints/best_model.pt
    ps_1/checkpoints/latest_model.pt
    ps_1/results/training_history.json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SEED,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    EPOCHS,
    THRESHOLD,
    LAMBDA_UTT,
    FOCAL_GAMMA,
    FOCAL_ALPHA,
)
from dataset import PartialSpoofDataset, collate_fn
from model import PartialSpoofBiLSTM
from losses import MultiTaskLoss
from metrics import precision_score, recall_score, f1_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def frame_metrics(all_preds: np.ndarray, all_targets: np.ndarray, threshold: float = THRESHOLD) -> tuple[float, float, float]:
    p = precision_score(all_targets, all_preds, threshold=threshold)
    r = recall_score(all_targets, all_preds, threshold=threshold)
    f = f1_score(all_targets, all_preds, threshold=threshold)
    return p, r, f


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    device: torch.device,
    is_train: bool = True,
) -> tuple[float, float, float, float]:
    if is_train:
        model.train()
        model.wav2vec2.eval()   # keep backbone frozen
    else:
        model.eval()

    total_loss = 0.0
    n_batches  = 0
    all_preds   = []
    all_targets = []

    pbar = tqdm(loader, desc="  Train" if is_train else "  Eval ", leave=False)
    for batch in pbar:
        waveforms   = batch['waveform'].to(device) if batch['waveform'] is not None else None
        attn_mask   = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
        features    = batch['features'].to(device) if batch['features'] is not None else None
        labels_pad  = batch['labels'].to(device)
        utt_targets = batch['utt_targets'].to(device)
        T_ests      = batch['T_ests']

        if is_train:
            optimizer.zero_grad()

        # Reconcile T
        T_label = labels_pad.shape[1]

        # Initial forward pass to get sequence length
        with torch.set_grad_enabled(is_train):
            # Construct preliminary loss mask
            B = labels_pad.shape[0]
            loss_mask = torch.zeros(B, T_label, dtype=torch.bool, device=device)
            for i, t_len in enumerate(T_ests):
                t_real = min(int(t_len.item()), T_label)
                loss_mask[i, :t_real] = True

            frame_logits, utt_logits = model(
                input_values=waveforms,
                attention_mask=attn_mask,
                features=features,
                loss_mask=loss_mask,
            )

            T_actual = frame_logits.shape[1]
            T = min(T_actual, T_label)

            frame_logits = frame_logits[:, :T]
            labels       = labels_pad[:, :T]
            loss_mask    = loss_mask[:, :T]

            total_b_loss, frame_loss, utt_loss = criterion(
                frame_logits=frame_logits,
                frame_targets=labels,
                loss_mask=loss_mask,
                utt_logits=utt_logits,
                utt_targets=utt_targets,
            )

            if is_train:
                total_b_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

        total_loss += total_b_loss.item()
        n_batches  += 1

        with torch.no_grad():
            proba = torch.sigmoid(frame_logits).cpu().numpy()
            tgts  = labels.cpu().numpy()

        for i, t_len in enumerate(T_ests):
            t_real = min(int(t_len.item()), T)
            all_preds.extend(proba[i, :t_real].tolist())
            all_targets.extend(tgts[i, :t_real].tolist())

        pbar.set_postfix({'loss': f"{total_b_loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)
    all_preds   = np.array(all_preds, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    prec, rec, f1 = frame_metrics(all_preds, all_targets)
    return avg_loss, prec, rec, f1


def train(args: argparse.Namespace):
    set_seed(args.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"PartialSpoof ps_1 (BiLSTM + Multi-Layer + Multi-Task)")
    print(f"{'='*65}")
    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"Loss type  : {args.loss_type}")
    print(f"Lambda Utt : {args.lambda_utt}")
    print(f"Seed       : {args.seed}")
    if args.max_samples:
        print(f"Max samples (debug): {args.max_samples}")
    print()

    # 1. Datasets & Loaders
    train_ds = PartialSpoofDataset('train', max_samples=args.max_samples, use_features=args.use_features)
    dev_ds   = PartialSpoofDataset('dev',   max_samples=args.max_samples, use_features=args.use_features)

    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    print(f"Train batches : {len(train_loader)}")
    print(f"Dev batches   : {len(dev_loader)}\n")

    # 2. Model & Optimizer
    model = PartialSpoofBiLSTM().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = MultiTaskLoss(
        loss_type=args.loss_type,
        lambda_utt=args.lambda_utt,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )

    # 3. Training Loop State & Resume Handling
    best_f1 = 0.0
    history = []
    start_epoch = 1
    history_path = os.path.join(RESULTS_DIR, 'training_history.json')

    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            if not os.path.exists(resume_path) and os.path.exists(os.path.join(CHECKPOINT_DIR, resume_path)):
                resume_path = os.path.join(CHECKPOINT_DIR, resume_path)

        if os.path.exists(resume_path):
            print(f"[Resume] Loading checkpoint from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)

            model.load_state_dict(ckpt['model_state_dict'])

            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"[Resume] Warning: Could not restore optimizer state ({e}).")

            completed_epoch = ckpt.get('epoch', 0)
            start_epoch = completed_epoch + 1

            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    for _ in range(completed_epoch):
                        scheduler.step()
            else:
                for _ in range(completed_epoch):
                    scheduler.step()

            best_f1 = ckpt.get('dev_f1', 0.0)
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            if os.path.exists(best_path):
                try:
                    best_ckpt = torch.load(best_path, map_location='cpu')
                    best_f1 = max(best_f1, best_ckpt.get('dev_f1', 0.0))
                except Exception:
                    pass

            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        prev_hist = json.load(f)
                    if isinstance(prev_hist, list):
                        history = [h for h in prev_hist if h.get('epoch', 0) <= completed_epoch]
                except Exception:
                    history = []

            print(f"[Resume] Resumed from completed Epoch {completed_epoch}.")
            print(f"[Resume] Starting next training run at Epoch {start_epoch}/{args.epochs}.")
            print(f"[Resume] Current best Dev F1: {best_f1*100:.2f}%\n")

            if start_epoch > args.epochs:
                print(f"[Resume] Training is already complete ({completed_epoch}/{args.epochs} epochs).")
                print(f"[Resume] To train further, pass --epochs > {completed_epoch} (e.g., --epochs {completed_epoch + 5}).")
                return
        else:
            print(f"[Resume] Warning: Checkpoint not found at '{resume_path}'. Starting from scratch.\n")
    else:
        latest_path = os.path.join(CHECKPOINT_DIR, 'latest_model.pt')
        if os.path.exists(latest_path):
            print(f"[NOTICE] Found existing checkpoint at {latest_path}.")
            print("[NOTICE] To resume from it, rerun with '--resume'.")
            print("[NOTICE] Starting fresh training from epoch 1 (existing checkpoints will be overwritten)...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch:02d}/{args.epochs:02d}  (LR: {scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_p, train_r, train_f1 = run_epoch(
            model, train_loader, optimizer, criterion, device, is_train=True
        )
        with torch.no_grad():
            dev_loss, dev_p, dev_r, dev_f1 = run_epoch(
                model, dev_loader, optimizer, criterion, device, is_train=False
            )

        scheduler.step()

        print(
            f"  Train -- Loss: {train_loss:.4f}  "
            f"P: {train_p*100:.2f}%  R: {train_r*100:.2f}%  F1: {train_f1*100:.2f}%\n"
            f"  Dev   -- Loss: {dev_loss:.4f}  "
            f"P: {dev_p*100:.2f}%  R: {dev_r*100:.2f}%  F1: {dev_f1*100:.2f}%"
        )

        record = {
            'epoch'     : epoch,
            'train_loss': train_loss,
            'train_f1'  : train_f1,
            'dev_loss'  : dev_loss,
            'dev_p'     : dev_p,
            'dev_r'     : dev_r,
            'dev_f1'    : dev_f1,
        }
        history.append(record)

        latest_path = os.path.join(CHECKPOINT_DIR, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'dev_f1': dev_f1,
            'args': vars(args),
        }, latest_path)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dev_f1': dev_f1,
                'args': vars(args),
            }, best_path)
            print(f"  --> Best model saved (Dev F1 = {best_f1*100:.2f}%)\n")
        else:
            print()

    print(f"Training completed! Best Dev F1: {best_f1*100:.2f}%")
    print(f"Checkpoints saved in {CHECKPOINT_DIR}")
    print(f"History saved to {history_path}")


def parse_args(args_list: list[str] | None = None):
    p = argparse.ArgumentParser(description="Train PartialSpoof ps_1")
    p.add_argument('--epochs',       type=int,   default=EPOCHS)
    p.add_argument('--batch_size',   type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',           type=float, default=LR)
    p.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    p.add_argument('--loss_type',    type=str,   default='focal', choices=['focal', 'bce'])
    p.add_argument('--lambda_utt',   type=float, default=LAMBDA_UTT)
    p.add_argument('--focal_gamma',  type=float, default=FOCAL_GAMMA)
    p.add_argument('--focal_alpha',  type=float, default=FOCAL_ALPHA)
    p.add_argument('--seed',         type=int,   default=SEED)
    p.add_argument('--max_samples',  type=int,   default=None)
    p.add_argument('--use_features', action='store_true')
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument(
        '--resume',
        nargs='?',
        const=os.path.join(CHECKPOINT_DIR, 'latest_model.pt'),
        default=None,
        help="Resume training from checkpoint. Defaults to ps_1/checkpoints/latest_model.pt if flag is passed without argument.",
    )
    return p.parse_args(args_list)


if __name__ == '__main__':
    args = parse_args()
    train(args)
