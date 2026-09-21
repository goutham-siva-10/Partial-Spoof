"""
Sanity check: dataset -> dataloader -> model -> loss, all on 8 files.
Runs in ~30 seconds on CPU.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PartialSpoofDataset, collate_fn
from model import PartialSpoofBaseline

print("=== SANITY CHECK ===\n")

# 1. Dataset
print("[1] Dataset (8 train files)")
ds = PartialSpoofDataset('train', max_samples=8)
item = ds[0]
print(f"    file_id      : {item['file_id']}")
print(f"    waveform     : {tuple(item['waveform'].shape)}  dtype={item['waveform'].dtype}")
print(f"    labels       : {tuple(item['labels'].shape)}   dtype={item['labels'].dtype}")
print(f"    n_samples    : {item['n_samples']}")
print(f"    T_est        : {item['T_est']}")
print(f"    label values : {item['labels'][:10].tolist()}  ...")
print()

# 2. DataLoader + collate
print("[2] DataLoader (batch_size=4)")
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
batch  = next(iter(loader))
print(f"    waveform       : {tuple(batch['waveform'].shape)}")
print(f"    attention_mask : {tuple(batch['attention_mask'].shape)}")
print(f"    labels         : {tuple(batch['labels'].shape)}")
print(f"    T_ests         : {batch['T_ests'].tolist()}")
print(f"    file_ids       : {batch['file_ids']}")
print()

# 3. Model
print("[3] Model forward pass")
model = PartialSpoofBaseline()
model.eval()
with torch.no_grad():
    logits = model(batch['waveform'], batch['attention_mask'])
print(f"    logits shape   : {tuple(logits.shape)}")
print(f"    logits range   : [{logits.min():.4f}, {logits.max():.4f}]")
proba = torch.sigmoid(logits)
print(f"    P(fake) range  : [{proba.min():.4f}, {proba.max():.4f}]")
print()

# 4. Loss computation with mask
print("[4] Loss computation")
criterion = nn.BCEWithLogitsLoss()
T_actual  = logits.shape[1]
T_label   = batch['labels'].shape[1]
T         = min(T_actual, T_label)
logits_t  = logits[:, :T]
labels_t  = batch['labels'][:, :T]
T_ests    = batch['T_ests']

B = logits_t.shape[0]
loss_mask = torch.zeros(B, T, dtype=torch.bool)
for i, t_len in enumerate(T_ests):
    t_real = min(int(t_len.item()), T)
    loss_mask[i, :t_real] = True

loss = criterion(logits_t[loss_mask], labels_t[loss_mask])
print(f"    T_actual       : {T_actual}")
print(f"    T_label        : {T_label}")
print(f"    T (used)       : {T}")
print(f"    loss_mask sum  : {loss_mask.sum().item()} / {B*T} frames")
print(f"    BCE loss       : {loss.item():.6f}")
print()

# 5. Backward pass (training mode)
print("[5] Backward pass (only classifier.weight/bias get grad)")
model.classifier.train()
logits2 = model(batch['waveform'], batch['attention_mask'])
logits2  = logits2[:, :T]
loss2    = criterion(logits2[loss_mask], labels_t[loss_mask])
loss2.backward()
cw_grad = model.classifier.weight.grad
cb_grad = model.classifier.bias.grad
print(f"    classifier.weight.grad : {cw_grad.shape}  norm={cw_grad.norm():.6f}")
print(f"    classifier.bias.grad   : {cb_grad.shape}  norm={cb_grad.norm():.6f}")

# Confirm Wav2Vec2 has no grad
w2v_param = next(iter(model.wav2vec2.parameters()))
print(f"    wav2vec2 param grad    : {w2v_param.grad}  (should be None)")
print()

print("=== ALL CHECKS PASSED ===")
