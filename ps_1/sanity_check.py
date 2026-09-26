"""
sanity_check.py — End-to-end verification for ps_1 pipeline
===========================================================
Runs on 8 files to test:
1. Dataset & Collate (waveforms, attention masks, frame labels, utterance targets)
2. PartialSpoofBiLSTM forward pass (Multi-layer aggregation, BiLSTM, dual heads)
3. MultiTaskLoss calculation (Focal frame loss + Utterance BCE loss)
4. Backward pass (verifies BiLSTM, classifier heads, and layer weights learn; Wav2Vec2 frozen)
5. Post-processing (Median filter + Minimum duration pruning)
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from dataset import PartialSpoofDataset, collate_fn
from model import PartialSpoofBiLSTM
from losses import MultiTaskLoss
from metrics import post_process_predictions

print("=== SANITY CHECK: ps_1 (BiLSTM + Multi-Layer + Multi-Task) ===\n")

# 1. Dataset
print("[1] Dataset (8 train files)")
ds = PartialSpoofDataset('train', max_samples=8)
item = ds[0]
print(f"    file_id      : {item['file_id']}")
print(f"    waveform     : {tuple(item['waveform'].shape)}  dtype={item['waveform'].dtype}")
print(f"    labels       : {tuple(item['labels'].shape)}   dtype={item['labels'].dtype}")
print(f"    utt_target   : {item['utt_target']}  (label='{item['utt_label']}')")
print(f"    T_est        : {item['T_est']}")
print(f"    label values : {item['labels'][:10].tolist()} ...\n")

# 2. DataLoader + collate
print("[2] DataLoader (batch_size=4)")
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
batch = next(iter(loader))
print(f"    waveform       : {tuple(batch['waveform'].shape)}")
print(f"    attention_mask : {tuple(batch['attention_mask'].shape)}")
print(f"    labels         : {tuple(batch['labels'].shape)}")
print(f"    utt_targets    : {batch['utt_targets'].tolist()}")
print(f"    T_ests         : {batch['T_ests'].tolist()}")
print(f"    file_ids       : {batch['file_ids']}\n")

# 3. Model Forward Pass
print("[3] Model forward pass")
model = PartialSpoofBiLSTM()
model.eval()

with torch.no_grad():
    frame_logits, utt_logits = model(
        input_values=batch['waveform'],
        attention_mask=batch['attention_mask'],
    )
print(f"    frame_logits shape : {tuple(frame_logits.shape)}")
print(f"    utt_logits shape   : {tuple(utt_logits.shape)}")
proba = torch.sigmoid(frame_logits)
utt_proba = torch.sigmoid(utt_logits)
print(f"    P(fake frame) range: [{proba.min():.4f}, {proba.max():.4f}]")
print(f"    P(spoof utt) range : [{utt_proba.min():.4f}, {utt_proba.max():.4f}]\n")

# 4. Loss Computation (MultiTaskLoss with Focal Loss)
print("[4] Loss computation (MultiTaskLoss: Focal Loss + Utterance BCE)")
criterion = MultiTaskLoss(loss_type='focal', lambda_utt=0.5)

T_actual = frame_logits.shape[1]
T_label  = batch['labels'].shape[1]
T        = min(T_actual, T_label)

frame_logits_t = frame_logits[:, :T]
labels_t       = batch['labels'][:, :T]
T_ests         = batch['T_ests']

B = frame_logits_t.shape[0]
loss_mask = torch.zeros(B, T, dtype=torch.bool)
for i, t_len in enumerate(T_ests):
    t_real = min(int(t_len.item()), T)
    loss_mask[i, :t_real] = True

total_loss, frame_loss, utt_loss = criterion(
    frame_logits=frame_logits_t,
    frame_targets=labels_t,
    loss_mask=loss_mask,
    utt_logits=utt_logits,
    utt_targets=batch['utt_targets'],
)

print(f"    valid frames mask : {loss_mask.sum().item()} / {B*T}")
print(f"    frame loss (focal): {frame_loss.item():.6f}")
print(f"    utterance loss    : {utt_loss.item():.6f}")
print(f"    total loss        : {total_loss.item():.6f}\n")

# 5. Backward Pass & Gradient Verification
print("[5] Backward pass (verifying trainable vs frozen parameters)")
model.train()
model.wav2vec2.eval()   # backbone remains frozen

fl2, ul2 = model(
    input_values=batch['waveform'],
    attention_mask=batch['attention_mask'],
    loss_mask=loss_mask,
)
fl2 = fl2[:, :T]
loss2, _, _ = criterion(
    frame_logits=fl2,
    frame_targets=labels_t,
    loss_mask=loss_mask,
    utt_logits=ul2,
    utt_targets=batch['utt_targets'],
)
loss2.backward()

# Trainable layers must have valid gradients
bilstm_grad = model.bilstm.weight_ih_l0.grad
layer_w_grad = model.layer_weights.grad
fc_grad = model.frame_classifier[0].weight.grad
uc_grad = model.utt_classifier[0].weight.grad

assert bilstm_grad is not None, "BiLSTM grad is missing!"
assert layer_w_grad is not None, "Layer weights grad is missing!"
assert fc_grad is not None, "Frame classifier grad is missing!"
assert uc_grad is not None, "Utterance classifier grad is missing!"

print(f"    layer_weights.grad norm      : {layer_w_grad.norm():.6f}")
print(f"    bilstm.weight_ih_l0.grad norm: {bilstm_grad.norm():.6f}")
print(f"    frame_classifier[0].grad norm: {fc_grad.norm():.6f}")
print(f"    utt_classifier[0].grad norm  : {uc_grad.norm():.6f}")

# Wav2Vec2 parameters must be None
w2v_param = next(iter(model.wav2vec2.parameters()))
print(f"    wav2vec2 backbone grad       : {w2v_param.grad} (should be None)")
assert w2v_param.grad is None, "Wav2Vec2 parameters must remain frozen!"
print()

# 6. Post-processing verification
print("[6] Post-processing check (Median filter + Duration pruning)")
example_probs = proba[0, :T_ests[0]].cpu().numpy()
smooth_p, intervals = post_process_predictions(
    example_probs,
    threshold=0.5,
    kernel_size=7,
    min_duration_s=0.16,
)
print(f"    Raw length       : {len(example_probs)} frames")
print(f"    Smoothed length  : {len(smooth_p)} frames")
print(f"    Detected intervals: {intervals}")
print()

print("=== ALL CHECKS PASSED FOR ps_1 ===")
