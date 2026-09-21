import os
import sys
import torch
import numpy as np
import soundfile as sf
import torchaudio.functional as AF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_ROOT, SEG_LABEL_DIR, CHECKPOINT_DIR
from model import PartialSpoofBaseline

# 1. Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PartialSpoofBaseline().to(device)
ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

file_id = "CON_D_0000009"
wav_path = os.path.join(DATASET_ROOT, "dev", "con_wav", f"{file_id}.wav")

# 2. Predict using the model
data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
waveform = torch.from_numpy(data.T).mean(0, keepdim=True)
if sr != 16000:
    waveform = AF.resample(waveform, sr, 16000)

with torch.no_grad():
    pred_timestamps = model.predict_segments(waveform.to(device), threshold=0.5)

# 3. Load the ground truth from the dataset
# (PartialSpoof annotations are every 160ms, where '0' = fake and '1' = real)
npy_path = os.path.join(SEG_LABEL_DIR, "dev_seglab_0.16.npy")
seg_labels = np.load(npy_path, allow_pickle=True).item()
gt_ann = seg_labels[file_id]

gt_timestamps = []
in_fake = False
start_t = 0.0

for i, label in enumerate(gt_ann):
    is_fake = (label == '0')
    if is_fake and not in_fake:
        start_t = i * 0.16
        in_fake = True
    elif not is_fake and in_fake:
        gt_timestamps.append((start_t, i * 0.16))
        in_fake = False

if in_fake:
    gt_timestamps.append((start_t, len(gt_ann) * 0.16))

# 4. Print the comparison
print(f"--- File: {file_id} ---")
print(f"Ground Truth Deepfake : {gt_timestamps}")
print(f"Model Predicted       : {pred_timestamps}")
