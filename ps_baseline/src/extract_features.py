import os
import sys
import numpy as np
import torch
import soundfile as sf
import torchaudio.functional as AF
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_DIR, SAMPLE_RATE
from dataset import PartialSpoofDataset
from model import PartialSpoofBaseline


def load_audio_sf(wav_path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a wav file using soundfile (no FFmpeg / torchcodec needed).

    Returns:
        waveform : (n_samples,) float32 tensor at target_sr.
    """
    data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
    # data: (n_samples, n_channels)
    waveform = torch.from_numpy(data.T)          # (n_channels, n_samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True) # mono
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)                   # (n_samples,)


def extract_features_for_split(
    split: str,
    model: PartialSpoofBaseline,
    device: torch.device,
    wav_dir: str,
):
    print(f"\nExtracting features for split: {split}")

    try:
        ds = PartialSpoofDataset(split, use_features=False)
    except FileNotFoundError:
        print(f"  Skipping {split} -- audio directory not found.")
        return

    os.makedirs(FEATURES_DIR, exist_ok=True)

    for i in tqdm(range(len(ds)), desc=f"  {split}"):
        file_id, _ = ds.samples[i]
        out_path = os.path.join(FEATURES_DIR, f"{file_id}.pt")
        if os.path.exists(out_path):
            continue

        wav_path = os.path.join(wav_dir, file_id + '.wav')
        try:
            waveform = load_audio_sf(wav_path)
        except Exception as e:
            print(f"\n  [WARN] Could not load {wav_path}: {e} -- skipping")
            continue

        waveform = waveform.unsqueeze(0).to(device)  # (1, T_audio)

        with torch.no_grad():
            w2v_out = model.wav2vec2(input_values=waveform, attention_mask=None)
            features = w2v_out.last_hidden_state.squeeze(0).cpu()  # (T_frames, 768)

        torch.save(features, out_path)


def main():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import DATASET_ROOT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PartialSpoofBaseline().to(device)
    model.eval()

    for split in ['train', 'dev', 'eval']:
        wav_dir = os.path.join(DATASET_ROOT, split, 'con_wav')
        extract_features_for_split(split, model, device, wav_dir)

    print("\nFeature extraction complete!")


if __name__ == '__main__':
    main()

