"""
model.py — PartialSpoof Improved Model (ps_1)
==============================================
Architecture:
1. Backbone: Pretrained Wav2Vec2-base (Frozen)
2. Multi-Layer Feature Aggregation: Learnable softmax weighting over all 13 transformer layers
3. Temporal Sequence Modeling: 2-layer Bidirectional LSTM (256-d output)
4. Multi-Task Heads:
   - Frame-level Fake Classification Head (Linear -> GELU -> Linear -> 1)
   - Utterance-level Spoof Classification Head (Mean pooling -> Linear -> GELU -> Linear -> 1)
5. Integrated Inference & Post-processing:
   - 1D Median filtering (suppresses high-frequency noise spikes)
   - Minimum segment duration pruning (removes sub-160ms false alarms)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WAV2VEC_MODEL,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    WAV2VEC_STRIDE_S,
    MEDIAN_FILTER_KERNEL,
    MIN_SEGMENT_DURATION_S,
)
from metrics import post_process_predictions


class PartialSpoofBiLSTM(nn.Module):
    """Wav2Vec2 + Multi-layer Aggregation + BiLSTM + Multi-task Heads."""

    def __init__(
        self,
        model_name: str = WAV2VEC_MODEL,
        hidden_dim: int = HIDDEN_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden

        # 1. Pretrained Backbone (Frozen)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True,
            use_safetensors=True,
        )
        for p in self.wav2vec2.parameters():
            p.requires_grad_(False)
        self.wav2vec2.eval()

        # 2. Learnable Multi-Layer Aggregation Weights
        # Initialized to zeros -> equal softmax probability across all 13 layers
        self.layer_weights = nn.Parameter(torch.zeros(NUM_HIDDEN_LAYERS))

        # 3. BiLSTM Temporal Sequence Model
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * lstm_hidden)

        # 4. Frame Classifier Head: (B, T, 256) -> (B, T, 1)
        self.frame_classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # 5. Utterance Classifier Head: (B, 256) -> (B, 1)
        self.utt_classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self._report_params()

    def _report_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total     = trainable + frozen
        print(f"[Model ps_1] Wav2Vec2 + MultiLayer + BiLSTM({self.lstm_hidden}x2) + MultiTask")
        print(f"[Model ps_1] Total params     : {total:,}")
        print(f"[Model ps_1] Trainable params : {trainable:,}  <- BiLSTM + Heads + Layer Weights")
        print(f"[Model ps_1] Frozen params    : {frozen:,}  <- Wav2Vec2 backbone")

    def forward(
        self,
        input_values: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        features: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_values  : (B, n_samples) raw audio waveforms at 16kHz
            attention_mask: (B, n_samples) 1=real audio, 0=padding
            features      : (B, T, 768) optional pre-extracted features
            loss_mask     : (B, T) boolean mask for valid audio frames (for utterance pooling)

        Returns:
            frame_logits: (B, T) frame-level fake logits
            utt_logits  : (B,) utterance-level spoof logits
        """
        if features is None:
            # Wav2Vec2 is frozen -- keep in eval mode and no_grad
            self.wav2vec2.eval()
            with torch.no_grad():
                outputs = self.wav2vec2(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states  # tuple of 13 tensors: (B, T, 768)
                # Stack to (13, B, T, 768)
                stacked = torch.stack(hidden_states, dim=0)

            # Learnable layer aggregation
            weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            aggregated = (stacked * weights).sum(dim=0)  # (B, T, 768)
        else:
            aggregated = features

        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(aggregated)  # (B, T, 2 * lstm_hidden)
        lstm_out = self.norm(lstm_out)

        # Frame-level prediction
        frame_logits = self.frame_classifier(lstm_out).squeeze(-1)  # (B, T)

        # Utterance-level prediction (Masked Mean Pooling)
        if loss_mask is not None:
            T_common = min(lstm_out.shape[1], loss_mask.shape[1])
            cur_lstm = lstm_out[:, :T_common]
            cur_mask = loss_mask[:, :T_common].unsqueeze(-1).float()
            sum_pooled = (cur_lstm * cur_mask).sum(dim=1)
            counts = cur_mask.sum(dim=1).clamp(min=1.0)
            pooled = sum_pooled / counts                  # (B, 2 * lstm_hidden)
        else:
            pooled = lstm_out.mean(dim=1)

        utt_logits = self.utt_classifier(pooled).squeeze(-1)       # (B,)

        return frame_logits, utt_logits

    def predict_segments(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.5,
        post_process: bool = True,
        kernel_size: int = MEDIAN_FILTER_KERNEL,
        min_duration_s: float = MIN_SEGMENT_DURATION_S,
        frame_stride_s: float = WAV2VEC_STRIDE_S,
    ) -> list[tuple[float, float]]:
        """Inference helper for single audio files.

        Returns list of (start_s, end_s) detected fake intervals.
        """
        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            frame_logits, _ = self.forward(input_values=waveform)
            probs = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()

        if post_process:
            _, intervals = post_process_predictions(
                probs,
                threshold=threshold,
                kernel_size=kernel_size,
                min_duration_s=min_duration_s,
                frame_stride_s=frame_stride_s,
            )
        else:
            binary = (probs >= threshold).astype(int)
            from metrics import frames_to_intervals
            intervals = frames_to_intervals(binary, frame_stride_s=frame_stride_s)

        return intervals
