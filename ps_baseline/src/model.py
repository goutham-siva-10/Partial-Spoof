"""
model.py -- PartialSpoof Baseline Model
========================================

Architecture:
    Raw waveform (16 kHz)
        |
        v
    Frozen Wav2Vec2-base  (94,371,712 params, NO grad)
        |
        v  (B, T, 768)
    Linear(768 -> 1)       (769 params, trainable)
        |
        v  (B, T)
    Logits per frame  --sigmoid--> P(fake) per frame

Only the Linear layer is trained. Wav2Vec2 is completely frozen:
    - requires_grad=False on all Wav2Vec2 parameters
    - torch.no_grad() wraps the Wav2Vec2 forward pass (saves memory)
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WAV2VEC_MODEL, HIDDEN_DIM


class PartialSpoofBaseline(nn.Module):
    """Frozen Wav2Vec2-base + Linear(768->1) frame-level fake detector.

    Trainable parameters: 769  (Linear weight 768 + bias 1).
    Frozen parameters:    94,371,712  (entire Wav2Vec2-base).
    """

    def __init__(self, model_name: str = WAV2VEC_MODEL):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Frozen backbone
        # ------------------------------------------------------------------ #
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True)
        for p in self.wav2vec2.parameters():
            p.requires_grad_(False)
        self.wav2vec2.eval()   # always in eval mode (no dropout, no mask)

        # ------------------------------------------------------------------ #
        # Trainable classifier  (the ONLY thing that learns)
        # ------------------------------------------------------------------ #
        self.classifier = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self._report_params()

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_values: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_values   : (B, T_audio)  raw waveform samples at 16 kHz.
            attention_mask : (B, T_audio)  1 = real audio, 0 = padding.
                             Pass None for single-file (no padding) inference.
            features       : (B, T_frames, 768) pre-extracted Wav2Vec2 features.

        Returns:
            logits : (B, T_frames)  raw logits before sigmoid.
                     Apply torch.sigmoid() to get P(fake) in [0, 1].
        """
        if features is None:
            # Wav2Vec2 forward -- no gradient, saves ~360 MB of activations
            with torch.no_grad():
                w2v_out = self.wav2vec2(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )

            # features: (B, T_frames, 768)
            features = w2v_out.last_hidden_state

        # Linear: (B, T_frames, 768) -> (B, T_frames, 1) -> (B, T_frames)
        logits = self.classifier(features).squeeze(-1)

        return logits

    # ---------------------------------------------------------------------- #
    # Convenience
    # ---------------------------------------------------------------------- #

    def predict_proba(
        self,
        input_values: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return P(fake) in [0, 1] for each Wav2Vec2 output frame."""
        return torch.sigmoid(self.forward(input_values, attention_mask, features))

    def predict_segments(
        self,
        input_values: torch.Tensor,
        threshold: float = 0.5,
        frame_stride_s: float = 0.020,
    ) -> list:
        """Return list of (start_s, end_s) fake time intervals.

        Args:
            input_values : (1, T_audio) -- single file, no batch padding needed.
            threshold    : P(fake) threshold for binary decision.
            frame_stride_s: seconds per Wav2Vec2 output frame (default 0.020).

        Returns:
            List of (start_s, end_s) tuples for predicted fake segments.
        """
        proba = self.predict_proba(input_values).squeeze(0).cpu().numpy()
        binary = proba >= threshold

        segments = []
        in_fake  = False
        for t, is_fake in enumerate(binary):
            if is_fake and not in_fake:
                start_s = t * frame_stride_s
                in_fake = True
            elif not is_fake and in_fake:
                segments.append((start_s, t * frame_stride_s))
                in_fake = False
        if in_fake:
            segments.append((start_s, len(binary) * frame_stride_s))

        return segments

    # ---------------------------------------------------------------------- #
    # Utility
    # ---------------------------------------------------------------------- #

    def _report_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"[Model] {self.wav2vec2.config.model_type} + Linear({HIDDEN_DIM}->1)")
        print(f"[Model] Total params     : {total:,}")
        print(f"[Model] Trainable params : {trainable:,}  <- only these are optimised")
        print(f"[Model] Frozen params    : {frozen:,}  <- Wav2Vec2 backbone")
