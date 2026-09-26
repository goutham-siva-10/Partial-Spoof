"""
losses.py — Advanced loss functions for PartialSpoof detection
==============================================================
Includes:
- BinaryFocalLossWithLogits: Focuses gradient updates on hard boundary frames
- MultiTaskLoss: Combines frame-level segmentation loss and utterance-level classification loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FOCAL_GAMMA, FOCAL_ALPHA, LAMBDA_UTT


class BinaryFocalLossWithLogits(nn.Module):
    """Numerically stable Focal Loss with Logits for binary classification.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the loss contribution from easy frames (p_t ~ 1.0) and concentrates
    gradients on ambiguous / transition frames where genuine and spoofed speech meet.
    """

    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : raw model predictions before sigmoid (arbitrary shape)
            targets: binary labels 0.0 or 1.0 (same shape as logits)
        """
        # Standard BCE with logits per element
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute p_t and alpha_t
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        else:
            focal_weight = torch.pow((1.0 - p_t), self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """Joint loss for frame-level segmentation and utterance-level spoof detection.

    Loss = Loss_frame + lambda_utt * Loss_utt
    """

    def __init__(
        self,
        loss_type: str = 'focal',
        lambda_utt: float = LAMBDA_UTT,
        focal_gamma: float = FOCAL_GAMMA,
        focal_alpha: float = FOCAL_ALPHA,
    ):
        super().__init__()
        self.lambda_utt = lambda_utt

        if loss_type == 'focal':
            self.frame_criterion = BinaryFocalLossWithLogits(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.frame_criterion = nn.BCEWithLogitsLoss()

        self.utt_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        frame_logits: torch.Tensor,
        frame_targets: torch.Tensor,
        loss_mask: torch.Tensor,
        utt_logits: torch.Tensor = None,
        utt_targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_logits : (B, T)
            frame_targets: (B, T)
            loss_mask    : (B, T) boolean mask of valid audio frames
            utt_logits   : (B,) utterance-level logits (optional)
            utt_targets  : (B,) utterance-level targets 0.0 or 1.0 (optional)

        Returns:
            total_loss, frame_loss, utt_loss
        """
        valid_logits = frame_logits[loss_mask]
        valid_targets = frame_targets[loss_mask]

        if valid_logits.numel() == 0:
            frame_loss = torch.tensor(0.0, device=frame_logits.device, requires_grad=True)
        else:
            frame_loss = self.frame_criterion(valid_logits, valid_targets)

        if utt_logits is not None and utt_targets is not None and self.lambda_utt > 0:
            utt_loss = self.utt_criterion(utt_logits, utt_targets)
            total_loss = frame_loss + self.lambda_utt * utt_loss
        else:
            utt_loss = torch.tensor(0.0, device=frame_logits.device)
            total_loss = frame_loss

        return total_loss, frame_loss, utt_loss
