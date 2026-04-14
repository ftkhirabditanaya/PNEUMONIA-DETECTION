"""
losses.py  –  Focal Loss for class-imbalanced medical image classification.

Why Focal Loss over Cross-Entropy for this dataset?
  - The dataset has ~3x more PNEUMONIA than NORMAL images.
  - Standard CE treats all samples equally, so the model learns to predict
    PNEUMONIA for almost everything (easy 74% accuracy, useless clinically).
  - Focal Loss down-weights easy correct predictions (p > 0.9) and
    up-weights hard, uncertain ones (p ≈ 0.5), forcing the model to
    focus on the difficult borderline cases — exactly where false
    negatives (missed pneumonia) occur.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weight. Scalar or list [w_normal, w_pneumonia].
               Use alpha > 0.5 to up-weight the minority class (NORMAL).
               Default 0.75 = 75% weight on NORMAL (minority).
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 → standard cross-entropy.
               gamma=2 → standard Focal Loss (recommended).
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Raw logits, shape [batch, num_classes]
            targets: Ground truth class indices, shape [batch]
        """
        # Standard cross-entropy (log-softmax + NLL)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # p_t = probability of the correct class
        p_t = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma
        # Easy examples (p_t → 1) get weight → 0
        # Hard examples (p_t → 0) get weight → 1
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        # For binary: alpha for class 1 (PNEUMONIA), (1-alpha) for class 0 (NORMAL)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(1.0 - self.alpha, device=inputs.device),
        )

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

