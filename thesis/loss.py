"""Custom loss functions for EEG classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification with optional class-weight alpha.

    Re-weights cross-entropy by (1 - p_t)^gamma so easy examples contribute
    less to the total loss. When alpha is provided it acts as a class-frequency
    prior (same as the weight argument of CrossEntropyLoss).

    :param torch.Tensor | None alpha: Per-class weight tensor of shape (num_classes,).
    :param float gamma: Focusing exponent. 0.0 reduces to weighted cross-entropy.
    :param str reduction: 'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: torch.Tensor | None = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        :param torch.Tensor logits: Raw model outputs of shape (N, C).
        :param torch.Tensor targets: Class indices of shape (N,).
        :return: Scalar loss value.
        :rtype: torch.Tensor
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p_t = F.softmax(logits, dim=1).gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = ((1.0 - p_t) ** self.gamma) * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()
