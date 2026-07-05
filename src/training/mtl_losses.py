import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.losses import DiceCELoss


class MTLLoss(nn.Module):
    def __init__(self, n_classes: int = 6,
                 seg_weight: float = 0.7,
                 ord_weight: float = 0.3):
        super().__init__()
        self.seg_loss = DiceCELoss(n_classes=n_classes)
        self.ord_loss = nn.CrossEntropyLoss()
        self.seg_weight = seg_weight
        self.ord_weight = ord_weight

    def forward(self, seg_output: torch.Tensor, masks: torch.Tensor,
                ordinal_output: torch.Tensor,
                ordinal_labels: torch.Tensor) -> tuple:
        seg = self.seg_loss(seg_output, masks)
        ordinal = self.ord_loss(ordinal_output, ordinal_labels)
        total = self.seg_weight * seg + self.ord_weight * ordinal
        return total, seg, ordinal