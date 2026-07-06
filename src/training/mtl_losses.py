import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.losses import DiceCELoss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = 0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target,
                             ignore_index=self.ignore_index,
                             reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, n_classes=6, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.focal = FocalLoss(gamma=gamma)
        self.dice_weight = dice_weight

    def dice_loss(self, pred, target):
        pred = F.softmax(pred, dim=1)
        total, n = 0, 0
        for c in range(1, self.n_classes):
            p = pred[:, c]
            t = (target == c).float()
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            if union > 0:
                total += 1 - (2 * intersection + 1e-7) / (union + 1e-7)
                n += 1
        return total / max(n, 1)

    def forward(self, pred, target):
        return (self.dice_weight * self.dice_loss(pred, target) +
                (1 - self.dice_weight) * self.focal(pred, target))


class MTLLoss(nn.Module):
    def __init__(self, n_classes=6, seg_weight=0.7,
                 ord_weight=0.3, focal_gamma=2.0):
        super().__init__()
        self.seg_loss = DiceFocalLoss(n_classes=n_classes, gamma=focal_gamma)
        self.ord_loss = nn.CrossEntropyLoss()
        self.seg_weight = seg_weight
        self.ord_weight = ord_weight

    def forward(self, seg_output, masks, ordinal_output, ordinal_labels):
        seg = self.seg_loss(seg_output, masks)
        ordinal = self.ord_loss(ordinal_output, ordinal_labels)
        total = self.seg_weight * seg + self.ord_weight * ordinal
        return total, seg, ordinal