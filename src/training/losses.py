import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, n_classes=6, ce_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

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
        return (self.ce_weight * self.ce(pred, target) +
                (1 - self.ce_weight) * self.dice_loss(pred, target))