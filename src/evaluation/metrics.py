import torch
from src.data.dataset import CLASS_NAMES

def compute_metrics(preds, targets, n_classes=6):
    per_class_dice = {}
    scores = []
    for c in range(1, n_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union > 0:
            dice = ((2 * intersection + 1e-7) / (union + 1e-7)).item()
            per_class_dice[CLASS_NAMES[c]] = dice
            scores.append(dice)
    return {
        "mean_dice": sum(scores) / max(len(scores), 1),
        "per_class_dice": per_class_dice,
    }