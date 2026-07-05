import yaml
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

from src.data.dataset import ColonDataset
from src.models.segmentation import build_model
from src.training.losses import DiceCELoss
from src.evaluation.metrics import compute_metrics


def get_dominant_class(label_path):
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    unique = np.unique(mask)
    for cls in [2, 3, 4, 5]:
        if cls in unique:
            return cls
    return 1


def main():
    with open("configs/seg_cncc.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv("outputs/manifest_valid.csv")
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"]
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_df["dominant_class"] = train_df["label_path"].apply(get_dominant_class)
    class_counts = train_df["dominant_class"].value_counts()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = train_df["dominant_class"].map(class_weights).values

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        ColonDataset(train_df, "train", cfg["image_size"]),
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        ColonDataset(val_df, "val", cfg["image_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = build_model(n_classes=6, encoder=cfg["encoder"]).to(device)
    criterion = DiceCELoss(n_classes=6)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    best_dice = 0
    Path("outputs").mkdir(exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                all_preds.append(outputs.argmax(1).cpu())
                all_masks.append(masks.cpu())

        metrics = compute_metrics(
            torch.cat(all_preds),
            torch.cat(all_masks),
        )

        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"train_loss={train_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss/len(val_loader):.4f} | "
              f"mean_dice={metrics['mean_dice']:.4f}")
        for cls, dice in metrics["per_class_dice"].items():
            print(f"  {cls}: {dice:.4f}")

        if metrics["mean_dice"] > best_dice:
            best_dice = metrics["mean_dice"]
            torch.save(model.state_dict(), "outputs/best_model.pt")
            print(f"  → best model saved (dice={best_dice:.4f})")


if __name__ == "__main__":
    main()