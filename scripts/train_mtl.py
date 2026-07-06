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
from src.models.segmentation import build_mtl_model
from src.training.mtl_losses import MTLLoss
from src.evaluation.metrics import compute_metrics


def get_dominant_class(label_path: str) -> int:
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
        ColonDataset(train_df, "train", cfg["image_size"], return_ordinal=True),
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        ColonDataset(val_df, "val", cfg["image_size"], return_ordinal=True),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = build_mtl_model(n_classes=6, encoder=cfg["encoder"]).to(device)
    criterion = MTLLoss(n_classes=6, seg_weight=0.7, ord_weight=0.3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    best_dice = 0
    Path("outputs").mkdir(exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss, train_seg_loss, train_ord_loss = 0, 0, 0

        for images, masks, ordinal_labels in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            ordinal_labels = ordinal_labels.to(device)

            optimizer.zero_grad()
            seg_output, ordinal_output = model(images)
            loss, seg_l, ord_l = criterion(
                seg_output, masks,
                ordinal_output, ordinal_labels
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_seg_loss += seg_l.item()
            train_ord_loss += ord_l.item()

        model.eval()
        val_loss = 0
        all_preds, all_masks = [], []

        with torch.no_grad():
            for images, masks, ordinal_labels in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                ordinal_labels = ordinal_labels.to(device)
                seg_output, ordinal_output = model(images)
                loss, _, _ = criterion(
                    seg_output, masks,
                    ordinal_output, ordinal_labels
                )
                val_loss += loss.item()
                all_preds.append(seg_output.argmax(1).cpu())
                all_masks.append(masks.cpu())

        metrics = compute_metrics(
            torch.cat(all_preds),
            torch.cat(all_masks),
        )

        n_train = len(train_loader)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"loss={train_loss/n_train:.4f} "
              f"(seg={train_seg_loss/n_train:.4f} "
              f"ord={train_ord_loss/n_train:.4f}) | "
              f"val_loss={val_loss/len(val_loader):.4f} | "
              f"mean_dice={metrics['mean_dice']:.4f}")
        for cls, dice in metrics["per_class_dice"].items():
            print(f"  {cls}: {dice:.4f}")

        if metrics["mean_dice"] > best_dice:
            best_dice = metrics["mean_dice"]
            torch.save(model.state_dict(), "outputs/best_mtl_model.pt")
            print(f"  → best MTL model saved (dice={best_dice:.4f})")


if __name__ == "__main__":
    main()