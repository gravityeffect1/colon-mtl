
from pathlib import Path

root = Path(r"C:\Users\saram\OneDrive\Desktop\colon-mtl\dataset_annotated_cncc\training_dataset")

local_img_train = root / "images" / "train"
local_img_val = root / "images" / "val"
local_lbl_train = root / "labels" / "train"
local_lbl_val = root / "labels" / "val"


train_images = list(local_img_train.glob("*.png"))
val_images = list(local_img_val.glob("*.png"))
print(f"Train images local: {len(train_images)}")
print(f"Val images local: {len(val_images)}")

# construieste manifest local
import pandas as pd

rows = []
for img_path in train_images:
    lbl_path = local_lbl_train / img_path.name
    rows.append({
        "image_path": str(img_path),
        "label_path": str(lbl_path) if lbl_path.exists() else None,
        "has_label": lbl_path.exists(),
        "split": "train",
        "slide_id": img_path.name.split("_patch")[0],
    })

for img_path in val_images:
    lbl_path = local_img_val.parent.parent / "labels" / "val" / img_path.name
    rows.append({
        "image_path": str(img_path),
        "label_path": str(lbl_path) if lbl_path.exists() else None,
        "has_label": lbl_path.exists(),
        "split": "val",
        "slide_id": img_path.name.split("_patch")[0],
    })

df = pd.DataFrame(rows)
df.to_csv("outputs/manifest_cncc_local.csv", index=False)
print(f"\nTotal local: {len(df)}")
print(f"Cu label: {df['has_label'].sum()}")
print(f"Split:\n{df['split'].value_counts()}")
print(f"\nSlide-uri unice: {df['slide_id'].nunique()}")