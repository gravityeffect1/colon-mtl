from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(r"C:\Users\saram\OneDrive\Desktop\colon-mtl\dataset_annotated_cncc\original_dataset\originals_no_necrosis")

images_dir = ROOT / "images"
labels_dir = ROOT / "labels"

rows = []
for img_path in sorted(images_dir.glob("*.png")):
    lbl_path = labels_dir / img_path.name
    if not lbl_path.exists():
        continue
    rows.append({
        "image_path": str(img_path),
        "label_path": str(lbl_path),
        "slide_id": img_path.name.split("_patch")[0],
    })

df = pd.DataFrame(rows)
print(f"Perechi valide: {len(df)}")
print(f"Slide-uri unice: {df['slide_id'].nunique()}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["slide_id"]))

df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"

overlap = set(df.loc[train_idx, "slide_id"]) & set(df.loc[val_idx, "slide_id"])
print(f"Overlap: {len(overlap)} (trebuie 0)")
print(f"Train: {df[df['split']=='train'].shape[0]} | Val: {df[df['split']=='val'].shape[0]}")

Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/manifest_valid.csv", index=False)
print("Salvat: outputs/manifest_valid.csv")