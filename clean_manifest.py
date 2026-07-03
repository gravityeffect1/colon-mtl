# adaugi in audit_quick.py sau fisier nou clean_manifest.py

import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs/manifest_colon.csv")

# 1. identifici duplicatul fizic
print("=== DUPLICAT FIZIC (acelasi path, clase diferite) ===")
dup_path = df[df["slide_id"] == "273013-1 HE"]
print(dup_path[["class", "slide_id", "path"]])

# 2. elimini una din duplicate - pastrezi Hplastic-3
# (intreaba Filip care e eticheta corecta - acum luam una arbitrar)
df = df.drop_duplicates(subset=["slide_id"], keep="first")
print(f"\nDupa eliminare duplicate: {len(df)} slide-uri")

# 3. marchezi pacientii multi-clasa
patient_classes = df.groupby("patient_id")["class"].unique()
multi_class_patients = patient_classes[patient_classes.apply(len) > 1]
print("\n=== PACIENTI CU CLASE MULTIPLE ===")
for pid, classes in multi_class_patients.items():
    slides = df[df["patient_id"] == pid][["class", "slide_id"]].to_string(index=False)
    print(f"\nPacient {pid}: {list(classes)}")
    print(slides)

df["multi_class_patient"] = df["patient_id"].isin(multi_class_patients.index)

# 4. salveaza manifestul curat
df.to_csv("outputs/manifest_clean.csv", index=False)
print(f"\nSalvat: outputs/manifest_clean.csv")
print(f"Total slide-uri curate: {len(df)}")
print(f"Pacienti unici: {df['patient_id'].nunique()}")
print(f"Pacienti multi-clasa: {df['multi_class_patient'].sum()} slide-uri afectate")

from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("outputs/manifest_clean.csv")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# verificare CRITICA
overlap = set(train_df["patient_id"]) & set(val_df["patient_id"])
print(f"Overlap pacienti intre train si val: {len(overlap)} (trebuie 0)")

print(f"\nTrain: {len(train_df)} slide-uri, {train_df['patient_id'].nunique()} pacienti")
print(f"Val: {len(val_df)} slide-uri, {val_df['patient_id'].nunique()} pacienti")

print("\nDistributie clase in train:")
print(train_df["class"].value_counts())

print("\nDistributie clase in val:")
print(val_df["class"].value_counts())

train_df.to_csv("outputs/train_split.csv", index=False)
val_df.to_csv("outputs/val_split.csv", index=False)