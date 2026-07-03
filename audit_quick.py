from pathlib import Path
import pandas as pd
import re

data_root = Path(r"C:\data-colon")
IGNORE_CLASSES = {"altele"}

rows = []
for class_folder in sorted(data_root.iterdir()):
    if not class_folder.is_dir():
        continue
    if class_folder.name in IGNORE_CLASSES:
        continue
    for svs in class_folder.glob("*.svs"):
        rows.append({
            "class": class_folder.name,
            "slide_id": svs.stem,
            "size_mb": round(svs.stat().st_size / 1e6, 1),
            "path": str(svs),
        })

df = pd.DataFrame(rows)

# patient ID din primele cifre ale numelui
def extract_patient_id(slide_id):
    match = re.match(r'^(\d+)', slide_id)
    return match.group(1) if match else slide_id

df["patient_id"] = df["slide_id"].apply(extract_patient_id)

print("=== DISTRIBUȚIE CLASE ===")
print(df.groupby("class").size().to_string())
print(f"\nTotal slide-uri colon: {len(df)}")
print(f"Pacienți unici: {df['patient_id'].nunique()}")

print("\n=== PACIENȚI CU MULTIPLE SLIDE-URI ===")
multi = df.groupby("patient_id").size()
print(multi[multi > 1])

print("\n=== SLIDE DUPLICAT (273013-1) ===")
dup = df[df["slide_id"].str.contains("273013-1")]
print(dup[["class", "slide_id"]])

# salveaza manifestul
df.to_csv("outputs/manifest_colon.csv", index=False)
print("\nSalvat: outputs/manifest_colon.csv")