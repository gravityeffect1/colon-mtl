# colon-mtl

Proiect de computational pathology — clasificare și segmentare pe histopatologie de colon.

## Research notebook

[`notebooks/ordinal_mtl_colon.ipynb`](notebooks/ordinal_mtl_colon.ipynb) răspunde la
întrebarea de cercetare *"Does ordinal auxiliary supervision improve segmentation
robustness and generalization in colorectal histopathology?"* printr-o comparație
controlată **STL vs MTL**:

- **STL** — UNet (encoder ResNet-50) doar pentru segmentare.
- **MTL** — același encoder partajat + cap ordinal **CORAL** (Consistent Rank
  Logits); obiectiv comun `L = L_seg + λ · L_ord`.

Setup identic (optimizer, scheduler, augmentări, epoci, batch) pentru ambele modele,
astfel încât diferența să fie atribuibilă doar supervizării ordinale auxiliare.
Notebook-ul este **direct executabil** (generator de date sintetice inclus) și
acoperă: pipeline de preprocesare cu split patient-wise, distribuția claselor,
modelele STL/MTL, metrici interne (Dice, IoU, HD95, QWK, MAE), testare statistică
(bootstrap CI + Wilcoxon), analiză de erori, t-SNE pe embeddings și validare externă
zero-shot (CNCC → TCGA / UniToPatho).
