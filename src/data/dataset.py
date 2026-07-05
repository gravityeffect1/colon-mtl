import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ['background', 'stroma', 'adk_invasion',
               'high_grade', 'low_grade', 'normal_gland']
N_CLASSES = len(CLASS_NAMES)


ordinal_map = {
    2: 3,  
    3: 2,  
    4: 1,  
    5: 0,  
}


def get_ordinal_label(mask_array: np.ndarray) -> int:
    present = set(np.unique(mask_array))
    for cls in [2, 3, 4, 5]:  # prioritate: ADK > HG > LG > NORM
        if cls in present:
            return ordinal_map[cls]
    return 0  


def get_transforms(split: str, image_size: int = 512) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


class ColonDataset(Dataset):
    def __init__(self, manifest_df, split: str = "train",
                 image_size: int = 512, return_ordinal: bool = False):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = get_transforms(split, image_size)
        self.return_ordinal = return_ordinal

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)

        if self.return_ordinal:
            ordinal_label = get_ordinal_label(mask)

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].long()

        if self.return_ordinal:
            return image_t, mask_t, torch.tensor(ordinal_label)
        return image_t, mask_t