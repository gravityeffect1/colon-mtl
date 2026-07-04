import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ['background', 'stroma', 'adk_invasion',
               'high_grade', 'low_grade', 'normal_gland']
N_CLASSES = len(CLASS_NAMES)

def get_transforms(split: str, image_size: int = 512):
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
    def __init__(self, manifest_df, split="train", image_size=512):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = get_transforms(split, image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()