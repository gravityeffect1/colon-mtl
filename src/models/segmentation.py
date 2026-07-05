import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BaseSegmentationModel(nn.Module):
    def __init__(self, n_classes: int = 6, encoder: str = "resnet50"):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


class MultiTaskModel(nn.Module):
    def __init__(self, n_classes: int = 6, encoder: str = "resnet50"):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ordinal_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),  # NORM=0, LG=1, HG=2, ADK=3
        )

    def forward(self, x: torch.Tensor):
        seg_output = self.unet(x)
        features = self.unet.encoder(x)[-1]
        pooled = self.gap(features).flatten(1)
        ordinal_output = self.ordinal_head(pooled)
        return seg_output, ordinal_output


def build_model(n_classes: int = 6, encoder: str = "resnet50") -> BaseSegmentationModel:
    """Factory pentru baseline — păstrează compatibilitatea cu train_baseline.py."""
    return BaseSegmentationModel(n_classes=n_classes, encoder=encoder)


def build_mtl_model(n_classes: int = 6, encoder: str = "resnet50") -> MultiTaskModel:
    """Factory pentru MTL."""
    return MultiTaskModel(n_classes=n_classes, encoder=encoder)