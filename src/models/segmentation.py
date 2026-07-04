import segmentation_models_pytorch as smp

def build_model(n_classes: int = 6, encoder: str = "resnet50"):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    )