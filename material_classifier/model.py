"""Shared model definition and preprocessing for training and preview."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


IMAGE_SIZE = 224
MODULE_DIR = Path(__file__).resolve().parent
TRAINED_CHECKPOINT = MODULE_DIR / "output" / "best.pt"
REPO_CHECKPOINT = MODULE_DIR / "models" / "trashnet-mobilenetv3-small.pt"
FOOD_ORGANIC_CHECKPOINT = MODULE_DIR / "models" / "food-organic-mobilenetv3-small.pt"
DEFAULT_CHECKPOINT = next(
    (path for path in (FOOD_ORGANIC_CHECKPOINT, TRAINED_CHECKPOINT, REPO_CHECKPOINT)
     if path.is_file()),
    FOOD_ORGANIC_CHECKPOINT,
)


def build_model(num_classes: int, *, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def image_transform(*, training: bool):
    if training:
        geometry = [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        geometry = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]
    return transforms.Compose([
        *geometry,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
