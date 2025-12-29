# XaiVisionCAM/utils/image_utils.py
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, normalize


__all__ = [
    "load_image",
    "sample_images_flat",
]


def load_image(path: Path, size: int) -> torch.Tensor:
    """
    Load and normalize an RGB image for ImageNet-pretrained models.

    Args:
        path: Path to the image file
        size: Target image size (H = W = size)

    Returns:
        Tensor of shape (1, 3, size, size)
    """
    img = Image.open(path).convert("RGB")
    img = resize(img, (size, size))
    img = to_tensor(img)
    img = normalize(
        img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return img.unsqueeze(0)


def sample_images_flat(
    folder: Path,
    n: int,
    seed: int | None = None,
) -> List[Path]:
    """
    Randomly sample images from a flat directory.

    Args:
        folder: Directory containing images
        n: Number of images to sample
        seed: Optional random seed for reproducibility

    Returns:
        List of sampled image paths
    """
    images: List[Path] = []
    for ext in ("*.JPEG", "*.jpg", "*.jpeg", "*.png"):
        images.extend(folder.glob(ext))

    if not images:
        raise RuntimeError(f"No images found in {folder}")

    if seed is not None:
        random.seed(seed)

    return random.sample(images, min(n, len(images)))


def load_class_index(path: Path):
    """Load ImageNet class index mapping."""
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v[1] for k, v in data.items()}
