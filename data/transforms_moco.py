# mocomotion/data/transforms_moco.py

import random
from typing import Callable

from PIL import ImageFilter
from torchvision import transforms as T


class TwoCropsTransform:
    """
    Take one PIL image and return two transformed versions.

    This is the standard MoCo / SimCLR pattern:
      - q: query view
      - k: key view
    """
    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class GaussianBlur(object):
    """
    Gaussian blur augmentation as in MoCo v2 / SimCLR.

    With probability p (handled by RandomApply outside), it blurs the image
    with a sigma sampled from [0.1, 2.0].
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def build_moco_v2_transform(image_size: int = 96):
    """
    MoCo v2 / SimCLR-style strong augmentation for 96x96 images.

    NOTE: This assumes images are 3-channel RGB and returns normalized tensors.
    """
    # Color jitter params from MoCo v2 paper
    color_jitter = T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )

    augmentation = T.Compose([
        # random resized crop to 96x96 (course requires 96x96 resolution)
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),

        # strong color jitter, applied with high prob
        T.RandomApply([color_jitter], p=0.8),

        # random grayscale
        T.RandomGrayscale(p=0.2),

        # gaussian blur as in MoCo v2 (p=0.5)
        T.RandomApply([GaussianBlur(0.1, 2.0)], p=0.5),

        # convert to tensor + normalize (ImageNet stats; standard even for non-ImageNet)
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return augmentation


def build_moco_v2_two_crops(image_size: int = 96) -> TwoCropsTransform:
    """
    Convenience helper: directly returns a TwoCropsTransform with MoCo v2 augs.
    """
    base_transform = build_moco_v2_transform(image_size=image_size)
    return TwoCropsTransform(base_transform)


if __name__ == "__main__":
    # Tiny manual sanity check (no dataset yet, just making sure it runs)
    from PIL import Image
    import torch

    dummy_img = Image.new("RGB", (96, 96), color=(128, 128, 128))
    transform = build_moco_v2_two_crops(96)
    q, k = transform(dummy_img)
    print("q shape:", q.shape, "k shape:", k.shape)  # expect [3, 96, 96]
    print("dtype:", q.dtype, k.dtype)                # expect torch.float32
