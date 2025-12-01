# mocomotion/data/datasets.py

import glob
import os
from typing import Callable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .transforms_moco import build_moco_v2_two_crops


class PretrainDataset(Dataset):
    """
    Unlabeled pretraining dataset for MoCo.

    Expects directory structure like:

        data/pretrain/
            shard_000/
                *.png / *.jpg
            shard_001/
                *.png / *.jpg
            ...
            shard_499/

    It recursively collects all image paths under data/pretrain/shard_*.
    """

    def __init__(
        self,
        root: str = "data/pretrain",
        transform: Optional[Callable] = None,
        extensions: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform

        if extensions is None:
            # handle both png and jpg just in case
            extensions = (".png", ".jpg", ".jpeg")

        # collect all image paths from shard_* subfolders
        img_paths: List[str] = []
        for ext in extensions:
            pattern = os.path.join(root, "shard_*", f"*{ext}")
            img_paths.extend(glob.glob(pattern))

        if len(img_paths) == 0:
            raise RuntimeError(
                f"No images found in {root} with extensions {extensions}. "
                f"Check that your dataset is correctly placed."
            )

        # sort for deterministic order
        img_paths.sort()
        self.img_paths = img_paths

        print(
            f"[PretrainDataset] Found {len(self.img_paths)} images "
            f"under {root} (extensions={extensions})."
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            return self.transform(img)

        # If no transform, just return a single image
        return img


def build_pretrain_dataset(
    root: str = "data/pretrain",
    image_size: int = 96,
) -> "PretrainDataset":
    """
    Convenience helper to build the standard MoCo v2 pretrain dataset:
    - loads from data/pretrain/shard_*
    - uses TwoCropsTransform with 96x96 MoCo v2 augmentations
    """
    transform = build_moco_v2_two_crops(image_size=image_size)
    return PretrainDataset(root=root, transform=transform)


if __name__ == "__main__":
    # Quick smoke test: run `python -m data.datasets` from project root.
    from torch.utils.data import DataLoader

    dataset = build_pretrain_dataset(root="data/pretrain", image_size=96)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for (im_q, im_k) in loader:
        print("im_q shape:", im_q.shape)  # expect [4, 3, 96, 96]
        print("im_k shape:", im_k.shape)  # expect [4, 3, 96, 96]
        break
