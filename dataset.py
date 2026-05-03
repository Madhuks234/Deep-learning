"""
AyuLeafNet Dataset
==================
Handles data loading, augmentation, and splitting for Ayurvedic leaf images.

Expected folder structure:
    data/
    ├── Tulsi/
    │   ├── img001.jpg
    │   └── ...
    ├── Neem/
    │   └── ...
    └── ...
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from config import (
    CLASSES, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT, VAL_SPLIT, MEAN, STD, DATA_DIR, AUGMENT_TRAIN
)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str = "train") -> transforms.Compose:
    """Return appropriate transforms for train / val / test split."""
    base = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]

    if split == "train" and AUGMENT_TRAIN:
        aug = [
            transforms.Resize((int(IMAGE_SIZE * 1.15), int(IMAGE_SIZE * 1.15))),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.08),
            transforms.RandomRotation(degrees=25),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
        return transforms.Compose(aug)

    return transforms.Compose(base)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
class AyuLeafDataset(Dataset):
    """
    Custom dataset for Ayurvedic leaf images.

    Parameters
    ----------
    root_dir  : Path to the folder containing class sub-folders.
    classes   : List of class names (must match folder names).
    transform : torchvision transforms to apply.
    split     : "train" | "val" | "test" (used for logging only here).
    """

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir  : str,
        classes   : List[str] = CLASSES,
        transform : Optional[transforms.Compose] = None,
        split     : str = "train",
    ):
        self.root_dir  = Path(root_dir)
        self.classes   = classes
        self.class_idx = {cls: i for i, cls in enumerate(classes)}
        self.transform = transform or get_transforms(split)
        self.split     = split

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        missing = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                missing.append(cls)
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in self.VALID_EXTS:
                    self.samples.append((f, self.class_idx[cls]))

        if missing:
            print(f"⚠️  Folders not found for classes: {missing}")
            print(f"   Expected at: {self.root_dir}")

        if not self.samples:
            print("⚠️  No image samples loaded. Check your data/ directory structure.")
        else:
            print(f"✅ [{self.split}] Loaded {len(self.samples)} images "
                  f"across {len(self.classes)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a blank tensor if image is corrupt
            print(f"⚠️  Could not open {img_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label

    def class_counts(self) -> dict:
        """Return number of samples per class."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(
    dataset   : AyuLeafDataset,
    train_pct : float = TRAIN_SPLIT,
    val_pct   : float = VAL_SPLIT,
    seed      : int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Randomly split dataset into train / val / test subsets."""
    n      = len(dataset)
    idxs   = list(range(n))
    random.seed(seed)
    random.shuffle(idxs)

    n_train = int(n * train_pct)
    n_val   = int(n * val_pct)

    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train: n_train + n_val]
    test_idx  = idxs[n_train + n_val:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(
    data_dir     : str   = DATA_DIR,
    batch_size   : int   = BATCH_SIZE,
    num_workers  : int   = NUM_WORKERS,
    pin_memory   : bool  = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the data directory.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    # Each split gets its own transform
    train_ds = AyuLeafDataset(data_dir, transform=get_transforms("train"), split="train")
    val_ds   = AyuLeafDataset(data_dir, transform=get_transforms("val"),   split="val")
    test_ds  = AyuLeafDataset(data_dir, transform=get_transforms("test"),  split="test")

    # Use the same random split indices across all three (split by indices)
    full = AyuLeafDataset(data_dir, split="all")
    n    = len(full)
    idxs = list(range(n))
    random.seed(42)
    random.shuffle(idxs)

    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    train_ids = idxs[:n_train]
    val_ids   = idxs[n_train: n_train + n_val]
    test_ids  = idxs[n_train + n_val:]

    pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        Subset(train_ds, train_ids),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_ids),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        Subset(test_ds, test_ids),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    print(f"\n📦 DataLoaders ready — Train: {len(train_ids)}, "
          f"Val: {len(val_ids)}, Test: {len(test_ids)}\n")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# DEMO SYNTHETIC DATA (for quick testing without a real dataset)
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticAyuDataset(Dataset):
    """
    Generates random tensors mimicking leaf images — useful for
    architecture testing without a real dataset.
    """
    def __init__(self, size: int = 200, num_classes: int = 10):
        self.size        = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img   = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
        label = idx % self.num_classes
        return img, label


def get_synthetic_loaders(batch_size: int = 16):
    """Return synthetic train/val/test loaders for quick testing."""
    train = DataLoader(SyntheticAyuDataset(200), batch_size=batch_size, shuffle=True)
    val   = DataLoader(SyntheticAyuDataset(60),  batch_size=batch_size)
    test  = DataLoader(SyntheticAyuDataset(40),  batch_size=batch_size)
    print("🧪 Using SYNTHETIC data for quick testing.")
    return train, val, test
