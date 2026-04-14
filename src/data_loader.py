"""
data_loader.py  –  Fixed data loading with proper splits and CLAHE.

ROOT CAUSE FIX:
  The original Kaggle dataset has only 16 validation images (8 per class).
  This script combines train+val, then re-splits into 80% train / 20% val
  using stratified sampling, giving ~1000+ validation samples.
  Test set is kept completely separate and untouched.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── CLAHE preprocessing (applied before tensor conversion) ────────────────────
def apply_clahe(image_pil: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to a PIL image.
    Improves visibility of subtle pneumonia patterns in X-rays.
    Converts to LAB colour space, equalises L-channel only, converts back.
    """
    img_array = np.array(image_pil.convert("RGB"))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_eq)


# ── Dataset class ─────────────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-ray images.

    Args:
        image_paths: List of absolute paths to image files.
        labels:      Corresponding integer labels (0=NORMAL, 1=PNEUMONIA).
        transform:   torchvision transform pipeline.
        use_clahe:   Apply CLAHE preprocessing before transforms.
    """

    CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, image_paths, labels, transform=None, use_clahe=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Could not open {img_path}: {e}. Using blank image.")
            image = Image.new("RGB", (224, 224), color=0)

        if self.use_clahe:
            image = apply_clahe(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(phase: str, image_size: int = 224):
    """
    Return augmentation pipeline for each phase.

    Train: aggressive augmentation to prevent overfitting.
    Val/Test: deterministic resize + normalize only.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Path collector ────────────────────────────────────────────────────────────
def collect_image_paths(root_dir: str, splits: list):
    """
    Collect all image paths and labels from the given split folders.

    Args:
        root_dir: Path to chest_xray/ directory.
        splits:   List of split names, e.g. ["train", "val"].

    Returns:
        paths:  List[str]
        labels: List[int]
    """
    root = Path(root_dir)
    paths, labels = [], []

    for split in splits:
        for cls_name, cls_idx in ChestXrayDataset.CLASS_MAP.items():
            cls_dir = root / split / cls_name
            if not cls_dir.exists():
                print(f"[WARNING] Directory not found: {cls_dir}")
                continue
            for ext in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG"):
                for img_path in cls_dir.glob(ext):
                    paths.append(str(img_path))
                    labels.append(cls_idx)

    return paths, labels


# ── Weighted sampler for class imbalance ─────────────────────────────────────
def make_weighted_sampler(labels):
    """
    Create a WeightedRandomSampler that oversamples the minority class
    (NORMAL) so each batch sees a roughly balanced mix.

    This works alongside Focal Loss — both address imbalance at different levels.
    """
    labels_tensor = torch.tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_tensor]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Main entry point ──────────────────────────────────────────────────────────
def get_dataloaders(
    data_root: str,
    batch_size: int = 16,       # 16 is safe for CPU RAM; increase to 32 on GPU
    image_size: int = 224,
    val_split: float = 0.20,    # 20% of train+val combined → validation
    num_workers: int = 0,       # MUST be 0 on Windows (multiprocessing bug)
    use_clahe: bool = True,
    seed: int = 42,
):
    """
    Build and return train / val / test DataLoaders.

    KEY FIX: Combines the original tiny val set (16 images) into training
    pool and re-splits with stratification to get a proper validation set.

    Args:
        data_root:   Path to the chest_xray/ directory.
        batch_size:  Images per batch. Use 16 for CPU, 32 for GPU.
        image_size:  Resize target (224 for pretrained models).
        val_split:   Fraction of (train+original_val) to use as validation.
        num_workers: Parallel loading workers. Use 0 on Windows.
        use_clahe:   Apply CLAHE contrast enhancement.
        seed:        Random seed for reproducible splits.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Collect paths — merge train + original val into one pool
    train_val_paths, train_val_labels = collect_image_paths(
        data_root, splits=["train", "val"]
    )

    # 2. Stratified split → proper train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_split,
        stratify=train_val_labels,
        random_state=seed,
    )

    # 3. Separate, untouched test set
    test_paths, test_labels = collect_image_paths(data_root, splits=["test"])

    # 4. Print class distribution for transparency
    _print_split_stats("train", train_labels)
    _print_split_stats("val",   val_labels)
    _print_split_stats("test",  test_labels)

    # 5. Build datasets
    train_dataset = ChestXrayDataset(
        train_paths, train_labels,
        transform=get_transforms("train", image_size),
        use_clahe=use_clahe,
    )
    val_dataset = ChestXrayDataset(
        val_paths, val_labels,
        transform=get_transforms("val", image_size),
        use_clahe=use_clahe,
    )
    test_dataset = ChestXrayDataset(
        test_paths, test_labels,
        transform=get_transforms("test", image_size),
        use_clahe=use_clahe,
    )

    # 6. Weighted sampler for training (handles class imbalance at batch level)
    sampler = make_weighted_sampler(train_labels)

    # 7. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,           # weighted sampler replaces shuffle=True
        num_workers=num_workers,
        pin_memory=False,          # False on CPU
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


def _print_split_stats(name: str, labels: list):
    n_normal    = labels.count(0)
    n_pneumonia = labels.count(1)
    total       = len(labels)
    ratio       = n_pneumonia / max(n_normal, 1)
    print(f"  [{name:5s}] NORMAL={n_normal:4d} | PNEUMONIA={n_pneumonia:4d} "
          f"| total={total:5d} | imbalance ratio={ratio:.2f}x")

