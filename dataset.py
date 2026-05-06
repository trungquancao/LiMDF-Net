import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder


CLASS_NAMES: List[str] = sorted(["ack", "bcc", "mel", "nev", "scc", "sek"])
IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

IMAGE_SIZE: int = 256

def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-45, 45), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(10.0/255.0, 30.0/255.0), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_eval_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

class PADDataset(Dataset):
    def __init__(
        self,
        image_ids: np.ndarray,
        labels: np.ndarray,
        clinical_features: np.ndarray,
        image_dir: str,
        transform: A.Compose,
    ):
        self.image_ids = image_ids
        self.labels = labels
        self.clinical_features = clinical_features
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        label = int(self.labels[idx])

        img_path = os.path.join(self.image_dir, str(img_id))
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"Error reading image {img_path}: {e}")

        augmented = self.transform(image=image)
        image_tensor = augmented["image"]

        clinical_tensor = torch.tensor(
            self.clinical_features[idx], dtype=torch.float32
        )

        return image_tensor, clinical_tensor, label

def prepare_dataloaders(
    metadata_csv: str,
    image_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    df = pd.read_csv(metadata_csv)
    ignore_cols = ['patient_id', 'lesion_id', 'img_id', 'biopsed', 'dx', 'split', 'label']
    clinical_columns = [c for c in df.columns if c not in ignore_cols]
    clinical_dim = len(clinical_columns)
    print(f"[DATA] Detected {clinical_dim} tabular clinical features.")

    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)
    df["label"] = le.transform(df["dx"])

    train_mask = df["split"] == "train"
    num_cols_to_norm = ['age', 'diameter_1', 'diameter_2']
    
    for col in num_cols_to_norm:
        if col in df.columns:
            mean_val = float(df.loc[train_mask, col].mean())
            std_val = float(df.loc[train_mask, col].std())
            df[col] = (df[col] - mean_val) / max(std_val, 1e-8)

    clinical_matrix = df[clinical_columns].values.astype(np.float32)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"[DATA] Samples — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    def _split_arrays(split_df: pd.DataFrame):
        idx = split_df.index.values
        return (
            split_df["img_id"].values,
            split_df["label"].values,
            clinical_matrix[idx],
        )

    train_ids, train_labels, train_clin = _split_arrays(train_df)
    val_ids, val_labels, val_clin = _split_arrays(val_df)
    test_ids, test_labels, test_clin = _split_arrays(test_df)

    train_dataset = PADDataset(
        train_ids, train_labels, train_clin, image_dir, get_train_transforms()
    )
    val_dataset = PADDataset(
        val_ids, val_labels, val_clin, image_dir, get_eval_transforms()
    )
    test_dataset = PADDataset(
        test_ids, test_labels, test_clin, image_dir, get_eval_transforms()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, clinical_dim