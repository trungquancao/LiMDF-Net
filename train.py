"""
train.py
=============================================================================
Training configuration for LiDF-Net (Lightweight Direct Fusion Network)
-----------------------------------------------------------------------------
  • Optimizer  : AdamW with Differential Learning Rates
  • Backbone LR: 1e-5 (base_lr * 0.1) -> For pre-trained Vision Backbone
  • Head LR    : 1e-4 (base_lr)       -> For Clinical MLP, Fusion, Classifier
  • Batch size : 32/64
  • Epochs     : 100
  • Loss       : Weighted Focal Loss (with Label Smoothing)

The single-phase end-to-end training utilizes Differential Learning Rates.
The pre-trained vision backbone is updated slowly to preserve generalized
feature extraction capabilities, while the newly initialized branches
learn at a standard rate.

CLI usage examples:
------------------
    # Default: MobileNetV4 + Direct Fusion (Simple Concat)
    python train.py --metadata_csv final_metadata.csv --image_dir PAD-UFES-20/images
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from dataset import prepare_dataloaders, CLASS_NAMES
from models import (
    MultimodalSkinLesionNet,
    count_trainable_params,
    count_total_params,
    SUPPORTED_FUSIONS,
)


class EarlyStopping:
    """Halts training when validation loss stops improving."""
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none', label_smoothing=0.1
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, clinical, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        logits = model(images, clinical)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    for images, clinical, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, clinical)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return running_loss / total, correct / total, macro_f1


def calculate_class_weights(metadata_csv: str, device: torch.device) -> torch.Tensor:
    """Calculates smoothed inverse frequency class weights to address imbalance."""
    df = pd.read_csv(metadata_csv)
    
    train_df = df[df['split'] == 'train'].copy()
    
    if 'diagnostic' in train_df.columns:
        train_df.rename(columns={'diagnostic': 'dx'}, inplace=True)
    train_df['dx'] = train_df['dx'].astype(str).str.strip().str.lower()
    
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_NAMES)
    train_labels = le.transform(train_df['dx'])
    
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    total_samples = len(train_labels)
    num_classes = len(CLASS_NAMES)
    
    weights = np.power(total_samples / (num_classes * class_counts), 0.25)
    
    class_weights_tensor = torch.FloatTensor(weights).to(device)
        
    return class_weights_tensor

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    output_dir: str,
    alpha: float,       
    fusion_name: str,
    use_gem: bool,
) -> None:
    path = os.path.join(output_dir, "best_model.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "alpha": alpha,  
            "fusion_name": fusion_name,
            "use_gem": use_gem,
        },
        path,
    )

def get_optimizer(model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    params = []
    existing_params = set()
    
    params.append({'params': model.backbone.parameters(), 'lr': base_lr * 0.1})
    existing_params.update(model.backbone.parameters())
    
    head_params = [p for p in model.parameters() if p not in existing_params and p.requires_grad]
    if head_params:
        params.append({'params': head_params, 'lr': base_lr})
    
    return AdamW(params, weight_decay=1e-2)

def train(
    metadata_csv: str,
    image_dir: str,
    output_dir: str = "checkpoints",
    alpha: float = 1.0,             
    fusion_name: str = "simple_concat",
    use_gem: bool = True,
    batch_size: int = 32,
    total_epochs: int = 100,
    learning_rate: float = 0.0001,
    patience: int = 15,
    num_workers: int = 4,
    seed: int = 42,
) -> None:
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, clinical_dim = prepare_dataloaders(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MultimodalSkinLesionNet(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        clinical_dim=clinical_dim,
        alpha=alpha,             
        fusion_name=fusion_name,
        use_gem=use_gem,
    ).to(device)


    class_weights = calculate_class_weights(metadata_csv, device)
    criterion = WeightedFocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = get_optimizer(model, learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    early_stopping = EarlyStopping(patience=patience)
    
    best_f1 = 0.0


    for epoch in range(1, total_epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        current_head_lr = optimizer.param_groups[-1]['lr']

        if val_f1 > best_f1:
            best_f1 = val_f1
            _save_checkpoint(
                model, optimizer, epoch, val_f1, output_dir,
                alpha, fusion_name, use_gem
            )

        if early_stopping(val_loss):
            break
        
        scheduler.step(val_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train for Skin Lesion Classification",
    )
    parser.add_argument("--metadata_csv", type=str, default="final_metadata.csv")
    parser.add_argument("--image_dir", type=str, default="PAD-UFES-20/images")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--disable_gem", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, choices=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument(
        "--fusion", type=str, default="simple_concat", choices=SUPPORTED_FUSIONS,
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,        
        fusion_name=args.fusion,
        use_gem=not args.disable_gem,
        batch_size=args.batch_size,
        total_epochs=args.total_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )