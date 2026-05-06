import os
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score, 
    balanced_accuracy_score,
    precision_score,
    recall_score,
    accuracy_score
)
from tqdm import tqdm

from dataset import prepare_dataloaders, CLASS_NAMES
from models import (
    MultimodalSkinLesionNet,
    count_trainable_params,
    count_total_params,
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray, 
    class_names: List[str],
) -> Dict:
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    bacc_score = balanced_accuracy_score(y_true, y_pred)
    
    try:
        auc_score = roc_auc_score(
            y_true, 
            y_prob, 
            multi_class="ovr", 
            average="macro",
            labels=list(range(n_classes)) 
        )
    except ValueError:
        auc_score = 0.0

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    overall_acc = accuracy_score(y_true, y_pred)

    per_class = {}
    acc_list, sens_list, spec_list, ppv_list, npv_list, f1_list = [], [], [], [], [], []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (
            2 * ppv * sensitivity / (ppv + sensitivity)
            if (ppv + sensitivity) > 0 else 0.0
        )

        per_class[name] = {
            "Accuracy": round(accuracy, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "PPV": round(ppv, 4),
            "NPV": round(npv, 4),
            "F1": round(f1, 4),
        }

        acc_list.append(accuracy)
        sens_list.append(sensitivity)
        spec_list.append(specificity)
        ppv_list.append(ppv)
        npv_list.append(npv)
        f1_list.append(f1)

    macro = {
        "Accuracy": round(np.mean(acc_list), 4),
        "Sensitivity": round(np.mean(sens_list), 4),
        "Specificity": round(np.mean(spec_list), 4),
        "PPV": round(np.mean(ppv_list), 4),
        "NPV": round(np.mean(npv_list), 4),
        "F1": round(np.mean(f1_list), 4),
    }

    weighted_f1 = round(
        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
    )

    return {
        "per_class": per_class,
        "macro": macro,
        "weighted_f1": weighted_f1,
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(float(bacc_score), 4),
        "auc": round(float(auc_score), 4),
        "std_precision": round(float(macro_precision), 4),
        "std_recall": round(float(macro_recall), 4),
        "std_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm,
    }


def print_metrics(results: Dict, class_names: List[str]) -> None:
    header = (
        f"{'Class':<12} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} "
        f"{'PPV':>12} {'NPV':>12} {'F1':>12}"
    )
    
    print("  [STANDARD METRICS FOR REPORTING]")
    print(f"  OVERALL ACCURACY (ACC) : {results['overall_accuracy']:.4f}")
    print(f"  MACRO PRECISION        : {results['std_precision']:.4f}")
    print(f"  MACRO RECALL           : {results['std_recall']:.4f}")
    print(f"  MACRO F1-SCORE         : {results['std_f1']:.4f}")
    print()
    print("  [ADDITIONAL METRICS]")
    print(f"  BALANCED ACC (BACC)    : {results['balanced_accuracy']:.4f}")
    print(f"  MACRO AUC              : {results['auc']:.4f}")
    print(f"  WEIGHTED F1            : {results['weighted_f1']:.4f}  (weighted by true class support)")
    print(header)

    for cls in class_names:
        m = results["per_class"][cls]
        print(
            f"{cls:<12} {m['Accuracy']:>10.4f} {m['Sensitivity']:>12.4f} {m['Specificity']:>12.4f} "
            f"{m['PPV']:>12.4f} {m['NPV']:>12.4f} {m['F1']:>12.4f}"
        )

    print(sep)
    macro = results["macro"]
    print(
        f"{'MACRO AVG':<12} {macro['Accuracy']:>10.4f} {macro['Sensitivity']:>12.4f} "
        f"{macro['Specificity']:>12.4f} {macro['PPV']:>12.4f} "
        f"{macro['NPV']:>12.4f} {macro['F1']:>12.4f}"
    )
    print(
        f"{'WEIGHTED F1':<12} {'—':>10} {'—':>12} {'—':>12} "
        f"{'—':>12} {'—':>12} {results['weighted_f1']:>12.4f}"
    )
    print(f"{'═'*90}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"{'':>8}", end="")
    for name in class_names:
        print(f"{name:>7}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>8}", end="")
        for j in range(len(class_names)):
            print(f"{results['confusion_matrix'][i, j]:>7d}", end="")
        print()


def compute_model_complexity(model: nn.Module, device: torch.device, clinical_dim: int) -> None:
    """Computes GFLOPs and parameter counts using the `thop` library."""
    try:
        from thop import profile, clever_format
    except ImportError:
        print("\n[WARN] `thop` is not installed — skipping FLOP calculation.")
        print(f"  Total parameters     : {count_total_params(model):,}")
        print(f"  Trainable parameters : {count_trainable_params(model):,}")
        return

    dummy_image = torch.randn(1, 3, 256, 256).to(device)
    dummy_clinical = torch.randn(1, clinical_dim).to(device)

    model.eval()
    
    try:
        flops, params = profile(
            model,
            inputs=(dummy_image, dummy_clinical),
            verbose=False,
        )
        flops_str, params_str = clever_format([flops, params], "%.3f")
        gflops = flops / 1e9

        print("  MODEL COMPLEXITY")
        print(f"{'═'*50}")
        print(f"  Total Parameters     : {count_total_params(model):,}")
        print(f"  Trainable Parameters : {count_trainable_params(model):,}")
        print(f"  FLOPs                : {flops_str}")
        print(f"  GFLOPs               : {gflops:.4f}")
    except Exception as e:
        print(f"\n[WARN] `thop` encountered an issue calculating FLOPs: {e}")
        print(f"  Total parameters     : {count_total_params(model):,}")
        print(f"  Trainable parameters : {count_trainable_params(model):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inference & Feature Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  
    all_features = [] 

    for images, clinical, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        feat_map = model.backbone(images)               
        clin_feat = model.clinical_mlp(clinical)       
        
        if getattr(model, "fusion_name", "") == "cross_attention":
            B, C, H, W = feat_map.shape
            img_seq = feat_map.view(B, C, H * W).permute(0, 2, 1) 
            fused_features = model.fusion(img_seq, clin_feat)
        else:
            img_feat = model.global_pool(feat_map)         
            if getattr(model, "img_projector", None):
                img_feat = model.img_projector(img_feat)       
            fused_features = model.fusion(img_feat, clin_feat)
            
        all_features.append(fused_features.cpu().numpy())

        # 2. Classification
        logits = model.classifier(fused_features)
        probs = F.softmax(logits, dim=1) 
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy()) 

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    features = np.vstack(all_features) 

    # Save plotting data 
    # os.makedirs("results", exist_ok=True)
    # np.savez("results/plot_data.npz", 
    #          y_true=y_true, 
    #          y_pred=y_pred, 
    #          y_prob=y_prob, 
    #          features=features)

    return compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LiDF-Net on PAD-UFES-20 test set",
    )
    parser.add_argument("--metadata_csv", type=str, default="final_metadata.csv")
    parser.add_argument("--image_dir", type=str, default="PAD-UFES-20/images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--disable_gem", action="store_true", help="Use GAP instead of GeM pooling.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fusion", type=str, default="simple_concat")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, clinical_dim = prepare_dataloaders(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    alpha = checkpoint.get("alpha", args.alpha)
    fusion = checkpoint.get("fusion_name", args.fusion)
    saved_use_gem = checkpoint.get("use_gem", None)
    use_gem = saved_use_gem if saved_use_gem is not None else not args.disable_gem

    model = MultimodalSkinLesionNet(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        clinical_dim=clinical_dim,
        alpha=alpha,
        fusion_name=fusion,
        use_gem=use_gem,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(
        f"[EVAL] Loaded checkpoint — epoch {checkpoint.get('epoch', 'N/A')}, "
        f"Val F1={checkpoint.get('val_f1', 0.0):.4f}, "
        f"arch=GeM-MobileNet v{alpha:.1f} + {fusion}"
    )

    results = evaluate_test_set(model, test_loader, device)
    print_metrics(results, CLASS_NAMES)
    compute_model_complexity(model, device, clinical_dim)

if __name__ == "__main__":
    main()