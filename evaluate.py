"""
PHASE 6: Final Evaluation for ISLR101 (BiGRU)
ارزیابی نهایی فقط روی Test set (LOCKED)

قوانین:
- فقط بعد از انتخاب best checkpoint اجرا شود
- Test هرگز در training / validation / model selection استفاده نشده
- از همان preprocessing و norm_stats آموزش استفاده می‌کند
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import pickle
import json

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import BiGRUModel, count_parameters
from preprocessing import create_dataloaders, ISLR101Dataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints"
RESULTS_PATH = CHECKPOINT_PATH / "evaluation_results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def load_norm_stats(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_best_model(checkpoint_path: Path, device: torch.device):
    """لود بهترین مدل و config آن"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    model = BiGRUModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded best model from epoch {ckpt['epoch']}")
    print(f"   Best Val Acc (during training): {ckpt['best_val_acc']:.2f}%")
    print(f"   Parameters: {count_parameters(model):,}")

    return model, config, ckpt


@torch.no_grad()
def evaluate_full(model, loader, device, desc="Test"):
    """ارزیابی کامل و جمع‌آوری پیش‌بینی‌ها"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=desc, leave=False)
    for data, targets in pbar:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, preds = outputs.max(1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = total_loss / len(loader)

    return all_preds, all_targets, avg_loss


def compute_metrics(y_true, y_pred):
    """محاسبه تمام متریک‌های مورد نیاز"""
    acc = accuracy_score(y_true, y_pred) * 100

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, digits=4, zero_division=0, output_dict=False
    )

    metrics = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # 1. لود بهترین مدل
    # --------------------------------------------------
    best_path = CHECKPOINT_PATH / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"best_model.pth not found at {best_path}")

    model, config, ckpt = load_best_model(best_path, device)

    # --------------------------------------------------
    # 2. لود داده و ساخت Test loader (با همان norm_stats)
    # --------------------------------------------------
    print("\n📂 Loading Test data (LOCKED)...")
    train_data = np.load(DATA_PATH / "islr101_train.npy")
    train_labels = np.load(DATA_PATH / "islr101_train_label.npy")
    test_data = np.load(DATA_PATH / "islr101_test.npy")
    test_labels = np.load(DATA_PATH / "islr101_test_label.npy")

    # از همان create_dataloaders استفاده می‌کنیم تا preprocessing یکسان باشد
    # فقط test_loader را نیاز داریم
    _, _, test_loader, norm_stats = create_dataloaders(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        batch_size=config.get("batch_size", 16),
        normalize=True,
        val_ratio=0.15,
        seed=config.get("seed", 42),
    )

    print(f"  Test samples: {len(test_loader.dataset)} (LOCKED)")

    # --------------------------------------------------
    # 3. ارزیابی نهایی
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("🔒 FINAL TEST EVALUATION (LOCKED)")
    print("=" * 60)

    preds, targets, test_loss = evaluate_full(model, test_loader, device, desc="Test")

    metrics = compute_metrics(targets, preds)

    # --------------------------------------------------
    # 4. چاپ نتایج
    # --------------------------------------------------
    print(f"\n📌 Test Loss          : {test_loss:.4f}")
    print(f"📌 Accuracy           : {metrics['accuracy']:.2f}%")
    print(f"📌 Precision (macro)  : {metrics['precision_macro']:.2f}%")
    print(f"📌 Recall    (macro)  : {metrics['recall_macro']:.2f}%")
    print(f"📌 F1-score  (macro)  : {metrics['f1_macro']:.2f}%")
    print(f"📌 Precision (weighted): {metrics['precision_weighted']:.2f}%")
    print(f"📌 Recall    (weighted): {metrics['recall_weighted']:.2f}%")
    print(f"📌 F1-score  (weighted): {metrics['f1_weighted']:.2f}%")

    print("\n" + "-" * 60)
    print("Classification Report:")
    print(metrics["classification_report"])

    # --------------------------------------------------
    # 5. ذخیره نتایج
    # --------------------------------------------------
    # Confusion matrix
    cm_path = RESULTS_PATH / "confusion_matrix.npy"
    np.save(cm_path, metrics["confusion_matrix"])
    print(f"\n💾 Confusion matrix saved → {cm_path}")

    # متریک‌ها به صورت json
    results = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val_acc": float(ckpt["best_val_acc"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(metrics["accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "precision_weighted": float(metrics["precision_weighted"]),
        "recall_weighted": float(metrics["recall_weighted"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "num_test_samples": int(len(targets)),
        "config": config,
    }

    json_path = RESULTS_PATH / "final_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Metrics saved → {json_path}")

    print("\n" + "=" * 60)
    print("✅ Final evaluation completed successfully!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()