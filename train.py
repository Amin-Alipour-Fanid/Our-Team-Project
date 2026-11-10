"""
PHASE 7: Clean Training Pipeline for ISLR101 (BiGRU)
بازنویسی کامل از صفر طبق قوانین داور

جریان صحیح:
Dataset → Train/Val split (از Train) → Test (LOCKED)
→ BiGRU
→ Train loop + Val loop
→ Best checkpoint (بر اساس Val)
→ Final Test Evaluation (فقط یک بار)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
import random
import sys
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import BiGRUModel, count_parameters
from preprocessing import create_dataloaders, save_norm_stats

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints"
LOG_PATH = CHECKPOINT_PATH / "logs"
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        set_seed(config.get("seed", 42))

        # Data
        self._load_data()

        # Model
        self.model = BiGRUModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        ).to(self.device)

        print(f"📊 Model parameters: {count_parameters(self.model):,}")

        # Loss / Optim / Scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=False,
        )

        self.writer = SummaryWriter(LOG_PATH)

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        print("✅ Trainer ready\n")

    def _load_data(self):
        print("📂 Loading data...")
        train_data = np.load(DATA_PATH / "islr101_train.npy")
        train_labels = np.load(DATA_PATH / "islr101_train_label.npy")
        test_data = np.load(DATA_PATH / "islr101_test.npy")
        test_labels = np.load(DATA_PATH / "islr101_test_label.npy")

        print(f"  Train raw: {train_data.shape}")
        print(f"  Test  raw: {test_data.shape}")

        self.train_loader, self.val_loader, self.test_loader, self.norm_stats = create_dataloaders(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            batch_size=self.config["batch_size"],
            normalize=True,
            val_ratio=self.config.get("val_ratio", 0.15),
            seed=self.config.get("seed", 42),
        )

        # ذخیره norm_stats برای evaluate و inference بعدی
        save_norm_stats(self.norm_stats, CHECKPOINT_PATH / "norm_stats.pkl")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self, loader, desc="Eval"):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=desc, leave=False)
        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        return total_loss / len(loader), 100. * correct / total

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }
        torch.save(ckpt, CHECKPOINT_PATH / "latest.pth")
        if is_best:
            torch.save(ckpt, CHECKPOINT_PATH / "best_model.pth")
            print(f"  🏆 Best model saved (Val Acc: {self.best_val_acc:.2f}%)")

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_acc = ckpt["best_val_acc"]
        return ckpt["epoch"]

    def train(self):
        print("=" * 60)
        print("PHASE 7: Starting Clean Training")
        print("=" * 60)
        print(f"  Epochs        : {self.config['num_epochs']}")
        print(f"  Batch size    : {self.config['batch_size']}")
        print(f"  LR            : {self.config['learning_rate']}")
        print(f"  Patience      : {self.config['patience']}")
        print(f"  input_size    : {self.config['input_size']}  (3×67)")
        print(f"  Sequence len  : 112")
        print("=" * 60 + "\n")

        start = time.time()

        for epoch in range(1, self.config["num_epochs"] + 1):
            t0 = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader, desc="Val")

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:03d}/{self.config['num_epochs']} "
                  f"({time.time()-t0:.1f}s) | "
                  f"Train {train_loss:.4f}/{train_acc:.2f}% | "
                  f"Val {val_loss:.4f}/{val_acc:.2f}% | "
                  f"LR {lr:.6f}")

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.patience_counter >= self.config["patience"]:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start
        print("\n" + "=" * 60)
        print("Training finished")
        print(f"  Best Val Acc : {self.best_val_acc:.2f}% @ epoch {self.best_epoch}")
        print(f"  Time         : {total_time/60:.1f} min")
        print("=" * 60)

        # Load best
        best_path = CHECKPOINT_PATH / "best_model.pth"
        if best_path.exists():
            ep = self.load_checkpoint(best_path)
            print(f"✅ Loaded best model (epoch {ep})")
        else:
            print("⚠️  best_model.pth not found")

        return self.model

    def final_test(self):
        """فقط یک بار بعد از انتخاب best checkpoint"""
        print("\n" + "=" * 60)
        print("🔒 FINAL TEST EVALUATION (LOCKED – never used in training)")
        print("=" * 60)

        test_loss, test_acc = self.evaluate(self.test_loader, desc="Test")

        print(f"\n📌 Test Loss : {test_loss:.4f}")
        print(f"📌 Test Acc  : {test_acc:.2f}%")
        print("=" * 60)

        self.writer.add_scalar("Accuracy/test_final", test_acc, self.best_epoch)
        self.writer.close()

        return test_loss, test_acc


def main():
    config = {
        # Shape صحیح
        "input_size": 3 * 67,       # 201
        "num_classes": 101,

        # Model
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,

        # Training (محافظه‌کارانه برای MX350 2GB)
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "patience": 10,
        "val_ratio": 0.15,
        "seed": 42,
    }

    trainer = Trainer(config)
    model = trainer.train()
    test_loss, test_acc = trainer.final_test()

    print("\n📊 FINAL SUMMARY")
    print(f"  Best Validation Acc : {trainer.best_val_acc:.2f}%")
    print(f"  Final Test Acc      : {test_acc:.2f}%")
    print(f"  Parameters          : {count_parameters(model):,}")
    print(f"  Checkpoints         : {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()