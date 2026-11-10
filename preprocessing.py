"""
PHASE 3 (Corrected): Preprocessing Pipeline for ISLR101
پیش‌پردازش داده‌های keypoint برای BiGRU

قوانین رعایت‌شده:
- Shape واقعی: (N, 3, 112, 67, 1) → بعد از squeeze: (N, 3, 112, 67)
- 112 = Frames (Temporal / Sequence Length)
- 67  = Keypoints
- Train / Validation / Test کاملاً جدا
- Mean/Std فقط از Train محاسبه می‌شود
- Test قفل است و فقط برای ارزیابی نهایی استفاده می‌شود
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import pickle
from typing import Tuple, Optional, Dict


class ISLR101Dataset(Dataset):
    """دیتاست ISLR101 برای PyTorch با representation صحیح"""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        """
        Args:
            data: numpy array با shape (N, 3, 112, 67, 1) یا (N, 3, 112, 67)
            labels: numpy array با shape (N,)
            mean, std: آمار نرمال‌سازی (باید فقط از Train آمده باشند)
            normalize: آیا نرمال‌سازی اعمال شود
        """
        # حذف بعد اضافی اگر وجود داشته باشد
        if data.ndim == 5 and data.shape[-1] == 1:
            data = data.squeeze(-1)  # (N, 3, 112, 67)

        assert data.ndim == 4, f"Expected 4D data after squeeze, got {data.shape}"
        assert data.shape[1:] == (3, 112, 67), (
            f"Expected (N, 3, 112, 67), got {data.shape}. "
            "112 must be frames, 67 must be keypoints."
        )

        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.normalize = normalize
        self.mean = mean
        self.std = std

        if normalize:
            assert mean is not None and std is not None, (
                "mean and std must be provided when normalize=True"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]  # (3, 112, 67)
        label = self.labels[idx]

        if self.normalize:
            sample = self._normalize(sample)

        sample = torch.from_numpy(sample)  # (3, 112, 67)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label

    def _normalize(self, sample: np.ndarray) -> np.ndarray:
        """
        نرمال‌سازی یک نمونه
        sample: (3, 112, 67)
        mean/std: (3*67,) = (201,)  که روی (C, K) محاسبه شده
        """
        C, T, K = sample.shape  # 3, 112, 67
        # reshape به (C*K, T) = (201, 112)
        flat = sample.reshape(C * K, T)
        normalized = (flat - self.mean[:, None]) / self.std[:, None]
        return normalized.reshape(C, T, K)


def compute_norm_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    محاسبه mean و std فقط از داده‌های Training
    data: (N, 3, 112, 67) یا (N, 3, 112, 67, 1)
    خروجی: mean, std با shape (3*67,) = (201,)
    """
    if data.ndim == 5 and data.shape[-1] == 1:
        data = data.squeeze(-1)

    # (N, 3, 112, 67) → (N*112, 3*67)
    N, C, T, K = data.shape
    flat = data.reshape(N * T, C * K)  # (N*112, 201)

    # حذف فریم‌های کاملاً صفر (padding)
    mask = ~np.all(flat == 0, axis=1)
    flat = flat[mask]

    mean = np.mean(flat, axis=0).astype(np.float32)
    std = np.std(flat, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    return mean, std


def collate_fn(batch):
    """stack کردن batch و اطمینان از shape نهایی"""
    samples, labels = zip(*batch)
    samples = torch.stack(samples, dim=0)  # (B, 3, 112, 67)
    labels = torch.stack(labels, dim=0)    # (B,)

    assert samples.ndim == 4 and samples.shape[1:] == (3, 112, 67), (
        f"collate_fn produced wrong shape: {samples.shape}"
    )
    return samples, labels


def create_dataloaders(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 16,
    normalize: bool = True,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    ایجاد سه DataLoader کاملاً جدا:

    TRAIN  → برای آموزش
    VAL    → برای model selection / early stopping / checkpoint
    TEST   → قفل شده، فقط برای ارزیابی نهایی

    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """
    # --------------------------------------------------
    # 1. محاسبه mean/std فقط از کل Train (قبل از split)
    # --------------------------------------------------
    if normalize:
        mean, std = compute_norm_stats(train_data)
        norm_stats = {"mean": mean, "std": std}
    else:
        mean, std = None, None
        norm_stats = {"mean": None, "std": None}

    # --------------------------------------------------
    # 2. ساخت Dataset کامل Train
    # --------------------------------------------------
    full_train_dataset = ISLR101Dataset(
        train_data, train_labels, mean=mean, std=std, normalize=normalize
    )

    # --------------------------------------------------
    # 3. Split Train → Train + Validation
    # --------------------------------------------------
    n_total = len(full_train_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    # --------------------------------------------------
    # 4. Dataset تست (قفل)
    # --------------------------------------------------
    test_dataset = ISLR101Dataset(
        test_data, test_labels, mean=mean, std=std, normalize=normalize
    )

    # --------------------------------------------------
    # 5. DataLoaders
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"[Preprocessing] Train samples : {n_train}")
    print(f"[Preprocessing] Val   samples : {n_val}")
    print(f"[Preprocessing] Test  samples : {len(test_dataset)} (LOCKED)")
    if normalize:
        print(f"[Preprocessing] Mean shape  : {mean.shape}")
        print(f"[Preprocessing] Std  shape  : {std.shape}")

    return train_loader, val_loader, test_loader, norm_stats


def save_norm_stats(norm_stats: Dict, path: Path):
    """ذخیره آمار نرمال‌سازی برای استفاده در inference / evaluate"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(norm_stats, f)
    print(f"[Preprocessing] norm_stats saved to {path}")


def test_preprocessing():
    """تست سریع preprocessing با shape واقعی"""
    print("=" * 60)
    print("PHASE 3 (Corrected): Testing Preprocessing")
    print("=" * 60)

    data_path = Path("E:/new-project/data/raw")
    train_data = np.load(data_path / "islr101_train.npy")
    train_labels = np.load(data_path / "islr101_train_label.npy")
    test_data = np.load(data_path / "islr101_test.npy")
    test_labels = np.load(data_path / "islr101_test_label.npy")

    print(f"Raw train shape: {train_data.shape}")
    print(f"Raw test  shape: {test_data.shape}")

    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        train_data, train_labels,
        test_data, test_labels,
        batch_size=16,
        normalize=True,
        val_ratio=0.15,
        seed=42,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    # تست یک batch
    samples, labels = next(iter(train_loader))
    print(f"\nOne train batch:")
    print(f"  samples shape : {samples.shape}")   # باید (B, 3, 112, 67) باشد
    print(f"  labels  shape : {labels.shape}")
    print(f"  samples dtype : {samples.dtype}")
    print(f"  min/max       : {samples.min().item():.4f} / {samples.max().item():.4f}")

    assert samples.shape[1:] == (3, 112, 67), (
        f"Wrong sample shape! Expected (B, 3, 112, 67), got {samples.shape}"
    )
    print("\n✅ Shape assertion passed: (B, 3, 112, 67)")
    print("=" * 60)
    print("✅ Preprocessing test completed successfully!")
    print("=" * 60)

    return train_loader, val_loader, test_loader, norm_stats


if __name__ == "__main__":
    test_preprocessing()