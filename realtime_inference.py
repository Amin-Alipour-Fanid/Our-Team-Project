"""
Real-time Inference Module for ISLR101 (BiGRU) - Fixed

ورودی‌ها:
    predict(keypoints)          : توالی آماده (3, 112, 67) مثل قبل
    predict_segment(frames, dt) : فریم‌های یک «اشاره» (L, 3, 67) از دوربین
                                  → دقیقاً مثل داده‌ی آموزش به 112 فریم تبدیل می‌شود

نکته مهم: تابع _normalize عمداً دقیقاً مثل preprocessing.py آموزش نگه داشته شده
(حتی reshape عجیبش)، چون مدل با همین آموزش دیده. آن را تغییر ندهید.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import BiGRUModel
from calibration import load_calibration

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw"
NORM_STATS_PATH = CHECKPOINT_PATH / "norm_stats.pkl"
BEST_MODEL_PATH = CHECKPOINT_PATH / "best_model.pth"

T_LEN = 112
TRAIN_FPS = 30.0          # fps ویدیوهای دیتاست (برای تبدیل مدت زمان اشاره به تعداد فریم)
HAND_BLOCKS = [(25, 46), (46, 67)]
BODY_BLOCK = (0, 25)
MAX_GAP_FILL = 6          # پر کردن گم شدن کوتاه دست‌ها توسط MediaPipe


class SignLanguagePredictor:
    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        norm_stats_path: Path = NORM_STATS_PATH,
        device: Optional[str] = None,
        calib: Optional[Dict] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if not norm_stats_path.exists():
            raise FileNotFoundError(f"norm_stats not found: {norm_stats_path}")
        with open(norm_stats_path, "rb") as f:
            stats = pickle.load(f)
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)

        if not model_path.exists():
            raise FileNotFoundError(f"best_model not found: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        config = ckpt["config"]
        self.model = BiGRUModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.num_classes = config["num_classes"]
        self.config = config

        self.calib = calib if calib is not None else load_calibration()

        print("✅ SignLanguagePredictor ready")
        print(f"   Device      : {self.device}")
        print(f"   Best epoch  : {ckpt.get('epoch')}")
        if "best_val_acc" in ckpt:
            print(f"   Val Acc     : {ckpt['best_val_acc']:.2f}%")
        print(f"   Classes     : {self.num_classes}")

    # ------------------------------------------------------------ preprocessing
    def _normalize(self, sample: np.ndarray) -> np.ndarray:
        """دقیقاً همان منطق preprocessing آموزش (تغییر ندهید)"""
        C, T, K = sample.shape
        flat = sample.reshape(C * K, T)
        normalized = (flat - self.mean[:, None]) / self.std[:, None]
        return normalized.reshape(C, T, K)

    def preprocess(self, keypoints: np.ndarray) -> torch.Tensor:
        x = np.asarray(keypoints, dtype=np.float32)
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim == 3:
            x = x[np.newaxis]
        assert x.ndim == 4 and x.shape[1:] == (3, T_LEN, 67), f"bad shape {keypoints.shape}"
        x = np.nan_to_num(x).copy()
        for i in range(x.shape[0]):
            x[i] = self._normalize(x[i])
        return torch.from_numpy(x).to(self.device)

    @staticmethod
    def _fill_gaps(frames: np.ndarray) -> np.ndarray:
        """frames: (L, 3, 67). گم‌شدن‌های کوتاه هر بلوک (بدن/دست) را با آخرین مقدار پر می‌کند"""
        out = frames.copy()
        for s, e in [BODY_BLOCK] + HAND_BLOCKS:
            present = np.any(out[:, :2, s:e] != 0, axis=(1, 2))
            last, gap = None, 0
            for t in range(len(out)):
                if present[t]:
                    last, gap = t, 0
                elif last is not None:
                    gap += 1
                    # فقط اگر بعداً دوباره دیده شود (گپ واقعی، نه پایین آمدن دست)
                    nxt = np.argmax(present[t:]) if present[t:].any() else -1
                    if nxt != -1 and gap + nxt <= MAX_GAP_FILL:
                        out[t, :, s:e] = out[last, :, s:e]
        return out

    def build_sequence(self, frames: np.ndarray, duration_sec: Optional[float] = None,
                       length_scale: float = 1.0) -> np.ndarray:
        """(L, 3, 67) → (3, 112, 67) با همان layout زمانی داده‌ی آموزش"""
        frames = np.asarray(frames, dtype=np.float32)
        L = len(frames)
        c = self.calib
        if c["pad_mode"] == "none":
            target = T_LEN
        else:
            if duration_sec and duration_sec > 0:
                target = duration_sec * TRAIN_FPS
            else:
                target = L
            lo = max(8.0, c["len_p10"] * 0.8)
            hi = min(float(T_LEN), c["len_p90"] * 1.15)
            target = float(np.clip(target * length_scale, lo, hi))
        target = int(round(min(target, T_LEN)))

        idx = np.round(np.linspace(0, L - 1, target)).astype(int)
        seg = frames[idx]                        # (target, 3, 67)

        seq = np.zeros((3, T_LEN, 67), dtype=np.float32)
        seg_t = seg.transpose(1, 0, 2)           # (3, target, 67)
        mode = c["pad_mode"]
        if mode in ("end", "none"):
            seq[:, :target] = seg_t
        elif mode == "start":
            seq[:, T_LEN - target:] = seg_t
        else:
            off = (T_LEN - target) // 2
            seq[:, off:off + target] = seg_t
        return seq

    # ------------------------------------------------------------ prediction
    @torch.no_grad()
    def _probs(self, batch: np.ndarray) -> np.ndarray:
        x = self.preprocess(batch)
        logits = self.model(x)
        return F.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def predict(self, keypoints: np.ndarray, return_probs: bool = True) -> Dict:
        probs = self._probs(keypoints)[0]
        k = int(np.argmax(probs))
        res = {"predicted_class": k, "confidence": float(probs[k])}
        if return_probs:
            res["probabilities"] = probs
        return res

    @torch.no_grad()
    def predict_segment(self, frames: np.ndarray, duration_sec: Optional[float] = None,
                        tta: bool = True) -> Dict:
        """پیش‌بینی یک اشاره‌ی کامل با Test-Time Augmentation زمانی"""
        frames = self._fill_gaps(np.asarray(frames, dtype=np.float32))
        scales = [0.8, 0.9, 1.0, 1.1, 1.25] if tta else [1.0]
        batch = np.stack([self.build_sequence(frames, duration_sec, s) for s in scales])
        probs = self._probs(batch)
        # میانگین هندسی پایدارتر از میانگین ساده است
        logp = np.log(np.clip(probs, 1e-8, 1.0)).mean(0)
        p = np.exp(logp - logp.max())
        p /= p.sum()
        k = int(np.argmax(p))
        return {"predicted_class": k, "confidence": float(p[k]), "probabilities": p}

    def predict_from_file(self, npy_path: Union[str, Path]) -> Dict:
        return self.predict(np.load(npy_path))

    # ------------------------------------------------------------ diagnostics
    def verify_on_test(self, n: int = 300) -> Optional[float]:
        """چک می‌کند مدل + normalization روی داده‌ی واقعی تست درست کار می‌کند"""
        tp, tl = DATA_PATH / "islr101_test.npy", DATA_PATH / "islr101_test_label.npy"
        if not (tp.exists() and tl.exists()):
            print("ℹ️ test data not found, skipping verification")
            return None
        data = np.load(tp, mmap_mode="r")
        labels = np.load(tl)
        idx = np.unique(np.linspace(0, len(data) - 1, min(n, len(data))).astype(int))
        correct = 0
        for s in range(0, len(idx), 32):
            b = np.asarray(data[idx[s:s + 32]], dtype=np.float32)
            if b.ndim == 5:
                b = b[..., 0]
            pred = self._probs(b).argmax(1)
            correct += int((pred == labels[idx[s:s + 32]]).sum())
        acc = 100.0 * correct / len(idx)
        print(f"🧪 Offline test accuracy on {len(idx)} samples: {acc:.2f}%")
        return acc


def test_inference():
    print("=" * 60)
    print("Testing Real-time Inference Module")
    print("=" * 60)
    predictor = SignLanguagePredictor()
    dummy = np.random.randn(3, T_LEN, 67).astype(np.float32)
    r = predictor.predict(dummy)
    print(f"  shape check OK, probs: {r['probabilities'].shape}")
    seg = np.random.rand(45, 3, 67).astype(np.float32)
    r = predictor.predict_segment(seg, duration_sec=1.5)
    print(f"  segment check OK, class {r['predicted_class']} ({r['confidence']:.3f})")
    predictor.verify_on_test()
    print("=" * 60)


if __name__ == "__main__":
    test_inference()
