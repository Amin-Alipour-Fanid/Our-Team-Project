"""
Calibration for ISLR101 real-time pipeline (بدون نیاز به Train مجدد)

مشکل اصلی: مدل روی keypoint های OpenPose (BODY_25 + دو دست) آموزش دیده،
ولی در real-time از MediaPipe استفاده می‌شود. مختصات، مقیاس، ترتیب نقاط،
confidence و طول زمانی با داده‌ی آموزش فرق دارد و مدل ورودی‌ای می‌بیند که
هیچ‌وقت ندیده، پس همیشه چند کلاس ثابت (مثل «کمک کردن» یا «کارگاه») را می‌دهد.

این ماژول یک بار داده‌ی Train را آنالیز می‌کند و مشخصات «فضای آموزش» را
ذخیره می‌کند تا ورودی دوربین دقیقاً به همان فضا منتقل شود:
    - مرکز گردن و پهنای شانه (واحد و مقیاس مختصات)
    - میانگین confidence هر keypoint
    - mirror بودن ویدیوها
    - ترتیب اسلات دست چپ/راست
    - نحوه‌ی padding زمانی و توزیع طول ویدیوها

خروجی در checkpoints/calibration.json کش می‌شود.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw"
CHECKPOINT_PATH = ROOT / "checkpoints"
CALIB_PATH = CHECKPOINT_PATH / "calibration.json"
NORM_STATS_PATH = CHECKPOINT_PATH / "norm_stats.pkl"

T_LEN = 112
K = 67

# BODY_25 indices
NOSE, NECK, RSH, RELB, RWR, LSH, LELB, LWR, MIDHIP = 0, 1, 2, 3, 4, 5, 6, 7, 8
LHAND_SLOT_A = 25   # wrist of first hand block (25..45)
HAND_SLOT_B = 46    # wrist of second hand block (46..66)

CALIB_VERSION = 2


def _nanmedian(a, default):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float(default)


def compute_from_train(train_path: Path, max_samples: int = 3000) -> Dict:
    data = np.load(train_path, mmap_mode="r")
    n_total = data.shape[0]
    idx = np.unique(np.linspace(0, n_total - 1, min(n_total, max_samples)).astype(int))
    X = np.asarray(data[idx], dtype=np.float32)
    if X.ndim == 5 and X.shape[-1] == 1:
        X = X[..., 0]
    assert X.shape[1:] == (3, T_LEN, K), f"unexpected train shape {X.shape}"
    X = np.nan_to_num(X)

    x, y, c = X[:, 0], X[:, 1], X[:, 2]            # (n, T, K)
    valid = (x != 0) | (y != 0)                     # (n, T, K)

    # ---------------- temporal layout ----------------
    frame_nz = np.any(X != 0, axis=(1, 3))          # (n, T)
    lengths = frame_nz.sum(1)
    has = lengths > 0
    first = np.argmax(frame_nz, axis=1)
    last = T_LEN - 1 - np.argmax(frame_nz[:, ::-1], axis=1)
    med_len = _nanmedian(lengths[has], T_LEN)
    if med_len >= T_LEN - 2:
        pad_mode = "none"            # همه‌ی نمونه‌ها به 112 فریم resample شده‌اند
    elif np.mean(first[has] <= 1) > 0.8:
        pad_mode = "end"             # داده اول، صفر در انتها
    elif np.mean(last[has] >= T_LEN - 2) > 0.8:
        pad_mode = "start"           # صفر در ابتدا، داده در انتها
    else:
        pad_mode = "center"

    # ---------------- confidence per keypoint ----------------
    conf_mean = np.zeros(K, dtype=np.float64)
    det_rate = np.zeros(K, dtype=np.float64)
    for k in range(K):
        v = valid[:, :, k]
        det_rate[k] = v[frame_nz].mean() if frame_nz.any() else 0.0
        conf_mean[k] = c[:, :, k][v].mean() if v.any() else 0.0

    # ---------------- geometry ----------------
    both_sh = valid[:, :, RSH] & valid[:, :, LSH]
    sw = np.hypot(x[:, :, RSH] - x[:, :, LSH], y[:, :, RSH] - y[:, :, LSH])
    shoulder_width = _nanmedian(sw[both_sh], 1.0)

    neck_ok = valid[:, :, NECK]
    neck_x = _nanmedian(x[:, :, NECK][neck_ok], 0.0)
    neck_y = _nanmedian(y[:, :, NECK][neck_ok], 0.0)

    # آیا نقطه‌ی 1 واقعاً «گردن» است؟ (چک سازگاری با BODY_25)
    mid_x = (x[:, :, RSH] + x[:, :, LSH]) / 2
    mid_y = (y[:, :, RSH] + y[:, :, LSH]) / 2
    m = both_sh & neck_ok
    neck_err = _nanmedian(np.hypot(x[:, :, NECK] - mid_x, y[:, :, NECK] - mid_y)[m], 0) / max(shoulder_width, 1e-6)
    body25_ok = bool(neck_err < 0.3)

    # mirror: در ویدیوی عادی (غیر آینه‌ای) شانه‌ی راست فرد سمت چپ تصویر است
    dx = (x[:, :, RSH] - x[:, :, LSH])[both_sh]
    r_shoulder_on_image_left = bool(_nanmedian(dx, -1.0) < 0)

    # هر بلوک دست به کدام مچِ بدن نزدیک‌تر است؟
    def _hand_dist(slot, wrist):
        mm = valid[:, :, slot] & valid[:, :, wrist]
        d = np.hypot(x[:, :, slot] - x[:, :, wrist], y[:, :, slot] - y[:, :, wrist])[mm]
        return _nanmedian(d, np.inf)

    a_to_l, a_to_r = _hand_dist(LHAND_SLOT_A, LWR), _hand_dist(LHAND_SLOT_A, RWR)
    b_to_l, b_to_r = _hand_dist(HAND_SLOT_B, LWR), _hand_dist(HAND_SLOT_B, RWR)
    score_default = a_to_l + b_to_r
    score_swapped = a_to_r + b_to_l
    first_hand_block_is_left = bool(not (score_swapped < score_default))

    # مقیاس تصویر (برای اطلاع)
    xs, ys = x[valid], y[valid]
    coord_range = [float(np.percentile(xs, 1)), float(np.percentile(xs, 99)),
                   float(np.percentile(ys, 1)), float(np.percentile(ys, 99))] if xs.size else [0, 1, 0, 1]

    lens = lengths[has]
    calib = {
        "version": CALIB_VERSION,
        "source": "train_data",
        "pad_mode": pad_mode,
        "len_median": float(med_len),
        "len_p10": float(np.percentile(lens, 10)) if lens.size else 30.0,
        "len_p90": float(np.percentile(lens, 90)) if lens.size else 100.0,
        "neck": [neck_x, neck_y],
        "shoulder_width": shoulder_width,
        "conf_mean": conf_mean.tolist(),
        "det_rate": det_rate.tolist(),
        "r_shoulder_on_image_left": r_shoulder_on_image_left,
        "first_hand_block_is_left": first_hand_block_is_left,
        "body25_ok": body25_ok,
        "coord_range": coord_range,
    }
    return calib


def load_calibration(force: bool = False, verbose: bool = True) -> Dict:
    if CALIB_PATH.exists() and not force:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            calib = json.load(f)
        if calib.get("version") == CALIB_VERSION:
            if verbose:
                print_calibration(calib)
            return calib

    train_path = DATA_PATH / "islr101_train.npy"
    if train_path.exists():
        print("🔧 Computing calibration from training data (one time)...")
        calib = compute_from_train(train_path)
    else:
        # norm_stats برای این کار قابل استفاده نیست (ستون‌هایش x/y/conf را با هم قاطی دارد)
        raise FileNotFoundError(
            f"Train data not found: {train_path}\n"
            "برای یک بار کالیبراسیون، فایل islr101_train.npy باید در data/raw باشد. "
            "بعد از ساخته شدن checkpoints/calibration.json دیگر لازم نیست."
        )

    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    if verbose:
        print_calibration(calib)
    return calib


def print_calibration(c: Dict):
    print("📏 Calibration (" + c["source"] + ")")
    print(f"   pad_mode          : {c['pad_mode']}")
    print(f"   length p10/med/p90: {c['len_p10']:.0f} / {c['len_median']:.0f} / {c['len_p90']:.0f}")
    print(f"   neck              : ({c['neck'][0]:.3f}, {c['neck'][1]:.3f})")
    print(f"   shoulder width    : {c['shoulder_width']:.4f}")
    print(f"   coord range x/y   : {c['coord_range']}")
    print(f"   R-shoulder on left: {c['r_shoulder_on_image_left']}")
    print(f"   hand block 25 = left hand: {c['first_hand_block_is_left']}")
    if not c.get("body25_ok", True):
        print("   ⚠️ keypoint 1 does not look like NECK, body layout may not be BODY_25")


if __name__ == "__main__":
    load_calibration(force=True)
