"""
Keypoint Extractor for ISLR101 Real-time Pipeline (Fixed)
MediaPipe Tasks API  →  OpenPose-style (BODY_25 + Left hand 21 + Right hand 21)

اصلاحات نسبت به نسخه قبل:
  1. mapping صحیح MediaPipe → BODY_25 (قبلاً ترتیب نقاط بدن کاملاً اشتباه بود؛
     مثلاً گردن، شانه‌ها و مچ‌ها جای هم بودند)
  2. گردن و MidHip از میانگین شانه‌ها/لگن ساخته می‌شوند (MediaPipe ندارد)
  3. دست چپ/راست: MediaPipe برچسب handedness را برای تصویر آینه‌ای می‌دهد؛
     در وبکم معمولی برعکس بود. حالا دست‌ها بر اساس نزدیکی به مچ بدن تخصیص می‌یابند
  4. confidence دست‌ها قبلاً 0 بود (مدل فکر می‌کرد دستی دیده نمی‌شود)؛
     حالا مثل داده‌ی آموزش مقدار می‌گیرد
  5. مختصات normalized [0..1] و غیرهمسان (x/W , y/H) بود؛ حالا به پیکسل برده،
     با گردن و پهنای شانه به همان مقیاس/مرکز داده‌ی آموزش منتقل می‌شود
  6. نقاط نامرئی (مثل پاها زیر میز) صفر می‌شوند، همان رفتار OpenPose
  7. timestamp واقعی و pose مدل full (دقیق‌تر)
"""

import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

POSE_MODELS = {
    "lite": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
    "full": ("pose_landmarker_full.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"),
}
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# BODY_25 index  ←  MediaPipe pose index (یا تاپل = میانگین دو نقطه)
# (چپ/راست = چپ/راستِ خود فرد، در هر دو کتابخانه)
BODY25_FROM_MP = [
    0,          # 0  Nose
    (11, 12),   # 1  Neck      = mid shoulders
    12,         # 2  RShoulder
    14,         # 3  RElbow
    16,         # 4  RWrist
    11,         # 5  LShoulder
    13,         # 6  LElbow
    15,         # 7  LWrist
    (23, 24),   # 8  MidHip
    24,         # 9  RHip
    26,         # 10 RKnee
    28,         # 11 RAnkle
    23,         # 12 LHip
    25,         # 13 LKnee
    27,         # 14 LAnkle
    5,          # 15 REye
    2,          # 16 LEye
    8,          # 17 REar
    7,          # 18 LEar
    31,         # 19 LBigToe
    31,         # 20 LSmallToe (MediaPipe فقط یک نقطه پنجه دارد)
    29,         # 21 LHeel
    32,         # 22 RBigToe
    32,         # 23 RSmallToe
    30,         # 24 RHeel
]
MP_L_SH, MP_R_SH, MP_L_WR, MP_R_WR = 11, 12, 15, 16


def download_model(url: str, path: Path):
    if path.exists():
        return
    print(f"Downloading {path.name} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  → saved to {path}")


@dataclass
class FrameData:
    keypoints: np.ndarray                 # (3, 67) در فضای داده‌ی آموزش، نقاط ناموجود = 0
    has_hands: bool
    num_hands: int
    wrists: Dict[str, Optional[np.ndarray]]  # موقعیت مچ‌ها (واحد: پهنای شانه)
    timestamp: float
    draw_pose: Optional[np.ndarray] = None   # (33, 3) px برای نمایش
    draw_hands: List[np.ndarray] = field(default_factory=list)


class KeypointExtractor:
    SEQUENCE_LENGTH = 112
    NUM_KEYPOINTS = 67
    NUM_CHANNELS = 3

    def __init__(
        self,
        calib: Optional[Dict] = None,
        target_width: int = 640,
        target_height: int = 480,
        pose_model: str = "full",
        min_pose_detection_confidence: float = 0.5,
        min_hand_detection_confidence: float = 0.5,
        visibility_threshold: float = 0.5,
        frame_is_mirrored: bool = False,
    ):
        if calib is None:
            from calibration import load_calibration
            calib = load_calibration(verbose=False)
        self.calib = calib
        self.target_width = target_width
        self.target_height = target_height
        self.vis_th = visibility_threshold
        self.frame_is_mirrored = frame_is_mirrored

        self.train_neck = np.array(calib["neck"], dtype=np.float32)
        self.train_sw = float(calib["shoulder_width"]) or 1.0
        conf = np.array(calib["conf_mean"], dtype=np.float32)
        self.train_conf = np.where(conf > 0.05, conf, 0.8).astype(np.float32)
        self.want_r_left = bool(calib["r_shoulder_on_image_left"])
        self.first_block_left = bool(calib["first_hand_block_is_left"])

        pose_name, pose_url = POSE_MODELS.get(pose_model, POSE_MODELS["full"])
        pose_path = MODEL_DIR / pose_name
        download_model(pose_url, pose_path)
        download_model(HAND_URL, HAND_MODEL)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # وضعیت هم‌ترازی (EMA تا لرزش گردن/شانه حرکت دست را خراب نکند)
        self.neck_px: Optional[np.ndarray] = None
        self.sw_px: Optional[float] = None
        self.flip_votes = 0
        self._last_ts_ms = -1

        # سازگاری با API قبلی (sliding window)
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.SEQUENCE_LENGTH)

        print("✅ KeypointExtractor ready (OpenPose-compatible mapping + calibration)")

    # ------------------------------------------------------------------ utils
    def _timestamp_ms(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    @staticmethod
    def _lm_array(landmarks, w, h, default_vis=1.0) -> np.ndarray:
        out = np.zeros((len(landmarks), 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            vis = getattr(lm, "visibility", None)
            if vis is None:
                vis = default_vis
            out[i] = (lm.x * w, lm.y * h, vis)
        return out

    def _update_alignment(self, pose_px: Optional[np.ndarray]):
        if pose_px is None:
            return
        ls, rs = pose_px[MP_L_SH], pose_px[MP_R_SH]
        if ls[2] < self.vis_th or rs[2] < self.vis_th:
            return
        neck = (ls[:2] + rs[:2]) / 2
        sw = float(np.hypot(*(ls[:2] - rs[:2])))
        if sw < 10:
            return
        # آیا تصویر باید آینه شود تا با جهت داده‌ی آموزش یکی شود؟
        r_left = rs[0] < ls[0]
        self.flip_votes = int(np.clip(self.flip_votes + (1 if r_left != self.want_r_left else -1), -15, 15))
        a = 0.15
        if self.neck_px is None:
            self.neck_px, self.sw_px = neck, sw
        else:
            self.neck_px = (1 - a) * self.neck_px + a * neck
            self.sw_px = (1 - a) * self.sw_px + a * sw

    def _to_train_space(self, pts_px: np.ndarray) -> np.ndarray:
        """pts_px: (n,2) پیکسل → فضای مختصات داده‌ی آموزش"""
        neck = self.neck_px if self.neck_px is not None else np.array(
            [self.target_width / 2, self.target_height * 0.45], dtype=np.float32)
        sw = self.sw_px if self.sw_px is not None else self.target_width * 0.3
        p = pts_px - neck
        if self.flip_votes > 0:
            p[:, 0] = -p[:, 0]
        return p * (self.train_sw / sw) + self.train_neck

    def _assign_hands(self, hand_result, pose_px, w, h):
        """خروجی: dict {'left': arr(21,3) px, 'right': ...} بر اساس دست واقعی فرد"""
        hands = {"left": None, "right": None}
        if not hand_result.hand_landmarks:
            return hands
        cands = []
        for i, lms in enumerate(hand_result.hand_landmarks):
            arr = self._lm_array(lms, w, h, default_vis=1.0)
            label = hand_result.handedness[i][0].category_name
            score = hand_result.handedness[i][0].score
            # MediaPipe فرض می‌کند تصویر آینه‌ای است؛ برای وبکم خام برعکس است
            side = label.lower()
            if not self.frame_is_mirrored:
                side = "right" if side == "left" else "left"
            cands.append({"arr": arr, "side": side, "score": score})

        wr = {}
        if pose_px is not None:
            if pose_px[MP_L_WR][2] > 0.3:
                wr["left"] = pose_px[MP_L_WR][:2]
            if pose_px[MP_R_WR][2] > 0.3:
                wr["right"] = pose_px[MP_R_WR][:2]

        if len(wr) == 2 and len(cands) == 2:
            d = lambda c, s: np.hypot(*(c["arr"][0, :2] - wr[s]))
            straight = d(cands[0], "left") + d(cands[1], "right")
            swapped = d(cands[0], "right") + d(cands[1], "left")
            if straight <= swapped:
                hands["left"], hands["right"] = cands[0]["arr"], cands[1]["arr"]
            else:
                hands["left"], hands["right"] = cands[1]["arr"], cands[0]["arr"]
            return hands

        sw = self.sw_px or w * 0.3
        for c in cands:
            side = c["side"]
            if wr:
                dists = {s: np.hypot(*(c["arr"][0, :2] - p)) for s, p in wr.items()}
                best = min(dists, key=dists.get)
                if dists[best] < 0.8 * sw:
                    side = best
            if hands[side] is None:
                hands[side] = c["arr"]
            else:
                other = "left" if side == "right" else "right"
                if hands[other] is None:
                    hands[other] = c["arr"]
        return hands

    # ------------------------------------------------------------------ main
    def extract(self, frame: np.ndarray) -> Optional[FrameData]:
        if frame is None or frame.size == 0:
            return None
        h0, w0 = frame.shape[:2]
        if (w0, h0) != (self.target_width, self.target_height):
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms()

        pose_result = self.pose_landmarker.detect_for_video(mp_image, ts)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, ts)

        pose_px = None
        if pose_result.pose_landmarks:
            pose_px = self._lm_array(pose_result.pose_landmarks[0], w, h, default_vis=0.0)
        self._update_alignment(pose_px)
        hands = self._assign_hands(hand_result, pose_px, w, h)

        kp = np.zeros((self.NUM_CHANNELS, self.NUM_KEYPOINTS), dtype=np.float32)

        # ---- body (BODY_25)
        if pose_px is not None:
            pts = np.zeros((25, 2), dtype=np.float32)
            ok = np.zeros(25, dtype=bool)
            for i, src in enumerate(BODY25_FROM_MP):
                if isinstance(src, tuple):
                    a, b = pose_px[src[0]], pose_px[src[1]]
                    vis = min(a[2], b[2])
                    xy = (a[:2] + b[:2]) / 2
                else:
                    vis = pose_px[src][2]
                    xy = pose_px[src][:2]
                inside = (-0.02 * w <= xy[0] <= 1.02 * w) and (-0.02 * h <= xy[1] <= 1.02 * h)
                if vis >= self.vis_th and inside:
                    pts[i], ok[i] = xy, True
            if ok.any():
                tp = self._to_train_space(pts[ok])
                kp[0, :25][ok] = tp[:, 0]
                kp[1, :25][ok] = tp[:, 1]
                kp[2, :25][ok] = self.train_conf[:25][ok]

        # ---- hands
        blocks = {"left": 25, "right": 46} if self.first_block_left else {"left": 46, "right": 25}
        wrists = {"left": None, "right": None}
        for side, arr in hands.items():
            if arr is None:
                continue
            s = blocks[side]
            tp = self._to_train_space(arr[:, :2].copy())
            kp[0, s:s + 21] = tp[:, 0]
            kp[1, s:s + 21] = tp[:, 1]
            kp[2, s:s + 21] = self.train_conf[s:s + 21]
            wrists[side] = tp[0] / self.train_sw

        kp = np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)
        n_hands = sum(v is not None for v in hands.values())

        return FrameData(
            keypoints=kp,
            has_hands=n_hands > 0,
            num_hands=n_hands,
            wrists=wrists,
            timestamp=time.monotonic(),
            draw_pose=pose_px,
            draw_hands=[a for a in hands.values() if a is not None],
        )

    # -------------------------------------------------- legacy sliding window
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        fd = self.extract(frame)
        if fd is None:
            return None
        self.buffer.append(fd.keypoints)
        if len(self.buffer) < self.SEQUENCE_LENGTH:
            return None
        return np.stack(list(self.buffer), axis=1)  # (3, 112, 67)

    def reset(self):
        self.buffer.clear()

    def get_buffer_status(self) -> Tuple[int, int]:
        return len(self.buffer), self.SEQUENCE_LENGTH

    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()
