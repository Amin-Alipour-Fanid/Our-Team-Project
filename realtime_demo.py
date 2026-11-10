"""
Real-time Demo for ISLR101 (BiGRU) - Fixed

تغییر اصلی: به جای sliding window ثابت 112 فریمی (که هیچ شباهتی به نمونه‌های
آموزش نداشت)، هر «اشاره» جداگانه تشخیص داده می‌شود (از بالا آمدن دست تا پایین
آمدنش) و مثل داده‌ی دیتاست به 112 فریم تبدیل می‌شود.

Keys:
    q      خروج
    r      ریست
    SPACE  حالت دستی: شروع/پایان ضبط یک اشاره
    m      تغییر بین حالت خودکار و دستی

برای نمایش درست فارسی (اختیاری):  pip install arabic-reshaper python-bidi
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration import load_calibration
from keypoint_extractor import KeypointExtractor
from realtime_inference import SignLanguagePredictor

# ---------------------------------------------------------------- settings
CAM_INDEX = 0
FRAME_IS_MIRRORED = False   # اگر تصویر وبکم شما از قبل آینه‌ای است True کنید
START_FRAMES = 3            # چند فریم پشت سر هم دست دیده شود تا ضبط شروع شود
END_NO_HAND_SEC = 0.35      # نبودن دست به این مدت = پایان اشاره
END_STILL_SEC = 0.9         # بی‌حرکت ماندن دست به این مدت = پایان اشاره
STILL_TH = 0.012            # آستانه حرکت مچ (واحد: پهنای شانه در هر فریم)
MIN_SIGN_SEC = 0.4
MAX_SIGN_SEC = 4.0
PRE_ROLL = 6                # چند فریم قبل از شروع هم اضافه شود
MIN_CONF = 0.20             # زیر این مقدار «نامطمئن» نمایش داده می‌شود

# ---------------------------------------------------------------- classes
CLASS_NAMES = {
    0: "آدرس",
    1: "آمد",
    2: "آسان",
    3: "آزمایشگاه",
    4: "دکتری",
    5: "باران",
    6: "باید",
    7: "باز",
    8: "بازدید کردن",
    9: "بازنشسته",
    10: "بد",
    11: "بد آمدن",
    12: "برف",
    13: "بسته",
    14: "به‌دست آوردن",
    15: "بیمار",
    16: "بزرگ",
    17: "چه خبر",
    18: "چرا",
    19: "چیه",
    20: "کلاس",
    21: "کامپیوتر",
    22: "کرونا",
    23: "دادن",
    24: "دانشگاه",
    25: "دانشجو",
    26: "داشتن",
    27: "درس",
    28: "دیدن",
    29: "دیر",
    30: "دیروز",
    31: "دکتر",
    32: "دوست داشتن",
    33: "امروز",
    34: "امسال",
    35: "امشب",
    36: "امتحان",
    37: "اسم",
    38: "فردا",
    39: "فارغ‌التحصیل",
    40: "فرستادن",
    41: "گرفتن",
    42: "قبول شدن",
    43: "ها",
    44: "حرف زدن",
    45: "هندسه",
    46: "هنر",
    47: "جواب دادن",
    48: "جغرافیا",
    49: "جست‌وجو",
    50: "جزوه",
    51: "کارگاه",
    52: "کم",
    53: "کتاب",
    54: "کتابخانه",
    55: "کلید",
    56: "خوابگاه",
    57: "خواندن",
    58: "خواستن",
    59: "خانواده",
    60: "خوب",
    61: "خوشحال",
    62: "کی",
    63: "کوچک",
    64: "کدام",
    65: "کجا",
    66: "کمک کردن",
    67: "ما",
    68: "مقاله",
    69: "من",
    70: "مترو",
    71: "معدل",
    72: "مهندس",
    73: "ناراحت",
    74: "نه",
    75: "نباید",
    76: "نوشتن",
    77: "استاد",
    78: "راهنمایی کردن",
    79: "رفتن",
    80: "رشته تحصیلی",
    81: "ریاضی",
    82: "ثبت‌نام کردن",
    83: "سخت",
    84: "سلام",
    85: "شنیدن",
    86: "شیمی",
    87: "شروع شدن",
    88: "صحبت کردن",
    89: "سخنرانی",
    90: "تازه",
    91: "تدریس کردن",
    92: "تمام شدن",
    93: "توانستن",
    94: "تو",
    95: "و",
    96: "یاد دادن",
    97: "یاد گرفتن",
    98: "ذخیره کردن",
    99: "زیاد",
    100: "زود",
}


def get_class_name(idx: int) -> str:
    return CLASS_NAMES.get(idx, f"کلاس_{idx}")


# ---------------------------------------------------------------- Persian text
def _load_font(size):
    for p in ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf",
              "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_RESHAPER = True
except ImportError:
    _HAS_RESHAPER = False
_HAS_RAQM = features.check("raqm")
_FONTS = {}


def draw_texts(img, items):
    """items: list of (text, (x, y), size, (B,G,R))"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, (x, y), size, bgr in items:
        font = _FONTS.setdefault(size, _load_font(size))
        rgb = (bgr[2], bgr[1], bgr[0])
        if _HAS_RAQM:
            d.text((x, y), text, font=font, fill=rgb, direction="rtl" if _is_rtl(text) else None)
        elif _HAS_RESHAPER:
            d.text((x, y), get_display(arabic_reshaper.reshape(text)), font=font, fill=rgb)
        else:
            d.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _is_rtl(t):
    return any("\u0600" <= ch <= "\u06ff" for ch in t)


def draw_landmarks(img, fd):
    if fd is None:
        return
    if fd.draw_pose is not None:
        for i in (0, 11, 12, 13, 14, 15, 16):
            x, y, v = fd.draw_pose[i]
            if v > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (255, 180, 0), -1)
    for hand in fd.draw_hands:
        for x, y, _ in hand:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)


# ---------------------------------------------------------------- segmenter
class SignSegmenter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.recording = False
        self.frames, self.times = [], []
        self.pre = deque(maxlen=PRE_ROLL)
        self.hand_streak = 0
        self.last_hand_t = 0.0
        self.last_move_t = 0.0
        self.prev_wrists = None

    def _motion(self, fd):
        if self.prev_wrists is None:
            m = 0.0
        else:
            ds = [np.linalg.norm(fd.wrists[s] - self.prev_wrists[s])
                  for s in ("left", "right")
                  if fd.wrists[s] is not None and self.prev_wrists.get(s) is not None]
            m = max(ds) if ds else 0.0
        self.prev_wrists = {k: (None if v is None else v.copy()) for k, v in fd.wrists.items()}
        return m

    def start(self, now):
        self.recording = True
        self.frames = [p[0] for p in self.pre]
        self.times = [p[1] for p in self.pre]
        self.last_hand_t = self.last_move_t = now

    def finish(self):
        """خروجی: (frames, duration) یا None"""
        self.recording = False
        frames, times = self.frames, self.times
        self.frames, self.times = [], []
        self.hand_streak = 0
        if len(frames) < 6:
            return None
        dur = times[-1] - times[0]
        if dur < MIN_SIGN_SEC:
            return None
        return np.stack(frames), dur

    def update(self, fd, auto=True):
        now = fd.timestamp
        motion = self._motion(fd)
        if fd.has_hands:
            self.hand_streak += 1
            self.last_hand_t = now
            if motion > STILL_TH:
                self.last_move_t = now
        else:
            self.hand_streak = 0

        if not self.recording:
            self.pre.append((fd.keypoints, now))
            if auto and self.hand_streak >= START_FRAMES and motion > STILL_TH:
                self.start(now)
            return None

        self.frames.append(fd.keypoints)
        self.times.append(now)
        if not auto:
            if now - self.times[0] > MAX_SIGN_SEC * 2:
                return self.finish()
            return None

        if (now - self.last_hand_t > END_NO_HAND_SEC
                or now - self.last_move_t > END_STILL_SEC
                or now - self.times[0] > MAX_SIGN_SEC):
            # فریم‌های انتهایی بدون دست/بی‌حرکت را حذف کن
            keep_until = max(self.last_hand_t, self.last_move_t) + 0.1
            while len(self.times) > 1 and self.times[-1] > keep_until:
                self.times.pop(); self.frames.pop()
            return self.finish()
        return None


# ---------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("ISLR101 Real-time Demo (sign segmentation + calibration)")
    print("Keys: q quit | r reset | SPACE manual record | m auto/manual")
    print("=" * 60)
    if not (_HAS_RAQM or _HAS_RESHAPER):
        print("ℹ️ برای نمایش درست فارسی: pip install arabic-reshaper python-bidi")

    calib = load_calibration()
    try:
        predictor = SignLanguagePredictor(calib=calib)
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        return
    predictor.verify_on_test(n=200)

    extractor = KeypointExtractor(calib=calib, target_width=640, target_height=480,
                                  frame_is_mirrored=FRAME_IS_MIRRORED)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        extractor.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    seg = SignSegmenter()
    auto = True
    last_top3, history = [], deque(maxlen=6)
    fps, t_prev = 0.0, time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fd = extractor.extract(frame)
            if fd is None:
                continue

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            out = seg.update(fd, auto=auto)
            if out is not None:
                frames, dur = out
                try:
                    r = predictor.predict_segment(frames, duration_sec=dur)
                    p = r["probabilities"]
                    top = np.argsort(p)[::-1][:3]
                    last_top3 = [(int(i), float(p[i])) for i in top]
                    if last_top3[0][1] >= MIN_CONF:
                        history.appendleft(get_class_name(last_top3[0][0]))
                    print(f"[{len(frames)} frames, {dur:.2f}s] " +
                          " | ".join(f"{get_class_name(i)} {c*100:.1f}%" for i, c in last_top3))
                except Exception as e:
                    print(f"Inference error: {e}")

            display = frame.copy()
            if display.shape[1] != 640:
                display = cv2.resize(display, (640, 480))
            draw_landmarks(display, fd)

            status = ("● REC" if seg.recording else ("AUTO" if auto else "MANUAL: SPACE"))
            col = (0, 0, 255) if seg.recording else (0, 200, 0)
            cv2.putText(display, f"{status}  fps:{fps:.0f}  hands:{fd.num_hands}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            texts = []
            if last_top3:
                for i, (cls, conf) in enumerate(last_top3):
                    name = get_class_name(cls)
                    if i == 0 and conf < MIN_CONF:
                        name += " (نامطمئن)"
                    texts.append((f"{i+1}. {name}  {conf*100:.1f}%", (10, 50 + i * 36),
                                  28 if i == 0 else 22,
                                  (0, 255, 0) if i == 0 else (220, 220, 220)))
            else:
                texts.append(("دست‌ها را بالا بیاورید و اشاره کنید", (10, 50), 24, (0, 165, 255)))
            if history:
                texts.append(("  ،  ".join(list(history)[::-1]), (10, display.shape[0] - 70), 22, (255, 255, 0)))
            texts.append(("q: خروج | r: ریست | SPACE: ضبط دستی | m: حالت", (10, display.shape[0] - 35), 18, (200, 200, 200)))
            display = draw_texts(display, texts)

            cv2.imshow("ISLR101 Real-time Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                seg.reset(); last_top3 = []; history.clear()
            elif key == ord("m"):
                auto = not auto; seg.reset()
                print("Mode:", "AUTO" if auto else "MANUAL")
            elif key == ord(" "):
                if not seg.recording:
                    seg.start(fd.timestamp)
                else:
                    out = seg.finish()
                    if out is not None:
                        frames, dur = out
                        r = predictor.predict_segment(frames, duration_sec=dur)
                        p = r["probabilities"]
                        top = np.argsort(p)[::-1][:3]
                        last_top3 = [(int(i), float(p[i])) for i in top]
                        history.appendleft(get_class_name(last_top3[0][0]))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("✅ Demo finished")


if __name__ == "__main__":
    main()
