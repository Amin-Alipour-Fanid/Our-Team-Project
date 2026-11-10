# -*- coding: utf-8 -*-
"""
PHASE 1 (retry) - FIX T5: MediaPipe TASKS API functional test

ERROR from the PHASE 1 run:
    mp.solutions.holistic not available in this build
    (mp.solutions missing entirely; mediapipe.tasks IS present)

PROBABLE CAUSE:
    This MediaPipe build (1.0.1, Python 3.13) ships only the modern
    Tasks API. The legacy Solutions API (mp.solutions.*) is not included.

SINGLE SOLUTION:
    Use the Tasks API instead:
        PoseLandmarker  -> 33 body landmarks       (VIDEO mode)
        HandLandmarker  -> up to 2 hands x 21      (VIDEO mode)
    These are exactly the components the ISLR101 camera pipeline needs
    (dataset = 25 body points + 2 x 21 hand points; no face mesh needed).

    Two model files are downloaded ONCE into E:\new-project\models\ :
        pose_landmarker_lite.task   (~5.5 MB)
        hand_landmarker.task        (~7.8 MB)

TEST SEQUENCE:
    [A] Tasks API structure check
    [B] model download (skipped if files already exist)
    [C] black frame  -> must run without crash, expect 0 poses / 0 hands
    [D] live camera  -> expect 1 pose with 33 landmarks; report hands

PASS: A + B + C + (camera pose == 33 landmarks).
Hands are reported; if no hand was visible, rerun this script once
with both hands raised before PHASE 9.

Internet is required for the first run (model download only).

Run from project root with venv active:
    python scripts\fix_t5_phase1.py
"""

import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(r"E:\new-project")
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = {
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    ),
}

RESULTS = {}


def head(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def record(name, ok, info=""):
    RESULTS[name] = (ok, info)
    print(f"  >>> [{name}] {'PASS' if ok else 'FAIL'}  {info}")


def to_mp_image(rgb):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb))


def download(fname, url, dest):
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  cached  : {fname}")
        return True
    print(f"  download: {url}")
    tmp = dest.with_suffix(".part")
    try:
        from urllib.request import urlopen
        with urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done / 1048576:5.1f} / {total / 1048576:.1f} MB",
                              end="")
        print()
        tmp.replace(dest)
        print(f"  saved   : {dest}")
        return True
    except Exception as e:
        print(f"\n  [DOWNLOAD ERROR] {type(e).__name__}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def main():
    print("PHASE 1 (retry) - FIX T5: MEDIAPIPE TASKS API")

    # ------------------------------------------------------------------ [A]
    head("[A] MEDIAPIPE TASKS API STRUCTURE")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
    except Exception as e:
        record("A_structure", False, f"import failed: {type(e).__name__}: {e}")
        return
    print(f"  mediapipe : {mp.__version__}")
    print(f"  location  : {mp.__file__}")
    missing = []
    if not hasattr(mp, "Image"):
        missing.append("mp.Image")
    if not hasattr(mp, "ImageFormat"):
        missing.append("mp.ImageFormat")
    if not hasattr(mp_tasks, "BaseOptions"):
        missing.append("tasks.BaseOptions")
    for n in ("PoseLandmarker", "PoseLandmarkerOptions",
              "HandLandmarker", "HandLandmarkerOptions", "RunningMode"):
        if not hasattr(vision, n):
            missing.append(f"vision.{n}")
    if missing:
        record("A_structure", False, f"missing: {missing}")
        return
    try:
        modes = [m.name for m in vision.RunningMode]
    except Exception:
        modes = ["(unavailable)"]
    print(f"  RunningMode : {modes}")
    record("A_structure", True, "PoseLandmarker + HandLandmarker available")

    # ------------------------------------------------------------------ [B]
    head("[B] MODEL FILES (one-time download)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ok_b = True
    for key, (fname, url) in MODELS.items():
        if not download(fname, url, MODELS_DIR / fname):
            ok_b = False
    if not ok_b:
        print("  internet is required ONCE for the model download;")
        print("  check the connection/firewall and rerun this script.")
    record("B_models", ok_b, "model files ready" if ok_b else "download failed")

    # ------------------------------------------------------------------ [C]
    head("[C] BLACK FRAME (must run without crash; expect no detections)")
    try:
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(MODELS_DIR / MODELS["pose"][0])),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(MODELS_DIR / MODELS["hand"][0])),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        pose_det = vision.PoseLandmarker.create_from_options(pose_opts)
        hand_det = vision.HandLandmarker.create_from_options(hand_opts)

        black = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = 33
        p_res = pose_det.detect_for_video(to_mp_image(black), ts)
        ts += 33
        h_res = hand_det.detect_for_video(to_mp_image(black), ts)
        n_pose = len(p_res.pose_landmarks)
        n_hands = len(h_res.hand_landmarks)
        print(f"  black frame -> poses={n_pose} hands={n_hands} (expected 0 / 0)")
        record("C_black_frame", n_pose == 0 and n_hands == 0,
               f"poses={n_pose}, hands={n_hands}, no crash")
    except Exception as e:
        import traceback
        traceback.print_exc()
        record("C_black_frame", False, f"{type(e).__name__}: {e}")
        return

    # ------------------------------------------------------------------ [D]
    head("[D] LIVE CAMERA FRAME (pose must give 33 landmarks)")
    import cv2
    cap = None
    for idx, api in ((0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF), (1, cv2.CAP_DSHOW)):
        c = cv2.VideoCapture(idx, api)
        if c.isOpened():
            cap = c
            break
        c.release()
    if cap is None:
        record("D_camera", False, "camera could not be opened")
        return
    print("  camera opened - SIT BACK so your UPPER BODY is visible,")
    print("  and keep BOTH HANDS raised in view of the camera.")
    print("  (up to 10 seconds of searching follows)")
    got, pose_lm, hand_info = False, None, None
    ts = 66
    t_end = time.time() + 10.0
    last_msg = 0.0
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = frame[:, :, ::-1]
        ts += 33
        p_res = pose_det.detect_for_video(to_mp_image(rgb), ts)
        h_res = hand_det.detect_for_video(to_mp_image(rgb), ts)
        n_pose = len(p_res.pose_landmarks)
        n_hands = len(h_res.hand_landmarks)
        now = time.time()
        if now - last_msg >= 1.0:
            last_msg = now
            left = t_end - now
            print(f"    searching... pose={'YES' if n_pose else 'no'} "
                  f"hands={n_hands}  ({left:.0f}s left)")
        if n_pose >= 1:
            got = True
            pose_lm = p_res.pose_landmarks[0]
            hand_info = []
            for hl, hd in zip(h_res.hand_landmarks, h_res.handedness):
                label = hd[0].category_name if hd else "?"
                hand_info.append((label, len(hl)))
            break
    cap.release()
    if not got:
        record("D_camera", False, "no pose detected within 10 s "
                                  "(lighting / distance / upper body not visible)")
        return
    n_pose_lm = len(pose_lm)
    print(f"  pose landmarks : {n_pose_lm}  (expected 33)")
    sample = pose_lm[0]
    print(f"  sample landmark 0 (nose): x={sample.x:.3f} y={sample.y:.3f} "
          f"z={sample.z:.3f} vis={getattr(sample, 'visibility', None)}")
    if hand_info:
        for label, n in hand_info:
            print(f"  hand: {label:<6s} {n} landmarks (expected 21)")
    else:
        print("  hand: none visible in this frame")
    pose_ok = n_pose_lm == 33
    hands_ok = bool(hand_info) and all(n == 21 for _, n in hand_info)
    if pose_ok and hands_ok:
        record("D_camera", True, f"pose={n_pose_lm}, hands={hand_info}")
    elif pose_ok:
        record("D_camera", True,
               f"pose={n_pose_lm} OK, but no valid hand detected - "
               f"RERUN this script once with both hands raised before PHASE 9")
    else:
        record("D_camera", False, f"unexpected pose landmark count {n_pose_lm}")

    # -------------------------------------------------------------- SUMMARY
    head("SUMMARY")
    ok_all = all(ok for ok, _ in RESULTS.values())
    for k in sorted(RESULTS):
        ok, info = RESULTS[k]
        print(f"  {'PASS' if ok else 'FAIL':4s}  {k:16s} {info}")
    print(f"\n  T5 FINAL: {'PASS' if ok_all else 'FAIL'}")
    if ok_all:
        print("  PHASE 1 can now be declared PASSED (T5 fixed via Tasks API).")
    print("  Send the FULL output back.")


if __name__ == "__main__":
    main()