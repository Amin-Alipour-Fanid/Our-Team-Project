# -*- coding: utf-8 -*-
"""
INPUT_SPEC - the single source of truth for the model input contract.

ISLR101 - Iranian Sign Language -> Persian Text -> Persian Speech

Every property in this module was MEASURED on the real dataset
(PHASE 0-2 audits); nothing here is assumed:

  raw dataset arrays : (B, 3, 112, 67, 1) float32
                       axis 1 = channels (x, y, confidence)
                       axis 2 = TIME (chronological; smoothness ratio 0.149)
                       axis 3 = 67 keypoints; trailing axis is a singleton
  padding            : exact zero tail on the time axis (no internal gaps)
  model input        : (B, T=112, F=201) float32, chronological, zero-padded
  feature layout     : f = 3*k + c ->
                       [x0,y0,c0, x1,y1,c1, ..., x66,y66,c66]
  keypoint layout    : k00-24 body (BODY_25 order; feet k19-24 ~always missing)
                       k25-45 hand A @ LEFT wrist k07 (wrist-first 21 pts)
                       k46-66 hand B @ RIGHT wrist k04 (wrist-first 21 pts)
  coordinate conv.   : x image-like, y UP-positive, image-normalized scale
  confidence channel : c=2; value 0 <=> keypoint missing
  normalization      : applied later by src/preprocessing.py (PHASE 5) using
                       TRAIN-ONLY statistics - identical for val/test/camera

DOCUMENTED LIMITATION (measured, PHASE 2 part 3): the dataset's original
anchor is a fixed image point near the frame center (frame bottom edge
measured at y ~ -0.494; exact value not uniquely determinable). It is
neutralized by the unified preprocessing (PHASE 5), which re-anchors every
active frame on the neck (presence 100%, measured) and rescales by
shoulder width - identically for train/val/test/camera.

ALL conversions between the dataset layout and the model layout MUST go
through this module. No other module may transpose/reshape on its own.
"""

import numpy as np

# --------------------------------------------------------------- lengths
T = 112             # timesteps per sequence
K = 67              # keypoints per frame
C = 3               # channels per keypoint (x, y, confidence)
F = K * C           # 201 features per timestep
NUM_CLASSES = 101   # class ids 0..100

# --------------------------------------------------------------- channels
CH_X, CH_Y, CH_CONF = 0, 1, 2
CHANNEL_NAMES = ("x", "y", "confidence")

# --------------------------------------------------------- keypoint groups
BODY = slice(0, 25)          # BODY_25-order body skeleton
HAND_A = slice(25, 46)       # hand attached to LEFT wrist (k07)
HAND_B = slice(46, 67)       # hand attached to RIGHT wrist (k04)

BODY_25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]
HAND_21_NAMES = [
    "Wrist",
    "ThumbCMC", "ThumbMCP", "ThumbIP", "ThumbTip",
    "IndexMCP", "IndexPIP", "IndexDIP", "IndexTip",
    "MiddleMCP", "MiddlePIP", "MiddleDIP", "MiddleTip",
    "RingMCP", "RingPIP", "RingDIP", "RingTip",
    "PinkyMCP", "PinkyPIP", "PinkyDIP", "PinkyTip",
]
KEYPOINT_NAMES = (
    list(BODY_25_NAMES)
    + ["HandA_" + n for n in HAND_21_NAMES]
    + ["HandB_" + n for n in HAND_21_NAMES]
)

# frequently used indices (measured anchors of the layout)
KP_NOSE = 0
KP_NECK = 1
KP_RSHO, KP_RELB, KP_RWRI = 2, 3, 4
KP_LSHO, KP_LELB, KP_LWRI = 5, 6, 7
KP_MHIP = 8
KP_HAND_A_WRIST = 25
KP_HAND_B_WRIST = 46


def feature_index(k, c):
    """Index of keypoint k / channel c inside the 201-dim feature vector."""
    if not (0 <= k < K):
        raise ValueError(f"keypoint {k} out of range 0..{K - 1}")
    if c not in (CH_X, CH_Y, CH_CONF):
        raise ValueError(f"channel {c} out of range 0..2")
    return 3 * k + c


# ------------------------------------------------------------ conversions
def normalize_array_shape(x):
    """
    Accept the dataset layout (B, 3, 112, 67[, 1...]) and return (B, C, T, K).
    Values are never changed - only singleton trailing axes are removed.
    Anything else raises ValueError.
    """
    a = np.asarray(x)
    if a.ndim >= 5:
        if any(s != 1 for s in a.shape[4:]):
            raise ValueError(f"rank-{a.ndim} shape {a.shape}: trailing axes "
                             "must all be singletons")
        a = a.reshape(tuple(a.shape[:4]))
    if a.ndim != 4:
        raise ValueError(f"expected a batch (B, {C}, {T}, {K}[, 1]); "
                         f"got rank {a.ndim} shape {a.shape}")
    if a.shape[1:] != (C, T, K):
        raise ValueError(
            f"dataset layout must be (B, C={C}, T={T}, K={K}); got {a.shape}. "
            "The TIME axis is the one of length 112 - never feed (B, 112, 3, 67)."
        )
    return a


def to_model_input(x):
    """
    Dataset batch (B, 3, 112, 67[, 1]) -> model input (B, 112, 201) float32.

    x_model[b, t, 3*k + c] = x_dataset[b, c, t, k]
    The chronological frame order (dataset axis 2) becomes the sequence
    axis (axis 1). Padding (zero tail) is preserved exactly.
    """
    a = normalize_array_shape(x)
    return np.ascontiguousarray(
        a.transpose(0, 2, 3, 1).reshape(a.shape[0], T, F),
        dtype=np.float32,
    )


def from_model_input(xm):
    """Exact inverse of to_model_input: (B, 112, 201) -> (B, 3, 112, 67)."""
    a = assert_model_input(xm)
    return np.ascontiguousarray(
        a.reshape(a.shape[0], T, K, C).transpose(0, 3, 1, 2),
        dtype=np.float32,
    )


# -------------------------------------------------------------- validation
def assert_model_input(xm, name="model_input"):
    """Validate the (B, 112, 201) model layout; return it as an ndarray."""
    a = np.asarray(xm)
    if a.ndim != 3 or a.shape[1:] != (T, F):
        raise ValueError(f"{name} must have shape (B, {T}, {F}); got {a.shape}")
    return a


def as_model_input(xm, name="model_input"):
    """Validate the model layout and return it float32 / C-contiguous."""
    return np.ascontiguousarray(assert_model_input(xm, name), dtype=np.float32)


# ------------------------------------------------------------------ masks
def frame_mask(xm, eps=1e-9):
    """Boolean (B, T) mask of active (non-padded) frames of a model input."""
    a = assert_model_input(xm)
    return np.abs(a).sum(axis=2) > eps


def effective_lengths(xm, eps=1e-9):
    """1-based index of the last active frame per sequence; shape (B,)."""
    m = frame_mask(xm, eps)
    idx = np.arange(1, T + 1)
    return (m * idx).max(axis=1)


def describe():
    """Human-readable contract summary (for logs and reports)."""
    return (
        "INPUT_SPEC: model input (B, 112, 201) float32 | "
        "f = 3*k + c (c: 0=x, 1=y, 2=confidence) | "
        "k00-24 body(BODY_25), k25-45 handA@Lwrist, k46-66 handB@Rwrist | "
        "chronological frames, zero-padded tail | "
        "x image-like, y UP-positive, image-normalized scale | "
        "conf 0 <=> missing | 101 classes"
    )