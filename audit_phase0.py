# -*- coding: utf-8 -*-
"""
PHASE 0 - Audit Current Project (READ-ONLY) - v2

ISLR101 - Iranian Sign Language -> Persian Text -> Persian Speech

v1 findings:
  - x arrays are RANK-5: (B, 3, 112, 67, 1)  -> trailing singleton axis
  - value statistics were skipped because of the unexpected rank
  - Python 3.13.4, venv at E:\new-project\.venv, no ML packages installed

v2 adds / fixes:
  1) squeezes singleton axes and re-checks axis interpretation
  2) per-channel statistics (coordinate space? confidence channel?)
  3) NaN / Inf check
  4) padding analysis along the candidate time axis (effective lengths)
  5) smoothness evidence: consecutive-frame delta vs consecutive-keypoint delta
  6) per-keypoint summary grids (blocks k00-24 / k25-45 / k46-66)
  7) label <-> word consistency from video names + full class->word list
  8) (subject, word) overlap between official train and test
  9) inspection of zip archives in the project root
 10) environment re-check

This script does NOT modify, create or delete any file.

Run from project root with venv active:
    python scripts\audit_phase0.py
"""

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(r"E:\new-project")

NPY = {
    "train_x": "islr101_train.npy",
    "test_x": "islr101_test.npy",
    "train_y": "islr101_train_label.npy",
    "test_y": "islr101_test_label.npy",
    "train_name": "islr101_train_name.npy",
    "test_name": "islr101_test_name.npy",
}

SRC_FILES = [
    r"src\dataset.py", r"src\preprocessing.py", r"src\keypoints.py",
    r"src\model.py", r"src\train.py", r"src\inference.py",
    r"src\camera.py", r"src\tts.py", r"src\utils.py",
    r"main.py", r"configs\bigru.yaml", r"requirements.txt", r"README.md",
]

EXPECTED = (3, 112, 67)          # (C, T, K) after squeezing singleton axes
NAME_RE = re.compile(r"^subject(\d+)_(.+)_video(\d+)$")


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def run_safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:
        print(f"\n  [AUDIT ERROR] in {fn.__name__}: {type(e).__name__}: {e}")
        print("  (audit continues with the next section)")


def load(path):
    try:
        return np.load(str(path), mmap_mode="r")
    except Exception:
        pass
    try:
        return np.load(str(path), allow_pickle=True)
    except Exception as e:
        print(f"  [ERROR] cannot load {path.name}: {type(e).__name__}: {e}")
        return None


def find_data_dir():
    for d in (PROJECT_ROOT / "data" / "raw",
              PROJECT_ROOT / "data" / "processed",
              PROJECT_ROOT / "data", PROJECT_ROOT):
        if d.exists() and all((d / f).exists() for f in NPY.values()):
            return d
    hits = sorted(PROJECT_ROOT.rglob(NPY["train_x"]))
    return hits[0].parent if hits else None


def squeeze_dims(shape):
    return tuple(d for d in shape if d != 1)


# ----------------------------------------------------------------------
def audit_files(data_dir):
    section("1-3) PROJECT / SOURCE FILES / DATASET FILES (condensed)")
    print(f"  project root exists : {PROJECT_ROOT.exists()}")
    missing = [f for f in SRC_FILES if not (PROJECT_ROOT / f).exists()]
    print(f"  source files        : {len(SRC_FILES) - len(missing)}/{len(SRC_FILES)} present, {len(missing)} missing")
    print(f"  data directory      : {data_dir}")
    for f in NPY.values():
        p = data_dir / f
        if p.exists():
            print(f"    EXISTS   {f}  ({p.stat().st_size / 1048576:.1f} MB)")
        else:
            print(f"    MISSING  {f}")
    print("  -> project is data-only; all source files will be built in later phases")


def audit_shapes(data_dir):
    section("4) SHAPES + SINGLETON-SQUEEZE + AXIS INTERPRETATION  [CRITICAL]")
    for key in ("train_x", "test_x"):
        arr = load(data_dir / NPY[key])
        if arr is None:
            continue
        raw = tuple(arr.shape)
        sq = squeeze_dims(raw)
        print(f"\n  {NPY[key]}")
        print(f"    raw shape = {raw}   (rank {len(raw)})")
        print(f"    squeezed  = {sq}   (rank {len(sq)})   dtype = {arr.dtype}")
        if len(sq) == 4 and sq[1:] == EXPECTED:
            print(f"    [OK] after removing singleton axes: (B={sq[0]}, C=3, T=112, K=67)")
            print(f"    -> BiGRU input = (B, 112, 3*67=201)")
        else:
            print(f"    [STOP] squeezed shape != (B, 3, 112, 67) -> do NOT proceed")


def audit_consistency(data_dir):
    section("5) ARRAY LENGTH CONSISTENCY (x / label / name)")
    for xk, yk, nk in (("train_x", "train_y", "train_name"),
                       ("test_x", "test_y", "test_name")):
        lens = {}
        for k in (xk, yk, nk):
            a = load(data_dir / NPY[k])
            if a is not None:
                lens[k] = a.shape[0]
        vals = set(lens.values())
        print(f"  {xk}/{yk}/{nk}: {lens} -> "
              f"{'[OK] match' if len(vals) == 1 else '[WARN] MISMATCH'}")


def audit_labels(data_dir):
    section("6) LABELS")
    for key in ("train_y", "test_y"):
        y = load(data_dir / NPY[key])
        if y is None:
            continue
        yv = np.asarray(y).ravel().astype(np.int64)
        counts = np.bincount(yv, minlength=101)
        print(f"  {NPY[key]}: n={yv.shape[0]} dtype={np.asarray(y).dtype} "
              f"min={yv.min()} max={yv.max()} unique={len(np.unique(yv))} "
              f"classes_present={int((counts > 0).sum())}/101")


def parse_names(arr):
    names = [str(x) for x in np.asarray(arr).ravel()]
    parsed, bad = [], []
    for nm in names:
        m = NAME_RE.match(nm)
        if m:
            parsed.append((int(m.group(1)), m.group(2), int(m.group(3))))
        else:
            bad.append(nm)
    return parsed, bad


def audit_names(data_dir):
    section("7) VIDEO NAMES: SIGNERS + LABEL<->WORD CONSISTENCY")
    data = {}
    for nk, yk in (("train_name", "train_y"), ("test_name", "test_y")):
        narr = load(data_dir / NPY[nk])
        yarr = load(data_dir / NPY[yk])
        if narr is None or yarr is None:
            continue
        parsed, bad = parse_names(narr)
        data[nk] = (parsed, np.asarray(yarr).ravel().astype(np.int64))
        print(f"\n  {NPY[nk]}: parsed {len(parsed)}, unparsed {len(bad)}")
        if bad:
            print(f"    unparsed examples: {bad[:5]}")
        subs = Counter(s for s, _, _ in parsed)
        print(f"    subjects: {len(subs)} unique -> {sorted(subs)}")
        if nk == "train_name":
            print("    train samples per subject: "
                  + ", ".join(f"s{k}={subs[k]}" for k in sorted(subs)))

    if "train_name" not in data or "test_name" not in data:
        return

    tr_parsed, tr_y = data["train_name"]
    te_parsed, te_y = data["test_name"]

    def build_map(parsed, ys):
        m, conflicts = {}, []
        for (s, w, v), lab in zip(parsed, ys.tolist()):
            if lab in m and m[lab] != w:
                conflicts.append((int(lab), m[lab], w))
            else:
                m[int(lab)] = w
        return m, conflicts

    tr_map, tr_conf = build_map(tr_parsed, tr_y)
    te_map, te_conf = build_map(te_parsed, te_y)
    print(f"\n  label->word conflicts inside train: {len(tr_conf)}"
          + (f"  examples: {tr_conf[:3]}" if tr_conf else "  [OK]"))
    print(f"  label->word conflicts inside test : {len(te_conf)}"
          + (f"  examples: {te_conf[:3]}" if te_conf else "  [OK]"))

    mism = [(c, tr_map[c], te_map.get(c)) for c in sorted(tr_map)
            if te_map.get(c) != tr_map[c]]
    print(f"  train/test label->word mismatches: {len(mism)}"
          + (f"  examples: {mism[:3]}" if mism else "  [OK]"))

    words = set(tr_map.values())
    print(f"  unique words in train map: {len(words)} (expected 101)")
    dupw = [w for w, c in Counter(tr_map.values()).items() if c > 1]
    if dupw:
        print(f"  [WARN] words shared by several classes: {dupw}")

    tr_pairs = {(s, w) for s, w, _ in tr_parsed}
    te_pairs = {(s, w) for s, w, _ in te_parsed}
    both = sorted(tr_pairs & te_pairs)
    print(f"  (subject, word) pairs in BOTH train and test: {len(both)}"
          f" / train {len(tr_pairs)} / test {len(te_pairs)}")
    if both:
        print(f"    examples: {both[:5]}")

    print("\n  -- full class -> word list (from names, verify visually) --")
    line = []
    for cid in sorted(tr_map):
        line.append(f"{cid:03d}={tr_map[cid]}")
        if len(line) == 3:
            print("    " + " | ".join(f"{x:<24}" for x in line))
            line = []
    if line:
        print("    " + " | ".join(f"{x:<24}" for x in line))


def audit_values(data_dir):
    section("8) VALUE STATS: CHANNELS / NaN / PADDING / SMOOTHNESS / KEYPOINTS")
    arr = load(data_dir / NPY["train_x"])
    if arr is None:
        return
    n = min(300, arr.shape[0])
    raw = np.array(arr[:n], dtype=np.float64)
    sample = raw.reshape(squeeze_dims(raw.shape))
    if sample.ndim != 4 or sample.shape[1:] != EXPECTED:
        print(f"  [SKIP] unexpected sample shape after squeeze: {sample.shape}")
        return
    print(f"  train sample: {sample.shape}")

    print("\n  -- per-channel statistics (train sample) --")
    for c in range(3):
        ch = sample[:, c]
        p1, p50, p99 = np.percentile(ch, [1.0, 50.0, 99.0])
        in01 = (ch.min() >= -1e-6) and (ch.max() <= 1.0 + 1e-6)
        print(f"    ch{c}: min={ch.min():10.4f} max={ch.max():10.4f} "
              f"mean={ch.mean():9.4f} std={ch.std():8.4f} | "
              f"p1={p1:8.4f} p50={p50:8.4f} p99={p99:8.4f} | within[0,1]={in01}")
    print(f"    NaNs={int(np.isnan(sample).sum())}  Infs={int(np.isinf(sample).sum())}")

    tarr = load(data_dir / NPY["test_x"])
    if tarr is not None:
        tn = min(150, tarr.shape[0])
        traw = np.array(tarr[:tn], dtype=np.float64)
        ts = traw.reshape(squeeze_dims(traw.shape))
        if ts.ndim == 4 and ts.shape[1:] == EXPECTED:
            print("\n  -- test sample channel ranges (must be compatible with train) --")
            for c in range(3):
                ch = ts[:, c]
                print(f"    ch{c}: min={ch.min():10.4f} max={ch.max():10.4f} "
                      f"mean={ch.mean():9.4f}")

    T = sample.shape[2]
    print("\n  -- padding analysis (time = axis 2) --")
    frame_nonzero = np.abs(sample).sum(axis=(1, 3)) > 1e-9      # (n, T)
    idx = np.arange(1, T + 1)
    eff = (frame_nonzero * idx).max(axis=1)                     # 1-based last active frame
    q0, q25, q50, q75, q100 = np.percentile(eff, [0, 25, 50, 75, 100])
    print(f"    effective length: min={q0:.0f} p25={q25:.1f} p50={q50:.1f} "
          f"p75={q75:.1f} max={q100:.0f}")
    print(f"    full-length (112 active frames): {int((eff == T).sum())}/{n}")
    print(f"    zero FIRST frame               : {int((~frame_nonzero[:, 0]).sum())}/{n}")
    print(f"    all frames zero                : {int((eff == 0).sum())}/{n}")
    gaps = sum(1 for i in range(n)
               if eff[i] > 0 and not bool(frame_nonzero[i, :int(eff[i])].all()))
    print(f"    zero frames INSIDE active region: {gaps}/{n}")

    kp_nonzero = np.abs(sample).sum(axis=(1, 2)) > 1e-9         # (n, K)
    print(f"    last keypoint (k=66) all-zero  : {int((~kp_nonzero[:, -1]).sum())}/{n} "
          f"(should be ~0 -> K is not a padded axis)")

    print("\n  -- smoothness evidence, channels 0-1, active frames only --")
    dts, dks = [], []
    for i in range(n):
        L = min(T, max(int(eff[i]), 2))
        seg = sample[i, :2, :L, :]
        dts.append(float(np.abs(np.diff(seg, axis=1)).mean()))   # along T
        dks.append(float(np.abs(np.diff(seg, axis=2)).mean()))   # along K
    mdt, mdk = float(np.mean(dts)), float(np.mean(dks))
    print(f"    mean |delta| consecutive FRAMES    (axis 2, T) = {mdt:.5f}")
    print(f"    mean |delta| consecutive KEYPOINTS (axis 3, K) = {mdk:.5f}")
    print(f"    ratio = {mdt / max(mdk, 1e-12):.4f}  (much smaller than 1 -> axis 2 is time)")

    print("\n  -- per-keypoint summary (train sample) --")
    mx = sample[:, 0].mean(axis=(0, 1))
    my = sample[:, 1].mean(axis=(0, 1))
    mc = sample[:, 2].mean(axis=(0, 1))
    zf = (~kp_nonzero).mean(axis=0)

    def grid(title, vals):
        print(f"    {title}")
        line = []
        for k in range(67):
            if k in (0, 25, 46):
                if line:
                    print("      " + " ".join(line))
                    line = []
                tag = "k00-24" if k == 0 else ("k25-45" if k == 25 else "k46-66")
                print(f"      [{tag}]")
            line.append(f"k{k:02d}={vals[k]:.2f}")
            if len(line) == 7:
                print("      " + " ".join(line))
                line = []
        if line:
            print("      " + " ".join(line))

    grid("mean channel-0 per keypoint:", mx)
    grid("mean channel-1 per keypoint:", my)
    grid("mean channel-2 per keypoint:", mc)
    grid("fraction of sequences where keypoint is ALL-ZERO:", zf)


def audit_zip():
    section("9) ZIP ARCHIVES IN PROJECT ROOT")
    zips = sorted(PROJECT_ROOT.glob("*.zip"))
    if not zips:
        print("  no zip files in project root")
        return
    for zp in zips[:3]:
        print(f"\n  {zp.name}  ({zp.stat().st_size / 1048576:.1f} MB)")
        try:
            with zipfile.ZipFile(zp) as z:
                names = z.namelist()
        except Exception as e:
            print(f"    [WARN] cannot open zip: {type(e).__name__}: {e}")
            print("    (multi-part/split zips must be joined before reading)")
            continue
        print(f"    total entries: {len(names)}")
        exts = Counter((Path(nm).suffix.lower() or "(none)") for nm in names)
        print(f"    extensions: {dict(exts.most_common(12))}")
        docs = [nm for nm in names
                if Path(nm).suffix.lower() in
                (".md", ".txt", ".pdf", ".json", ".py", ".csv", ".ipynb")]
        if docs:
            print(f"    documentation/code entries ({len(docs)}):")
            for nm in docs[:40]:
                print(f"      {nm}")
        else:
            print("    no documentation/code entries found inside the zip")


def audit_environment():
    section("10) ENVIRONMENT")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  Interpreter : {sys.executable}")
    print(f"  venv active : {sys.prefix != getattr(sys, 'base_prefix', sys.prefix)}")
    print(f"  numpy       : {np.__version__}")
    for mod in ("torch", "cv2", "mediapipe", "pyttsx3", "yaml"):
        try:
            m = __import__(mod)
            print(f"  {mod:12s}: {getattr(m, '__version__', 'installed')}")
        except Exception:
            print(f"  {mod:12s}: NOT INSTALLED")
    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  GPU           : {p.name}")
            print(f"  VRAM          : {p.total_memory / 1073741824:.2f} GB")
    except Exception:
        pass


def audit_summary():
    section("11) AUDIT DECISIONS")
    print("""  Facts this v2 audit must establish before any architecture work:
    - squeezed shape is exactly (B, 3, 112, 67)             (section 4)
    - channel semantics: which channel is x / y / confidence,
      and coordinate space (pixel vs normalized)            (section 8)
    - padding convention on the time axis                   (section 8)
    - label -> word mapping is consistent, 101 unique words (section 7)
    - signer layout for a leakage-free train/val split     (section 7)
  Known limitation already documented:
    - official test shares signers 1..10 with train
      -> signer-independent evaluation is NOT possible with this split
  No file was modified by this script.""")


def main():
    print("PHASE 0 - AUDIT CURRENT PROJECT (READ-ONLY) - v2")
    print(f"Root: {PROJECT_ROOT}")
    data_dir = None
    try:
        data_dir = find_data_dir()
    except Exception as e:
        print(f"[AUDIT ERROR] find_data_dir: {type(e).__name__}: {e}")
    if data_dir is None:
        section("DATASET FILES")
        print("  [FAIL] no directory contains all 6 islr101 npy files")
    else:
        run_safe(audit_files, data_dir)
        run_safe(audit_shapes, data_dir)
        run_safe(audit_consistency, data_dir)
        run_safe(audit_labels, data_dir)
        run_safe(audit_names, data_dir)
        run_safe(audit_values, data_dir)
    run_safe(audit_zip)
    run_safe(audit_environment)
    run_safe(audit_summary)
    print("\nPHASE 0 v2 finished. Send the FULL output back.")


if __name__ == "__main__":
    main()