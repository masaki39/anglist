"""
頚椎学習曲線 ローカル実行スクリプト

Usage:
  uv run python train/learning_curve/cervical/run_local.py

結果は LC_DIR/results/<VARIANT>/fold{N}_size{S:03d}.json に保存される。
既存のジョブはスキップされるので途中再開も可能。
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

# ===== 設定 =====
DATA_DIR   = "/Volumes/T7 Shield/dicom/omuro/merged_dataset"
LC_DIR     = "/Volumes/T7 Shield/dicom/omuro/cervical_lc"
FOLDS      = [1, 2, 3, 4, 5]
SIZES      = [20, 40, 80, 160, 220, 259]
AUGMENT    = True
LOSS       = "awl"
SIGMA      = 15.0
EPOCHS      = 100
BATCH_SIZE  = 4   # 重い場合は2に下げる
NUM_WORKERS = 0   # 0=メインスレッドのみ（マルチプロセス負荷を排除）
NICE        = 19  # プロセス優先度（19=最低。他の作業を妨げない）
LANDMARKS   = "C2_center,C2_ant,C2_post,C7_sup_post,C7_inf_ant,C7_inf_post,T1_ant,T1_post"
N_FOLDS     = 5
SEED        = 42
# =================

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
LANDMARK_ORDER = LANDMARKS.split(",")
VARIANT = f"smallunet_aug{int(AUGMENT)}_{LOSS}_s{int(SIGMA)}"

# デバイス選択（MPS優先）
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# 前処理・推論ヘルパー（eval_cervical.py と同一ロジック）
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(REPO, "train"))
from dataset import _percentile_clip_norm, _resize_with_padding  # noqa: E402
import onnxruntime as ort  # noqa: E402


def preprocess(img_np, resize=(512, 512)):
    if img_np.ndim == 3:
        img_np = img_np[0]
    img_np = _percentile_clip_norm(img_np)
    t = torch.from_numpy(img_np).unsqueeze(0)
    t, scale, pad_x, pad_y = _resize_with_padding(t, resize)
    return t.unsqueeze(0), scale, pad_x, pad_y


def postprocess(hm):
    hm = hm[0]
    return [(float(np.unravel_index(np.argmax(c), c.shape)[1]),
             float(np.unravel_index(np.argmax(c), c.shape)[0])) for c in hm]


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _signed_slope(v):
    ang = math.degrees(math.atan2(v[1], v[0]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return -ang


def _wrap(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a


def compute_cervical_angles(pts, spacing_mm=1.0):
    if not all(k in pts for k in LANDMARK_ORDER):
        return None
    v_C2  = _vec(pts["C2_ant"],     pts["C2_post"])
    v_C7i = _vec(pts["C7_inf_ant"], pts["C7_inf_post"])
    v_T1  = _vec(pts["T1_ant"],     pts["T1_post"])
    c2c7 = _wrap(_signed_slope(v_C7i) - _signed_slope(v_C2))
    t1s  = _signed_slope(v_T1)
    sva  = (pts["C2_center"][0] - pts["C7_sup_post"][0]) * spacing_mm
    return {"C2C7_angle": c2c7, "T1S": t1s, "C2C7_SVA": sva}


def run_subprocess(cmd):
    result = subprocess.run(["nice", "-n", str(NICE)] + cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print("=== STDOUT ===")
        print(result.stdout[-3000:])
        print("=== STDERR ===")
        print(result.stderr[-3000:])
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    lines = (result.stdout + result.stderr).strip().split("\n")
    print("  " + "\n  ".join(lines[-3:]))


# ---------------------------------------------------------------------------
# Fold 準備
# ---------------------------------------------------------------------------
def prepare_folds():
    folds_path = os.path.join(LC_DIR, "cervical_folds.json")
    if os.path.exists(folds_path):
        print(f"[folds] 既存のfolds.jsonを使用: {folds_path}")
        with open(folds_path) as f:
            return json.load(f)

    cases = sorted(
        f.replace("_image.npy", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith("_image.npy")
    )
    print(f"[folds] 全症例数: {len(cases)}")

    random.seed(SEED)
    shuffled = cases[:]
    random.shuffle(shuffled)

    folds = {}
    for k in range(N_FOLDS):
        folds[str(k + 1)] = shuffled[k::N_FOLDS]

    for k, v in folds.items():
        print(f"  fold {k}: {len(v)} cases")

    os.makedirs(LC_DIR, exist_ok=True)
    data = {"n_folds": N_FOLDS, "seed": SEED, "folds": folds}
    with open(folds_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[folds] 保存: {folds_path}")
    return data


# ---------------------------------------------------------------------------
# 1ジョブ実行
# ---------------------------------------------------------------------------
def run_job(fold, size, test_ids, train_ids, tmp_dir):
    results_dir = os.path.join(LC_DIR, "results", VARIANT)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"fold{fold}_size{size:03d}.json")

    if os.path.exists(out_path):
        print(f"  [skip] fold{fold}_size{size:03d}")
        return

    t0 = time.time()
    print(f"  --- fold={fold}  size={size}  device={DEVICE} ---")

    # 訓練データを一時ディレクトリにコピー
    train_dir = os.path.join(tmp_dir, f"train_f{fold}_s{size}")
    os.makedirs(train_dir, exist_ok=True)
    for cid in train_ids:
        for ext in ["_image.npy", "_landmarks.json"]:
            shutil.copy(os.path.join(DATA_DIR, cid + ext), train_dir)

    save_dir = os.path.join(tmp_dir, f"runs/{VARIANT}/fold{fold}_size{size:03d}")
    cmd = [
        "uv", "run", "python", "train/train.py",
        "--data-dir", train_dir,
        "--save-dir", save_dir,
        "--landmarks", LANDMARKS,
        "--backbone", "smallunet",
        "--sigma", str(SIGMA),
        "--loss", LOSS,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--split-seed", str(fold * 7 + size),
        "--device", DEVICE,
    ]
    if AUGMENT:
        cmd.append("--augment")
    run_subprocess(cmd)

    onnx_path = os.path.join(save_dir, "lc.onnx")
    run_subprocess([
        "uv", "run", "python", "train/export_lumbar.py",
        "--checkpoint", os.path.join(save_dir, "best.pt"),
        "--output", onnx_path,
    ])

    # テストデータは常駐ディレクトリから直接読む（コピー不要）
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    all_errors = {k: [] for k in LANDMARK_ORDER}
    angle_errs = {"C2C7_angle": [], "T1S": [], "C2C7_SVA": []}

    for cid in test_ids:
        img_np = np.load(os.path.join(DATA_DIR, f"{cid}_image.npy"))
        inp_t, scale, pad_x, pad_y = preprocess(img_np)
        ort_out = sess.run(None, {"image": inp_t.numpy()})
        pred = postprocess(ort_out[0])

        with open(os.path.join(DATA_DIR, f"{cid}_landmarks.json")) as fp:
            meta = json.load(fp)
        spacing = meta.get("metadata", {}).get("spacing", [1.0])[0]
        lm = meta["landmarks_ijk"]
        gt_ang = meta.get("angles_deg", {})

        pred_ij = {}
        for (px, py), name in zip(pred, LANDMARK_ORDER):
            gx, gy = lm[name]["i"], lm[name]["j"]
            x_o = (px - pad_x) / scale
            y_o = (py - pad_y) / scale
            err = math.sqrt((x_o - gx) ** 2 + (y_o - gy) ** 2) * spacing
            all_errors[name].append(err)
            pred_ij[name] = (x_o, y_o)

        ai_ang = compute_cervical_angles(pred_ij, spacing_mm=spacing)
        if ai_ang and gt_ang:
            for a in angle_errs:
                if a in ai_ang and a in gt_ang:
                    angle_errs[a].append(abs(ai_ang[a] - gt_ang[a]))

    # 外れ値検出（>10mm）
    is_outlier = [max(all_errors[k][i] for k in LANDMARK_ORDER) > 10.0
                  for i in range(len(test_ids))]
    n_outliers = sum(is_outlier)
    ok_idx = [i for i, o in enumerate(is_outlier) if not o]

    all_vals = [all_errors[k][i] for i in range(len(test_ids)) for k in LANDMARK_ORDER]
    ok_vals  = [all_errors[k][i] for i in ok_idx for k in LANDMARK_ORDER]

    results = {
        "fold": fold, "size": size, "variant": VARIANT,
        "n_test": len(test_ids), "device": DEVICE,
        "elapsed_sec": round(time.time() - t0),
        "overall": {
            "mre_mm": sum(all_vals) / len(all_vals),
            "sdr2": sum(1 for e in all_vals if e <= 2.0) / len(all_vals) * 100,
            "sdr4": sum(1 for e in all_vals if e <= 4.0) / len(all_vals) * 100,
            "angle_mae": {a: sum(v) / len(v) for a, v in angle_errs.items() if v},
        },
        "overall_excl_outliers": {
            "n": len(ok_idx), "n_outliers": n_outliers,
            "mre_mm": sum(ok_vals) / len(ok_vals) if ok_vals else None,
        },
        "landmarks": {
            k: {
                "mre_mm": sum(all_errors[k]) / len(all_errors[k]),
                "sdr4": sum(1 for e in all_errors[k] if e <= 4.0) / len(all_errors[k]) * 100,
            }
            for k in LANDMARK_ORDER
        },
    }

    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)

    mre = results["overall"]["mre_mm"]
    sdr = results["overall"]["sdr4"]
    elapsed = results["elapsed_sec"]
    print(f"  MRE={mre:.2f}mm  SDR@4={sdr:.1f}%  outliers={n_outliers}/{len(test_ids)}  {elapsed}s  -> saved")

    # 訓練データ一時ディレクトリを削除（ディスク節約）
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(save_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    print(f"=== 頚椎学習曲線 ローカル実行 ===")
    print(f"variant: {VARIANT}  device: {DEVICE}")
    print(f"folds: {FOLDS}  sizes: {SIZES}")
    print(f"epochs: {EPOCHS}  batch: {BATCH_SIZE}  sigma: {SIGMA}")
    print()

    fold_data = prepare_folds()

    total_jobs = len(FOLDS) * len(SIZES)
    done = 0

    with tempfile.TemporaryDirectory(prefix="cervical_lc_") as tmp_dir:
        for fold in FOLDS:
            print(f"\n{'='*60}")
            print(f"FOLD {fold}  ({FOLDS.index(fold)+1}/{len(FOLDS)})")
            print(f"{'='*60}")

            test_ids = fold_data["folds"][str(fold)]
            all_train_ids = [
                cid for k, ids in fold_data["folds"].items()
                for cid in ids if k != str(fold)
            ]

            for size in SIZES:
                actual_size = min(size, len(all_train_ids))
                random.seed(fold * 1000 + size)
                train_ids = sorted(all_train_ids)
                random.shuffle(train_ids)
                train_ids = train_ids[:actual_size]

                run_job(fold, actual_size, test_ids, train_ids, tmp_dir)
                done += 1
                print(f"  進捗: {done}/{total_jobs} jobs")

    print("\n=== 全ジョブ完了 ===")
    print(f"結果: {os.path.join(LC_DIR, 'results', VARIANT)}/")


if __name__ == "__main__":
    os.nice(NICE)
    main()
