"""
ONNX inference and evaluation for cervical spine heatmap model.
Usage:
  uv run python train/infer_onnx_cervical.py --model cervical.onnx --image sample_image.npy --json sample_landmarks.json
  uv run python train/infer_onnx_cervical.py --model cervical.onnx --dir dataset/cervical/
"""

import argparse
import io
import json
import math
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import onnxruntime as ort
import torch

import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from dataset_cervical import LANDMARK_ORDER
from dataset import _percentile_clip_norm, _resize_with_padding

# ---------------------------------------------------------------------------
# Angle computation (mirrors CervicalMeasureAssist/lib/logic_angles_cervical.py)
# ---------------------------------------------------------------------------

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


def compute_cervical_angles(coords, landmark_keys, spacing_mm=1.0):
    """
    Compute C2C7_angle, T1S, and C2C7_SVA from predicted (x,y) coords.
    Mirrors compute_cervical_measurements() in logic_angles_cervical.py.
    spacing_mm: mm per pixel for SVA conversion. Use 1.0 for mm-space coords.
    """
    pts = {k: xy for k, xy in zip(landmark_keys, coords)}
    required = LANDMARK_ORDER
    if not all(k in pts for k in required):
        return None

    v_C2  = _vec(pts["C2_ant"],     pts["C2_post"])
    v_C7i = _vec(pts["C7_inf_ant"], pts["C7_inf_post"])
    v_T1  = _vec(pts["T1_ant"],     pts["T1_post"])

    c2c7 = _wrap(_signed_slope(v_C7i) - _signed_slope(v_C2))
    t1s  = _signed_slope(v_T1)
    sva  = (pts["C2_center"][0] - pts["C7_sup_post"][0]) * spacing_mm

    return {"C2C7_angle": c2c7, "T1S": t1s, "C2C7_SVA": sva}


# ---------------------------------------------------------------------------
# Statistics helpers (shared with infer_onnx.py)
# ---------------------------------------------------------------------------

def _ci95(values):
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    return mean - 1.96 * se, mean + 1.96 * se


def _sd(values):
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def bland_altman_stats(ai_vals, gt_vals):
    diffs = [a - g for a, g in zip(ai_vals, gt_vals)]
    n = len(diffs)
    bias = sum(diffs) / n
    sd = _sd(diffs)
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def icc_3_1(ai_vals, gt_vals):
    n = len(ai_vals)
    if n < 2:
        return float("nan")
    all_vals = ai_vals + gt_vals
    grand = sum(all_vals) / (2 * n)
    row_means = [(a + g) / 2 for a, g in zip(ai_vals, gt_vals)]
    ss_between = 2 * sum((r - grand) ** 2 for r in row_means)
    col_means = [sum(ai_vals) / n, sum(gt_vals) / n]
    ss_rater = n * sum((c - grand) ** 2 for c in col_means)
    ss_total = sum((v - grand) ** 2 for v in all_vals)
    ss_error = ss_total - ss_between - ss_rater
    ms_b = ss_between / (n - 1)
    ms_e = max(ss_error / (n - 1), 1e-12)
    return (ms_b - ms_e) / (ms_b + ms_e)


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", help=".npy image path")
    p.add_argument("--json", help="Optional landmarks json to compare")
    p.add_argument("--dir", help="Dataset directory for batch MRE evaluation")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--splits", default=None, help="splits.json for test-set evaluation")
    p.add_argument("--subset", default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--report", default=None, help="Save evaluation report to this text file")
    return p.parse_args()


def get_model_resize(sess, args_resize):
    if args_resize is not None:
        return args_resize
    shape = sess.get_inputs()[0].shape
    return [shape[2], shape[3]]


def preprocess(img_np, resize):
    if img_np.ndim == 3:
        img_np = img_np[0]
    img_np = _percentile_clip_norm(img_np)
    t = torch.from_numpy(img_np).unsqueeze(0)
    t, scale, pad_x, pad_y = _resize_with_padding(t, tuple(resize))
    return t.unsqueeze(0), scale, pad_x, pad_y


def postprocess_heatmaps(hm: np.ndarray):
    hm = hm[0]
    coords, confs = [], []
    for c in hm:
        idx = np.argmax(c)
        y, x = np.unravel_index(idx, c.shape)
        coords.append((float(x), float(y)))
        confs.append(float(c[y, x]))
    return coords, confs


def _load_gt(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    lm = meta["landmarks_ijk"]
    gt_coords = [(lm[k]["i"], lm[k]["j"]) for k in LANDMARK_ORDER if k in lm]
    gt_keys = [k for k in LANDMARK_ORDER if k in lm]
    spacing = None
    if "metadata" in meta and "spacing" in meta["metadata"]:
        spacing = meta["metadata"]["spacing"][0]
    gt_angles = meta.get("angles_deg", {})
    return gt_coords, gt_keys, spacing, gt_angles


def _compute_errors(pred_coords, gt_coords, scale, pad_x, pad_y, spacing):
    errors_px, errors_mm = [], []
    for (px, py), (gx, gy) in zip(pred_coords, gt_coords):
        x_orig = (px - pad_x) / scale
        y_orig = (py - pad_y) / scale
        re_px = math.sqrt((x_orig - gx) ** 2 + (y_orig - gy) ** 2)
        errors_px.append(re_px)
        errors_mm.append(re_px * spacing if spacing is not None else None)
    return errors_px, errors_mm


def _back_project(coords, scale, pad_x, pad_y):
    return [((x - pad_x) / scale, (y - pad_y) / scale) for x, y in coords]


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(args):
    ort_sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    resize = get_model_resize(ort_sess, args.resize)
    data_dir = args.dir

    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith("_image.npy"):
            continue
        base = fname.replace("_image.npy", "")
        json_path = os.path.join(data_dir, f"{base}_landmarks.json")
        npy_path = os.path.join(data_dir, fname)
        if os.path.exists(json_path):
            samples.append((base, npy_path, json_path))

    if not samples:
        print(f"No samples found in {data_dir}")
        return

    if args.splits and args.subset != "all":
        with open(args.splits, "r") as fp:
            splits = json.load(fp)
        allowed = set(splits[args.subset])
        samples = [(b, n, j) for b, n, j in samples if b in allowed]
        print(f"Evaluating subset='{args.subset}' ({len(samples)} cases)")

    n = len(samples)
    all_errors_px = {k: [] for k in LANDMARK_ORDER}
    all_errors_mm = {k: [] for k in LANDMARK_ORDER}
    all_confs = {k: [] for k in LANDMARK_ORDER}

    angle_names = ["C2C7_angle", "T1S", "C2C7_SVA"]
    ai_angles = {a: [] for a in angle_names}
    gt_angles_store = {a: [] for a in angle_names}
    angle_units = {"C2C7_angle": "°", "T1S": "°", "C2C7_SVA": "mm"}

    low_conf_cases = []
    outlier_cases = []

    for case_id, npy_path, json_path in samples:
        img_np = np.load(npy_path)
        inp_t, scale, pad_x, pad_y = preprocess(img_np, resize)
        ort_out = ort_sess.run(None, {"image": inp_t.numpy()})
        pred_coords, confs = postprocess_heatmaps(ort_out[0])

        gt_coords, gt_keys, spacing, gt_ang = _load_gt(json_path)
        errors_px, errors_mm = _compute_errors(pred_coords, gt_coords, scale, pad_x, pad_y, spacing)

        for i, k in enumerate(LANDMARK_ORDER):
            if i < len(errors_px):
                all_errors_px[k].append(errors_px[i])
                if errors_mm[i] is not None:
                    all_errors_mm[k].append(errors_mm[i])
                all_confs[k].append(confs[i])

        if any(c < 0.05 for c in confs):
            low_conf_cases.append((case_id, min(confs)))

        valid_mm = [e for e in errors_mm if e is not None]
        if valid_mm and max(valid_mm) > 10.0:
            outlier_cases.append((case_id, max(valid_mm)))

        pred_orig = _back_project(pred_coords, scale, pad_x, pad_y)
        spacing_mm = spacing if spacing is not None else 1.0
        ai_ang = compute_cervical_angles(pred_orig, LANDMARK_ORDER, spacing_mm=spacing_mm)
        if ai_ang and gt_ang:
            for ang in angle_names:
                if ang in ai_ang and ang in gt_ang:
                    ai_angles[ang].append(ai_ang[ang])
                    gt_angles_store[ang].append(gt_ang[ang])

    # -----------------------------------------------------------------------
    # Print: Landmark MRE
    # -----------------------------------------------------------------------
    sep = "-" * 72
    print(f"\n=== Landmark Detection (N={n}) ===")
    hdr = f"{'Landmark':<14}  {'MRE(mm)':>8}  {'SD(mm)':>7}  {'95%CI':>14}  {'SDR@2':>6}  {'SDR@4':>6}  {'Conf':>5}"
    print(hdr)
    print(sep)

    overall_mm, overall_conf = [], []
    for k in LANDMARK_ORDER:
        emm = all_errors_mm[k]
        cvals = all_confs[k]
        if not emm:
            continue
        mean_mm = sum(emm) / len(emm)
        sd_mm = _sd(emm)
        ci_lo, ci_hi = _ci95(emm)
        sdr2 = sum(1 for v in emm if v <= 2.0) / len(emm) * 100
        sdr4 = sum(1 for v in emm if v <= 4.0) / len(emm) * 100
        mean_conf = sum(cvals) / len(cvals)
        print(f"{k:<14}  {mean_mm:>8.2f}  {sd_mm:>7.2f}  [{ci_lo:>5.2f},{ci_hi:>5.2f}]  {sdr2:>5.1f}%  {sdr4:>5.1f}%  {mean_conf:>5.3f}")
        overall_mm.extend(emm)
        overall_conf.extend(cvals)

    print(sep)
    if overall_mm:
        m = sum(overall_mm) / len(overall_mm)
        s = _sd(overall_mm)
        ci_lo, ci_hi = _ci95(overall_mm)
        sdr2 = sum(1 for v in overall_mm if v <= 2.0) / len(overall_mm) * 100
        sdr4 = sum(1 for v in overall_mm if v <= 4.0) / len(overall_mm) * 100
        mc = sum(overall_conf) / len(overall_conf)
        print(f"{'Overall':<14}  {m:>8.2f}  {s:>7.2f}  [{ci_lo:>5.2f},{ci_hi:>5.2f}]  {sdr2:>5.1f}%  {sdr4:>5.1f}%  {mc:>5.3f}")

    if outlier_cases:
        print(f"\nOutliers (any landmark >10mm): {len(outlier_cases)}/{n}")
    else:
        print(f"\nOutliers (any landmark >10mm): 0/{n}")

    # -----------------------------------------------------------------------
    # Print: Angle / SVA evaluation
    # -----------------------------------------------------------------------
    print(f"\n=== Angle / SVA Accuracy — AI vs GT (Bland-Altman) ===")
    hdr2 = f"{'Metric':<12}  {'Unit':>4}  {'N':>4}  {'MAE':>7}  {'SD':>6}  {'Bias':>8}  {'LoA_lo':>7}  {'LoA_hi':>7}  {'ICC':>5}"
    print(hdr2)
    print(sep)

    for ang in angle_names:
        unit = angle_units[ang]
        ai_v = ai_angles[ang]
        gt_v = gt_angles_store[ang]
        if len(ai_v) < 2:
            print(f"{ang:<12}  {unit:>4}  {'n/a':>4}")
            continue
        abs_errs = [abs(a - g) for a, g in zip(ai_v, gt_v)]
        mae = sum(abs_errs) / len(abs_errs)
        sd = _sd(abs_errs)
        bias, sd_diff, loa_lo, loa_hi = bland_altman_stats(ai_v, gt_v)
        icc = icc_3_1(ai_v, gt_v)
        print(f"{ang:<12}  {unit:>4}  {len(ai_v):>4}  {mae:>7.2f}  {sd:>6.2f}  {bias:>8.2f}  {loa_lo:>7.2f}  {loa_hi:>7.2f}  {icc:>5.3f}")

    if low_conf_cases:
        print(f"\n=== Low-confidence cases (peak < 0.05): {len(low_conf_cases)}/{n} ===")
    else:
        print(f"\nAll {n} cases had heatmap confidence >= 0.05")


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dir:
        if args.report:
            buf = io.StringIO()
            with redirect_stdout(buf):
                evaluate_dataset(args)
            output = buf.getvalue()
            sys.stdout.write(output)
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Report saved to {args.report}")
        else:
            evaluate_dataset(args)
        return

    if not args.image:
        print("Error: --image or --dir is required")
        return

    ort_sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    resize = get_model_resize(ort_sess, args.resize)

    img_np = np.load(args.image)
    inp_t, scale, pad_x, pad_y = preprocess(img_np, resize)
    ort_out = ort_sess.run(None, {"image": inp_t.numpy()})
    coords, confs = postprocess_heatmaps(ort_out[0])

    print("Predicted coords (x,y) in resized space:")
    for name, (x, y), c in zip(LANDMARK_ORDER, coords, confs):
        print(f"  {name}: ({x:.1f}, {y:.1f})  conf={c:.4f}")

    orig_coords = _back_project(coords, scale, pad_x, pad_y)
    angles = compute_cervical_angles(orig_coords, LANDMARK_ORDER)
    if angles:
        print("\nPredicted measurements:")
        units = {"C2C7_angle": "°", "T1S": "°", "C2C7_SVA": "mm"}
        for k, v in angles.items():
            print(f"  {k}: {v:.2f}{units.get(k, '')}")


if __name__ == "__main__":
    main()
