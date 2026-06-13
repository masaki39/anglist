"""
ディレクトリを再帰的に走査してDICOMファイルをデータセットに変換する。
SQLite DB不要。各サブディレクトリ内に dataset/ を作成する（--out 指定時は1ディレクトリに集約）。

使用法:
  uv run python scripts/extract_dicom_dir.py --dir "/Volumes/T7 Shield/dicom/omuro"
  uv run python scripts/extract_dicom_dir.py --dir "/Volumes/T7 Shield/dicom/omuro" --dry-run
  uv run python scripts/extract_dicom_dir.py --dir "/Volumes/T7 Shield/dicom/omuro" --limit 3
  uv run python scripts/extract_dicom_dir.py --dir "/Volumes/T7 Shield/dicom/omuro" --force
  uv run python scripts/extract_dicom_dir.py --dir "/Volumes/T7 Shield/dicom/omuro" --out "/Volumes/T7 Shield/dicom/omuro/demo_dataset"
  uv run python scripts/extract_dicom_dir.py --dir "..." --landmark-type phase2

出力:
  <subdir>/dataset/<filename_stem>_image.npy        (--out なし)
  <subdir>/dataset/<filename_stem>_landmarks.json   (--out なし)
  <out>/<filename_stem>_image.npy                   (--out あり)
  <out>/<filename_stem>_landmarks.json              (--out あり)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_landmark_template(landmark_type: str) -> dict:
    if landmark_type == "phase2":
        from train.landmark_scheme import make_landmark_template
        return make_landmark_template()
    else:  # cervical (default)
        from train.dataset_cervical import LANDMARK_ORDER
        return {k: {"i": None, "j": None, "k": 0} for k in LANDMARK_ORDER}


def _load_dicom_pixel(dcm_path: Path) -> tuple[np.ndarray, list[float]]:
    import pydicom

    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D pixel array, got shape {arr.shape}")
    spacing = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    return arr, spacing


def _build_metadata(spacing: list[float]) -> dict:
    sp_i, sp_j = spacing[0], spacing[1]
    return {
        "spacing": [sp_i, sp_j, 1.0],
        "ijk_to_ras": [
            [-sp_i, 0.0, 0.0],
            [0.0, -sp_j, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "origin_ras": [0.0, 0.0, 0.0],
    }


def extract_one(dcm_path: Path, out_dir: Path, dry_run: bool, force: bool, landmark_type: str = "cervical") -> bool:
    case_id = dcm_path.stem
    npy_path = out_dir / f"{case_id}_image.npy"
    json_path = out_dir / f"{case_id}_landmarks.json"

    if npy_path.exists() and json_path.exists() and not force:
        print(f"  [SKIP] {case_id}")
        return True

    print(f"  {case_id}: {dcm_path.name}")
    if dry_run:
        return True

    try:
        arr, spacing = _load_dicom_pixel(dcm_path)
    except Exception as e:
        print(f"  [ERROR] {dcm_path.name}: {e}")
        return False

    H, W = arr.shape
    np.save(str(npy_path), arr)

    meta = {
        "case_id": case_id,
        "source": str(dcm_path),
        "image_shape": [H, W],
        "metadata": _build_metadata(spacing),
        "landmarks_ijk": _make_landmark_template(landmark_type),
    }
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    return True


def process_subdir(subdir: Path, dry_run: bool, force: bool, limit: int | None, out: Path | None = None, landmark_type: str = "cervical") -> tuple[int, int]:
    dcm_files = sorted(subdir.glob("*.dcm"))
    if not dcm_files:
        return 0, 0

    if limit is not None:
        dcm_files = dcm_files[:limit]

    out_dir = out if out is not None else subdir / "dataset"
    print(f"\n[{subdir.name}] {len(dcm_files)} 件 -> {out_dir}")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    ok = sum(extract_one(f, out_dir, dry_run, force, landmark_type) for f in dcm_files)
    return ok, len(dcm_files)


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM files in subdirs to npy+json dataset")
    parser.add_argument("--dir", required=True, help="Root directory containing DICOM subdirectories")
    parser.add_argument("--out", default=None, help="全サブディレクトリの出力を1つのディレクトリに集約 (省略時は各<subdir>/dataset/)")
    parser.add_argument("--landmark-type", default="cervical", choices=["cervical", "phase2"],
                        help="ランドマークテンプレートの種類 (default: cervical)")
    parser.add_argument("--dry-run", action="store_true", help="ファイル書き込みなし")
    parser.add_argument("--force", action="store_true", help="既存ファイルを上書き")
    parser.add_argument("--limit", type=int, default=None, help="各サブディレクトリの処理上限数")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        sys.exit(f"Directory not found: {root}")

    out_dir = Path(args.out) if args.out else None
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and p != out_dir and p.name != "dataset")
    if not subdirs:
        sys.exit(f"No subdirectories found in {root}")

    print(f"対象ディレクトリ: {root}")
    print(f"サブディレクトリ: {[d.name for d in subdirs]}")
    if out_dir:
        print(f"出力先: {out_dir}")
    print(f"ランドマーク種別: {args.landmark_type}")
    if args.dry_run:
        print("[dry-run] ファイル書き込みはスキップします")

    total_ok = total_n = 0
    for subdir in subdirs:
        ok, n = process_subdir(subdir, args.dry_run, args.force, args.limit, out_dir, args.landmark_type)
        total_ok += ok
        total_n += n

    print(f"\n完了: {total_ok}/{total_n} 件")


if __name__ == "__main__":
    main()
