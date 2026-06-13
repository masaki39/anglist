"""
Phase 2 データセットの I/O ユーティリティ。
npy/JSON の読み書きと、Slicer ノードへの変換を行う。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import slicer
    import vtk
    _IN_SLICER = True
except ImportError:
    _IN_SLICER = False

from train.landmark_scheme import ALL_LANDMARK_KEYS, OPTIONAL_LANDMARKS


# ---- ランドマークグループ定義 ----

LANDMARK_GROUPS: List[Dict] = [
    {
        "name": "Skull",
        "keys": ["EAC"],
        "color": (1.0, 0.9, 0.0),   # yellow
        "tab_label": "Skull",
    },
    {
        "name": "C-spine",
        "keys": (
            ["C2_3", "C2_4"]
            + [f"{v}_{i}" for v in ["C3", "C4", "C5", "C6", "C7"] for i in range(1, 5)]
        ),
        "color": (0.2, 0.85, 0.2),  # green
        "tab_label": "C",
    },
    {
        "name": "T-spine",
        "keys": [f"T{n}_{i}" for n in range(1, 13) for i in range(1, 5)],
        "color": (0.3, 0.6, 1.0),   # blue
        "tab_label": "T",
    },
    {
        "name": "L-spine",
        "keys": [f"L{n}_{i}" for n in range(1, 6) for i in range(1, 5)],
        "optional_keys": [f"L6_{i}" for i in range(1, 5)],
        "color": (1.0, 0.55, 0.0),  # orange
        "tab_label": "L",
    },
    {
        "name": "Sacrum/Pelvis",
        "keys": ["S1_1", "S1_2", "FH", "femur_prox", "femur_dist"],
        "color": (1.0, 0.3, 0.3),   # red
        "tab_label": "S/P",
    },
]

# key → group color のマップ
KEY_COLOR: Dict[str, Tuple[float, float, float]] = {}
for _g in LANDMARK_GROUPS:
    for _k in _g["keys"]:
        KEY_COLOR[_k] = _g["color"]
    for _k in _g.get("optional_keys", []):
        KEY_COLOR[_k] = _g["color"]


# ---- データセット探索 ----

def discover_cases(dataset_dir: str) -> List[str]:
    """ディレクトリから case_id リストを返す（ソート済み）。"""
    p = Path(dataset_dir)
    return sorted(
        f.name.replace("_image.npy", "")
        for f in p.glob("*_image.npy")
    )


def npy_path(dataset_dir: str, case_id: str) -> str:
    return os.path.join(dataset_dir, f"{case_id}_image.npy")


def json_path(dataset_dir: str, case_id: str) -> str:
    return os.path.join(dataset_dir, f"{case_id}_landmarks.json")


def load_json(dataset_dir: str, case_id: str) -> Dict:
    with open(json_path(dataset_dir, case_id), encoding="utf-8") as f:
        return json.load(f)


def save_json(meta: Dict, dataset_dir: str, case_id: str) -> None:
    with open(json_path(dataset_dir, case_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def detect_variant(placed_keys: set) -> str:
    """配置済みキーからバリアントを自動検出する。保存時・進捗集計時に使用。"""
    if any(f"L6_{i}" in placed_keys for i in range(1, 5)):
        return "lumbarization"
    t12 = any(f"T12_{i}" in placed_keys for i in range(1, 5))
    t11 = any(f"T11_{i}" in placed_keys for i in range(1, 5))
    l1  = any(f"L1_{i}"  in placed_keys for i in range(1, 5))
    if not t12 and t11 and l1:
        return "t12_missing"
    l5 = any(f"L5_{i}" in placed_keys for i in range(1, 5))
    l4 = any(f"L4_{i}" in placed_keys for i in range(1, 5))
    s1 = any(k in placed_keys for k in ("S1_1", "S1_2"))
    if not l5 and l4 and s1:
        return "sacralization"
    return "normal"


def count_annotated(dataset_dir: str, case_id: str) -> Tuple[int, int]:
    """(設定済み点数, バリアント対応合計点数) を返す。"""
    try:
        meta = load_json(dataset_dir, case_id)
        lm = meta.get("landmarks_ijk", {})
        placed = {k for k in ALL_LANDMARK_KEYS if (lm.get(k) or {}).get("i") is not None}
        variant = detect_variant(placed)
        active = set(active_keys_for_variant(variant))
        return len(placed & active), len(active)
    except Exception:
        return 0, len(ALL_LANDMARK_KEYS)


# ---- Slicer ノード操作 ----

def create_volume_node(case_id: str, dataset_dir: str, meta: Dict):
    """npy を読み込み vtkMRMLScalarVolumeNode を生成する。"""
    path = npy_path(dataset_dir, case_id)
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]  # (1, H, W)

    node_name = f"WSA_{case_id}"
    old = slicer.mrmlScene.GetFirstNodeByName(node_name)
    if old:
        slicer.mrmlScene.RemoveNode(old)

    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", node_name)
    slicer.util.updateVolumeFromArray(node, array)

    # IJK to RAS
    md = meta.get("metadata", {})
    ijk_to_ras = md.get("ijk_to_ras", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = md.get("origin_ras", [0.0, 0.0, 0.0])
    mat = vtk.vtkMatrix4x4()
    mat.Identity()
    for r in range(3):
        for c in range(3):
            mat.SetElement(r, c, ijk_to_ras[r][c])
        mat.SetElement(r, 3, origin[r])
    node.SetIJKToRASMatrix(mat)
    node.CreateDefaultDisplayNodes()

    # Window/level
    dn = node.GetDisplayNode()
    if dn:
        flat = array.ravel()
        lo = float(np.percentile(flat, 1))
        hi = float(np.percentile(flat, 99))
        dn.SetWindowLevelMinMax(lo, hi)

    return node


def create_markup_node(case_id: str):
    """空の vtkMRMLMarkupsFiducialNode を生成する。"""
    node_name = f"WSA_LM_{case_id}"
    old = slicer.mrmlScene.GetFirstNodeByName(node_name)
    if old:
        slicer.mrmlScene.RemoveNode(old)

    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
    node.CreateDefaultDisplayNodes()
    dn = node.GetDisplayNode()
    if dn:
        dn.SetGlyphScale(1.5)
        dn.SetTextScale(3.0)
        dn.SetSelectedColor(1.0, 0.5, 0.0)  # orange selected
        dn.SetColor(0.5, 0.9, 1.0)          # light blue default
    return node


def load_landmarks_into_node(meta: Dict, markup_node, volume_node) -> None:
    """JSON の landmarks_ijk を markup_node にロードする。"""
    lm = meta.get("landmarks_ijk", {})

    ijk_to_ras_mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras_mat)

    markup_node.RemoveAllControlPoints()

    for key in ALL_LANDMARK_KEYS:
        entry = lm.get(key) or {}
        i_val = entry.get("i")
        if i_val is None:
            continue
        j_val = entry.get("j", 0.0)
        k_val = entry.get("k", 0.0)
        ras_h = ijk_to_ras_mat.MultiplyPoint([i_val, j_val, k_val, 1.0])
        idx = markup_node.AddControlPoint(ras_h[0], ras_h[1], ras_h[2])
        markup_node.SetNthControlPointLabel(idx, key)


def save_landmarks_from_node(
    meta: Dict,
    markup_node,
    volume_node,
    lumbar_variant: str = "normal",
) -> Dict:
    """markup_node の座標を meta に書き込み、更新済み meta を返す。"""
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(ras_to_ijk)

    # label → index
    label_to_idx: Dict[str, int] = {}
    for i in range(markup_node.GetNumberOfControlPoints()):
        lbl = markup_node.GetNthControlPointLabel(i)
        if lbl:
            label_to_idx[lbl] = i

    lm: Dict = meta.get("landmarks_ijk", {})
    ras = [0.0, 0.0, 0.0]
    for key in ALL_LANDMARK_KEYS:
        if key in label_to_idx:
            markup_node.GetNthControlPointPositionWorld(label_to_idx[key], ras)
            ijk_h = ras_to_ijk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])
            lm[key] = {
                "i": round(ijk_h[0], 3),
                "j": round(ijk_h[1], 3),
                "k": round(ijk_h[2], 3),
            }
        else:
            lm[key] = {"i": None, "j": None, "k": 0}

    meta["landmarks_ijk"] = lm
    meta["lumbar_variant"] = lumbar_variant
    return meta


def get_placed_keys(markup_node) -> set:
    """markup_node に存在するランドマークキーのセットを返す。"""
    keys = set()
    for i in range(markup_node.GetNumberOfControlPoints()):
        lbl = markup_node.GetNthControlPointLabel(i)
        if lbl in ALL_LANDMARK_KEYS:
            keys.add(lbl)
    return keys


def active_keys_for_variant(variant: str) -> List[str]:
    """
    バリアントに応じてアノテーション対象キーリストを返す（進捗計算・保存用）。
    - normal: L6 を除く
    - lumbarization: L6 を含む
    - sacralization: L5 を除く
    - t12_missing: T12 を除く
    """
    exclude = set()
    if variant != "lumbarization":
        exclude.update(OPTIONAL_LANDMARKS)  # L6_1〜L6_4
    if variant == "sacralization":
        exclude.update(f"L5_{i}" for i in range(1, 5))
    if variant == "t12_missing":
        exclude.update(f"T12_{i}" for i in range(1, 5))
    return [k for k in ALL_LANDMARK_KEYS if k not in exclude]
