"""
ONNX推論ロジック：Volumeから2D画像を取り出し、パディングリサイズでモデル入力に合わせ、
ヒートマップ最大値を元画像座標に戻してMarkupsに配置する。
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import slicer
import vtk

from logic_angles_cervical import REQUIRED_KEYS


def _percentile_clip_norm(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    eps = 1e-6
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + eps)
    return img.astype(np.float32)


def _resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Bilinear resize with align_corners=False to match torch.nn.functional.interpolate.
    img: (H,W)
    """
    h, w = img.shape
    x_new = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    x_new = np.clip(x_new, 0, w - 1)
    y_new = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
    y_new = np.clip(y_new, 0, h - 1)
    tmp = np.zeros((h, new_w), dtype=np.float32)
    for i in range(h):
        tmp[i] = np.interp(x_new, np.arange(w), img[i])
    out = np.zeros((new_h, new_w), dtype=np.float32)
    for j in range(new_w):
        out[:, j] = np.interp(y_new, np.arange(h), tmp[:, j])
    return out


def _pad_resize(img: np.ndarray, target_hw: Tuple[int, int]):
    """縦横比を維持してリサイズし、余白ゼロパディング。返り値: 画像, scale, pad_x, pad_y。"""
    h, w = img.shape
    th, tw = target_hw
    scale = min(th / h, tw / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = _resize_bilinear(img, new_h, new_w)

    pad_y = (th - new_h) // 2
    pad_x = (tw - new_w) // 2
    padded = np.pad(resized, ((pad_y, th - new_h - pad_y), (pad_x, tw - new_w - pad_x)), mode="constant")
    return padded.astype(np.float32), scale, pad_x, pad_y


def _load_landmark_keys(model_path: str) -> Optional[List[str]]:
    """ONNX横に置かれた .meta.json からランドマーク名を読む。なければ None。"""
    meta_path = model_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("landmark_keys")
    return None


class OnnxInferenceLogic:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_path = None
        self.target_hw = (512, 512)
        self.landmark_keys = list(REQUIRED_KEYS)

    def load_model(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            slicer.util.pip_install("onnxruntime")
            import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_path = model_path
        model_shape = self.session.get_inputs()[0].shape
        if isinstance(model_shape[2], int) and isinstance(model_shape[3], int):
            self.target_hw = (model_shape[2], model_shape[3])

        loaded_keys = _load_landmark_keys(model_path)
        if loaded_keys is not None:
            self.landmark_keys = loaded_keys
        else:
            self.landmark_keys = list(REQUIRED_KEYS)

    def _extract_slice(self, volumeNode):
        arr = slicer.util.arrayFromVolume(volumeNode)
        if arr.ndim == 3:
            img2d = arr[0]
        else:
            raise ValueError(f"期待するshape (D,H,W) ですが取得: {arr.shape}")
        return img2d

    def _preprocess(self, img2d: np.ndarray):
        img_norm = _percentile_clip_norm(img2d)
        img_pad, scale, pad_x, pad_y = _pad_resize(img_norm, self.target_hw)
        input_tensor = img_pad[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return input_tensor, scale, pad_x, pad_y

    def _postprocess(self, heatmaps: np.ndarray, scale: float, pad_x: float, pad_y: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        hm = heatmaps[0]
        coords, confs = [], []
        for c in hm:
            idx = np.argmax(c)
            y, x = np.unravel_index(idx, c.shape)
            x_orig = (x - pad_x) / scale
            y_orig = (y - pad_y) / scale
            coords.append((x_orig, y_orig))
            confs.append(float(c[y, x]))
        return coords, confs

    def _ijk_to_ras(self, volumeNode, i, j, k=0.0):
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        ijk_h = [i, j, k, 1.0]
        ras_h = mat.MultiplyPoint(ijk_h)
        return ras_h[:3]

    def _build_overlay_heatmap(self, heatmaps: np.ndarray, orig_shape: Tuple[int, int], scale: float, pad_x: float, pad_y: float) -> np.ndarray:
        """各ランドマークのheatmapを元画像サイズにリサイズして返す。返り値: (L, H_orig, W_orig)"""
        h_orig, w_orig = orig_shape
        hm = heatmaps[0]
        new_h = int(round(h_orig * scale))
        new_w = int(round(w_orig * scale))
        channels = []
        for c in hm:
            crop = c[int(pad_y): int(pad_y) + new_h, int(pad_x): int(pad_x) + new_w]
            channels.append(_resize_bilinear(crop, h_orig, w_orig))
        return np.stack(channels, axis=0)

    def _check_anatomy(self, coords_ij: List[Tuple[float, float]], confs: List[float], flip_x: bool = False) -> List[str]:
        """Return list of warning strings for anatomically implausible predictions."""
        warnings = []
        pts = {k: xy for k, xy in zip(self.landmark_keys, coords_ij)}

        # Anterior landmark should have smaller x (i) than posterior in standard orientation.
        # When flip_x is True the image is mirrored, so anterior has larger x.
        for prefix in ("C2", "C7_inf", "T1"):
            ant = f"{prefix}_ant"
            post = f"{prefix}_post"
            if ant in pts and post in pts:
                ant_x, post_x = pts[ant][0], pts[post][0]
                reversed_x = (ant_x <= post_x) if flip_x else (ant_x >= post_x)
                if reversed_x:
                    warnings.append(f"{ant} が {post} より後方（x座標が逆転）")

        # Low confidence landmarks
        LOW_CONF_THRESHOLD = 0.05
        for key, conf in zip(self.landmark_keys, confs):
            if conf < LOW_CONF_THRESHOLD:
                warnings.append(f"{key} の信頼度が低い（peak={conf:.4f}）")

        return warnings

    def predict_and_place(self, volumeNode, markupNode, flip_x: bool = False) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        if self.session is None:
            raise RuntimeError("モデルがロードされていません。")
        img2d = self._extract_slice(volumeNode)
        inp, scale, pad_x, pad_y = self._preprocess(img2d)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        coords_ij, confs = self._postprocess(outputs[0], scale, pad_x, pad_y)
        heatmap_2d = self._build_overlay_heatmap(outputs[0], img2d.shape, scale, pad_x, pad_y)

        anatomy_warnings = self._check_anatomy(coords_ij, confs, flip_x=flip_x)
        if anatomy_warnings:
            msg = "【警告】AI予測に問題の可能性があります:\n" + "\n".join(f"  ・{w}" for w in anatomy_warnings)
            slicer.util.warningDisplay(msg, windowTitle="CervicalMeasureAssist")

        label_to_idx = {}
        for i in range(markupNode.GetNumberOfControlPoints()):
            label_to_idx[markupNode.GetNthControlPointLabel(i)] = i

        was_modifying = markupNode.StartModify()
        try:
            for key, (x, y) in zip(self.landmark_keys, coords_ij):
                ras = self._ijk_to_ras(volumeNode, x, y, 0.0)
                if key in label_to_idx:
                    markupNode.SetNthControlPointPosition(label_to_idx[key], ras[0], ras[1], ras[2])
                else:
                    markupNode.AddControlPoint(ras[0], ras[1], ras[2])
                    markupNode.SetNthControlPointLabel(markupNode.GetNumberOfControlPoints() - 1, key)
        finally:
            markupNode.EndModify(was_modifying)
        return coords_ij, heatmap_2d
