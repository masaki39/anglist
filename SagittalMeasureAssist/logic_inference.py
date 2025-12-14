"""
ONNX推論ロジック：Volumeから2D画像を取り出し、パディングリサイズでモデル入力に合わせ、
ヒートマップ最大値を元画像座標に戻してMarkupsに配置する。
"""

import os
from typing import List, Tuple

import numpy as np
import slicer
import vtk

from logic_angles import REQUIRED_KEYS


def _percentile_clip_norm(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    eps = 1e-6
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + eps)
    return img.astype(np.float32)


def _resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Minimal bilinear resize using separable 1D interpolation (no external deps).
    img: (H,W)
    """
    h, w = img.shape
    x_old = np.arange(w)
    x_new = np.linspace(0, w - 1, new_w)
    # interpolate along x for each row
    tmp = np.zeros((h, new_w), dtype=np.float32)
    for i in range(h):
        tmp[i] = np.interp(x_new, x_old, img[i])
    y_old = np.arange(h)
    y_new = np.linspace(0, h - 1, new_h)
    out = np.zeros((new_h, new_w), dtype=np.float32)
    for j in range(new_w):
        out[:, j] = np.interp(y_new, y_old, tmp[:, j])
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


class OnnxInferenceLogic:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_path = None
        self.target_hw = (512, 512)

    def load_model(self, model_path: str, target_hw: Tuple[int, int]):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime がインストールされていません。`uv sync --extra ml` を実行してください。") from exc

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
        self.target_hw = target_hw
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_path = model_path

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
        # ONNXには (1,1,H,W)
        input_tensor = img_pad[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return input_tensor, scale, pad_x, pad_y

    def _postprocess(self, heatmaps: np.ndarray, scale: float, pad_x: float, pad_y: float) -> List[Tuple[float, float]]:
        # heatmaps: (1, L, H, W)
        hm = heatmaps[0]
        coords = []
        for c in hm:
            idx = np.argmax(c)
            y, x = np.unravel_index(idx, c.shape)
            # 逆変換（paddingとスケールを戻す）
            x_orig = (x - pad_x) / scale
            y_orig = (y - pad_y) / scale
            coords.append((x_orig, y_orig))
        return coords

    def _ijk_to_ras(self, volumeNode, i, j, k=0.0):
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        ijk_h = [i, j, k, 1.0]
        ras_h = mat.MultiplyPoint(ijk_h)
        return ras_h[:3]

    def _coords_ij_to_ras(self, volumeNode, coords_ij):
        coords_ras = []
        for (i, j) in coords_ij:
            coords_ras.append(self._ijk_to_ras(volumeNode, i, j, 0.0))
        return coords_ras

    def predict_and_place(self, volumeNode, markupNode):
        if self.session is None:
            raise RuntimeError("モデルがロードされていません。")
        img2d = self._extract_slice(volumeNode)
        inp, scale, pad_x, pad_y = self._preprocess(img2d)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        coords_ij = self._postprocess(outputs[0], scale, pad_x, pad_y)
        # 書き込み
        markupNode.RemoveAllControlPoints()
        for idx, (x, y) in enumerate(coords_ij):
            ras = self._ijk_to_ras(volumeNode, x, y, 0.0)
            markupNode.AddControlPoint(ras[0], ras[1], ras[2])
            markupNode.SetNthControlPointLabel(idx, REQUIRED_KEYS[idx])
        return coords_ij
