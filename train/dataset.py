import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


LANDMARK_ORDER = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH", "L1_center"]


def _percentile_clip_norm(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    eps = 1e-6
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + eps)
    return img.astype(np.float32)


def _resize_with_padding(img: torch.Tensor, target_size: Tuple[int, int]):
    """
    Resize with aspect ratio preserved, pad with zeros to target_size (H, W).
    Returns resized+pad image and (scale, pad_x, pad_y) for coordinate mapping.
    """
    # img: (1, H, W)
    _, h, w = img.shape
    th, tw = target_size
    scale = min(th / h, tw / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    img = img.unsqueeze(0)  # (1,1,H,W)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    img = img.squeeze(0)

    pad_y = (th - new_h) // 2
    pad_x = (tw - new_w) // 2
    img = F.pad(img, (pad_x, tw - new_w - pad_x, pad_y, th - new_h - pad_y))
    return img, scale, pad_x, pad_y


def _make_heatmaps(coords: List[Tuple[float, float]], size: Tuple[int, int], sigma: float) -> torch.Tensor:
    h, w = size
    device = "cpu"
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    heatmaps = []
    sigma2 = 2 * sigma * sigma
    for (x, y) in coords:
        g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / sigma2)
        heatmaps.append(g)
    return torch.stack(heatmaps, dim=0)  # (L,H,W)


def _build_augmentation():
    import albumentations as A
    return A.Compose(
        [
            A.Rotate(limit=15, border_mode=0, p=0.8),
            A.ElasticTransform(alpha=50, sigma=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # X線ドメイン固有: 量子ノイズ・検出器ガンマ特性のシミュレーション
            A.GaussNoise(p=0.4),
            A.RandomGamma(gamma_limit=(75, 130), p=0.4),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class HeatmapDataset(Dataset):
    """
    Loads .npy image and .json landmarks (IJK). Generates normalized image and heatmaps.
    """

    def __init__(
        self,
        data_dir: str,
        resize: Tuple[int, int] = (512, 512),
        sigma: float = 3.0,
        percentile_clip: Tuple[float, float] = (1.0, 99.0),
        landmark_keys: Optional[List[str]] = None,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.resize = resize
        self.sigma = sigma
        self.percentile_clip = percentile_clip
        self.landmark_keys = landmark_keys if landmark_keys is not None else LANDMARK_ORDER
        self.samples = self._discover_samples()
        self._transform = _build_augmentation() if augment else None

    def _discover_samples(self):
        out = []
        skipped = 0
        for fname in os.listdir(self.data_dir):
            if not fname.endswith("_image.npy"):
                continue
            base = fname.replace("_image.npy", "")
            json_path = os.path.join(self.data_dir, f"{base}_landmarks.json")
            npy_path = os.path.join(self.data_dir, fname)
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            lm = meta.get("landmarks_ijk", {})
            if any(lm.get(k) is None for k in self.landmark_keys):
                skipped += 1
                continue
            out.append((base, npy_path, json_path))
        out.sort()
        if skipped:
            print(f"WARNING: {skipped} samples skipped (null landmarks)")
        if not out:
            raise RuntimeError(f"No samples found in {self.data_dir}")
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, npy_path, json_path = self.samples[idx]
        img_np = np.load(npy_path)
        # Accept shape (H,W) or (D,H,W); use first slice if 3D.
        if img_np.ndim == 3:
            if img_np.shape[0] > 1:
                print(f"WARNING: {npy_path} は {img_np.shape[0]} スライスありますが、最初の1枚のみ使用します。")
            img_np = img_np[0]
        if img_np.ndim != 2:
            raise ValueError(f"Unsupported image shape {img_np.shape} for {npy_path}")

        with open(json_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        coords = self._extract_coords(meta, img_np.shape)

        img_np = _percentile_clip_norm(img_np, *self.percentile_clip)
        img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1,H,W)
        img_t, scale, pad_x, pad_y = _resize_with_padding(img_t, self.resize)  # (1,Ht,Wt)

        # Rescale coords to resized+pad space
        coords_resized = [(x * scale + pad_x, y * scale + pad_y) for (x, y) in coords]

        if self._transform is not None:
            img_np_512 = img_t.squeeze(0).numpy()  # (H,W) float32
            result = self._transform(image=img_np_512, keypoints=coords_resized)
            # albumentations 2.x may drop out-of-bounds keypoints even with remove_invisible=False;
            # fall back to unaugmented coords if count changes to keep tensor shapes consistent.
            if len(result["keypoints"]) == len(coords_resized):
                img_t = torch.from_numpy(result["image"].copy()).unsqueeze(0)
                th, tw = self.resize
                coords_resized = [
                    (max(0.0, min(float(kp[0]), tw - 1.0)), max(0.0, min(float(kp[1]), th - 1.0)))
                    for kp in result["keypoints"]
                ]

        hr, wr = self.resize
        heatmap = _make_heatmaps(coords_resized, (hr, wr), sigma=self.sigma)
        coords_t = torch.tensor(coords_resized, dtype=torch.float32)
        return {
            "image": img_t,
            "heatmap": heatmap,
            "coords": coords_t,
            "case_id": case_id,
        }

    def _extract_coords(self, meta: Dict, shape_hw: Tuple[int, int]) -> List[Tuple[float, float]]:
        coords = []
        if "landmarks_ijk" not in meta:
            raise ValueError("Missing landmarks_ijk in json")
        lm = meta["landmarks_ijk"]
        for name in self.landmark_keys:
            if name not in lm:
                raise ValueError(f"Missing landmark {name}")
            coords.append((lm[name]["i"], lm[name]["j"]))
        return coords
