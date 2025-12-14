import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


LANDMARK_ORDER = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"]


def _percentile_clip_norm(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    eps = 1e-6
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + eps)
    return img.astype(np.float32)


def _resize_tensor(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # img: (1, H, W)
    img = img.unsqueeze(0)  # (1,1,H,W)
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    return img.squeeze(0)


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
    ):
        self.data_dir = data_dir
        self.resize = resize
        self.sigma = sigma
        self.percentile_clip = percentile_clip
        self.samples = self._discover_samples()

    def _discover_samples(self):
        out = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith("_image.npy"):
                continue
            base = fname.replace("_image.npy", "")
            json_path = os.path.join(self.data_dir, f"{base}_landmarks.json")
            npy_path = os.path.join(self.data_dir, fname)
            if os.path.exists(json_path):
                out.append((base, npy_path, json_path))
        out.sort()
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
            img_np = img_np[0]
        if img_np.ndim != 2:
            raise ValueError(f"Unsupported image shape {img_np.shape} for {npy_path}")

        with open(json_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        coords = self._extract_coords(meta, img_np.shape)

        img_np = _percentile_clip_norm(img_np, *self.percentile_clip)
        img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1,H,W)
        img_t = _resize_tensor(img_t, self.resize)  # (1,Hr,Wr)

        # Rescale coords to resized space
        h0, w0 = img_np.shape
        hr, wr = self.resize
        coords_resized = []
        for (x, y) in coords:
            coords_resized.append((x * wr / w0, y * hr / h0))

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
        for name in LANDMARK_ORDER:
            if name not in lm:
                raise ValueError(f"Missing landmark {name}")
            coords.append((lm[name]["i"], lm[name]["j"]))
        return coords
