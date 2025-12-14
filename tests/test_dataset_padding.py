import json
import numpy as np
import torch

from train.dataset import HeatmapDataset, LANDMARK_ORDER


def _write_sample(tmp_path):
    img = np.zeros((100, 50), dtype=np.float32)
    img[0, 0] = 1.0  # corner pixel
    np.save(tmp_path / "case001_image.npy", img)

    lm = {
        name: {"i": float(x), "j": float(y), "k": 0.0}
        for name, (x, y) in zip(
            LANDMARK_ORDER,
            [
                (0, 0),       # maps to padding only
                (49, 99),     # bottom-right corner within bounds
                (10, 20),
                (25, 50),
                (5, 75),
            ],
        )
    }
    with open(tmp_path / "case001_landmarks.json", "w", encoding="utf-8") as fp:
        json.dump({"landmarks_ijk": lm, "metadata": {}}, fp)


def test_padding_preserves_aspect_and_coords(tmp_path):
    _write_sample(tmp_path)
    ds = HeatmapDataset(data_dir=str(tmp_path), resize=(512, 512), sigma=2.0)
    sample = ds[0]

    img = sample["image"]
    coords = sample["coords"]
    heatmap = sample["heatmap"]

    assert img.shape == (1, 512, 512)
    assert heatmap.shape == (len(LANDMARK_ORDER), 512, 512)
    assert coords.shape == (len(LANDMARK_ORDER), 2)

    # Original shape (H=100, W=50) -> scale=5.12 -> new (512,256), pad_x=128, pad_y=0
    expected_first = torch.tensor([128.0, 0.0])
    assert torch.allclose(coords[0], expected_first, atol=1e-3)

    # Heatmap peak should align with coords (within 1px tolerance)
    y0, x0 = torch.nonzero(heatmap[0] == heatmap[0].max(), as_tuple=True)
    y0 = y0[0].item()
    x0 = x0[0].item()
    assert abs(x0 - expected_first[0].item()) <= 1
    assert abs(y0 - expected_first[1].item()) <= 1
