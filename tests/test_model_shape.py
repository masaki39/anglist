import torch

from train.model import SmallUNet


def test_small_unet_output_shape():
    model = SmallUNet(num_landmarks=5)
    x = torch.randn(2, 1, 128, 96)
    y = model(x)
    assert y.shape == (2, 5, 128, 96)
