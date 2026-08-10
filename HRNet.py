"""Holographic reconstruction network (HRNet) used by the project baseline.

This module implements the amplitude-reconstruction architecture described in
Z. Ren, Z. Xu, and E. Y. Lam, Advanced Photonics 1, 016004 (2019).  HRNet in
that paper means *holographic reconstruction network*; it is unrelated to the
later high-resolution representation network that uses the same acronym.

The reference architecture is

    Conv(1, 32)
    ResUnit(64, downsample=True)
    ResUnit(64)
    ResUnit(128, downsample=True)
    ResUnit(128)
    ResUnit(256, downsample=True)
    ResUnit(256)
    Conv(256, 64) + PixelShuffle(8)

All convolutions use 3 x 3 kernels and are followed by batch normalization and
ReLU.  The three max-pooling operations reduce each spatial dimension by eight;
the final 64 channels are periodically shuffled into one full-resolution
amplitude image.  The paper reports 2,857,248 convolutional weights when biases
and batch-normalization parameters are excluded, which this implementation
reproduces exactly.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


REFERENCE_CONV_WEIGHT_COUNT = 2_857_248


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic FFTs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize01(array: np.ndarray) -> np.ndarray:
    """Finite min-max normalization to [0, 1]."""
    x = np.asarray(array, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def standardize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-standard-deviation normalization."""
    x = np.asarray(array, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return ((x - float(x.mean())) / (float(x.std()) + eps)).astype(np.float32)


def normalize_hologram(array: np.ndarray, mode: str) -> np.ndarray:
    """Apply the input transform recorded in an HRNet checkpoint."""
    mode = mode.lower()
    x = np.asarray(array, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "minmax":
        return normalize01(x)
    if mode == "standard":
        return standardize(x)
    if mode == "none":
        return x.astype(np.float32)
    raise ValueError(f"Unknown input normalization mode: {mode}")


def load_2d_array(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a two-dimensional .npy array or grayscale image as float32."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.lower().endswith(".npy"):
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    array = np.squeeze(np.asarray(array, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {array.shape}: {path}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class ConvBNReLU(nn.Module):
    """The 3 x 3 convolution--BN--ReLU unit used throughout the paper."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualUnit(nn.Module):
    """Projection-free residual unit from Table 4 of the HRNet paper.

    The reference parameter table contains no learnable 1 x 1 projection in the
    identity branch.  When the number of channels doubles, the identity is
    therefore zero-padded.  In the three downsampling units, the same 2 x 2
    max-pooling is applied to the convolutional and identity branches.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("ResidualUnit does not support reducing channel count.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_channels == self.in_channels:
            return x
        missing = self.out_channels - self.in_channels
        left = missing // 2
        right = missing - left
        zeros_left = x.new_zeros((x.shape[0], left, x.shape[2], x.shape[3]))
        zeros_right = x.new_zeros((x.shape[0], right, x.shape[2], x.shape[3]))
        return torch.cat((zeros_left, x, zeros_right), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        identity = self._identity(x)
        residual = self.conv2(self.conv1(x))
        return residual + identity


class HolographicReconstructionNet(nn.Module):
    """Paper-faithful amplitude HRNet with an eightfold subpixel output layer."""

    scale_factor: int = 8

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = ConvBNReLU(1, 32)
        self.layer2 = ResidualUnit(32, 64, downsample=True)
        self.layer3 = ResidualUnit(64, 64, downsample=False)
        self.layer4 = ResidualUnit(64, 128, downsample=True)
        self.layer5 = ResidualUnit(128, 128, downsample=False)
        self.layer6 = ResidualUnit(128, 256, downsample=True)
        self.layer7 = ResidualUnit(256, 256, downsample=False)
        self.pre_shuffle = ConvBNReLU(256, 64)
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, hologram: torch.Tensor) -> torch.Tensor:
        if hologram.ndim != 4 or hologram.shape[1] != 1:
            raise ValueError(f"HRNet expects [B, 1, H, W], got {tuple(hologram.shape)}")
        h, w = hologram.shape[-2:]
        if h % self.scale_factor or w % self.scale_factor:
            raise ValueError(f"HRNet input dimensions must be divisible by 8, got {(h, w)}")
        x = self.layer1(hologram)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return self.pixel_shuffle(self.pre_shuffle(x))

    def convolutional_weight_count(self) -> int:
        """Count only 2D convolution weights, matching the paper's convention."""
        return sum(
            module.weight.numel()
            for module in self.modules()
            if isinstance(module, nn.Conv2d)
        )


def initialize_as_paper(model: nn.Module, std: float = 0.1, bias: float = 1.0) -> None:
    """Apply the truncated-normal convolution initialization reported by Ren et al."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if module.bias is not None:
                nn.init.constant_(module.bias, bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 8,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Reflect-pad [B,C,H,W] symmetrically and return (left,right,top,bottom)."""
    h, w = tensor.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0, 0)
    mode = "reflect" if h > max(top, bottom) and w > max(left, right) else "replicate"
    return F.pad(tensor, (left, right, top, bottom), mode=mode), (left, right, top, bottom)


def crop_padding(tensor: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """Undo :func:`pad_to_multiple`."""
    left, right, top, bottom = padding
    h_end = tensor.shape[-2] - bottom if bottom else tensor.shape[-2]
    w_end = tensor.shape[-1] - right if right else tensor.shape[-1]
    return tensor[..., top:h_end, left:w_end]


def infer_sample_patch_ids(path: str | os.PathLike[str]) -> Tuple[str, str]:
    """Infer project sample and patch IDs from ``.../<sample>/<patch>/<file>``."""
    p = Path(path)
    patch_id = p.parent.name or p.stem
    sample_id = p.parent.parent.name if p.parent.parent.name else "sample_unknown"
    return sample_id, patch_id


def checkpoint_model_config() -> Dict[str, object]:
    """Return the immutable architecture description stored in checkpoints."""
    return {
        "architecture": "Ren-Xu-Lam amplitude HRNet",
        "input_channels": 1,
        "output_channels": 1,
        "residual_depths": [64, 64, 128, 128, 256, 256],
        "downsampling_factor": 8,
        "upsampling": "pixel_shuffle",
        "reference_conv_weight_count": REFERENCE_CONV_WEIGHT_COUNT,
    }

