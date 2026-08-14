"""Final project implementation of the DH-GAN reconstruction baseline.

Reference
---------
X. Chen, H. Wang, A. Razi, M. Kozicki, and C. Mann, "DH-GAN: a
physics-driven untrained generative adversarial network for holographic
imaging," Optics Express 31, 10114-10135 (2023), doi:10.1364/OE.480894.

Method retained from the article
--------------------------------
* a back-propagated hologram is the input of an untrained hourglass
  autoencoder that directly predicts object amplitude and phase;
* angular-spectrum forward propagation regenerates the sensor hologram;
* hologram MSE, an adversarial distance, and background-only complex TV are
  optimized per hologram;
* a K=2 proposal and simulated annealing update the background mask;
* one generator update is followed by at most five discriminator updates.

Project adaptations for the 256 x 256 patches
----------------------------------------------
The article does not report all numerical hyperparameters. The following
explicit, auditable adaptations are used identically for every hologram.

1. A finite ``bootstrap_iters`` stage teaches the direct-output autoencoder to
   reproduce the back-propagated complex field.  The bootstrap target is then
   completely removed; no BP skip/residual remains in the final DH-GAN field.
   This prevents weak particles from disappearing during random initialization
   while avoiding a permanent BP identity mapping.
2. A conservative particle-protection map is extracted once from locally
   normalized BP amplitude.  It only excludes likely particles from background
   TV.  K-means/annealing may add foreground but cannot classify the protected
   pixels as background.  It is not used in the hologram loss or network output.
3. Label smoothing, linearly decaying instance noise, and loss-gated
   discriminator updates prevent discriminator saturation.  The original
   maximum of five discriminator updates is retained.
4. The adversarial contribution is linearly introduced and capped at 5% of
   the physics-plus-background objective so that adversarial gradients cannot
   override sensor-plane consistency.
5. An exponential moving average is used for inference.  Checkpoints are
   ranked without reference objects using sensor MSE, background complex TV,
   and near-zero amplitude collapse.  Ground truth is never loaded.

The numerical loss weights, Adam rates, and annealing temperature are exposed
and written to ``meta.json`` because the article does not report them.

"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Required by CUDA for repeatable GEMM when deterministic algorithms are
# requested.  It must be set before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


EPS = 1.0e-8


@dataclass
class DHGANConfig:
    wavelength: float = 632.8e-9
    pixel_size: float = 6.9e-6
    z_distance: float = 0.021
    iterations: int = 3000
    bootstrap_iters: int = 300
    bootstrap_lr: float = 1.0e-3
    generator_lr: float = 2.0e-4
    discriminator_lr: float = 1.0e-5
    d_steps: int = 5
    lambda_hologram: float = 50.0
    lambda_adversarial: float = 0.01
    lambda_background: float = 5.0
    # Fixed, ground-truth-free stability controls.  These affect every sample
    # identically and are recorded in meta.json for reproducibility.
    adversarial_ratio_cap: float = 0.05
    adversarial_ramp_iters: int = 300
    real_label: float = 0.90
    fake_label: float = 0.10
    discriminator_gate_low: float = 0.50
    discriminator_gate_high: float = 0.65
    instance_noise: float = 0.03
    instance_noise_iters: int = 1000
    gradient_clip: float = 5.0
    ema_decay: float = 0.995
    selection_warmup: int = 300
    selection_interval: int = 50
    selection_saturation_margin: float = 0.03
    selection_saturation_weight: float = 0.50
    width_scale: float = 1.0
    # The paper's extra super-resolution stage maps a lower-resolution
    # hologram to a larger object grid.  Project patches and their references
    # are already matched at 256 x 256, so the stage is disabled by default.
    super_resolution: bool = False
    discriminator_input_mode: str = "duplicated"
    mask_update_interval: int = 100
    annealing_temperature: float = 0.01
    support_percentile: float = 3.0
    support_local_sigma: float = 4.0
    support_trend_sigma: float = 20.0
    support_min_area: int = 32
    support_dilation: int = 4
    border: int = 6
    max_amplitude: float = 1.0
    seed: int = 999
    log_interval: int = 100
    checkpoint_interval: int = 500


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
    return ((array - lo) / (hi - lo + EPS)).astype(np.float32)


def load_intensity(path_value: str, image_size: int = 0) -> np.ndarray:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Input hologram not found: {path}")
    if path.suffix.lower() == ".npy":
        image = np.load(path).astype(np.float32)
    else:
        image = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D hologram, got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Input contains NaN or infinite values")
    if image_size > 0:
        if min(image.shape) < image_size:
            raise ValueError(f"img_size={image_size} exceeds input shape {image.shape}")
        image = image[:image_size, :image_size]
    return normalize01(image)


def infer_ids(input_path: str) -> Tuple[str, str]:
    path = Path(input_path)
    return path.parent.parent.name or "sample", path.parent.name or path.stem


def resolve_z_distance(input_path: str, explicit_value: Optional[float]) -> Tuple[float, str]:
    """Resolve propagation distance without applying a measured-data default to simulations."""
    if explicit_value is not None:
        return float(explicit_value), "command line"
    input_file = Path(input_path)
    patch_dir = input_file.parent
    metadata = patch_dir / "meta.json"
    if metadata.is_file():
        with metadata.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in ("z_distance", "z_distance_m", "best_z"):
            if key in payload:
                return float(payload[key]), f"{metadata.name}:{key}"
    best_z = patch_dir / "best_z.txt"
    if best_z.is_file():
        contents = best_z.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"best_z_m\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            contents,
        )
        if match is None:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", contents)
        if match:
            value = match.group(1) if match.lastindex else match.group(0)
            return float(value), best_z.name
    return 0.02275, "project measured-data fallback"


def make_valid_mask(height: int, width: int, border: int, device: torch.device) -> torch.Tensor:
    if border < 0 or 2 * border >= min(height, width):
        raise ValueError(f"Invalid border={border} for {(height, width)}")
    mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    if border == 0:
        mask.fill_(1.0)
    else:
        mask[:, :, border:-border, border:-border] = 1.0
    return mask


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b).square() * mask) / (torch.sum(mask) + EPS)


class AngularSpectrum(nn.Module):
    def __init__(self, height: int, width: int, z: float, wavelength: float, pixel_size: float, device: torch.device):
        super().__init__()
        fx = fft.fftfreq(width, d=pixel_size, device=device)
        fy = fft.fftfreq(height, d=pixel_size, device=device)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        root = 1.0 - (wavelength * fx_grid).square() - (wavelength * fy_grid).square()
        phase = (2.0 * torch.pi / wavelength) * z * torch.sqrt(torch.clamp(root, min=0.0))
        self.register_buffer("transfer", torch.exp(1j * phase).to(torch.complex64)[None, None])

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return fft.ifft2(fft.fft2(field) * self.transfer)


class ConvNormAct(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int = 3, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel, padding=kernel // 2, bias=False)
        # Batch normalization follows Tables 1 and 2 of the source article.
        # A batch of one is valid here because statistics are accumulated over
        # all spatial pixels of every feature channel.
        self.norm = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation: nn.Module = nn.ReLU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


def scaled_channels(value: int, scale: float) -> int:
    return max(4, int(round(value * scale / 4.0)) * 4)


class DirectGenerator(nn.Module):
    """Direct amplitude/phase hourglass based on Table 1 of Chen et al."""

    def __init__(self, width_scale: float, super_resolution: bool, max_amplitude: float):
        super().__init__()
        c16, c32 = scaled_channels(16, width_scale), scaled_channels(32, width_scale)
        c64, c128 = scaled_channels(64, width_scale), scaled_channels(128, width_scale)
        self.super_resolution = bool(super_resolution)
        self.max_amplitude = float(max_amplitude)
        self.encoder = nn.Sequential(
            ConvNormAct(2, c32, 5), ConvNormAct(c32, c32), nn.MaxPool2d(2),
            ConvNormAct(c32, c64), ConvNormAct(c64, c64), nn.MaxPool2d(2),
            ConvNormAct(c64, c128), ConvNormAct(c128, c128), nn.MaxPool2d(2),
            ConvNormAct(c128, c128), ConvNormAct(c128, c16, activation="tanh"),
        )
        self.middle = nn.Sequential(ConvNormAct(c16, c128), ConvNormAct(c128, c128))
        # A 2 x 2 stride-2 kernel performs non-overlapping learned upsampling.
        # It is a transposed convolution as specified in Table 1, but avoids
        # the overlap pattern that produces checkerboard texture with 4 x 4.
        self.up1 = nn.ConvTranspose2d(c128, c128, 2, stride=2)
        self.dec1 = nn.Sequential(ConvNormAct(c128, c64), ConvNormAct(c64, c64))
        self.up2 = nn.ConvTranspose2d(c64, c64, 2, stride=2)
        self.dec2 = nn.Sequential(ConvNormAct(c64, c32), ConvNormAct(c32, c32))
        self.up3 = nn.ConvTranspose2d(c32, c32, 2, stride=2)
        if self.super_resolution:
            self.sr = nn.Sequential(
                ConvNormAct(c32, c16), ConvNormAct(c16, c16),
                nn.ConvTranspose2d(c16, c16, 2, stride=2), nn.ReLU(inplace=False),
            )
        else:
            self.sr = ConvNormAct(c32, c16)
        self.output = nn.Sequential(ConvNormAct(c16, c16), ConvNormAct(c16, c16), nn.Conv2d(c16, 2, 3, padding=1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, network_input: torch.Tensor, output_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        x = self.middle(self.encoder(network_input))
        x = self.dec1(F.relu(self.up1(x), inplace=False))
        x = self.dec2(F.relu(self.up2(x), inplace=False))
        x = F.relu(self.up3(x), inplace=False)
        raw = self.output(self.sr(x))
        if raw.shape[-2:] != output_size:
            raw = F.interpolate(raw, size=output_size, mode="bilinear", align_corners=False)
        amplitude = self.max_amplitude * torch.sigmoid(raw[:, 0:1])
        phase = torch.pi * torch.tanh(raw[:, 1:2])
        return {"amplitude": amplitude, "phase": phase, "field": amplitude * torch.exp(1j * phase)}


class DiscBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class Discriminator(nn.Module):
    """Global discriminator following Table 2 of Chen et al."""

    def __init__(self, input_channels: int, width_scale: float):
        super().__init__()
        c16, c32 = scaled_channels(16, width_scale), scaled_channels(32, width_scale)
        c64, c128 = scaled_channels(64, width_scale), scaled_channels(128, width_scale)
        self.body = nn.Sequential(
            DiscBlock(input_channels, c32, 5), DiscBlock(c32, c32), nn.MaxPool2d(2),
            DiscBlock(c32, c64), DiscBlock(c64, c64), nn.MaxPool2d(2),
            DiscBlock(c64, c128), DiscBlock(c128, c128), nn.MaxPool2d(2),
            DiscBlock(c128, c128), nn.Conv2d(c128, c16, 3, padding=1),
        )
        self.fc = nn.Linear(c16, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(F.adaptive_avg_pool2d(self.body(image), 1).flatten(1))


def prepare_network_input(
    target: torch.Tensor,
    backward: AngularSpectrum,
    valid: torch.Tensor,
    super_resolution: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_field = backward(torch.sqrt(torch.clamp(target, min=0.0)).to(torch.complex64)).detach()
    amplitude = torch.abs(base_field)
    scale = torch.median(amplitude[valid > 0.5])
    amplitude_channel = torch.clamp(amplitude / (scale + EPS), 0.0, 2.0) - 1.0
    phase_channel = torch.angle(base_field) / torch.pi
    network_input = torch.cat([amplitude_channel, phase_channel], dim=1)
    if super_resolution:
        network_input = F.interpolate(network_input, scale_factor=0.5, mode="bilinear", align_corners=False)
    return network_input.detach(), base_field


def build_protection_mask(base_amplitude: np.ndarray, valid_np: np.ndarray, config: DHGANConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative dark-object support from local/broad BP amplitude ratio."""
    amplitude = np.asarray(base_amplitude, dtype=np.float32)
    local = ndi.gaussian_filter(amplitude, config.support_local_sigma)
    trend = ndi.gaussian_filter(amplitude, config.support_trend_sigma)
    ratio = local / (trend + EPS)
    values = ratio[valid_np > 0.5]
    threshold = float(np.percentile(values, config.support_percentile))
    foreground = (ratio <= threshold) & (valid_np > 0.5)
    foreground = ndi.binary_closing(foreground, iterations=2)
    foreground = ndi.binary_fill_holes(foreground)
    labels, count = ndi.label(foreground)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= config.support_min_area
        keep[0] = False
        foreground = keep[labels]
    if config.support_dilation > 0:
        foreground = ndi.binary_dilation(foreground, iterations=config.support_dilation)
    foreground = foreground.astype(np.float32) * valid_np
    background = valid_np * (1.0 - foreground)
    return foreground.astype(np.float32), background.astype(np.float32)


def discriminator_view(intensity: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "single":
        return intensity
    if mode == "duplicated":
        return torch.cat([intensity, intensity], dim=1)
    raise ValueError(mode)


def complex_background_tv(field: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    real, imag = field.real, field.imag
    mask_x = background[:, :, :, 1:] * background[:, :, :, :-1]
    mask_y = background[:, :, 1:, :] * background[:, :, :-1, :]
    dx = (real[:, :, :, 1:] - real[:, :, :, :-1]).square()
    dx += (imag[:, :, :, 1:] - imag[:, :, :, :-1]).square()
    dy = (real[:, :, 1:, :] - real[:, :, :-1, :]).square()
    dy += (imag[:, :, 1:, :] - imag[:, :, :-1, :]).square()
    return torch.sum(dx * mask_x) / (torch.sum(mask_x) + EPS) + torch.sum(dy * mask_y) / (torch.sum(mask_y) + EPS)


def kmeans_background(amplitude: torch.Tensor, valid: torch.Tensor, protection: torch.Tensor) -> torch.Tensor:
    values = amplitude.detach()[valid > 0.5]
    centers = torch.stack([torch.quantile(values, 0.25), torch.quantile(values, 0.75)])
    for _ in range(25):
        labels = torch.argmin(torch.abs(values[:, None] - centers[None, :]), dim=1)
        updated = centers.clone()
        for index in range(2):
            selected = values[labels == index]
            if selected.numel():
                updated[index] = selected.mean()
        if torch.max(torch.abs(updated - centers)) < 1.0e-6:
            centers = updated
            break
        centers = updated
    full_labels = torch.argmin(torch.abs(amplitude.detach()[..., None] - centers), dim=-1)
    counts = torch.stack([(full_labels[valid > 0.5] == index).sum() for index in range(2)])
    background_label = int(torch.argmax(counts).item())
    proposal = (full_labels == background_label).float() * valid
    # The K-means object class may contain sparse pixels across the field.  A
    # max-pool dilation here would join those pixels and can erase the entire
    # background-TV domain.  Only the independently protected particle support
    # is forced to foreground; the raw K-means labels remain otherwise intact.
    return (proposal * (1.0 - protection) * valid).detach()


def annealed_update(
    current: torch.Tensor,
    proposal: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    update_index: int,
) -> Tuple[torch.Tensor, float, bool, float, float]:
    # Compare complete K-means proposals instead of repeatedly intersecting
    # them with the current mask.  Repeated intersections monotonically shrink
    # the background and can reach the empty-mask trivial solution.
    valid_count = float(valid.sum().item())
    proposal_fraction = float(proposal.sum().item() / max(valid_count, 1.0))
    if proposal_fraction < 0.35:
        current_score = float(masked_mse(prediction, target, current).item())
        proposal_score = float(masked_mse(prediction, target, proposal).item())
        return current, temperature, False, current_score, proposal_score
    current_score = float(masked_mse(prediction, target, current).item())
    proposal_score = float(masked_mse(prediction, target, proposal).item())
    if proposal_score < current_score:
        return proposal, temperature, True, current_score, proposal_score
    cooled = temperature / max(math.log1p(update_index), 1.0)
    probability = math.exp(-max(proposal_score - current_score, 0.0) / max(cooled, EPS))
    accepted = random.random() < probability
    return (proposal if accepted else current), cooled, accepted, current_score, proposal_score


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def update_ema(ema: nn.Module, source: nn.Module, decay: float) -> None:
    """Update the inference copy without introducing an extra training loss."""
    with torch.no_grad():
        source_state = source.state_dict()
        for name, value in ema.state_dict().items():
            source_value = source_state[name]
            if torch.is_floating_point(value):
                value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                value.copy_(source_value)


def add_instance_noise(view: torch.Tensor, standard_deviation: float) -> torch.Tensor:
    if standard_deviation <= 0.0:
        return view
    return view + standard_deviation * torch.randn_like(view)


def discriminator_loss(
    discriminator: nn.Module,
    real_view: torch.Tensor,
    fake_view: torch.Tensor,
    bce: nn.Module,
    real_label: torch.Tensor,
    fake_label: torch.Tensor,
) -> torch.Tensor:
    logits = discriminator(torch.cat([real_view, fake_view], dim=0))
    return 0.5 * (
        bce(logits[0:1], real_label) + bce(logits[1:2], fake_label)
    )


def choose_discriminator_steps(probe_loss: float, config: DHGANConfig) -> int:
    """Gate D updates once its binary task is already confidently solved."""
    if probe_loss < config.discriminator_gate_low:
        return 0
    if probe_loss < config.discriminator_gate_high:
        return 1
    return config.d_steps


def low_amplitude_fraction(
    amplitude: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Detect near-zero sigmoid collapse without using a reference image."""
    return torch.sum((amplitude <= margin).float() * valid) / (torch.sum(valid) + EPS)


def ground_truth_free_selection(
    model: nn.Module,
    network_input: torch.Tensor,
    output_size: Tuple[int, int],
    propagator: nn.Module,
    target: torch.Tensor,
    valid: torch.Tensor,
    background: torch.Tensor,
    config: DHGANConfig,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        output = model(network_input, output_size)
        prediction = torch.abs(propagator(output["field"])).square()
        hologram = masked_mse(prediction, target, valid)
        background_tv = complex_background_tv(output["field"], background)
        saturation = low_amplitude_fraction(
            output["amplitude"], valid, config.selection_saturation_margin
        )
        score = (
            config.lambda_hologram * hologram
            + config.lambda_background * background_tv
            + config.selection_saturation_weight * saturation
        )
    metrics = {
        "score": float(score.item()),
        "hologram": float(hologram.item()),
        "background_tv": float(background_tv.item()),
        "low_amplitude_fraction": float(saturation.item()),
    }
    return metrics["score"], metrics


def state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def reconstruct(
    intensity: np.ndarray,
    config: DHGANConfig,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, object]:
    if config.iterations <= 0 or not 0 <= config.bootstrap_iters < config.iterations:
        raise ValueError("Require iterations > 0 and 0 <= bootstrap_iters < iterations")
    if not 0.0 <= config.fake_label < config.real_label <= 1.0:
        raise ValueError("Require 0 <= fake_label < real_label <= 1")
    if not 0.0 <= config.ema_decay < 1.0:
        raise ValueError("Require 0 <= ema_decay < 1")
    if not 0.0 <= config.discriminator_gate_low < config.discriminator_gate_high:
        raise ValueError("Require 0 <= discriminator_gate_low < discriminator_gate_high")
    if config.d_steps < 0 or config.selection_interval < 0:
        raise ValueError("d_steps and selection_interval must be non-negative")
    if config.adversarial_ratio_cap < 0.0 or config.gradient_clip < 0.0:
        raise ValueError("Stability coefficients must be non-negative")
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = intensity.shape
    output_size = (height, width)
    target = torch.from_numpy(intensity.astype(np.float32))[None, None].to(device)
    valid = make_valid_mask(height, width, config.border, device)
    forward = AngularSpectrum(
        height, width, config.z_distance, config.wavelength, config.pixel_size, device
    ).to(device)
    backward = AngularSpectrum(
        height, width, -config.z_distance, config.wavelength, config.pixel_size, device
    ).to(device)
    network_input, base_field = prepare_network_input(
        target, backward, valid, config.super_resolution
    )
    base_amplitude = torch.abs(base_field)
    bootstrap_amplitude = torch.clamp(
        base_amplitude, max=config.max_amplitude
    ).detach()
    bootstrap_phase = torch.angle(base_field).detach()

    valid_np = valid.squeeze().cpu().numpy().astype(np.float32)
    protection_np, initial_background_np = build_protection_mask(
        base_amplitude.squeeze().cpu().numpy(), valid_np, config
    )
    protection = torch.from_numpy(protection_np)[None, None].to(device)
    background = torch.from_numpy(initial_background_np)[None, None].to(device)
    # A fixed selection domain makes scores comparable across iterations even
    # while the training mask is updated by K-means and annealing.
    selection_background = background.clone()

    generator = DirectGenerator(
        config.width_scale, config.super_resolution, config.max_amplitude
    ).to(device)
    ema_generator = copy.deepcopy(generator).to(device).eval()
    set_requires_grad(ema_generator, False)
    disc_channels = 1 if config.discriminator_input_mode == "single" else 2
    discriminator = Discriminator(disc_channels, config.width_scale).to(device)
    optimizer_g = optim.Adam(
        generator.parameters(), lr=config.bootstrap_lr, betas=(0.9, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999)
    )
    bce = nn.BCEWithLogitsLoss()
    real_label = torch.full((1, 1), config.real_label, device=device)
    fake_label = torch.full((1, 1), config.fake_label, device=device)

    history_names = [
        "generator", "bootstrap", "hologram", "adversarial", "background",
        "adversarial_term", "adversarial_scale", "discriminator",
        "discriminator_probe", "discriminator_updates", "instance_noise",
        "background_fraction", "selection_score",
    ]
    histories: Dict[str, List[float]] = {name: [] for name in history_names}
    mask_events: List[dict] = []
    selection_events: List[dict] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_background: Optional[torch.Tensor] = None
    best_score = math.inf
    best_iteration = 0
    best_metrics: Dict[str, float] = {}
    total_discriminator_updates = 0
    temperature, start = config.annealing_temperature, 0

    if resume:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("--resume requested but checkpoint is missing")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state.get("stability_version") != 1:
            raise RuntimeError(
                "This checkpoint predates stable DH-GAN training; rerun without --resume."
            )
        generator.load_state_dict(state["generator"])
        ema_generator.load_state_dict(state["ema_generator"])
        discriminator.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        background = state["background"].to(device)
        temperature, start = float(state["temperature"]), int(state["iteration"])
        histories = {
            key: [float(value) for value in values]
            for key, values in state["histories"].items()
        }
        mask_events = list(state.get("mask_events", []))
        selection_events = list(state.get("selection_events", []))
        best_state = state.get("best_state")
        best_background = state.get("best_background")
        best_score = float(state.get("best_score", math.inf))
        best_iteration = int(state.get("best_iteration", 0))
        best_metrics = dict(state.get("best_metrics", {}))
        total_discriminator_updates = int(state.get("total_discriminator_updates", 0))

    timer = time.perf_counter()
    update_index = len(mask_events)
    print("Start stability-controlled direct-output DH-GAN reconstruction...")
    print(
        f"device={device}; bootstrap={config.bootstrap_iters}; "
        f"protected_fraction={protection.mean().item():.4f}; "
        f"G={count_parameters(generator):,}; D={count_parameters(discriminator):,}"
    )
    for iteration in range(start, config.iterations):
        completed = iteration + 1
        post_bootstrap = max(completed - config.bootstrap_iters, 0)
        in_bootstrap = completed <= config.bootstrap_iters
        if completed == config.bootstrap_iters + 1 and start <= config.bootstrap_iters:
            # Remove bootstrap momentum and initialize EMA at the start of the
            # actual DH-GAN objective.
            optimizer_g = optim.Adam(
                generator.parameters(), lr=config.generator_lr, betas=(0.5, 0.999)
            )
            ema_generator.load_state_dict(generator.state_dict())

        generator.train()
        discriminator.eval()
        set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)
        output = generator(network_input, output_size)
        prediction = torch.abs(forward(output["field"])).square()
        loss_hologram = masked_mse(prediction, target, valid)
        zero = torch.zeros((), device=device)
        adversarial_scale = zero
        effective_adversarial = zero
        if in_bootstrap:
            phase_difference = torch.atan2(
                torch.sin(output["phase"] - bootstrap_phase),
                torch.cos(output["phase"] - bootstrap_phase),
            )
            loss_bootstrap = masked_mse(
                output["amplitude"], bootstrap_amplitude, valid
            ) + 0.1 * masked_mse(
                phase_difference, torch.zeros_like(phase_difference), valid
            )
            loss_adversarial, loss_background = zero, zero
            loss_generator = loss_bootstrap
        else:
            loss_bootstrap = zero
            fake_logits = discriminator(
                discriminator_view(prediction, config.discriminator_input_mode)
            )
            loss_adversarial = bce(fake_logits, real_label)
            loss_background = complex_background_tv(output["field"], background)
            physical_term = (
                config.lambda_hologram * loss_hologram
                + config.lambda_background * loss_background
            )
            ramp = min(
                1.0, post_bootstrap / max(float(config.adversarial_ramp_iters), 1.0)
            )
            raw_adversarial = config.lambda_adversarial * ramp * loss_adversarial
            cap = config.adversarial_ratio_cap * torch.clamp(
                physical_term.detach(), min=1.0e-6
            )
            adversarial_scale = torch.clamp(
                cap / (raw_adversarial.detach() + EPS), max=1.0
            )
            effective_adversarial = raw_adversarial * adversarial_scale
            loss_generator = physical_term + effective_adversarial
        if not torch.isfinite(loss_generator):
            raise FloatingPointError(f"Non-finite generator loss at iteration {completed}")
        loss_generator.backward()
        if config.gradient_clip > 0.0:
            nn.utils.clip_grad_norm_(generator.parameters(), config.gradient_clip)
        optimizer_g.step()
        if in_bootstrap:
            ema_generator.load_state_dict(generator.state_dict())
        else:
            update_ema(ema_generator, generator, config.ema_decay)

        should_update_mask = (
            not in_bootstrap
            and config.mask_update_interval > 0
            and post_bootstrap % config.mask_update_interval == 0
        )
        if should_update_mask:
            with torch.no_grad():
                refreshed = ema_generator(network_input, output_size)
                refreshed_prediction = torch.abs(forward(refreshed["field"])).square()
                proposal = kmeans_background(
                    refreshed["amplitude"], valid, protection
                )
                update_index += 1
                background, temperature, accepted, current_score, proposal_score = annealed_update(
                    background, proposal, target, refreshed_prediction, valid,
                    temperature, update_index,
                )
                mask_events.append({
                    "iteration": completed,
                    "accepted": accepted,
                    "temperature": temperature,
                    "current_score": current_score,
                    "proposal_score": proposal_score,
                    "background_fraction": float(
                        background.sum().item() / (valid.sum().item() + EPS)
                    ),
                })

        discriminator_value = 0.0
        probe_value = 0.0
        d_updates = 0
        noise_std = 0.0
        if not in_bootstrap:
            discriminator.train()
            set_requires_grad(discriminator, True)
            with torch.no_grad():
                fake_output = ema_generator(network_input, output_size)
                fake_intensity = torch.abs(forward(fake_output["field"])).square()
            noise_fraction = max(
                0.0,
                1.0 - post_bootstrap / max(float(config.instance_noise_iters), 1.0),
            )
            noise_std = config.instance_noise * noise_fraction
            real_view = add_instance_noise(
                discriminator_view(target, config.discriminator_input_mode), noise_std
            )
            fake_view = add_instance_noise(
                discriminator_view(fake_intensity.detach(), config.discriminator_input_mode),
                noise_std,
            )
            with torch.no_grad():
                probe_value = float(
                    discriminator_loss(
                        discriminator, real_view, fake_view, bce,
                        real_label, fake_label,
                    ).item()
                )
            d_updates = choose_discriminator_steps(probe_value, config)
            values = []
            for _ in range(d_updates):
                optimizer_d.zero_grad(set_to_none=True)
                loss_d = discriminator_loss(
                    discriminator, real_view, fake_view, bce,
                    real_label, fake_label,
                )
                loss_d.backward()
                if config.gradient_clip > 0.0:
                    nn.utils.clip_grad_norm_(
                        discriminator.parameters(), config.gradient_clip
                    )
                optimizer_d.step()
                values.append(float(loss_d.detach().item()))
            discriminator_value = float(np.mean(values)) if values else probe_value
            total_discriminator_updates += d_updates

        selection_score_value = math.nan
        should_select = (
            not in_bootstrap
            and post_bootstrap >= config.selection_warmup
            and config.selection_interval > 0
            and post_bootstrap % config.selection_interval == 0
        )
        if should_select:
            selection_score_value, candidate_metrics = ground_truth_free_selection(
                ema_generator, network_input, output_size, forward, target, valid,
                selection_background, config,
            )
            improved = selection_score_value < best_score
            selection_events.append({
                "iteration": completed,
                "selected": improved,
                **candidate_metrics,
            })
            if improved:
                best_score = selection_score_value
                best_iteration = completed
                best_metrics = candidate_metrics
                best_state = state_dict_to_cpu(ema_generator)
                best_background = background.detach().cpu().clone()

        recorded = {
            "generator": float(loss_generator.detach().item()),
            "bootstrap": float(loss_bootstrap.detach().item()),
            "hologram": float(loss_hologram.detach().item()),
            "adversarial": float(loss_adversarial.detach().item()),
            "background": float(loss_background.detach().item()),
            "adversarial_term": float(effective_adversarial.detach().item()),
            "adversarial_scale": float(adversarial_scale.detach().item()),
            "discriminator": discriminator_value,
            "discriminator_probe": probe_value,
            "discriminator_updates": float(d_updates),
            "instance_noise": float(noise_std),
            "background_fraction": float(
                background.sum().item() / (valid.sum().item() + EPS)
            ),
            "selection_score": selection_score_value,
        }
        for name, value in recorded.items():
            histories[name].append(value)

        if completed == 1 or completed % config.log_interval == 0 or completed == config.iterations:
            stage = "BP-bootstrap" if in_bootstrap else "DH-GAN"
            best_text = "none" if best_iteration == 0 else f"{best_iteration}:{best_score:.5f}"
            print(
                f"iter={completed:04d}/{config.iterations:04d} stage={stage:<12s} "
                f"holo={recorded['hologram']:.6f} adv={recorded['adversarial']:.4f} "
                f"adv_scale={recorded['adversarial_scale']:.3f} "
                f"D={probe_value:.4f}/{d_updates} bg={recorded['background']:.6f} "
                f"best={best_text} t={time.perf_counter()-timer:.1f}s"
            )

        if checkpoint_path and config.checkpoint_interval > 0 and (
            completed % config.checkpoint_interval == 0 or completed == config.iterations
        ):
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save({
                "stability_version": 1,
                "generator": generator.state_dict(),
                "ema_generator": ema_generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "background": background.detach().cpu(),
                "temperature": temperature,
                "iteration": completed,
                "histories": histories,
                "mask_events": mask_events,
                "selection_events": selection_events,
                "best_state": best_state,
                "best_background": best_background,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "best_metrics": best_metrics,
                "total_discriminator_updates": total_discriminator_updates,
                "config": asdict(config),
            }, checkpoint_path)

    if best_state is None:
        best_score, best_metrics = ground_truth_free_selection(
            ema_generator, network_input, output_size, forward, target, valid,
            selection_background, config,
        )
        best_iteration = config.iterations
        best_state = state_dict_to_cpu(ema_generator)
        best_background = background.detach().cpu().clone()
    ema_generator.load_state_dict(best_state)
    ema_generator.eval()
    if best_background is not None:
        background = best_background.to(device)
    with torch.no_grad():
        final = ema_generator(network_input, output_size)
        final_prediction = torch.abs(forward(final["field"])).square()
    return {
        "obj_amp": final["amplitude"].squeeze().cpu().numpy().astype(np.float32),
        "obj_phase": final["phase"].squeeze().cpu().numpy().astype(np.float32),
        "object_real": final["field"].real.squeeze().cpu().numpy().astype(np.float32),
        "object_imag": final["field"].imag.squeeze().cpu().numpy().astype(np.float32),
        "I_pred": final_prediction.squeeze().cpu().numpy().astype(np.float32),
        "error_map": torch.abs(final_prediction - target).squeeze().cpu().numpy().astype(np.float32),
        "base_amp": base_amplitude.squeeze().cpu().numpy().astype(np.float32),
        "base_phase": torch.angle(base_field).squeeze().cpu().numpy().astype(np.float32),
        "valid_mask": valid_np,
        "protection_mask": protection_np,
        "background_mask": background.squeeze().cpu().numpy().astype(np.float32),
        "histories": {
            key: np.asarray(values, dtype=np.float32)
            for key, values in histories.items()
        },
        "mask_events": mask_events,
        "selection_events": selection_events,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "total_discriminator_updates": total_discriminator_updates,
        "elapsed_seconds": float(time.perf_counter() - timer),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "device": str(device),
        "config": asdict(config),
    }


def display_limits(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    values = np.asarray(array)[np.isfinite(array)]
    lo, hi = np.percentile(values, [low, high]).astype(float)
    return lo, hi if hi > lo else lo + 1.0e-6


def save_png(array: np.ndarray, path: str, cmap: str = "gray", fixed: Optional[Tuple[float, float]] = None) -> None:
    figure, axis = plt.subplots(figsize=(4, 4))
    vmin, vmax = fixed if fixed is not None else display_limits(array)
    axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.axis("off")
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def save_outputs(results: Dict[str, object], intensity: np.ndarray, out_dir: str, input_path: str, sample: str, patch: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    arrays = {
        "table1_img": results["obj_amp"], "clean_amp": results["obj_amp"],
        "clean_phase": results["obj_phase"], "object_real": results["object_real"],
        "object_imag": results["object_imag"], "I_pred": results["I_pred"],
        "error_map": results["error_map"], "base_amp": results["base_amp"],
        "base_phase": results["base_phase"], "background_mask": results["background_mask"],
        "protection_mask": results["protection_mask"], "valid_mask": results["valid_mask"],
        "input_intensity": intensity,
    }
    for name, array in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), np.asarray(array, dtype=np.float32))
    for name, values in results["histories"].items():
        np.save(os.path.join(out_dir, f"history_{name}.npy"), values)
    metadata = {
        "method": "DH-GAN", "implementation": "stability-controlled direct-output project adaptation",
        "reference_doi": "10.1364/OE.480894", "input_path": input_path,
        "sample_id": sample, "patch_id": patch, "table1_image": "obj_amp",
        "z_distance_source": results.get("z_distance_source", "unspecified"),
        "config": results["config"], "mask_events": results["mask_events"],
        "selection_events": results["selection_events"],
        "selected_iteration": results["best_iteration"],
        "selection_score": results["best_score"],
        "selection_metrics": results["best_metrics"],
        "total_discriminator_updates": results["total_discriminator_updates"],
        "elapsed_seconds": results["elapsed_seconds"], "device": results["device"],
        "generator_parameters": results["generator_parameters"],
        "discriminator_parameters": results["discriminator_parameters"],
        "implementation_checks": [
            "direct generator output; no permanent BP skip or residual",
            "finite BP complex-field bootstrap removed after bootstrap_iters",
            "fixed support affects only the background-TV domain",
            "adaptive foreground may grow but cannot overwrite protected particles",
            "generator and discriminator use the paper-specified BatchNorm layers",
            "optional 2x super-resolution stage disabled for matched 256 x 256 project patches",
            "one-sided label smoothing and decaying instance noise stabilize adversarial training",
            "loss-gated discriminator uses zero, one, or at most five updates per generator update",
            "adversarial contribution is capped relative to physics and background-TV losses",
            "EMA checkpoint is selected without ground truth using sensor MSE, background TV, and collapse rate",
        ],
        "undisclosed_paper_parameters": [
            "loss weights", "Adam learning rates", "annealing temperature",
            "two-channel discriminator encoding", "random seed and stopping rule",
        ],
        "fair_comparison_policy": [
            "one fixed configuration is used for simulated and measured holograms",
            "the default seed is declared before evaluation and is not selected using ground truth",
            "neither training nor checkpoint selection reads a ground-truth array",
            "all paper-unspecified constants and stability adaptations are serialized here",
        ],
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def save_visuals(results: Dict[str, object], intensity: np.ndarray, png_dir: str) -> None:
    os.makedirs(png_dir, exist_ok=True)
    panels = {
        "input_intensity": (intensity, "gray", None), "base_amp": (results["base_amp"], "gray", None),
        "base_phase": (results["base_phase"], "twilight_shifted", (-np.pi, np.pi)),
        "clean_amp": (results["obj_amp"], "gray", None),
        "clean_phase": (results["obj_phase"], "twilight_shifted", (-np.pi, np.pi)),
        "I_pred": (results["I_pred"], "gray", None), "error_map": (results["error_map"], "magma", None),
        "background_mask": (results["background_mask"], "gray", (0.0, 1.0)),
        "protection_mask": (results["protection_mask"], "gray", (0.0, 1.0)),
        "valid_mask": (results["valid_mask"], "gray", (0.0, 1.0)),
        "object_real": (results["object_real"], "gray", None),
        "table1_img": (results["obj_amp"], "gray", None),
    }
    for name, (array, cmap, fixed) in panels.items():
        save_png(np.asarray(array), os.path.join(png_dir, f"{name}.png"), cmap, fixed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final direct-output DH-GAN baseline")
    parser.add_argument("--input", "--input_root", dest="input", default="data/sample_007/patch_0013/patch_0013.npy")
    parser.add_argument("--sample_id", default="")
    parser.add_argument("--patch_id", default="")
    parser.add_argument("--out_root", default="outputs/DH-GAN")
    parser.add_argument("--png_root", default="PNG/DH-GAN")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument(
        "--z_distance", type=float, default=None,
        help="Propagation distance in metres; otherwise read meta.json/best_z.txt",
    )
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--bootstrap_iters", type=int, default=300)
    parser.add_argument("--bootstrap_lr", type=float, default=1.0e-3)
    parser.add_argument("--generator_lr", type=float, default=2.0e-4)
    parser.add_argument("--discriminator_lr", type=float, default=1.0e-5)
    parser.add_argument("--d_steps", type=int, default=5)
    parser.add_argument("--lambda_hologram", type=float, default=50.0)
    parser.add_argument("--lambda_adversarial", type=float, default=0.01)
    parser.add_argument("--lambda_background", type=float, default=5.0)
    parser.add_argument("--adversarial_ratio_cap", type=float, default=0.05)
    parser.add_argument("--adversarial_ramp_iters", type=int, default=300)
    parser.add_argument("--real_label", type=float, default=0.90)
    parser.add_argument("--fake_label", type=float, default=0.10)
    parser.add_argument("--discriminator_gate_low", type=float, default=0.50)
    parser.add_argument("--discriminator_gate_high", type=float, default=0.65)
    parser.add_argument("--instance_noise", type=float, default=0.03)
    parser.add_argument("--instance_noise_iters", type=int, default=1000)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--selection_warmup", type=int, default=300)
    parser.add_argument("--selection_interval", type=int, default=50)
    parser.add_argument("--selection_saturation_margin", type=float, default=0.03)
    parser.add_argument("--selection_saturation_weight", type=float, default=0.50)
    parser.add_argument("--width_scale", type=float, default=1.0)
    parser.add_argument(
        "--super_resolution", action="store_true",
        help="Enable the optional 2x output stage from the paper; off for matched project patches",
    )
    parser.add_argument(
        "--no_super_resolution", dest="super_resolution", action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(super_resolution=False)
    parser.add_argument("--disc_input_mode", choices=["single", "duplicated"], default="duplicated")
    parser.add_argument("--mask_update_interval", type=int, default=100)
    parser.add_argument("--annealing_temperature", type=float, default=0.01)
    parser.add_argument("--support_percentile", type=float, default=3.0)
    parser.add_argument("--support_local_sigma", type=float, default=4.0)
    parser.add_argument("--support_trend_sigma", type=float, default=20.0)
    parser.add_argument("--support_min_area", type=int, default=32)
    parser.add_argument("--support_dilation", type=int, default=4)
    parser.add_argument("--border", type=int, default=6)
    parser.add_argument("--max_amplitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_png", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_parser().parse_args(argv)
    intensity = load_intensity(args.input, args.img_size)
    inferred_sample, inferred_patch = infer_ids(args.input)
    sample, patch = args.sample_id or inferred_sample, args.patch_id or inferred_patch
    out_dir = os.path.join(args.out_root, sample, patch)
    z_distance, z_source = resolve_z_distance(args.input, args.z_distance)
    print(f"Resolved z_distance={z_distance:.8g} m from {z_source}")
    config = DHGANConfig(
        wavelength=args.wavelength, pixel_size=args.pixel_size, z_distance=z_distance,
        iterations=args.iters, bootstrap_iters=args.bootstrap_iters,
        bootstrap_lr=args.bootstrap_lr,
        generator_lr=args.generator_lr, discriminator_lr=args.discriminator_lr,
        d_steps=args.d_steps, lambda_hologram=args.lambda_hologram,
        lambda_adversarial=args.lambda_adversarial, lambda_background=args.lambda_background,
        adversarial_ratio_cap=args.adversarial_ratio_cap,
        adversarial_ramp_iters=args.adversarial_ramp_iters,
        real_label=args.real_label, fake_label=args.fake_label,
        discriminator_gate_low=args.discriminator_gate_low,
        discriminator_gate_high=args.discriminator_gate_high,
        instance_noise=args.instance_noise, instance_noise_iters=args.instance_noise_iters,
        gradient_clip=args.gradient_clip, ema_decay=args.ema_decay,
        selection_warmup=args.selection_warmup,
        selection_interval=args.selection_interval,
        selection_saturation_margin=args.selection_saturation_margin,
        selection_saturation_weight=args.selection_saturation_weight,
        width_scale=args.width_scale, super_resolution=args.super_resolution,
        discriminator_input_mode=args.disc_input_mode, mask_update_interval=args.mask_update_interval,
        annealing_temperature=args.annealing_temperature, support_percentile=args.support_percentile,
        support_local_sigma=args.support_local_sigma, support_trend_sigma=args.support_trend_sigma,
        support_min_area=args.support_min_area, support_dilation=args.support_dilation,
        border=args.border, max_amplitude=args.max_amplitude, seed=args.seed,
        log_interval=args.log_interval, checkpoint_interval=args.checkpoint_interval,
    )
    results = reconstruct(
        intensity, config, checkpoint_path=os.path.join(out_dir, "dhgan_checkpoint.pt"), resume=args.resume
    )
    results["z_distance_source"] = z_source
    save_outputs(results, intensity, out_dir, args.input, sample, patch)
    if not args.no_png:
        save_visuals(results, intensity, os.path.join(args.png_root, sample, patch))
    print(f"[DH-GAN] finished: {sample}/{patch}")
    print(f"Outputs: {out_dir}")
    return results


if __name__ == "__main__":
    main()
