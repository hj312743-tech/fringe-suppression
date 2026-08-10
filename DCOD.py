"""
Deep Compressive Object Decoder (DCOD) baseline for this project.

This is a PyTorch reimplementation adapted to the input/output conventions of
the existing baselines in this repository.  It follows the central method in:

    F. Niknam, H. Qazvini, and H. Latifi,
    "Holographic optical field recovery using a regularized untrained deep
    decoder network," Scientific Reports 11, 10903 (2021).
    https://doi.org/10.1038/s41598-021-90312-5

The implementation is intentionally self-contained and does not depend on the
authors' TensorFlow/Fringe.Py code.  The main DCOD ingredients are retained:

1. a low-parameter Deep Decoder maps a fixed low-resolution random tensor to
   object amplitude and phase;
2. the predicted complex field is propagated by the angular-spectrum method;
3. the regenerated sensor intensity is fitted to the recorded hologram;
4. AdamW supplies decoupled weight-decay regularization;
5. during the first stage, the latent tensor is periodically perturbed and the
   amplitude coefficient alternates between 1.3 and 1.4;
6. a second stage disables randomization and reduces weight decay by 10x.

Project interface
-----------------
The default input is ``data/sample_007/patch_0013/patch_0013.npy``.  Outputs
are written to ``outputs/DCOD/<sample_id>/<patch_id>`` and include the common
``table1_img.npy`` and ``I_pred.npy`` files used by the evaluation pipeline.

For a quick smoke test, for example:

    python DCOD.py --stage1_iters 2 --stage2_iters 1 --channels 16 --no_png

The paper-oriented schedule is the default (30,000 + 5,000 iterations).  The
channel count is scaled to 128 by default for the project's 256 x 256 patches,
so that the decoder remains under-parameterized.  Use ``--channels 256`` for a
closer match to the public 512 x 512 implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.titlesize": 14,
        "figure.autolayout": True,
    }
)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class DCODConfig:
    wavelength: float = 632.8e-9
    pixel_size: float = 6.9e-6
    z_distance: float = 0.02275

    stage1_iters: int = 30000
    stage2_iters: int = 5000
    learning_rate: float = 1.0e-2
    weight_decay: float = 2.0e-3
    refine_weight_decay: float = 2.0e-4

    channels: int = 128
    decoder_depth: int = 5
    latent_std: float = 0.1
    latent_perturb_std: float = 0.02
    randomization_interval: int = 500
    amp_coefficients: Tuple[float, ...] = (1.3, 1.4)

    border: int = 6
    seed: int = 999
    log_interval: int = 300
    checkpoint_interval: int = 5000
    eps: float = 1.0e-8


# =============================================================================
# Basic utilities
# =============================================================================
def normalize_np(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Min-max normalize a 2-D image to [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return ((x - xmin) / (xmax - xmin + eps)).astype(np.float32)


def percentile_limits(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    finite = np.asarray(x)[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if vmax <= vmin:
        vmax = vmin + 1.0e-6
    return vmin, vmax


def load_intensity(input_path: str, img_size: int = 0) -> np.ndarray:
    """Load a hologram from NPY or a common image format."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input hologram not found: {path}")

    if path.suffix.lower() == ".npy":
        intensity = np.load(path).astype(np.float32)
    else:
        intensity = np.asarray(Image.open(path).convert("F"), dtype=np.float32)

    intensity = np.squeeze(intensity)
    if intensity.ndim != 2:
        raise ValueError(f"Expected a 2-D hologram, got shape {intensity.shape}")

    if img_size > 0:
        if intensity.shape[0] < img_size or intensity.shape[1] < img_size:
            raise ValueError(
                f"img_size={img_size} exceeds input shape {intensity.shape}; "
                "use --img_size 0 to retain the native size."
            )
        intensity = intensity[:img_size, :img_size]

    if not np.all(np.isfinite(intensity)):
        raise ValueError("The input hologram contains NaN or infinite values.")

    return normalize_np(intensity)


def infer_sample_patch_ids(input_path: str) -> Tuple[str, str]:
    """Infer sample and patch IDs from .../<sample>/<patch>/<file>."""
    path = Path(input_path)
    patch_id = path.parent.name or path.stem
    sample_id = path.parent.parent.name if path.parent.parent.name else "sample"
    return sample_id, patch_id


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_valid_mask(height: int, width: int, border: int, device: torch.device) -> torch.Tensor:
    """Create the same central valid-region convention used by project baselines."""
    if border < 0:
        raise ValueError("border must be non-negative")
    if border * 2 >= height or border * 2 >= width:
        raise ValueError(f"border={border} is too large for image shape {(height, width)}")
    mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    if border == 0:
        mask.fill_(1.0)
    else:
        mask[:, :, border : height - border, border : width - border] = 1.0
    return mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    return torch.sum((prediction - target).square() * mask) / (torch.sum(mask) + eps)


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Map phase to [-pi, pi] without changing its complex exponential."""
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


# =============================================================================
# Angular-spectrum propagation
# =============================================================================
class AngularSpectrumPropagator(nn.Module):
    """Precomputed differentiable ASM propagator for a fixed image geometry."""

    def __init__(
        self,
        height: int,
        width: int,
        z: float,
        wavelength: float,
        pixel_size: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        fx = fft.fftfreq(width, d=pixel_size, device=device)
        fy = fft.fftfreq(height, d=pixel_size, device=device)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

        wave_number = 2.0 * torch.pi / wavelength
        root_term = 1.0 - (wavelength * fx_grid).square() - (wavelength * fy_grid).square()

        # The sampling used in this project has no relevant evanescent support.
        # Clamping matches the ASM implementation used by the other baselines.
        root_term = torch.clamp(root_term, min=0.0)
        phase = wave_number * z * torch.sqrt(root_term)
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer("transfer", transfer[None, None, :, :])

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError(f"ASM expects [B,C,H,W], got {tuple(field.shape)}")
        return fft.ifft2(fft.fft2(field) * self.transfer)


def propagate_asm(
    field: torch.Tensor,
    z: float,
    wavelength: float,
    pixel_size: float,
) -> torch.Tensor:
    """Compatibility helper used for the one-time BP visualization."""
    _, _, height, width = field.shape
    propagator = AngularSpectrumPropagator(
        height=height,
        width=width,
        z=z,
        wavelength=wavelength,
        pixel_size=pixel_size,
        device=field.device,
    )
    return propagator(field)


# =============================================================================
# Deep Decoder
# =============================================================================
class DecoderBlock(nn.Module):
    """1 x 1 channel mixing, 2x bilinear upsampling, ReLU, and BatchNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(channels, eps=1.0e-3, momentum=0.01, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = F.relu(x, inplace=False)
        return self.bn(x)


class DeepDecoder(nn.Module):
    """
    Low-parameter decoder used by DCOD.

    The public implementation uses five 256-channel upsampling blocks for
    512 x 512 images.  This implementation exposes both values as parameters
    and automatically chooses a latent spatial size for the requested output.
    """

    def __init__(self, channels: int = 128, depth: int = 5, out_channels: int = 2) -> None:
        super().__init__()
        if channels <= 0 or depth <= 0:
            raise ValueError("channels and depth must be positive")
        self.channels = int(channels)
        self.depth = int(depth)
        self.blocks = nn.ModuleList([DecoderBlock(self.channels) for _ in range(self.depth)])
        self.final_conv = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True)
        self.final_bn = nn.BatchNorm2d(self.channels, eps=1.0e-3, momentum=0.01, affine=True)
        self.output_conv = nn.Conv2d(self.channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras Conv2D uses Glorot uniform by default; use the same family here.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        x = latent
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = F.relu(x, inplace=False)
        x = self.final_bn(x)
        x = torch.sigmoid(self.output_conv(x))
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class DCODModel(nn.Module):
    """Deep Decoder plus the fixed holographic forward model."""

    def __init__(
        self,
        image_shape: Tuple[int, int],
        z: float,
        wavelength: float,
        pixel_size: float,
        channels: int,
        depth: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(int(v) for v in image_shape)
        self.decoder = DeepDecoder(channels=channels, depth=depth, out_channels=2)
        self.propagator = AngularSpectrumPropagator(
            height=self.image_shape[0],
            width=self.image_shape[1],
            z=z,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=device,
        )

    def forward(self, latent: torch.Tensor, amp_coefficient: float | torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.decoder(latent, output_size=self.image_shape)

        # The public code uses channel 0 for phase in [0, 2pi] and channel 1
        # for amplitude, multiplied by a coefficient greater than one.
        phase_0_2pi = output[:, 0:1] * (2.0 * torch.pi)
        amplitude = output[:, 1:2] * amp_coefficient
        object_field = amplitude * torch.exp(1j * phase_0_2pi)

        sensor_field = self.propagator(object_field)
        predicted_intensity = torch.abs(sensor_field).square()
        return {
            "I_pred": predicted_intensity,
            "obj_amp": amplitude,
            "obj_phase_0_2pi": phase_0_2pi,
            "obj_phase": wrap_phase(phase_0_2pi),
            "object_field": object_field,
        }


# =============================================================================
# Reconstruction
# =============================================================================
def _set_optimizer_weight_decay(optimizer: optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["weight_decay"] = float(value)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    latent: torch.Tensor,
    amp_coefficient: float,
    loss_history: Sequence[float],
    amp_history: Sequence[float],
    config: DCODConfig,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
        "latent": latent.detach().cpu(),
        "amp_coefficient": float(amp_coefficient),
        "loss_history": list(float(v) for v in loss_history),
        "amp_history": list(float(v) for v in amp_history),
        "config": asdict(config),
    }


def reconstruct_dcod(
    intensity_target: np.ndarray,
    z: float,
    wavelength: float,
    pixel_size: float,
    stage1_iters: int = 30000,
    stage2_iters: int = 5000,
    border: int = 6,
    seed: int = 999,
    channels: int = 128,
    decoder_depth: int = 5,
    lr: float = 1.0e-2,
    weight_decay: float = 2.0e-3,
    refine_weight_decay: float = 2.0e-4,
    latent_std: float = 0.1,
    latent_perturb_std: float = 0.02,
    amp_coefficients: Sequence[float] = (1.3, 1.4),
    randomization_interval: int = 500,
    log_interval: int = 300,
    checkpoint_interval: int = 5000,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    eps: float = 1.0e-8,
) -> Dict[str, np.ndarray | float | int | str]:
    """
    Reconstruct one hologram using the adapted DCOD method.

    The data term is the normalized masked sensor-intensity MSE.  The public
    implementation uses ordinary MSE over the full sensor plane; restricting
    it to the common valid region is the only loss-domain adaptation made for
    consistency with the other project baselines.
    """
    intensity_target = np.asarray(intensity_target, dtype=np.float32)
    if intensity_target.ndim != 2:
        raise ValueError(f"intensity_target must be 2-D, got {intensity_target.shape}")
    if stage1_iters < 0 or stage2_iters < 0:
        raise ValueError("Iteration counts must be non-negative")
    if stage1_iters + stage2_iters <= 0:
        raise ValueError("At least one optimization iteration is required")
    if not amp_coefficients or any(value <= 0 for value in amp_coefficients):
        raise ValueError("amp_coefficients must contain positive values")
    if randomization_interval <= 0:
        raise ValueError("randomization_interval must be positive")

    config = DCODConfig(
        wavelength=wavelength,
        pixel_size=pixel_size,
        z_distance=z,
        stage1_iters=stage1_iters,
        stage2_iters=stage2_iters,
        learning_rate=lr,
        weight_decay=weight_decay,
        refine_weight_decay=refine_weight_decay,
        channels=channels,
        decoder_depth=decoder_depth,
        latent_std=latent_std,
        latent_perturb_std=latent_perturb_std,
        randomization_interval=randomization_interval,
        amp_coefficients=tuple(float(v) for v in amp_coefficients),
        border=border,
        seed=seed,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        eps=eps,
    )

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = intensity_target.shape
    target_I = torch.from_numpy(intensity_target).to(device=device, dtype=torch.float32)[None, None]
    valid_mask = make_valid_mask(height, width, border=border, device=device)

    # Conventional BP is used only as a visualization reference.
    with torch.no_grad():
        sensor_field_zero_phase = torch.sqrt(torch.clamp(target_I, min=0.0)).to(torch.complex64)
        baseline_field = propagate_asm(sensor_field_zero_phase, -z, wavelength, pixel_size)
        baseline_amp = torch.abs(baseline_field).cpu().squeeze().numpy().astype(np.float32)
        baseline_phase = torch.angle(baseline_field).cpu().squeeze().numpy().astype(np.float32)

    model = DCODModel(
        image_shape=(height, width),
        z=z,
        wavelength=wavelength,
        pixel_size=pixel_size,
        channels=channels,
        depth=decoder_depth,
        device=device,
    ).to(device)

    downsample_factor = 2**decoder_depth
    latent_height = max(1, math.ceil(height / downsample_factor))
    latent_width = max(1, math.ceil(width / downsample_factor))
    latent_reference = torch.randn(
        (1, channels, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    ) * latent_std
    latent = latent_reference.clone()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_iters = stage1_iters + stage2_iters
    start_iteration = 0
    amp_coefficient = float(amp_coefficients[0])
    loss_history: List[float] = []
    amp_history: List[float] = []

    if resume:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("--resume was requested but no checkpoint file was found")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"])
        latent = checkpoint["latent"].to(device=device, dtype=torch.float32)
        amp_coefficient = float(checkpoint["amp_coefficient"])
        loss_history = [float(v) for v in checkpoint.get("loss_history", [])]
        amp_history = [float(v) for v in checkpoint.get("amp_history", [])]
        print(f"[DCOD] Resuming from iteration {start_iteration}")

    parameter_count = count_trainable_parameters(model)
    output_values = 2 * height * width
    print("Start DCOD iterative reconstruction...")
    print(f"Device: {device}")
    print(
        f"Decoder: depth={decoder_depth}, channels={channels}, "
        f"latent={latent_height}x{latent_width}, parameters={parameter_count:,}, "
        f"two-channel output values={output_values:,}"
    )
    if parameter_count >= output_values:
        print(
            "[DCOD warning] The decoder is not under-parameterized at this image size. "
            "Consider reducing --channels for a closer Deep Decoder prior."
        )

    model.train()
    timer_start = time.perf_counter()
    running_losses: List[float] = []
    latest_out: Optional[Dict[str, torch.Tensor]] = None

    for iteration in range(start_iteration, total_iters):
        in_stage1 = iteration < stage1_iters

        if in_stage1:
            _set_optimizer_weight_decay(optimizer, weight_decay)
            # Match the public schedule: perturb every 500 steps, excluding 0.
            if iteration > 0 and iteration % randomization_interval == 0:
                latent = latent_reference + torch.randn_like(latent_reference) * latent_perturb_std
                coefficient_index = (iteration // randomization_interval) % len(amp_coefficients)
                amp_coefficient = float(amp_coefficients[coefficient_index])
                print(
                    f"[DCOD] Randomization at iter {iteration}: "
                    f"latent std={latent_perturb_std:g}, amp coefficient={amp_coefficient:g}"
                )
        else:
            # The paper's final 5,000 iterations use 0.1x weight decay and no randomization.
            _set_optimizer_weight_decay(optimizer, refine_weight_decay)

        optimizer.zero_grad(set_to_none=True)
        latest_out = model(latent, amp_coefficient=amp_coefficient)
        loss = masked_mse(latest_out["I_pred"], target_I, valid_mask, eps=eps)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DCOD loss at iteration {iteration + 1}")

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        loss_history.append(loss_value)
        amp_history.append(float(amp_coefficient))
        running_losses.append(loss_value)
        if len(running_losses) > 100:
            running_losses.pop(0)

        completed = iteration + 1
        if completed == 1 or completed % log_interval == 0 or completed == total_iters:
            stage_name = "randomized" if in_stage1 else "refinement"
            elapsed = time.perf_counter() - timer_start
            print(
                f"Iter {completed:05d}/{total_iters:05d} | stage={stage_name:<10s} | "
                f"loss={loss_value:.7f} | avg100={np.mean(running_losses):.7f} | "
                f"amp={amp_coefficient:.3f} | elapsed={elapsed:.1f}s"
            )

        if (
            checkpoint_path
            and checkpoint_interval > 0
            and (completed % checkpoint_interval == 0 or completed == total_iters)
        ):
            checkpoint_parent = os.path.dirname(checkpoint_path)
            if checkpoint_parent:
                os.makedirs(checkpoint_parent, exist_ok=True)
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed,
                    latent=latent,
                    amp_coefficient=amp_coefficient,
                    loss_history=loss_history,
                    amp_history=amp_history,
                    config=config,
                ),
                checkpoint_path,
            )
            print(f"[DCOD] Checkpoint saved: {checkpoint_path}")

    elapsed_seconds = time.perf_counter() - timer_start

    # Recompute the final result after the last optimizer update.  BatchNorm is
    # deliberately kept in training mode to mirror Keras training=True.
    with torch.no_grad():
        latest_out = model(latent, amp_coefficient=amp_coefficient)

    I_pred_np = latest_out["I_pred"].cpu().squeeze().numpy().astype(np.float32)
    obj_amp_np = latest_out["obj_amp"].cpu().squeeze().numpy().astype(np.float32)
    obj_phase_np = latest_out["obj_phase"].cpu().squeeze().numpy().astype(np.float32)
    phase_0_2pi_np = latest_out["obj_phase_0_2pi"].cpu().squeeze().numpy().astype(np.float32)
    valid_np = valid_mask.cpu().squeeze().numpy().astype(np.float32)
    error_map = np.abs(I_pred_np - intensity_target).astype(np.float32)

    return {
        "base_amp": baseline_amp,
        "base_phase": baseline_phase,
        "obj_amp": obj_amp_np,
        "obj_phase": obj_phase_np,
        "obj_phase_0_2pi": phase_0_2pi_np,
        "obj_contrast": (1.0 - obj_amp_np).astype(np.float32),
        "I_pred": I_pred_np,
        "error_map": error_map,
        "valid_mask": valid_np,
        "loss_history": np.asarray(loss_history, dtype=np.float32),
        "amp_coefficient_history": np.asarray(amp_history, dtype=np.float32),
        "final_amp_coefficient": float(amp_coefficient),
        "elapsed_seconds": float(elapsed_seconds),
        "parameter_count": int(parameter_count),
        "latent_shape": np.asarray(latent.shape, dtype=np.int32),
        "device": str(device),
        "config": asdict(config),
    }


# =============================================================================
# Output and visualization
# =============================================================================
def save_single_image(
    image: np.ndarray,
    save_path: str,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 300,
    colorbar: bool = False,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    handle = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if colorbar:
        plt.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_results_dcod(
    intensity: np.ndarray,
    results: Dict[str, np.ndarray | float | int | str],
    save_path: str,
    out_dir: str,
    prefix: str,
    save_single: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    valid_mask = np.asarray(results["valid_mask"])
    obj_amp_vis = np.asarray(results["obj_amp"]).copy()
    obj_phase_vis = np.asarray(results["obj_phase"]).copy()
    error_map_vis = np.asarray(results["error_map"]).copy()
    obj_amp_vis[valid_mask < 0.5] = np.nan
    obj_phase_vis[valid_mask < 0.5] = np.nan
    error_map_vis[valid_mask < 0.5] = np.nan

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    panels = [
        (axes[0, 0], intensity, "(a) Raw Hologram", "gray", *percentile_limits(intensity, 1, 99)),
        (
            axes[0, 1],
            np.asarray(results["I_pred"]),
            "(b) Predicted Hologram",
            "gray",
            *percentile_limits(np.asarray(results["I_pred"]), 1, 99),
        ),
        (
            axes[0, 2],
            error_map_vis,
            "(c) Hologram Error",
            "magma",
            *percentile_limits(error_map_vis, 1, 99),
        ),
        (
            axes[1, 0],
            np.asarray(results["base_amp"]),
            "(e) Baseline ASM Amp",
            "gray",
            *percentile_limits(np.asarray(results["base_amp"]), 1, 99),
        ),
        (
            axes[1, 1],
            obj_amp_vis,
            "(f) DCOD Amp",
            "gray",
            *percentile_limits(obj_amp_vis, 1, 99.5),
        ),
        (
            axes[1, 2],
            1.0 - obj_amp_vis,
            "(g) DCOD Contrast (1-Amp)",
            "cividis",
            *percentile_limits(1.0 - obj_amp_vis, 1, 99.5),
        ),
        (axes[1, 3], obj_phase_vis, "(h) DCOD Phase", "twilight_shifted", -np.pi, np.pi),
    ]

    for axis, image, title, cmap, vmin, vmax in panels:
        handle = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        plt.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    loss_history = np.asarray(results["loss_history"])
    axes[0, 3].plot(loss_history, linewidth=1.4)
    axes[0, 3].set_title("(d) DCOD Physics Loss")
    axes[0, 3].set_xlabel("Iteration")
    axes[0, 3].set_ylabel("Masked MSE")
    axes[0, 3].set_yscale("log")
    axes[0, 3].grid(alpha=0.3)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not save_single:
        return

    save_single_image(
        intensity,
        os.path.join(out_dir, f"{prefix}_a_raw_hologram.png"),
        "gray",
        *percentile_limits(intensity, 1, 99),
    )
    save_single_image(
        np.asarray(results["I_pred"]),
        os.path.join(out_dir, f"{prefix}_b_I_pred.png"),
        "gray",
        *percentile_limits(np.asarray(results["I_pred"]), 1, 99),
    )
    save_single_image(
        error_map_vis,
        os.path.join(out_dir, f"{prefix}_c_error_map.png"),
        "magma",
        *percentile_limits(error_map_vis, 1, 99),
    )
    save_single_image(
        obj_amp_vis,
        os.path.join(out_dir, f"{prefix}_f_dcod_amp.png"),
        "gray",
        *percentile_limits(obj_amp_vis, 1, 99.5),
    )
    save_single_image(
        obj_phase_vis,
        os.path.join(out_dir, f"{prefix}_h_dcod_phase.png"),
        "twilight_shifted",
        -np.pi,
        np.pi,
    )


def save_outputs(
    results: Dict[str, np.ndarray | float | int | str],
    intensity: np.ndarray,
    out_dir: str,
    sample_id: str,
    patch_id: str,
    input_path: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    array_keys = {
        "table1_img.npy": "obj_amp",
        "I_pred.npy": "I_pred",
        "clean_amp.npy": "obj_amp",
        "clean_phase.npy": "obj_phase",
        "phase_0_2pi.npy": "obj_phase_0_2pi",
        "base_amp.npy": "base_amp",
        "base_phase.npy": "base_phase",
        "error_map.npy": "error_map",
        "valid_mask.npy": "valid_mask",
        "loss_history.npy": "loss_history",
        "amp_coefficient_history.npy": "amp_coefficient_history",
    }
    for filename, key in array_keys.items():
        np.save(os.path.join(out_dir, filename), np.asarray(results[key]))
    np.save(os.path.join(out_dir, "input_intensity.npy"), intensity.astype(np.float32))

    config = dict(results["config"])
    meta = {
        "method": "DCOD",
        "implementation": "PyTorch project adaptation based on Niknam et al. (2021)",
        "reference_doi": "10.1038/s41598-021-90312-5",
        "reference_code": "https://github.com/farhadnkm/DCOD",
        "input_path": str(input_path),
        "sample_id": sample_id,
        "patch_id": patch_id,
        "table1_image": "obj_amp",
        "complex_field_output": True,
        "parameter_count": int(results["parameter_count"]),
        "latent_shape": np.asarray(results["latent_shape"]).astype(int).tolist(),
        "final_amp_coefficient": float(results["final_amp_coefficient"]),
        "elapsed_seconds": float(results["elapsed_seconds"]),
        "device": str(results["device"]),
        "config": config,
        "adaptations": [
            "PyTorch implementation",
            "project angular-spectrum geometry",
            "project valid-region masked intensity MSE",
            "project output-directory and Table 1 interface",
            "default decoder width scaled to 128 channels for 256x256 patches",
        ],
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


# =============================================================================
# Command-line interface
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DCOD untrained Deep Decoder baseline for lensless holographic reconstruction."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "sample_007", "patch_0013", "patch_0013.npy"),
        help="Input hologram (.npy or image).",
    )
    parser.add_argument("--sample_id", type=str, default="", help="Output sample ID; inferred if omitted.")
    parser.add_argument("--patch_id", type=str, default="", help="Output patch ID; inferred if omitted.")
    parser.add_argument("--out_root", type=str, default=os.path.join("outputs", "DCOD"))
    parser.add_argument("--png_root", type=str, default=os.path.join("PNG", "DCOD"))
    parser.add_argument("--img_size", type=int, default=256, help="Top-left crop size; 0 keeps native size.")

    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument("--z_distance", type=float, default=0.02275)

    parser.add_argument("--stage1_iters", type=int, default=30000)
    parser.add_argument("--stage2_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--weight_decay", type=float, default=2.0e-3)
    parser.add_argument("--refine_weight_decay", type=float, default=2.0e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--decoder_depth", type=int, default=5)
    parser.add_argument("--latent_std", type=float, default=0.1)
    parser.add_argument("--latent_perturb_std", type=float, default=0.02)
    parser.add_argument("--randomization_interval", type=int, default=500)
    parser.add_argument("--amp_coefficients", type=float, nargs="+", default=[1.3, 1.4])
    parser.add_argument("--border", type=int, default=6)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--resume", action="store_true", help="Resume from the patch checkpoint.")
    parser.add_argument("--no_png", action="store_true", help="Skip PNG visualization output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray | float | int | str]:
    args = build_parser().parse_args(argv)

    intensity = load_intensity(args.input, img_size=args.img_size)
    inferred_sample, inferred_patch = infer_sample_patch_ids(args.input)
    sample_id = args.sample_id or inferred_sample
    patch_id = args.patch_id or inferred_patch

    out_dir = os.path.join(args.out_root, sample_id, patch_id)
    checkpoint_path = os.path.join(out_dir, "dcod_checkpoint.pt")

    results = reconstruct_dcod(
        intensity_target=intensity,
        z=args.z_distance,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        stage1_iters=args.stage1_iters,
        stage2_iters=args.stage2_iters,
        border=args.border,
        seed=args.seed,
        channels=args.channels,
        decoder_depth=args.decoder_depth,
        lr=args.lr,
        weight_decay=args.weight_decay,
        refine_weight_decay=args.refine_weight_decay,
        latent_std=args.latent_std,
        latent_perturb_std=args.latent_perturb_std,
        amp_coefficients=args.amp_coefficients,
        randomization_interval=args.randomization_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    save_outputs(
        results=results,
        intensity=intensity,
        out_dir=out_dir,
        sample_id=sample_id,
        patch_id=patch_id,
        input_path=args.input,
    )

    if not args.no_png:
        os.makedirs(args.png_root, exist_ok=True)
        prefix = f"DCOD_{sample_id}_{patch_id}"
        draw_results_dcod(
            intensity=intensity,
            results=results,
            save_path=os.path.join(args.png_root, f"{prefix}_canvas.png"),
            out_dir=args.png_root,
            prefix=prefix,
            save_single=True,
        )

    print(f"[DCOD] finished: {sample_id}/{patch_id}")
    print(f"Outputs saved to: {out_dir}")
    return results


if __name__ == "__main__":
    main()
