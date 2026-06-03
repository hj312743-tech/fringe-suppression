import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.titlesize": 14,
    "figure.autolayout": True,
})


# =========================================================
# Basic utilities
# =========================================================
def normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x: np.ndarray, low=1.0, high=99.0):
    vals = np.asarray(x, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vals, low)
    vmax = np.percentile(vals, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


# =========================================================
# Angular-spectrum propagation (ASM)
# =========================================================
def propagate_asm(field: torch.Tensor, z: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    _, _, H, W = field.shape
    fx = fft.fftfreq(W, d=pixel_size)
    fy = fft.fftfreq(H, d=pixel_size)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    FX, FY = FX.to(field.device), FY.to(field.device)

    k = 2 * torch.pi / wavelength
    term = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    term = torch.clamp(term, min=0.0)
    phase_shift = k * z * torch.sqrt(term)

    H_transfer = torch.exp(1j * phase_shift)
    field_prop = fft.ifft2(fft.fft2(field) * H_transfer)
    return field_prop


# =========================================================
# Loss functions
# =========================================================
def make_valid_mask(H: int, W: int, border: int, device: torch.device) -> torch.Tensor:
    """
    Create a valid field-of-view mask consistent with the full model.

    Border pixels are excluded from optimization, evaluation, and visualization
    to suppress ASM periodic-boundary artifacts and support-padding artifacts.
    """
    if border < 0:
        raise ValueError(f"border must be non-negative: {border}")
    if border * 2 >= min(H, W):
        raise ValueError(f"border is too large: image={(H, W)}, border={border}")

    m = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    if border == 0:
        m[:, :, :, :] = 1.0
    else:
        m[:, :, border:H-border, border:W-border] = 1.0
    return m


def masked_l1(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(torch.abs(x) * mask) / (torch.sum(mask) + eps)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(((x - y) ** 2) * mask) / (torch.sum(mask) + eps)


def masked_tv_loss(img: torch.Tensor, weight_map=None, valid_mask=None, eps: float = 1e-8) -> torch.Tensor:
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])

    if weight_map is None:
        wy = 1.0
        wx = 1.0
    else:
        wy = 0.5 * (weight_map[:, :, 1:, :] + weight_map[:, :, :-1, :])
        wx = 0.5 * (weight_map[:, :, :, 1:] + weight_map[:, :, :, :-1])

    if valid_mask is None:
        my = 1.0
        mx = 1.0
    else:
        my = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
        mx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]

    dy = dy * wy * my
    dx = dx * wx * mx

    if torch.is_tensor(my):
        denom_y = torch.sum(my) + eps
        denom_x = torch.sum(mx) + eps
    else:
        denom_y = dy.numel()
        denom_x = dx.numel()

    return torch.sum(dx) / denom_x + torch.sum(dy) / denom_y


def multi_scale_masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = masked_mse(pred, target, valid_mask)

    pred_2 = F.avg_pool2d(pred, 2, 2)
    tar_2 = F.avg_pool2d(target, 2, 2)
    mask_2 = (F.avg_pool2d(valid_mask, 2, 2) > 0.999).float()
    loss = loss + 0.5 * masked_mse(pred_2, tar_2, mask_2)

    pred_4 = F.avg_pool2d(pred, 4, 4)
    tar_4 = F.avg_pool2d(target, 4, 4)
    mask_4 = (F.avg_pool2d(valid_mask, 4, 4) > 0.999).float()
    loss = loss + 0.25 * masked_mse(pred_4, tar_4, mask_4)
    return loss


# =========================================================
# Network modules
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SingleFieldResUNet(nn.Module):
    """
    Single-field ablation network.

    The network outputs amplitude and phase maps, forms one complex field,
    and propagates it to the sensor plane.
    """
    def __init__(self, in_channels=1, base_f=24):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_f)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_f, base_f * 2)
        self.bottleneck = ResidualBlock(base_f * 2, base_f * 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(base_f * 3, base_f)
        self.out_conv = nn.Conv2d(base_f, 2, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(e2)
        d1 = self.dec1(torch.cat([self.up(b), e1], dim=1))
        out = self.out_conv(d1)

        amp = torch.sigmoid(out[:, 0:1, :, :])
        phase = torch.tanh(out[:, 1:2, :, :]) * torch.pi
        return amp, phase


class SingleModelAblationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_net = SingleFieldResUNet()

    def forward(self, noise, z, wavelength, pixel_size):
        A_single, phi_single = self.obj_net(noise)
        U_single = A_single * torch.exp(1j * phi_single)
        U_sensor = propagate_asm(U_single, z, wavelength, pixel_size)
        I_pred = torch.abs(U_sensor) ** 2
        U_single_amp = torch.abs(U_single)
        return {
            "I_pred": I_pred,
            "U_single_amp": U_single_amp,
            "A_single": A_single,
            "phi_single": phi_single,
            "U_single": U_single,
            "U_sensor": U_sensor,
        }


# =========================================================
# Particle-likelihood map: use the same strategy as the full model for a fair ablation.
# =========================================================
def build_soft_support_from_baseline_hp(
    baseline_amp,
    device,
    percentile=93.4,
    hp_kernel=31,
    smooth_kernel=7,
    smooth_iters=2,
    boost=1.35,
    edge_exclude=14,
    dilate_kernel=5,
):
    x = normalize_np(baseline_amp)
    obj_like = 1.0 - x

    t = torch.tensor(obj_like, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    low = F.avg_pool2d(t, kernel_size=hp_kernel, stride=1, padding=hp_kernel // 2)
    hp = torch.relu(t - low)

    hp_np = hp.detach().cpu().squeeze().numpy()
    H, W = hp_np.shape

    valid_np = np.zeros_like(hp_np, dtype=np.float32)
    valid_np[edge_exclude:H-edge_exclude, edge_exclude:W-edge_exclude] = 1.0

    vals = hp_np[valid_np > 0.5]
    if vals.size == 0:
        vals = hp_np.reshape(-1)
    thr = np.percentile(vals, percentile)

    hard_mask = ((hp_np > thr) & (valid_np > 0.5)).astype(np.float32)
    hard_mask = ndi.binary_fill_holes(hard_mask > 0.5).astype(np.float32)

    mask_t = torch.tensor(hard_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    for _ in range(smooth_iters):
        mask_t = F.avg_pool2d(mask_t, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)

    mask_t = torch.clamp(mask_t * boost, 0.0, 1.0)

    if dilate_kernel > 1:
        pad = dilate_kernel // 2
        mask_t = F.max_pool2d(mask_t, kernel_size=dilate_kernel, stride=1, padding=pad)

    mask_t = torch.clamp(mask_t, 0.0, 1.0)
    return mask_t, hp_np


# =========================================================
# Main reconstruction
# =========================================================
def reconstruct_single_model(
    intensity_target,
    z,
    wavelength,
    pixel_size,
    iters=2200,
    border=16,
    support_percentile=92.8,
    support_boost=1.45,
    support_edge_exclude=12,
    support_dilate_kernel=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = intensity_target.shape
    target_I = torch.tensor(intensity_target, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    valid_mask = make_valid_mask(H, W, border=border, device=device)

    # Baseline ASM
    U_raw = torch.sqrt(torch.clamp(target_I, min=0.0)) + 0j
    U_baseline = propagate_asm(U_raw, -z, wavelength, pixel_size)
    baseline_amp = torch.abs(U_baseline).detach().cpu().squeeze().numpy()
    baseline_phase = torch.angle(U_baseline).detach().cpu().squeeze().numpy()

    # Particle-likelihood map
    support_mask, support_seed = build_soft_support_from_baseline_hp(
        baseline_amp=baseline_amp,
        device=device,
        percentile=support_percentile,
        hp_kernel=31,
        smooth_kernel=7,
        smooth_iters=2,
        boost=support_boost,
        edge_exclude=support_edge_exclude,
        dilate_kernel=support_dilate_kernel,
    )
    support_mask = support_mask * valid_mask
    outside_mask = (1.0 - support_mask) * valid_mask
    tv_weight_map = (0.25 + 0.75 * outside_mask) * valid_mask

    torch.manual_seed(42)
    fixed_noise = torch.rand((1, 1, H, W), device=device)
    net = SingleModelAblationModel().to(device)

    optimizer = optim.Adam([
        {"params": net.obj_net.parameters(), "lr": 7e-4},
    ])

    # Use the same single-field constraints as the full model where applicable.
    w_tv_amp = 0.015
    w_tv_phase = 0.012
    w_support_out_amp = 0.18
    w_support_out_phase = 0.03
    w_sparse_inside = 0.008
    w_phase_inside = 0.000
    w_amp_inside_pull = 0.010

    print("Start single-model ablation reconstruction...")
    for i in range(iters):
        optimizer.zero_grad()

        out = net(fixed_noise, z, wavelength, pixel_size)
        I_pred_tensor = out["I_pred"]
        A_single = out["A_single"]
        phi_single = out["phi_single"]
        U_single_amp = out["U_single_amp"]

        loss_data = multi_scale_masked_mse(I_pred_tensor, target_I, valid_mask)

        loss_support = (
            masked_l1((1.0 - A_single), outside_mask) * w_support_out_amp +
            masked_l1(phi_single, outside_mask) * w_support_out_phase
        )

        loss_sparse_inside = masked_l1((1.0 - A_single), support_mask) * w_sparse_inside
        loss_phase_inside = masked_l1(phi_single, support_mask) * w_phase_inside

        desired_contrast = 0.10 * support_mask
        loss_amp_inside_pull = (
            masked_l1(F.relu(desired_contrast - (1.0 - A_single)), support_mask) * w_amp_inside_pull
        )

        loss_tv = (
            masked_tv_loss(A_single, weight_map=tv_weight_map, valid_mask=valid_mask) * w_tv_amp +
            masked_tv_loss(phi_single, weight_map=tv_weight_map, valid_mask=valid_mask) * w_tv_phase
        )

        loss = (
            loss_data +
            loss_support +
            loss_sparse_inside +
            loss_phase_inside +
            loss_amp_inside_pull +
            loss_tv
        )
        loss.backward()
        optimizer.step()

        if (i + 1) % 300 == 0:
            print(
                f"Iter {i + 1:04d} | "
                f"Data: {loss_data.item():.5f} | "
                f"Support: {loss_support.item():.5f} | "
                f"Sparse-in: {loss_sparse_inside.item():.5f} | "
                f"TV: {loss_tv.item():.5f}"
            )

    A_single_np = A_single.detach().cpu().squeeze().numpy().astype(np.float32)
    phi_single_np = phi_single.detach().cpu().squeeze().numpy().astype(np.float32)
    U_single_amp_np = U_single_amp.detach().cpu().squeeze().numpy().astype(np.float32)
    I_pred_np = I_pred_tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    error_map = np.abs(I_pred_np - intensity_target).astype(np.float32)
    support_np = support_mask.detach().cpu().squeeze().numpy().astype(np.float32)
    valid_np = valid_mask.detach().cpu().squeeze().numpy().astype(np.float32)
    U_single_amp_contrast = (1.0 - normalize_np(U_single_amp_np)).astype(np.float32)

    return {
        "base_amp": baseline_amp.astype(np.float32),
        "base_phase": baseline_phase.astype(np.float32),
        "U_single_amp": U_single_amp_np,
        "paper_main_output": U_single_amp_np,
        "table1_img": U_single_amp_np,
        "U_single_amp_contrast": U_single_amp_contrast,
        "A_single": A_single_np,
        "phi_single": phi_single_np,
        "single_contrast": (1.0 - A_single_np).astype(np.float32),
        "I_pred": I_pred_np,
        "error_map": error_map,
        "support_mask": support_np,
        "valid_mask": valid_np,
        "support_seed": support_seed.astype(np.float32),
        "border": int(border),
    }



# =========================================================
# Saving, metrics, and visualization
# =========================================================
def crop2d(arr: np.ndarray, crop: int) -> np.ndarray:
    """Centrally crop a 2D array by the specified border width."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Only 2D arrays are supported, got shape={x.shape}")
    if crop <= 0:
        return x
    h, w = x.shape
    if 2 * crop >= min(h, w):
        raise ValueError(f"display_crop is too large: image={(h, w)}, crop={crop}")
    return x[crop:h - crop, crop:w - crop]


def crop_with_valid(arr: np.ndarray, valid: np.ndarray, crop: int, fill_nan: bool = True) -> np.ndarray:
    """
    Mask invalid borders with NaN or zeros, then apply central cropping.

    When display_crop equals border, invalid border pixels are not shown.
    """
    x = np.asarray(arr, dtype=np.float32).copy()
    v = np.asarray(valid).astype(bool)
    if x.shape != v.shape:
        raise ValueError(f"Shape mismatch between arr and valid: arr={x.shape}, valid={v.shape}")
    if fill_nan:
        x[~v] = np.nan
    else:
        x[~v] = 0.0
    return crop2d(x, crop)


def raw_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    return crop2d(np.asarray(arr, dtype=np.float32), crop)


def resolve_display_crop(display_crop: int, border: int) -> int:
    """Use border as the display crop when display_crop is negative."""
    return int(border if display_crop is None or display_crop < 0 else display_crop)


def save_cropped_arrays(results, out_dir: str, display_crop: int):
    """
    Save additionally cropped arrays for visualization and supplementary figures.

    The full-size table1_img.npy is still saved for the unified evaluation script.
    Cropped arrays are intended only for direct figure export.
    """
    crop_dir = os.path.join(out_dir, f"crop{display_crop}")
    os.makedirs(crop_dir, exist_ok=True)

    valid = results["valid_mask"] > 0.5
    save_keys = [
        "table1_img",
        "paper_main_output",
        "U_single_amp",
        "U_single_amp_contrast",
        "A_single",
        "phi_single",
        "I_pred",
        "error_map",
        "support_mask",
    ]

    for key in save_keys:
        arr = results.get(key, None)
        if arr is None:
            continue
        # Save all cropped arrays using the same crop rule and avoid NaNs for later export.
        cropped = crop_with_valid(arr, valid, display_crop, fill_nan=False)
        np.save(os.path.join(crop_dir, f"{key}.npy"), cropped.astype(np.float32))

    with open(os.path.join(crop_dir, "crop_info.txt"), "w", encoding="utf-8") as f:
        h, w = results["table1_img"].shape
        f.write(f"original_shape = {(h, w)}\n")
        f.write(f"display_crop   = {display_crop}\n")
        f.write(f"cropped_shape  = {crop2d(results['table1_img'], display_crop).shape}\n")
        f.write("These cropped arrays are for visualization/supplementary figures only.\n")

    print(f"Cropped arrays saved to: {crop_dir}")


def save_outputs(results, out_dir, display_crop: int | None = None):
    os.makedirs(out_dir, exist_ok=True)

    # Full-size outputs are retained for the unified evaluation script.
    np.save(os.path.join(out_dir, "table1_img.npy"), results["table1_img"].astype(np.float32))
    np.save(os.path.join(out_dir, "paper_main_output.npy"), results["paper_main_output"].astype(np.float32))
    np.save(os.path.join(out_dir, "U_single_amp.npy"), results["U_single_amp"].astype(np.float32))
    np.save(os.path.join(out_dir, "U_single_amp_contrast.npy"), results["U_single_amp_contrast"].astype(np.float32))
    np.save(os.path.join(out_dir, "A_single.npy"), results["A_single"].astype(np.float32))
    np.save(os.path.join(out_dir, "phi_single.npy"), results["phi_single"].astype(np.float32))
    np.save(os.path.join(out_dir, "I_pred.npy"), results["I_pred"].astype(np.float32))
    np.save(os.path.join(out_dir, "error_map.npy"), results["error_map"].astype(np.float32))
    np.save(os.path.join(out_dir, "support_mask.npy"), results["support_mask"].astype(np.float32))
    np.save(os.path.join(out_dir, "valid_mask.npy"), results["valid_mask"].astype(np.float32))

    if display_crop is not None and display_crop > 0:
        save_cropped_arrays(results, out_dir, display_crop)

    print(f"Outputs saved to: {out_dir}")


def prepare_cmap(name: str, bad_color: str = 'white'):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=bad_color)
    return cmap


def valid_residual_metrics(raw: np.ndarray, sim: np.ndarray, valid: np.ndarray):
    m = valid.astype(bool)
    x = raw[m].astype(np.float64)
    y = sim[m].astype(np.float64)
    if x.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "valid_pixels": 0}
    diff = x - y
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"mae": mae, "rmse": rmse, "valid_pixels": int(x.size)}


def save_metric_summary(intensity, results, out_dir, full_model_dir=None, display_crop: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    valid = results["valid_mask"] > 0.5
    obj_metrics = valid_residual_metrics(intensity, results["I_pred"], valid)

    lines = [
        "Single-model ablation metrics",
        f"Border / valid crop = {results.get('border', 'NA')}",
        f"Display crop        = {display_crop if display_crop is not None else 'NA'}",
        f"Valid pixels        = {obj_metrics['valid_pixels']}",
        f"MAE(single-model)  = {obj_metrics['mae']:.6f}",
        f"RMSE(single-model) = {obj_metrics['rmse']:.6f}",
    ]

    summary = {
        "border": int(results.get("border", -1)),
        "display_crop": None if display_crop is None else int(display_crop),
        "single_model": obj_metrics,
    }

    if full_model_dir is not None:
        sim_path = os.path.join(full_model_dir, "I_pred.npy")
        a_path = os.path.join(full_model_dir, "U_ob_amp.npy")
        if os.path.exists(sim_path):
            full_sim = np.load(sim_path).astype(np.float32)
            full_metrics = valid_residual_metrics(intensity, full_sim, valid)
            summary["full_model"] = full_metrics
            summary["delta_single_model_minus_full"] = {
                "mae": float(obj_metrics["mae"] - full_metrics["mae"]),
                "rmse": float(obj_metrics["rmse"] - full_metrics["rmse"]),
            }
            lines += [
                f"MAE(full)         = {full_metrics['mae']:.6f}",
                f"RMSE(full)        = {full_metrics['rmse']:.6f}",
                f"Delta MAE         = {obj_metrics['mae'] - full_metrics['mae']:.6f}",
                f"Delta RMSE        = {obj_metrics['rmse'] - full_metrics['rmse']:.6f}",
            ]
        if os.path.exists(a_path):
            full_U_ob_amp = np.load(a_path).astype(np.float32)
            dA = np.abs(results["U_single_amp"] - full_U_ob_amp)
            dA_valid = dA[valid]
            summary["U_single_amp_abs_diff"] = {
                "mean": float(np.mean(dA_valid)),
                "max": float(np.max(dA_valid)),
            }
            lines += [
                f"Mean ||U_single|-|U_ob|| = {np.mean(dA_valid):.6f}",
                f"Max  ||U_single|-|U_ob|| = {np.max(dA_valid):.6f}",
            ]

    txt_path = os.path.join(out_dir, "object_only_metric_summary.txt")
    json_path = os.path.join(out_dir, "object_only_metric_summary.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def draw_results(intensity, results, save_path, full_model_dir=None, display_crop=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid = results["valid_mask"] > 0.5

    raw_vis = raw_crop(intensity, display_crop)
    base_amp_vis = crop_with_valid(results["base_amp"], valid, display_crop, fill_nan=True)
    U_single_amp_vis = crop_with_valid(results["U_single_amp"], valid, display_crop, fill_nan=True)
    U_single_amp_contrast_vis = crop_with_valid(results["U_single_amp_contrast"], valid, display_crop, fill_nan=True)
    phi_single_vis = crop_with_valid(results["phi_single"], valid, display_crop, fill_nan=True)
    support_vis = crop_with_valid(results["support_mask"], valid, display_crop, fill_nan=True)
    I_pred_vis = crop_with_valid(results["I_pred"], valid, display_crop, fill_nan=True)

    full_U_ob_amp_vis = None
    dA_vis = None
    if full_model_dir is not None:
        a_path = os.path.join(full_model_dir, "U_ob_amp.npy")
        if os.path.exists(a_path):
            full_U_ob_amp = np.load(a_path).astype(np.float32)
            full_U_ob_amp_vis = crop_with_valid(full_U_ob_amp, valid, display_crop, fill_nan=True)
            dA_vis = np.abs(U_single_amp_vis - full_U_ob_amp_vis)

    gray_cmap = prepare_cmap("gray", bad_color="white")
    phase_cmap = prepare_cmap("twilight_shifted", bad_color="white")
    aux_cmap = prepare_cmap("cividis", bad_color="white")
    diff_cmap = prepare_cmap("magma", bad_color="white")

    fig, axs = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    rvmin, rvmax = percentile_limits(raw_vis, 1, 99.5)
    im0 = axs[0, 0].imshow(raw_vis, cmap=gray_cmap, vmin=rvmin, vmax=rvmax)
    axs[0, 0].set_title("(a) Raw hologram")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    bmin, bmax = percentile_limits(base_amp_vis, 1, 99.5)
    im1 = axs[0, 1].imshow(base_amp_vis, cmap=gray_cmap, vmin=bmin, vmax=bmax)
    axs[0, 1].set_title("(b) Baseline ASM")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    amin, amax = percentile_limits(U_single_amp_vis, 1, 99.5)
    im2 = axs[0, 2].imshow(U_single_amp_vis, cmap=gray_cmap, vmin=amin, vmax=amax)
    axs[0, 2].set_title(r"(c) Single-model $|U_{single}|$")
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    if full_U_ob_amp_vis is not None:
        fvmin, fvmax = percentile_limits(full_U_ob_amp_vis, 1, 99.5)
        im3 = axs[0, 3].imshow(full_U_ob_amp_vis, cmap=gray_cmap, vmin=fvmin, vmax=fvmax)
        axs[0, 3].set_title(r"(d) Full model $|U_{ob}|$")
        plt.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)
    else:
        cvmin, cvmax = percentile_limits(U_single_amp_contrast_vis, 1, 99.5)
        im3 = axs[0, 3].imshow(U_single_amp_contrast_vis, cmap=aux_cmap, vmin=cvmin, vmax=cvmax)
        axs[0, 3].set_title(r"(d) Single-model contrast")
        plt.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

    if dA_vis is not None:
        dvmin, dvmax = percentile_limits(dA_vis, 1, 99.5)
        im4 = axs[1, 0].imshow(dA_vis, cmap=diff_cmap, vmin=dvmin, vmax=dvmax)
        axs[1, 0].set_title(r"(e) $||U_{single}|-|U_{ob}||$")
        plt.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)
    else:
        cvmin, cvmax = percentile_limits(U_single_amp_contrast_vis, 1, 99.5)
        im4 = axs[1, 0].imshow(U_single_amp_contrast_vis, cmap=aux_cmap, vmin=cvmin, vmax=cvmax)
        axs[1, 0].set_title(r"(e) Single-model contrast")
        plt.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)

    cvmin, cvmax = percentile_limits(U_single_amp_contrast_vis, 1, 99.5)
    im5 = axs[1, 1].imshow(U_single_amp_contrast_vis, cmap=aux_cmap, vmin=cvmin, vmax=cvmax)
    axs[1, 1].set_title(r"(f) Single-model contrast")
    plt.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im6 = axs[1, 2].imshow(phi_single_vis, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    axs[1, 2].set_title("(g) Single-field phase")
    plt.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04)

    im7 = axs[1, 3].imshow(support_vis, cmap=aux_cmap, vmin=0, vmax=1)
    axs[1, 3].set_title("(h) Particle-likelihood map")
    plt.colorbar(im7, ax=axs[1, 3], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save an additional diagnostic figure for checking the predicted hologram; not intended for the manuscript.
    debug_path = os.path.splitext(save_path)[0] + "_debug.png"
    fig2, axs2 = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    imd0 = axs2[0].imshow(U_single_amp_vis, cmap=gray_cmap, vmin=amin, vmax=amax)
    axs2[0].set_title(r"Single-model $|U_{single}|$")
    plt.colorbar(imd0, ax=axs2[0], fraction=0.046, pad=0.04)
    isim_min, isim_max = percentile_limits(I_pred_vis, 1, 99.5)
    imd1 = axs2[1].imshow(I_pred_vis, cmap=gray_cmap, vmin=isim_min, vmax=isim_max)
    axs2[1].set_title(r"Predicted $I_{pred}$")
    plt.colorbar(imd1, ax=axs2[1], fraction=0.046, pad=0.04)
    err = crop_with_valid(results["error_map"], valid, display_crop, fill_nan=True)
    evmin, evmax = percentile_limits(err, 1, 99.5)
    imd2 = axs2[2].imshow(err, cmap=diff_cmap, vmin=evmin, vmax=evmax)
    axs2[2].set_title("Sensor residual")
    plt.colorbar(imd2, ax=axs2[2], fraction=0.046, pad=0.04)
    for ax in axs2.flat:
        ax.axis("off")
    plt.savefig(debug_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)


# =========================================================
# Main entry point
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Single-model ablation: use one complex field without explicit background or fringe components.")
    parser.add_argument("--input", type=str, default="data/sample_01/patch_0006/patch_0006.npy")
    parser.add_argument("--out_dir", type=str, default="outputs/Ours_objectOnly/sample_01/patch_0006")
    parser.add_argument("--png", type=str, default="PNG/ablation_sample_01/patch_0006_object_only.png")
    parser.add_argument("--full_model_dir", type=str, default="outputs/Ours/sample_01/patch_0006")
    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument("--z_distance", type=float, default=0.0245)
    parser.add_argument("--iters", type=int, default=2200)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--border", type=int, default=0)
    parser.add_argument("--display_crop", type=int, default=-1, help="Display-only crop for figure arrays; -1 means using --border.")
    parser.add_argument("--support_percentile", type=float, default=92.8)
    parser.add_argument("--support_boost", type=float, default=1.45)
    parser.add_argument("--support_edge_exclude", type=int, default=12)
    parser.add_argument("--support_dilate_kernel", type=int, default=5)
    args = parser.parse_args()

    if os.path.exists(args.input):
        if args.input.lower().endswith(".npy"):
            intensity = np.load(args.input).astype(np.float32)
        else:
            intensity = np.array(Image.open(args.input).convert("L"), dtype=np.float32)
    else:
        Y, X = np.ogrid[-1:1:args.img_size * 1j, -1:1:args.img_size * 1j]
        intensity = np.exp(-(X ** 2 + Y ** 2)) + np.random.randn(args.img_size, args.img_size) * 0.05

    intensity = intensity[:args.img_size, :args.img_size]
    intensity = normalize_np(intensity)

    results = reconstruct_single_model(
        intensity_target=intensity,
        z=args.z_distance,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        iters=args.iters,
        border=args.border,
        support_percentile=args.support_percentile,
        support_boost=args.support_boost,
        support_edge_exclude=args.support_edge_exclude,
        support_dilate_kernel=args.support_dilate_kernel,
    )

    display_crop = resolve_display_crop(args.display_crop, args.border)
    if display_crop != args.border:
        print(f"[WARN] display_crop={display_crop} differs from border={args.border}; matching them is recommended.")

    save_outputs(results, args.out_dir, display_crop=display_crop)
    full_dir = args.full_model_dir if os.path.isdir(args.full_model_dir) else None
    save_metric_summary(intensity, results, args.out_dir, full_model_dir=full_dir, display_crop=display_crop)
    draw_results(intensity, results, args.png, full_model_dir=full_dir, display_crop=display_crop)
    print("Done.")


if __name__ == "__main__":
    main()
