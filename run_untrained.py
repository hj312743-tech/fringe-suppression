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

# Basic utilities
def normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
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


def smooth1d_circular(arr, k=7):
    pad = k // 2
    ext = np.concatenate([arr[-pad:], arr, arr[:pad]], axis=0)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.convolve(ext, kernel, mode='same')[pad:-pad]
    return out



# Angular-spectrum propagation (ASM)
def propagate_asm(field: torch.Tensor, z: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    """
    field: [B, C, H, W] complex tensor
    z: propagation distance (m)
    """
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



# Loss functions
def make_valid_mask(H: int, W: int, border: int, device: torch.device) -> torch.Tensor:
    """Create a valid-region mask by excluding image borders."""
    m = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
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



# FFT-based dominant fringe-orientation estimation
def estimate_dominant_stripe_angles(intensity, topk=2, min_sep_deg=18):
    img = normalize_np(intensity)
    img = img - np.mean(img)
    H, W = img.shape

    Fmag = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    y = yy - cy
    x = xx - cx
    r = np.sqrt(x ** 2 + y ** 2)
    r_norm = r / (min(H, W) / 2.0 + 1e-8)

    mask = (r_norm > 0.08) & (r_norm < 0.45)
    angle = (np.rad2deg(np.arctan2(y, x)) + 180.0) % 180.0

    hist = np.zeros(180, dtype=np.float64)
    for ang in range(180):
        sel = mask & (np.abs(angle - ang) < 0.5)
        if np.any(sel):
            hist[ang] = np.sum(Fmag[sel])
    hist = smooth1d_circular(hist.astype(np.float32), k=9)

    chosen = []
    tmp = hist.copy()
    for _ in range(topk):
        idx = int(np.argmax(tmp))
        chosen.append(float(idx))
        for j in range(180):
            d = min(abs(j - idx), 180 - abs(j - idx))
            if d < min_sep_deg:
                tmp[j] = -1.0

    if len(chosen) == 0:
        chosen = [45.0]
    return chosen


# Network modules
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


class ObjectResUNet(nn.Module):
    """
    Output:
    - A_obj: object amplitude, close to 1 in the background and lower in object regions.
    - phi_obj: object phase.
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


class SmoothBackgroundModel(nn.Module):
    """Smooth-background field parameterized by low-order polynomial basis functions."""
    def __init__(self, H, W):
        super().__init__()
        y = torch.linspace(-1, 1, H)
        x  = torch.linspace(-1, 1, W)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        R2 = X ** 2 + Y ** 2

        basis_list = [
            torch.ones_like(X),
            X, Y,
            X ** 2, Y ** 2, X * Y,
            X ** 3, Y ** 3,
            X * R2, Y * R2,
            R2, (X ** 2 - Y ** 2),
            X ** 4, Y ** 4, (X ** 2) * (Y ** 2),
            X * Y * R2, R2 ** 2
        ]
        self.num_modes = len(basis_list)
        basis = torch.stack(basis_list, dim=0).unsqueeze(0)
        self.register_buffer('basis', basis)

        self.amp_coeffs = nn.Parameter(torch.zeros(1, self.num_modes, 1, 1))
        self.amp_coeffs.data[0, 0, 0, 0] = 0.8
        self.phase_coeffs = nn.Parameter(torch.zeros(1, self.num_modes, 1, 1))

    def forward(self):
        A_bg = torch.sum(self.amp_coeffs * self.basis, dim=1, keepdim=True)
        phi_bg = torch.sum(self.phase_coeffs * self.basis, dim=1, keepdim=True)

        A_bg = F.softplus(A_bg) + 1e-4
        U_bg = A_bg * torch.exp(1j * phi_bg)
        return U_bg, A_bg, phi_bg


class ParasiticFringeModel(nn.Module):
    """Zero-baseline directional parasitic-fringe field parameterization."""
    def __init__(self, H, W, angles_deg, freq_list=(5.0, 9.0, 13.0), amp_mod_max=0.45, phase_mod_max=0.70):
        super().__init__()
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('X', X.unsqueeze(0).unsqueeze(0))
        self.register_buffer('Y', Y.unsqueeze(0).unsqueeze(0))

        env_basis = [
            torch.ones_like(X),
            X, Y,
            X ** 2, Y ** 2, X * Y,
            (X ** 2 + Y ** 2)
        ]
        env_basis = torch.stack(env_basis, dim=0).unsqueeze(0)
        self.register_buffer('env_basis', env_basis)
        self.num_env = env_basis.shape[1]

        carrier_angles = []
        carrier_freqs = []
        for ang in angles_deg:
            for f in freq_list:
                carrier_angles.append(float(ang))
                carrier_freqs.append(float(f))
        self.num_carriers = len(carrier_angles)

        theta = torch.tensor(np.deg2rad(carrier_angles), dtype=torch.float32).view(1, self.num_carriers, 1, 1)
        freqs = torch.tensor(carrier_freqs, dtype=torch.float32).view(1, self.num_carriers, 1, 1)
        self.register_buffer('theta', theta)
        self.register_buffer('freqs', freqs)

        self.carrier_phase = nn.Parameter(torch.zeros(1, self.num_carriers, 1, 1))
        self.amp_env_coeffs = nn.Parameter(torch.zeros(1, self.num_carriers, self.num_env, 1, 1))
        self.phase_env_coeffs = nn.Parameter(torch.zeros(1, self.num_carriers, self.num_env, 1, 1))

        self.amp_env_coeffs.data[:, :, 0, :, :] = 0.02
        self.phase_env_coeffs.data[:, :, 0, :, :] = 0.01

        self.amp_mod_max = amp_mod_max
        self.phase_mod_max = phase_mod_max

    def forward(self, gain=1.0):
        proj = torch.cos(self.theta) * self.X + torch.sin(self.theta) * self.Y
        amp_env = torch.sum(self.amp_env_coeffs * self.env_basis.unsqueeze(1), dim=2)
        phase_env = torch.sum(self.phase_env_coeffs * self.env_basis.unsqueeze(1), dim=2)

        carrier_arg = 2.0 * torch.pi * self.freqs * proj + self.carrier_phase
        carrier = torch.cos(carrier_arg)
        carrier_q = torch.sin(carrier_arg)

        stripe_amp_raw = torch.sum(amp_env * carrier, dim=1, keepdim=True)
        Phi_par_raw = torch.sum(phase_env * carrier_q, dim=1, keepdim=True)

        C_par = gain * self.amp_mod_max * torch.tanh(stripe_amp_raw)
        Phi_par = gain * self.phase_mod_max * torch.tanh(Phi_par_raw)

        parasitic_coeff = torch.expm1(C_par)
        U_par = parasitic_coeff * torch.exp(1j * Phi_par)
        return U_par, C_par, Phi_par


class PhysicsDrivenDecompositionModel(nn.Module):
    def __init__(self, H, W, stripe_angles_deg):
        super().__init__()
        self.obj_net = ObjectResUNet()
        self.smooth_background = SmoothBackgroundModel(H, W)
        self.parasitic_fringe = ParasiticFringeModel(H, W, stripe_angles_deg)

    def forward(self, noise, z, wavelength, pixel_size, stripe_gain=1.0):
        A_obj, phi_obj = self.obj_net(noise)
        U_bg, A_bg, phi_bg = self.smooth_background()
        U_par, C_par, Phi_par = self.parasitic_fringe(gain=stripe_gain)

        U_obj = A_obj * torch.exp(1j * phi_obj)

        # Product field: smooth background multiplied by sample-induced modulation.
        U_ob = U_bg * U_obj

        # Sensor plane: propagate the product field and parasitic field, then add them coherently.
        U_ob_sensor = propagate_asm(U_ob, z, wavelength, pixel_size)
        U_par_sensor = propagate_asm(U_par, z, wavelength, pixel_size)
        U_sensor = U_ob_sensor + U_par_sensor
        I_pred = torch.abs(U_sensor) ** 2

        A_ob = torch.abs(U_ob)

        return {
            "I_pred": I_pred,
            "A_obj": A_obj,
            "phi_obj": phi_obj,
            "U_bg": U_bg,
            "A_bg": A_bg,
            "phi_bg": phi_bg,
            "U_par": U_par,
            "U_ob": U_ob,
            "U_ob_amp": A_ob,
            "U_ob_sensor": U_ob_sensor,
            "U_par_sensor": U_par_sensor,
            "C_par": C_par,
            "Phi_par": Phi_par,
        }


# Particle-likelihood map
# - Excludes image borders when estimating the threshold.
# - Uses mild dilation to avoid over-expanding the object prior.
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

    # Mild dilation avoids excessive object bridging and boundary pulling.
    if dilate_kernel > 1:
        pad = dilate_kernel // 2
        mask_t = F.max_pool2d(mask_t, kernel_size=dilate_kernel, stride=1, padding=pad)

    mask_t = torch.clamp(mask_t, 0.0, 1.0)
    return mask_t, hp_np


# Reconstruction loop
def reconstruct(
    I_raw_np,
    z,
    wavelength,
    pixel_size,
    iters=2200,
    border=6,
    support_percentile=93.5,
    support_boost=1.35,
    support_edge_exclude=14,
    support_dilate_kernel=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = I_raw_np.shape
    I_raw = torch.tensor(I_raw_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    valid_mask = make_valid_mask(H, W, border=border, device=device)

    # ---------- baseline ----------
    U_raw = torch.sqrt(torch.clamp(I_raw, min=0.0)) + 0j
    U_baseline = propagate_asm(U_raw, -z, wavelength, pixel_size)
    baseline_amp = torch.abs(U_baseline).detach().cpu().squeeze().numpy()
    baseline_phase = torch.angle(U_baseline).detach().cpu().squeeze().numpy()

    # ---------- support ----------
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

    # ---------- Fringe-orientation estimation ----------
    stripe_angles_deg = estimate_dominant_stripe_angles(I_raw_np, topk=2, min_sep_deg=18)
    print(f"Estimated carrier orientation angles: {stripe_angles_deg}")

    # ---------- Model initialization ----------
    torch.manual_seed(42)
    fixed_noise = torch.rand((1, 1, H, W), device=device)
    net = PhysicsDrivenDecompositionModel(H, W, stripe_angles_deg).to(device)

    optimizer = optim.Adam([
        {"params": net.obj_net.parameters(), "lr": 7e-4},
        {"params": net.smooth_background.parameters(), "lr": 3.5e-3},
        {"params": net.parasitic_fringe.parameters(), "lr": 2.0e-3},
    ])

    # ---------- Loss weights ----------
    w_tv_amp = 0.015
    w_tv_phase = 0.012

    # Outside the soft support, keep the object modulation close to transparent.
    w_support_out_amp = 0.18
    w_support_out_phase = 0.03

    # Inside the soft support, allow the object branch to carry object modulation.
    w_sparse_inside = 0.008
    w_phase_inside = 0.000
    w_amp_inside_pull = 0.010

    # Smooth-background regularization
    w_A_bg_reg = 0.003
    w_phi_bg_reg = 0.003

    # Prevent the smooth background from absorbing object-region structures.
    w_bg_support_amp = 0.016
    w_bg_support_phase = 0.005

    # Parasitic-fringe regularization
    w_stripe_inside_amp = 0.050
    w_stripe_inside_phase = 0.024
    w_stripe_l1_amp = 0.015
    w_stripe_l1_phase = 0.008
    w_stripe_tv_amp = 0.006
    w_stripe_tv_phase = 0.003

    stripe_start_iter = 160
    stripe_ramp_iters = 400

    print("Start the iterative reconstruction process...")
    for i in range(iters):
        optimizer.zero_grad()

        if i < stripe_start_iter:
            stripe_gain = 0.0
        else:
            stripe_gain = min(1.0, (i - stripe_start_iter) / float(stripe_ramp_iters))

        out = net(
            fixed_noise,
            z,
            wavelength,
            pixel_size,
            stripe_gain=stripe_gain,
        )

        I_pred = out["I_pred"]
        A_obj = out["A_obj"]
        phi_obj = out["phi_obj"]
        A_bg = out["A_bg"]
        phi_bg = out["phi_bg"]
        C_par = out["C_par"]
        Phi_par = out["Phi_par"]
        A_ob = out["U_ob_amp"]

        # 1) Hologram-domain physics consistency.
        loss_data = multi_scale_masked_mse(I_pred, I_raw, valid_mask)

        # 2) Outside the soft support, keep amplitude near 1 and phase near 0.
        loss_support = (
            masked_l1((1.0 - A_obj), outside_mask) * w_support_out_amp +
            masked_l1(phi_obj, outside_mask) * w_support_out_phase
        )

        # 3) Inside the soft support, encourage amplitude modulation.
        loss_sparse_inside = masked_l1((1.0 - A_obj), support_mask) * w_sparse_inside

        # 4) Mild phase regularization inside the soft support.
        loss_phase_inside = masked_l1(phi_obj, support_mask) * w_phase_inside

        # 5) Weak amplitude pull inside the soft support.
        desired_contrast = 0.10 * support_mask
        loss_amp_inside_pull = (
            masked_l1(F.relu(desired_contrast - (1.0 - A_obj)), support_mask) * w_amp_inside_pull
        )

        # 6) Object-branch TV, stronger in background regions and weaker in object regions.
        loss_tv = (
            masked_tv_loss(A_obj, weight_map=tv_weight_map, valid_mask=valid_mask) * w_tv_amp +
            masked_tv_loss(phi_obj, weight_map=tv_weight_map, valid_mask=valid_mask) * w_tv_phase
        )

        # 7) Global smooth-background regularization.
        bg_mean = torch.sum(A_bg * valid_mask) / (torch.sum(valid_mask) + 1e-8)
        loss_bg_reg = (
            torch.sum(((A_bg - bg_mean) ** 2) * valid_mask) / (torch.sum(valid_mask) + 1e-8) * w_A_bg_reg +
            masked_tv_loss(phi_bg, valid_mask=valid_mask) * w_phi_bg_reg
        )

        # 8) Discourage the smooth background from absorbing low-frequency object shape.
        loss_bg_in_support = (
            masked_l1(A_bg - bg_mean, support_mask) * w_bg_support_amp +
            masked_l1(phi_bg, support_mask) * w_bg_support_phase
        )

        # 9) Parasitic-fringe regularization.
        loss_stripe_reg = (
            masked_l1(C_par, support_mask) * w_stripe_inside_amp +
            masked_l1(Phi_par, support_mask) * w_stripe_inside_phase +
            masked_l1(C_par, valid_mask) * w_stripe_l1_amp +
            masked_l1(Phi_par, valid_mask) * w_stripe_l1_phase +
            masked_tv_loss(C_par, valid_mask=valid_mask) * w_stripe_tv_amp +
            masked_tv_loss(Phi_par, valid_mask=valid_mask) * w_stripe_tv_phase
        )

        loss = (
            loss_data +
            loss_support +
            loss_sparse_inside +
            loss_phase_inside +
            loss_amp_inside_pull +
            loss_tv +
            loss_bg_reg +
            loss_bg_in_support +
            loss_stripe_reg
        )
        loss.backward()
        optimizer.step()

        if (i + 1) % 300 == 0:
            print(
                f"Iter {i + 1:04d} | "
                f"Data: {loss_data.item():.5f} | "
                f"Support: {loss_support.item():.5f} | "
                f"BG-in-support: {loss_bg_in_support.item():.5f} | "
                f"Stripe: {loss_stripe_reg.item():.5f} | "
                f"gain={stripe_gain:.2f}"
            )

    # ---------- Outputs ----------
    A_obj_np = A_obj.detach().cpu().squeeze().numpy()
    phi_obj_np = phi_obj.detach().cpu().squeeze().numpy()
    A_bg_np = A_bg.detach().cpu().squeeze().numpy()
    phi_bg_np = phi_bg.detach().cpu().squeeze().numpy()
    C_par_np = C_par.detach().cpu().squeeze().numpy()
    Phi_par_np = Phi_par.detach().cpu().squeeze().numpy()
    A_ob_np = A_ob.detach().cpu().squeeze().numpy()
    I_pred_np = I_pred.detach().cpu().squeeze().numpy()
    error_map = np.abs(I_pred_np - I_raw_np)
    support_np = support_mask.detach().cpu().squeeze().numpy()
    valid_np = valid_mask.detach().cpu().squeeze().numpy()

    A_ob_contrast = 1.0 - normalize_np(A_ob_np)
    A_obj_contrast = 1.0 - A_obj_np
    U_par_strength = np.abs(np.expm1(C_par_np))
    A_bg_display = A_bg_np.copy()

    return {
        "base_amp": baseline_amp,
        "base_phase": baseline_phase,
        "U_ob_amp": A_ob_np,
        "table1_img": A_ob_np,
        "U_ob_amp_contrast": A_ob_contrast,
        "A_obj": A_obj_np,
        "phi_obj": phi_obj_np,
        "A_obj_contrast": A_obj_contrast,
        "A_bg": A_bg_np,
        "phi_bg": phi_bg_np,
        "C_par": C_par_np,
        "Phi_par": Phi_par_np,
        "U_par_strength": U_par_strength,
        "A_bg_display": A_bg_display,
        "I_pred": I_pred_np,
        "error_map": error_map,
        "support_mask": support_np,
        "valid_mask": valid_np,
        "support_seed": support_seed,
        "stripe_angles_deg": stripe_angles_deg,
        "border": border,
    }

# Output saving
def save_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # table1_img.npy is kept as a compatibility output for the evaluation script.
    np.save(os.path.join(out_dir, "table1_img.npy"), results["table1_img"].astype(np.float32))
    np.save(os.path.join(out_dir, "U_ob_amp.npy"), results["U_ob_amp"].astype(np.float32))
    np.save(os.path.join(out_dir, "A_ob_contrast.npy"), results["U_ob_amp_contrast"].astype(np.float32))
    np.save(os.path.join(out_dir, "A_obj.npy"), results["A_obj"].astype(np.float32))
    np.save(os.path.join(out_dir, "phi_obj.npy"), results["phi_obj"].astype(np.float32))
    np.save(os.path.join(out_dir, "A_bg.npy"), results["A_bg"].astype(np.float32))
    np.save(os.path.join(out_dir, "phi_bg.npy"), results["phi_bg"].astype(np.float32))
    np.save(os.path.join(out_dir, "I_pred.npy"), results["I_pred"].astype(np.float32))
    np.save(os.path.join(out_dir, "support_mask.npy"), results["support_mask"].astype(np.float32))
    print(f"Outputs saved to: {out_dir}")


# Visualization
def draw_results(intensity, results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    border = int(results.get("border", 0))

    def crop_border(arr):
        x = np.asarray(arr, dtype=np.float32)
        if border <= 0:
            return x
        h, w = x.shape[:2]
        if (2 * border) >= h or (2 * border) >= w:
            raise ValueError(f"border={border} is too large for cropping shape={x.shape}")
        return x[border:h-border, border:w-border]

    # Crop all panels by the same border to exclude FFT boundary artifacts.
    intensity_vis = crop_border(intensity)
    ipred_vis = crop_border(results["I_pred"])
    error_map_vis = crop_border(results["error_map"])
    U_par_strength = crop_border(results["U_par_strength"])
    base_amp_vis = crop_border(results["base_amp"])
    base_phase_vis = crop_border(results["base_phase"])
    a_ob_vis = crop_border(results["U_ob_amp"])
    a_ob_contrast_vis = crop_border(results["U_ob_amp_contrast"])
    phi_obj_vis = crop_border(results["phi_obj"])
    A_bg_display_vis = crop_border(results["A_bg_display"])
    support_vis = crop_border(results["support_mask"])
    A_bg_sq_vis = crop_border(results["A_bg"] ** 2)

    fig, axs = plt.subplots(2, 6, figsize=(28, 9), constrained_layout=True)

    # Use a fixed aspect ratio for all panels.
    imshow_kwargs = dict(interpolation="nearest", aspect="equal")

    ivmin, ivmax = percentile_limits(intensity_vis, 1, 99)
    im0 = axs[0, 0].imshow(intensity_vis, cmap="gray", vmin=ivmin, vmax=ivmax, **imshow_kwargs)
    axs[0, 0].set_title("(a) Raw Hologram")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    svmin, svmax = percentile_limits(ipred_vis, 1, 99)
    im1 = axs[0, 1].imshow(ipred_vis, cmap="gray", vmin=svmin, vmax=svmax, **imshow_kwargs)
    axs[0, 1].set_title(r"(b) Predicted $I_{pred}$")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    evmin, evmax = percentile_limits(error_map_vis, 1, 99)
    im2 = axs[0, 2].imshow(error_map_vis, cmap="magma", vmin=evmin, vmax=evmax, **imshow_kwargs)
    axs[0, 2].set_title("(c) Error Map")
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    bvmin, bvmax = percentile_limits(A_bg_sq_vis, 1, 99)
    im3 = axs[0, 3].imshow(A_bg_sq_vis, cmap="viridis", vmin=bvmin, vmax=bvmax, **imshow_kwargs)
    axs[0, 3].set_title(r"(d) Smooth Background $A_{bg}^2$")
    plt.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

    pvmin, pvmax = percentile_limits(U_par_strength, 1, 99)
    im4 = axs[0, 4].imshow(U_par_strength, cmap="cividis", vmin=pvmin, vmax=pvmax, **imshow_kwargs)
    axs[0, 4].set_title("(e) Parasitic Field Strength")
    plt.colorbar(im4, ax=axs[0, 4], fraction=0.046, pad=0.04)

    im5 = axs[0, 5].imshow(support_vis, cmap="cividis", vmin=0, vmax=1, **imshow_kwargs)
    axs[0, 5].set_title("(f) Soft Particle-likelihood Map")
    plt.colorbar(im5, ax=axs[0, 5], fraction=0.046, pad=0.04)

    bmin, bmax = percentile_limits(base_amp_vis, 1, 99)
    im6 = axs[1, 0].imshow(base_amp_vis, cmap="gray", vmin=bmin, vmax=bmax, **imshow_kwargs)
    axs[1, 0].set_title("(g) Baseline ASM (Amp)")
    plt.colorbar(im6, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im7 = axs[1, 1].imshow(base_phase_vis, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, **imshow_kwargs)
    axs[1, 1].set_title("(h) Baseline ASM (Phase)")
    plt.colorbar(im7, ax=axs[1, 1], fraction=0.046, pad=0.04)

    amin, amax = percentile_limits(a_ob_vis, 1, 99.5)
    im8 = axs[1, 2].imshow(a_ob_vis, cmap="gray", vmin=amin, vmax=amax, **imshow_kwargs)
    axs[1, 2].set_title(r"(i) Product-field Amp. $|U_{ob}|$")
    plt.colorbar(im8, ax=axs[1, 2], fraction=0.046, pad=0.04)

    cmin, cmax = percentile_limits(a_ob_contrast_vis, 1, 99.5)
    im9 = axs[1, 3].imshow(a_ob_contrast_vis, cmap="cividis", vmin=cmin, vmax=cmax, **imshow_kwargs)
    axs[1, 3].set_title(r"(j) $|U_{ob}|$ Contrast View")
    plt.colorbar(im9, ax=axs[1, 3], fraction=0.046, pad=0.04)

    im10 = axs[1, 4].imshow(phi_obj_vis, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, **imshow_kwargs)
    axs[1, 4].set_title(r"(k) Object Phase $\phi_{obj}$")
    plt.colorbar(im10, ax=axs[1, 4], fraction=0.046, pad=0.04)

    tvmin, tvmax = percentile_limits(A_bg_display_vis, 1, 99)
    im11 = axs[1, 5].imshow(A_bg_display_vis, cmap="viridis", vmin=tvmin, vmax=tvmax, **imshow_kwargs)
    axs[1, 5].set_title(r"(l) Smooth Background $A_{bg}$")
    plt.colorbar(im11, ax=axs[1, 5], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.axis("off")
        ax.set_box_aspect(1)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Physics-driven self-supervised parasitic-fringe suppression in lensless holographic reconstruction.")
    parser.add_argument("--input", type=str, default="data/sample_01/patch_0006/patch_0006.npy")
    parser.add_argument("--out_dir", type=str, default="outputs/Ours/sample_01/patch_0006")
    parser.add_argument("--png", type=str, default="PNG/reconstruction_sample_01/patch_0006.png")
    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument("--z_distance", type=float, default=0.0245)
    parser.add_argument("--iters", type=int, default=2200)
    parser.add_argument("--img_size", type=int, default=256)

    # Default parameters used in the manuscript
    parser.add_argument("--border", type=int, default=16)
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

    results = reconstruct(
        I_raw_np=intensity,
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

    save_outputs(results, args.out_dir)
    draw_results(intensity, results, args.png)
    print("Done.")


if __name__ == "__main__":
    main()
