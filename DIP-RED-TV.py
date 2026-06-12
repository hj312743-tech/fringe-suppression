import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.optim as optim
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'figure.autolayout': True
})


# ==========================================
# Basic utilities
# ==========================================
def normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x: np.ndarray, low=1.0, high=99.0):
    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


# ==========================================
# Angular-spectrum propagation (ASM)
# ==========================================
def propagate_asm(field: torch.Tensor, z: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    _, _, H, W = field.shape
    fx = fft.fftfreq(W, d=pixel_size)
    fy = fft.fftfreq(H, d=pixel_size)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    FX, FY = FX.to(field.device), FY.to(field.device)

    k = 2 * torch.pi / wavelength
    term = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    term = torch.clamp(term, min=0)
    phase_shift = k * z * torch.sqrt(term)

    H_transfer = torch.exp(1j * phase_shift)
    field_prop = fft.ifft2(fft.fft2(field) * H_transfer)
    return field_prop


# ==========================================
# Valid-region mask and physics consistency
# ==========================================
def make_valid_mask(H: int, W: int, border: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    m[:, :, border:H-border, border:W-border] = 1.0
    return m


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(((x - y) ** 2) * mask) / (torch.sum(mask) + eps)


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


# ==========================================
# Network: same backbone as U-Net-PC for a fair comparison
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ObjectResUNet(nn.Module):
    """
    Outputs:
    - obj_amp   : transmission amplitude in (0, 1)
    - obj_phase : phase in (-pi, pi)

    Only the DIP backbone is retained; no explicit background branch is introduced.
    The DIP-RED-TV improvement comes from the explicit TV denoiser with ADMM/variable splitting.
    """
    def __init__(self, in_channels: int = 1, base_f: int = 24):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_f)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_f, base_f * 2)
        self.bottleneck = ResidualBlock(base_f * 2, base_f * 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(base_f * 3, base_f)
        self.out_conv = nn.Conv2d(base_f, 2, 3, padding=1)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(e2)
        d1 = self.dec1(torch.cat([self.up(b), e1], dim=1))
        out = self.out_conv(d1)

        amp = torch.sigmoid(out[:, 0:1, :, :])
        phase = torch.tanh(out[:, 1:2, :, :]) * torch.pi
        return amp, phase


class DIPREDTVModel(nn.Module):
    def __init__(self, in_channels: int = 1, base_f: int = 24):
        super().__init__()
        self.obj_net = ObjectResUNet(in_channels=in_channels, base_f=base_f)

    def forward(self, noise: torch.Tensor, z: float, wavelength: float, pixel_size: float):
        obj_amp, obj_phase = self.obj_net(noise)
        U_obj = obj_amp * torch.exp(1j * obj_phase)
        U_sensor = propagate_asm(U_obj, z, wavelength, pixel_size)
        I_pred = torch.abs(U_sensor) ** 2
        return {
            'I_pred': I_pred,
            'obj_amp': obj_amp,
            'obj_phase': obj_phase,
        }


# ==========================================
# TV denoiser based on the Chambolle projection algorithm
# Used here as an explicit denoiser f(.).
# ==========================================
def tv_denoise_chambolle_2d(img: np.ndarray, weight: float = 0.08, n_iter_max: int = 60) -> np.ndarray:
    """
    A lightweight 2D TV denoiser without additional dependencies.
    The formulation is close to ROF denoising / Chambolle projection.

    Parameters:
        img       : 2D float32 array
        weight    : TV strength; larger values give stronger smoothing
        n_iter_max: number of iterations
    """
    img = img.astype(np.float32, copy=False)
    px = np.zeros_like(img, dtype=np.float32)
    py = np.zeros_like(img, dtype=np.float32)
    tau = 0.25

    for _ in range(n_iter_max):
        div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = img - weight * div_p

        grad_u_x = np.roll(u, -1, axis=1) - u
        grad_u_y = np.roll(u, -1, axis=0) - u

        px_new = px + (tau / weight) * grad_u_x
        py_new = py + (tau / weight) * grad_u_y
        norm_new = np.maximum(1.0, np.sqrt(px_new * px_new + py_new * py_new))

        px = px_new / norm_new
        py = py_new / norm_new

    div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
    u = img - weight * div_p
    return u.astype(np.float32)


# ==========================================
# DIP-RED-TV reconstruction
# ==========================================
def reconstruct_dip_red_tv(
    intensity_target: np.ndarray,
    z: float,
    wavelength: float,
    pixel_size: float,
    iters: int = 2200,
    border: int = 6,
    seed: int = 42,
    base_f: int = 24,
    lr: float = 7e-4,
    beta_amp: float = 0.09,
    beta_phase: float = 0.06,
    denoise_interval: int = 100,
    tv_weight_amp: float = 0.080,
    tv_weight_phase: float = 0.045,
    tv_iters_amp: int = 60,
    tv_iters_phase: int = 45,
):
    """
    DIP-RED-TV runnable implementation.

    Design notes:
    1) The main network is still an untrained DIP/U-Net backbone;
    2) the physics-consistency term fits the input hologram;
    3) variable splitting introduces an external explicit TV denoiser;
    4) the network output x is coupled to the TV-denoised auxiliary variable v through a proximity term;
    5) the denoiser is called every denoise_interval iterations to reduce computational cost.

    This follows the core idea of DIP-RED:
    - ADMM/variable splitting separates data fidelity from the explicit denoiser;
    - the denoiser does not require explicit backpropagation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = intensity_target.shape

    target_I = torch.tensor(intensity_target, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    valid_mask = make_valid_mask(H, W, border=border, device=device)

    # Baseline ASM is used only for visualization and comparison
    U_raw = torch.sqrt(torch.clamp(target_I, min=0.0)) + 0j
    U_baseline = propagate_asm(U_raw, -z, wavelength, pixel_size)
    baseline_amp = torch.abs(U_baseline).detach().cpu().squeeze().numpy()
    baseline_phase = torch.angle(U_baseline).detach().cpu().squeeze().numpy()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    fixed_noise = torch.rand((1, 1, H, W), device=device)
    net = DIPREDTVModel(in_channels=1, base_f=base_f).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # ADMM / variable-splitting auxiliary and dual variables
    # External TV denoising is applied separately to amplitude and phase.
    v_amp = np.ones((H, W), dtype=np.float32) * 0.95
    v_phase = np.zeros((H, W), dtype=np.float32)
    u_amp = np.zeros((H, W), dtype=np.float32)
    u_phase = np.zeros((H, W), dtype=np.float32)

    loss_phy_history = []
    loss_prox_history = []
    denoise_step_marks = []

    print('Start DIP-RED-TV iterative reconstruction...')
    for i in range(iters):
        optimizer.zero_grad()

        out = net(fixed_noise, z, wavelength, pixel_size)
        I_pred = out['I_pred']
        obj_amp = out['obj_amp']
        obj_phase = out['obj_phase']

        loss_phy = multi_scale_masked_mse(I_pred, target_I, valid_mask)

        # Proximity term: the network output is encouraged to stay close to the externally denoised auxiliary variable with dual correction.
        ref_amp = torch.tensor(v_amp - u_amp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        ref_phase = torch.tensor(v_phase - u_phase, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        loss_prox_amp = torch.mean((obj_amp - ref_amp) ** 2)
        loss_prox_phase = torch.mean((obj_phase - ref_phase) ** 2)
        loss_prox = beta_amp * loss_prox_amp + beta_phase * loss_prox_phase

        loss = loss_phy + loss_prox
        loss.backward()
        optimizer.step()

        loss_phy_history.append(float(loss_phy.item()))
        loss_prox_history.append(float(loss_prox.item()))

        # Call the explicit TV denoiser every few iterations
        if (i + 1) % denoise_interval == 0 or i == 0:
            amp_np = obj_amp.detach().cpu().squeeze().numpy().astype(np.float32)
            phase_np = obj_phase.detach().cpu().squeeze().numpy().astype(np.float32)

            z_amp = amp_np + u_amp
            z_phase = phase_np + u_phase

            v_amp = tv_denoise_chambolle_2d(z_amp, weight=tv_weight_amp, n_iter_max=tv_iters_amp)
            v_phase = tv_denoise_chambolle_2d(z_phase, weight=tv_weight_phase, n_iter_max=tv_iters_phase)

            v_amp = np.clip(v_amp, 0.0, 1.0)
            v_phase = np.clip(v_phase, -np.pi, np.pi)

            # dual update
            u_amp = u_amp + amp_np - v_amp
            u_phase = u_phase + phase_np - v_phase

            denoise_step_marks.append(i + 1)

        if (i + 1) % 300 == 0:
            print(
                f'Iter {i + 1:04d} | '
                f'Physics: {loss_phy.item():.6f} | '
                f'Prox: {loss_prox.item():.6f} | '
                f'Denoiser every {denoise_interval} iters'
            )

    # Run one final forward pass to ensure the latest network output is saved
    with torch.no_grad():
        out = net(fixed_noise, z, wavelength, pixel_size)
        I_pred = out['I_pred']
        obj_amp = out['obj_amp']
        obj_phase = out['obj_phase']

    I_pred_np = I_pred.detach().cpu().squeeze().numpy()
    obj_amp_np = obj_amp.detach().cpu().squeeze().numpy()
    obj_phase_np = obj_phase.detach().cpu().squeeze().numpy()
    valid_np = valid_mask.detach().cpu().squeeze().numpy()

    error_map = np.abs(I_pred_np - intensity_target)
    obj_contrast = 1.0 - obj_amp_np

    # Export auxiliary denoiser variables for analyzing the effect of DIP-RED
    denoised_amp = v_amp.copy()
    denoised_phase = v_phase.copy()
    denoised_contrast = 1.0 - denoised_amp

    return {
        'base_amp': baseline_amp,
        'base_phase': baseline_phase,
        'obj_amp': obj_amp_np,
        'obj_phase': obj_phase_np,
        'obj_contrast': obj_contrast,
        'I_pred': I_pred_np,
        'error_map': error_map,
        'valid_mask': valid_np,
        'loss_phy_history': np.array(loss_phy_history, dtype=np.float32),
        'loss_prox_history': np.array(loss_prox_history, dtype=np.float32),
        'denoise_step_marks': np.array(denoise_step_marks, dtype=np.int32),
        'denoised_amp': denoised_amp.astype(np.float32),
        'denoised_phase': denoised_phase.astype(np.float32),
        'denoised_contrast': denoised_contrast.astype(np.float32),
    }


# ==========================================
# Visualization
# ==========================================
def draw_results_dip_red_tv(
    intensity: np.ndarray,
    results: dict,
    save_path: str = 'PNG/dip_red_tv_canvas.png',
    out_dir: str = 'PNG',
    prefix: str = 'sample_sim_001_patch_0001_dipredtv',
    save_single: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 5, figsize=(26, 10))
    valid_mask = results['valid_mask']

    obj_amp_vis = results['obj_amp'].copy()
    obj_amp_vis[valid_mask < 0.5] = 1.0

    obj_contrast_vis = results['obj_contrast'].copy()
    obj_contrast_vis[valid_mask < 0.5] = 0.0

    obj_phase_vis = results['obj_phase'].copy()
    obj_phase_vis[valid_mask < 0.5] = 0.0

    den_amp_vis = results['denoised_amp'].copy()
    den_amp_vis[valid_mask < 0.5] = 1.0

    den_contrast_vis = results['denoised_contrast'].copy()
    den_contrast_vis[valid_mask < 0.5] = 0.0

    error_map_vis = results['error_map'].copy()
    error_map_vis[valid_mask < 0.5] = 0.0

    def save_single_panel(img, filename, cmap='gray', vmin=None, vmax=None, dpi=300):
        fig_single = plt.figure(figsize=(6, 6))
        ax = fig_single.add_subplot(111)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig_single.savefig(
            os.path.join(out_dir, filename),
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.02
        )
        plt.close(fig_single)

    # First row
    im0 = axs[0, 0].imshow(intensity, cmap='gray')
    axs[0, 0].set_title('(a) Raw Hologram')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(results['I_pred'], cmap='gray')
    axs[0, 1].set_title('(b) Predicted hologram $I_{pred}$')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    evmin, evmax = percentile_limits(error_map_vis, 1, 99)
    im2 = axs[0, 2].imshow(error_map_vis, cmap='magma', vmin=evmin, vmax=evmax)
    axs[0, 2].set_title('(c) Error Map')
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    axs[0, 3].plot(results['loss_phy_history'], label='Physics', linewidth=1.5)
    axs[0, 3].plot(results['loss_prox_history'], label='Proximity', linewidth=1.2)
    for mark in results['denoise_step_marks']:
        axs[0, 3].axvline(mark, color='gray', alpha=0.12, linewidth=0.8)
    axs[0, 3].set_title('(d) Loss Curves')
    axs[0, 3].set_xlabel('Iteration')
    axs[0, 3].set_ylabel('Loss')
    axs[0, 3].legend()
    axs[0, 3].grid(alpha=0.3)

    dmin, dmax = percentile_limits(den_amp_vis, 1, 99.5)
    im4 = axs[0, 4].imshow(den_amp_vis, cmap='gray', vmin=dmin, vmax=dmax)
    axs[0, 4].set_title('(e) TV Auxiliary Amp')
    plt.colorbar(im4, ax=axs[0, 4], fraction=0.046, pad=0.04)

    # Second row
    bmin, bmax = percentile_limits(results['base_amp'], 1, 99)
    im5 = axs[1, 0].imshow(results['base_amp'], cmap='gray', vmin=bmin, vmax=bmax)
    axs[1, 0].set_title('(f) Baseline ASM (Amp)')
    plt.colorbar(im5, ax=axs[1, 0], fraction=0.046, pad=0.04)

    amin, amax = percentile_limits(obj_amp_vis, 1, 99.5)
    im6 = axs[1, 1].imshow(obj_amp_vis, cmap='gray', vmin=amin, vmax=amax)
    axs[1, 1].set_title('(g) DIP-RED-TV Amp')
    plt.colorbar(im6, ax=axs[1, 1], fraction=0.046, pad=0.04)

    cmin, cmax = percentile_limits(obj_contrast_vis, 1, 99.5)
    im7 = axs[1, 2].imshow(obj_contrast_vis, cmap='cividis', vmin=cmin, vmax=cmax)
    axs[1, 2].set_title('(h) DIP-RED-TV Contrast')
    plt.colorbar(im7, ax=axs[1, 2], fraction=0.046, pad=0.04)

    im8 = axs[1, 3].imshow(obj_phase_vis, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    axs[1, 3].set_title('(i) DIP-RED-TV Phase')
    plt.colorbar(im8, ax=axs[1, 3], fraction=0.046, pad=0.04)

    dcmin, dcmax = percentile_limits(den_contrast_vis, 1, 99.5)
    im9 = axs[1, 4].imshow(den_contrast_vis, cmap='cividis', vmin=dcmin, vmax=dcmax)
    axs[1, 4].set_title('(j) TV Auxiliary Contrast')
    plt.colorbar(im9, ax=axs[1, 4], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        if ax not in [axs[0, 3]]:
            ax.axis('off')

    if save_single:
        save_single_panel(intensity, f'{prefix}_a_raw_hologram.png', cmap='gray')
        save_single_panel(results['I_pred'], f'{prefix}_b_I_pred.png', cmap='gray')
        save_single_panel(error_map_vis, f'{prefix}_c_error_map.png', cmap='magma', vmin=evmin, vmax=evmax)
        save_single_panel(den_amp_vis, f'{prefix}_e_tv_aux_amp.png', cmap='gray', vmin=dmin, vmax=dmax)
        save_single_panel(results['base_amp'], f'{prefix}_f_baseline_amp.png', cmap='gray', vmin=bmin, vmax=bmax)
        save_single_panel(obj_amp_vis, f'{prefix}_g_dipredtv_amp.png', cmap='gray', vmin=amin, vmax=amax)
        save_single_panel(obj_contrast_vis, f'{prefix}_h_dipredtv_contrast.png', cmap='cividis', vmin=cmin, vmax=cmax)
        save_single_panel(obj_phase_vis, f'{prefix}_i_dipredtv_phase.png', cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
        save_single_panel(den_contrast_vis, f'{prefix}_j_tv_aux_contrast.png', cmap='cividis', vmin=dcmin, vmax=dcmax)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ==========================================
# Main script
# ==========================================
if __name__ == '__main__':
    # -------- Optical parameters --------
    WAVELENGTH = 632.8e-9
    PIXEL_SIZE = 6.9e-6
    Z_DISTANCE = 0.02275

    # -------- Optimization parameters --------
    ITERS = 2200
    BORDER = 6
    BASE_F = 24
    LR = 7e-4
    IMG_SIZE = 256
    SEED = 42

    # -------- DIP-RED-TV parameters --------
    BETA_AMP = 0.09
    BETA_PHASE = 0.06
    DENOISE_INTERVAL = 100
    TV_WEIGHT_AMP = 0.080
    TV_WEIGHT_PHASE = 0.045
    TV_ITERS_AMP = 60
    TV_ITERS_PHASE = 45

    # -------- Data loading --------
    # input_npy = os.path.join('data', 'sample_007', 'patch_0013', 'patch_0013.npy')
    input_npy = os.path.join('data', 'sample_007', 'patch_0013', 'patch_0013.npy')
    input_png = 'holo.jpg'

    if os.path.exists(input_npy):
        intensity = np.load(input_npy).astype(np.float32)
    elif os.path.exists(input_png):
        intensity = np.array(Image.open(input_png).convert('L'), dtype=np.float32)
    else:
        Y, X = np.ogrid[-1:1:IMG_SIZE * 1j, -1:1:IMG_SIZE * 1j]
        intensity = np.exp(-(X ** 2 + Y ** 2)) + np.random.randn(IMG_SIZE, IMG_SIZE) * 0.05

    intensity = intensity[:IMG_SIZE, :IMG_SIZE]
    intensity = normalize_np(intensity)

    # -------- Reconstruction --------
    results = reconstruct_dip_red_tv(
        intensity_target=intensity,
        z=Z_DISTANCE,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        iters=ITERS,
        border=BORDER,
        seed=SEED,
        base_f=BASE_F,
        lr=LR,
        beta_amp=BETA_AMP,
        beta_phase=BETA_PHASE,
        denoise_interval=DENOISE_INTERVAL,
        tv_weight_amp=TV_WEIGHT_AMP,
        tv_weight_phase=TV_WEIGHT_PHASE,
        tv_iters_amp=TV_ITERS_AMP,
        tv_iters_phase=TV_ITERS_PHASE,
    )

    # -------- Visualization --------
    draw_results_dip_red_tv(
        intensity,
        results,
        # save_path='PNG/dip_red_tv_recon_canvas2_patch_0006.png',
        out_dir='PNG/DIP-RED-TV',
        #prefix='sample_01_patch_0006_dipredtv',
        save_single=True,
    )

    # -------- Evaluation interface output --------
    out_eval_dir = os.path.join('outputs', 'DIP-RED-TV', 'sample_007', 'patch_0013')
    os.makedirs(out_eval_dir, exist_ok=True)

    # Keep the same evaluation interface as Ours and U-Net-PC
    np.save(os.path.join(out_eval_dir, 'table1_img.npy'), results['obj_amp'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'I_pred.npy'), results['I_pred'].astype(np.float32))

    # Optional intermediate results
    np.save(os.path.join(out_eval_dir, 'clean_amp.npy'), results['obj_amp'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'clean_phase.npy'), results['obj_phase'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'clean_contrast.npy'), results['obj_contrast'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'tv_aux_amp.npy'), results['denoised_amp'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'tv_aux_phase.npy'), results['denoised_phase'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'base_amp.npy'), results['base_amp'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'base_phase.npy'), results['base_phase'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'loss_phy_history.npy'), results['loss_phy_history'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'loss_prox_history.npy'), results['loss_prox_history'].astype(np.float32))
    np.save(os.path.join(out_eval_dir, 'denoise_step_marks.npy'), results['denoise_step_marks'].astype(np.int32))

    print(f'DIP-RED-TV results saved to: {out_eval_dir}')
