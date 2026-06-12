import os
import json
import numpy as np
from PIL import Image

import torch
import torch.fft as fft
import matplotlib.pyplot as plt

try:
    from bm3d import bm3d
except ImportError:
    raise ImportError(
        "The bm3d package is not installed. Please run: pip install bm3d"
    )


# ==========================================
# Basic utilities
# ==========================================
def normalize_np(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x, low=1.0, high=99.0):
    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def save_single_image(img, save_path, cmap='gray', vmin=None, vmax=None, dpi=300):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# ==========================================
# ASM
# ==========================================
def propagate_asm(field, z, wavelength, pixel_size):
    """
    field: [1, 1, H, W] complex tensor
    z: propagation distance (m)
    """
    _, _, H, W = field.shape

    fx = fft.fftfreq(W, d=pixel_size, device=field.device)
    fy = fft.fftfreq(H, d=pixel_size, device=field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')

    k = 2 * torch.pi / wavelength
    term = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    term = torch.clamp(term, min=0.0)
    phase_shift = k * z * torch.sqrt(term)

    H_transfer = torch.exp(1j * phase_shift)
    field_prop = fft.ifft2(fft.fft2(field) * H_transfer)
    return field_prop


# ==========================================
# Noise-level estimation
# ==========================================
def estimate_sigma_mad(img, hp_ksize=7):
    """
    Estimate the noise level using a simple high-pass residual and MAD.
    Return sigma on the normalized image.
    """
    x = img.astype(np.float32)
    low = x.copy()

    # Use a simple mean filter as the low-pass estimate
    pad = hp_ksize // 2
    x_pad = np.pad(x, pad, mode='reflect')
    H, W = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            patch = x_pad[i:i+hp_ksize, j:j+hp_ksize]
            out[i, j] = np.mean(patch)

    hp = x - out
    mad = np.median(np.abs(hp - np.median(hp)))
    sigma = 1.4826 * mad
    return float(max(sigma, 1e-4))


# ==========================================
# Temporary BP reconstruction from the raw hologram
# ==========================================
def compute_bp_amp_from_raw(intensity, z_distance, wavelength, pixel_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    I_raw = torch.tensor(intensity, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    U_sensor = torch.sqrt(torch.clamp(I_raw, min=0.0)) + 0j
    U_bp = propagate_asm(U_sensor, -z_distance, wavelength, pixel_size)

    amp_bp = torch.abs(U_bp).detach().cpu().squeeze().numpy().astype(np.float32)
    phase_bp = torch.angle(U_bp).detach().cpu().squeeze().numpy().astype(np.float32)
    return amp_bp, phase_bp


# ==========================================
# BM3D
# ==========================================
def run_bm3d(
    input_path,
    sample_id,
    patch_id,
    z_distance,
    wavelength=632.8e-9,
    pixel_size=6.9e-6,
    out_root='outputs/BM3D',
    png_root='PNG',
    bp_root='outputs/BP',
    sigma_mode='auto',      # 'auto' or float
    sigma_scale=1.0,
):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(png_root, exist_ok=True)

    out_dir = os.path.join(out_root, sample_id, patch_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- load raw hologram ----------
    if input_path.lower().endswith('.npy'):
        intensity = np.load(input_path).astype(np.float32)
    else:
        intensity = np.array(Image.open(input_path).convert('L'), dtype=np.float32)

    intensity = normalize_np(intensity)

    # ---------- Prefer the existing BP result if available ----------
    bp_dir = os.path.join(bp_root, sample_id, patch_id)
    bp_amp_path = os.path.join(bp_dir, 'amp.npy')
    bp_phase_path = os.path.join(bp_dir, 'phase.npy')

    if os.path.exists(bp_amp_path):
        amp_bp = np.load(bp_amp_path).astype(np.float32)
        if os.path.exists(bp_phase_path):
            phase_bp = np.load(bp_phase_path).astype(np.float32)
        else:
            _, phase_bp = compute_bp_amp_from_raw(intensity, z_distance, wavelength, pixel_size)
        print(f'[BM3D] using existing BP result: {bp_amp_path}')
    else:
        print('[BM3D] BP result not found, computing BP from raw hologram...')
        amp_bp, phase_bp = compute_bp_amp_from_raw(intensity, z_distance, wavelength, pixel_size)

    # ---------- Normalize the BM3D input ----------
    amp_bp_norm = normalize_np(amp_bp)

    if sigma_mode == 'auto':
        sigma_est = estimate_sigma_mad(amp_bp_norm)
    else:
        sigma_est = float(sigma_mode)

    sigma_est *= float(sigma_scale)
    print(f'[BM3D] estimated sigma = {sigma_est:.6f}')

    # ---------- BM3D denoising ----------
    bm3d_amp = bm3d(amp_bp_norm, sigma_psd=sigma_est)
    bm3d_amp = np.clip(bm3d_amp, 0.0, 1.0).astype(np.float32)

    # Use the BM3D-denoised amplitude as the fixed Table 1 interface
    table1_img = bm3d_amp.copy()

    # ---------- save npy ----------
    np.save(os.path.join(out_dir, 'table1_img.npy'), table1_img)
    np.save(os.path.join(out_dir, 'bp_amp.npy'), amp_bp.astype(np.float32))
    np.save(os.path.join(out_dir, 'bm3d_amp.npy'), bm3d_amp.astype(np.float32))

    meta = {
        'method': 'BM3D',
        'full_name': 'Block-Matching and 3D Filtering',
        'display_name': 'Block-Matching and 3D Filtering',
        'sample_id': sample_id,
        'patch_id': patch_id,
        'table1_image': 'bm3d_amp',
        'complex_field_output': False,
        'sigma_mode': sigma_mode,
        'sigma_scale': float(sigma_scale),
        'sigma_est': float(sigma_est),
        'input_from': 'BP amplitude',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---------- PNG ----------
    prefix = f'BM3D_{sample_id}_{patch_id}'

    rmin, rmax = percentile_limits(intensity, 1, 99)
    save_single_image(
        intensity,
        os.path.join(png_root, f'{prefix}_raw_hologram.png'),
        cmap='gray',
        vmin=rmin,
        vmax=rmax,
        dpi=300
    )

    amin, amax = percentile_limits(amp_bp, 1, 99)
    save_single_image(
        amp_bp,
        os.path.join(png_root, f'{prefix}_bp_amp.png'),
        cmap='gray',
        vmin=amin,
        vmax=amax,
        dpi=300
    )

    bmin, bmax = percentile_limits(bm3d_amp, 1, 99)
    save_single_image(
        bm3d_amp,
        os.path.join(png_root, f'{prefix}_bm3d_amp.png'),
        cmap='gray',
        vmin=bmin,
        vmax=bmax,
        dpi=300
    )

    print(f'[BM3D] finished: {sample_id}/{patch_id}')
    print(f'Outputs saved to: {out_dir}')

    return {
        'bp_amp': amp_bp,
        'phase_bp': phase_bp,
        'bm3d_amp': bm3d_amp,
        'table1_img': table1_img,
        'sigma_est': sigma_est,
    }


if __name__ == '__main__':
    input_path = r'data/sample_007/patch_0013/patch_0013.npy'
    sample_id = 'sample_007'
    patch_id = 'patch_0013'

    WAVELENGTH = 632.8e-9
    PIXEL_SIZE = 6.9e-6
    Z_DISTANCE = 0.02275

    run_bm3d(
        input_path=input_path,
        sample_id=sample_id,
        patch_id=patch_id,
        z_distance=Z_DISTANCE,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        out_root='outputs/BM3D',
        png_root='PNG/BM3D_reconstruction_sample_01/patch_0006',
        bp_root='outputs/BP',
        sigma_mode='auto',   # A fixed value such as 0.03 can also be used
        sigma_scale=1.0,
    )