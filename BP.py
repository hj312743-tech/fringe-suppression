import os
import json
import numpy as np
from PIL import Image

import torch
import torch.fft as fft
import matplotlib.pyplot as plt


def normalize_np(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x, low=1.0, high=99.0):
    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


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


def save_single_image(img, save_path, cmap='gray', vmin=None, vmax=None, dpi=300):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def run_bp(
    input_path,
    sample_id,
    patch_id,
    z_distance,
    wavelength=632.8e-9,
    pixel_size=6.9e-6,
    out_root='outputs/BP',
    png_root='PNG',
):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(png_root, exist_ok=True)

    out_dir = os.path.join(out_root, sample_id, patch_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- load hologram ----------
    if input_path.lower().endswith('.npy'):
        intensity = np.load(input_path).astype(np.float32)
    else:
        intensity = np.array(Image.open(input_path).convert('L'), dtype=np.float32)

    intensity = normalize_np(intensity)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    I_raw = torch.tensor(intensity, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # ---------- BP reconstruction ----------
    # Sensor-plane field: amplitude is sqrt(I), and phase is initialized to zero
    U_sensor = torch.sqrt(torch.clamp(I_raw, min=0.0)) + 0j

    # Back-propagate to the object plane
    U_obj = propagate_asm(U_sensor, -z_distance, wavelength, pixel_size)

    # BP outputs
    amp = torch.abs(U_obj).detach().cpu().squeeze().numpy().astype(np.float32)
    phase = torch.angle(U_obj).detach().cpu().squeeze().numpy().astype(np.float32)

    # Forward-propagate once more to obtain I_pred for consistency checking
    U_forward = propagate_asm(U_obj, z_distance, wavelength, pixel_size)
    I_pred = (torch.abs(U_forward) ** 2).detach().cpu().squeeze().numpy().astype(np.float32)

    # ---------- Fixed Table 1 interface ----------
    # BP uses the amplitude image amp for Table 1
    table1_img = amp.copy()

    np.save(os.path.join(out_dir, 'table1_img.npy'), table1_img)
    np.save(os.path.join(out_dir, 'I_pred.npy'), I_pred)
    np.save(os.path.join(out_dir, 'amp.npy'), amp)
    np.save(os.path.join(out_dir, 'phase.npy'), phase)

    meta = {
        'method': 'BP',
        'sample_id': sample_id,
        'patch_id': patch_id,
        'z_distance_m': float(z_distance),
        'wavelength_m': float(wavelength),
        'pixel_size_m': float(pixel_size),
        'table1_image': 'amp',
        'complex_field_output': True,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---------- PNG preview ----------
    prefix = f'BP_{sample_id}_{patch_id}'

    amin, amax = percentile_limits(amp, 1, 99)
    save_single_image(
        amp,
        os.path.join(png_root, f'{prefix}_amp.png'),
        cmap='gray',
        vmin=amin,
        vmax=amax,
        dpi=300
    )

    save_single_image(
        phase,
        os.path.join(png_root, f'{prefix}_phase.png'),
        cmap='twilight_shifted',
        vmin=-np.pi,
        vmax=np.pi,
        dpi=300
    )

    irmin, irmax = percentile_limits(intensity, 1, 99)
    save_single_image(
        intensity,
        os.path.join(png_root, f'{prefix}_raw_hologram.png'),
        cmap='gray',
        vmin=irmin,
        vmax=irmax,
        dpi=300
    )

    ipred_min, ipred_max = percentile_limits(I_pred, 1, 99)
    save_single_image(
        I_pred,
        os.path.join(png_root, f'{prefix}_I_pred.png'),
        cmap='gray',
        vmin=ipred_min,
        vmax=ipred_max,
        dpi=300
    )

    print(f'[BP] finished: {sample_id}/{patch_id}')
    print(f'Outputs saved to: {out_dir}')
    return {
        'amp': amp,
        'phase': phase,
        'I_pred': I_pred,
        'table1_img': table1_img,
    }


if __name__ == '__main__':
    # Run BP for a single patch
    input_path = r'data/sample_007/patch_0013/patch_0013.npy'
    sample_id = 'sample_007'
    patch_id = 'patch_0013'

    WAVELENGTH = 632.8e-9
    PIXEL_SIZE = 6.9e-6
    Z_DISTANCE = 0.02275
    run_bp(
        input_path=input_path,
        sample_id=sample_id,
        patch_id=patch_id,
        z_distance=Z_DISTANCE,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        out_root='outputs/BP',
        png_root='PNG/BP',
    )