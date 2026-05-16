import os
import json
import numpy as np
from PIL import Image

import torch
import torch.fft as fft
import matplotlib.pyplot as plt


# ==========================================
# Utility functions
def normalize_np(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x, low=1.0, high=99.0):
    vmin = np.percentile(x, low)
    vmax = np.percentile(x, high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def smooth1d_circular(arr, k=7):
    pad = k // 2
    ext = np.concatenate([arr[-pad:], arr, arr[:pad]], axis=0)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.convolve(ext, kernel, mode='same')[pad:-pad]
    return out


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
# FFT-based orientation estimation
# ==========================================
def estimate_dominant_stripe_angles_from_image(img, topk=2, min_sep_deg=18, rmin=0.06, rmax=0.45):
    """
    Estimate the frequency-normal angles corresponding to the dominant stripe directions from the image spectrum.
    Return angles in the range [0, 180).
    """
    x = normalize_np(img)
    x = x - np.mean(x)
    H, W = x.shape

    Fmag = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2

    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    y = yy - cy
    xcoord = xx - cx
    r = np.sqrt(xcoord ** 2 + y ** 2)
    r_norm = r / (min(H, W) / 2.0 + 1e-8)

    angle = (np.rad2deg(np.arctan2(y, xcoord)) + 180.0) % 180.0
    mask = (r_norm > rmin) & (r_norm < rmax)

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


# ==========================================
# Smooth notch mask
# ==========================================
def build_smooth_notch_mask(
    H,
    W,
    angles_deg,
    sigma_deg=6.0,
    notch_depth=0.85,
    rmin=0.06,
    rmax=0.48,
):
    """
    Construct a smooth angular notch mask in the FFT plane.
    angles_deg: list of frequency-normal angles (0-180)
    sigma_deg: notch angular width; larger values suppress a wider angular range
    notch_depth: 0-1; larger values indicate stronger suppression
    """
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    y = yy - cy
    xcoord = xx - cx

    r = np.sqrt(xcoord ** 2 + y ** 2)
    r_norm = r / (min(H, W) / 2.0 + 1e-8)
    angle = (np.rad2deg(np.arctan2(y, xcoord)) + 180.0) % 180.0

    radial_mask = ((r_norm > rmin) & (r_norm < rmax)).astype(np.float32)

    mask = np.ones((H, W), dtype=np.float32)
    for ang in angles_deg:
        d = np.abs(angle - ang)
        d = np.minimum(d, 180.0 - d)
        angular_notch = np.exp(-0.5 * (d / sigma_deg) ** 2).astype(np.float32)
        mask *= (1.0 - notch_depth * angular_notch * radial_mask)

    mask = np.clip(mask, 0.0, 1.0)
    return mask.astype(np.float32)


# ==========================================
# BP + FFT Notch
# ==========================================
def run_bp_fft_notch(
    input_path,
    sample_id,
    patch_id,
    z_distance,
    wavelength=632.8e-9,
    pixel_size=6.9e-6,
    out_root='outputs/BP_FFT_Notch',
    png_root='PNG',
    topk=2,
    min_sep_deg=18,
    sigma_deg=6.0,
    notch_depth=0.85,
    rmin=0.06,
    rmax=0.48,
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

    # ---------- BP ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    I_raw = torch.tensor(intensity, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    U_sensor = torch.sqrt(torch.clamp(I_raw, min=0.0)) + 0j
    U_bp = propagate_asm(U_sensor, -z_distance, wavelength, pixel_size)

    amp_bp = torch.abs(U_bp).detach().cpu().squeeze().numpy().astype(np.float32)
    phase_bp = torch.angle(U_bp).detach().cpu().squeeze().numpy().astype(np.float32)

    # ---------- FFT notch on BP amplitude ----------
    x = amp_bp.astype(np.float32)
    x_mean = float(np.mean(x))
    x0 = x - x_mean

    angles_deg = estimate_dominant_stripe_angles_from_image(
        x0,
        topk=topk,
        min_sep_deg=min_sep_deg,
        rmin=rmin,
        rmax=rmax
    )
    print(f'[BP+FFT Notch] estimated stripe angles: {angles_deg}')

    F = np.fft.fftshift(np.fft.fft2(x0))
    notch_mask = build_smooth_notch_mask(
        H=x.shape[0],
        W=x.shape[1],
        angles_deg=angles_deg,
        sigma_deg=sigma_deg,
        notch_depth=notch_depth,
        rmin=rmin,
        rmax=rmax,
    )

    F_filtered = F * notch_mask
    x_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered))) + x_mean
    x_filtered = x_filtered.astype(np.float32)

    # Table 1 uses the final image after notch filtering.
    table1_img = x_filtered.copy()

    # ---------- save npy ----------
    np.save(os.path.join(out_dir, 'table1_img.npy'), table1_img)
    np.save(os.path.join(out_dir, 'bp_amp.npy'), amp_bp)
    np.save(os.path.join(out_dir, 'bp_phase.npy'), phase_bp)
    np.save(os.path.join(out_dir, 'notch_amp.npy'), x_filtered)
    np.save(os.path.join(out_dir, 'notch_mask.npy'), notch_mask.astype(np.float32))

    meta = {
        'method': 'BP+FFT Notch',
        'sample_id': sample_id,
        'patch_id': patch_id,
        'z_distance_m': float(z_distance),
        'wavelength_m': float(wavelength),
        'pixel_size_m': float(pixel_size),
        'table1_image': 'notch_amp',
        'complex_field_output': False,
        'topk': int(topk),
        'min_sep_deg': float(min_sep_deg),
        'sigma_deg': float(sigma_deg),
        'notch_depth': float(notch_depth),
        'rmin': float(rmin),
        'rmax': float(rmax),
        'angles_deg': [float(a) for a in angles_deg],
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---------- PNG ----------
    prefix = f'BP_FFT_Notch_{sample_id}_{patch_id}'

    rmin_v, rmax_v = percentile_limits(intensity, 1, 99)
    save_single_image(
        intensity,
        os.path.join(png_root, f'{prefix}_raw_hologram.png'),
        cmap='gray',
        vmin=rmin_v,
        vmax=rmax_v,
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

    nmin, nmax = percentile_limits(x_filtered, 1, 99)
    save_single_image(
        x_filtered,
        os.path.join(png_root, f'{prefix}_notch_amp.png'),
        cmap='gray',
        vmin=nmin,
        vmax=nmax,
        dpi=300
    )

    # Spectrum visualization
    spec_before = np.log1p(np.abs(F)).astype(np.float32)
    spec_after = np.log1p(np.abs(F_filtered)).astype(np.float32)

    s1min, s1max = percentile_limits(spec_before, 1, 99)
    save_single_image(
        spec_before,
        os.path.join(png_root, f'{prefix}_spectrum_before.png'),
        cmap='magma',
        vmin=s1min,
        vmax=s1max,
        dpi=300
    )

    s2min, s2max = percentile_limits(spec_after, 1, 99)
    save_single_image(
        spec_after,
        os.path.join(png_root, f'{prefix}_spectrum_after.png'),
        cmap='magma',
        vmin=s2min,
        vmax=s2max,
        dpi=300
    )

    save_single_image(
        notch_mask,
        os.path.join(png_root, f'{prefix}_notch_mask.png'),
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        dpi=300
    )

    print(f'[BP+FFT Notch] finished: {sample_id}/{patch_id}')
    print(f'Outputs saved to: {out_dir}')

    return {
        'bp_amp': amp_bp,
        'bp_phase': phase_bp,
        'notch_amp': x_filtered,
        'table1_img': table1_img,
        'notch_mask': notch_mask,
        'angles_deg': angles_deg,
    }


if __name__ == '__main__':
    input_path = r'data/sample_01/patch_0006/patch_0006.npy'
    sample_id = 'sample_01'
    patch_id = 'patch_0006'

    WAVELENGTH = 632.8e-9
    PIXEL_SIZE = 6.9e-6
    Z_DISTANCE = 0.0235

    run_bp_fft_notch(
        input_path=input_path,
        sample_id=sample_id,
        patch_id=patch_id,
        z_distance=Z_DISTANCE,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        out_root='outputs/BP_FFT_Notch',
        png_root='PNG/BP_FFT_Notch/reconstruction_sample_01/patch_0006',
        topk=2,
        min_sep_deg=18,
        sigma_deg=6.0,
        notch_depth=0.85,
        rmin=0.06,
        rmax=0.48,
    )