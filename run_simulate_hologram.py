import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import ndimage as ndi
except Exception as e:
    raise ImportError("scipy is required to run this script.") from e


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def percentile_limits(x, low=1.0, high=99.0):
    vmin = float(np.percentile(x, low))
    vmax = float(np.percentile(x, high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_img(arr: np.ndarray, path: Path, cmap='gray', mode='percentile', dpi=220):
    arr = np.asarray(arr, dtype=np.float32)
    if mode == 'percentile':
        vmin, vmax = percentile_limits(arr, 1.0, 99.0)
    else:
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def propagate_asm_np(field: np.ndarray, z: float, wavelength: float, pixel_size: float) -> np.ndarray:
    H, W = field.shape
    fx = np.fft.fftfreq(W, d=pixel_size)
    fy = np.fft.fftfreq(H, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    k = 2.0 * np.pi / wavelength
    term = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    term = np.clip(term, 0.0, None)
    phase_shift = k * z * np.sqrt(term)

    H_transfer = np.exp(1j * phase_shift).astype(np.complex64)
    field_prop = np.fft.ifft2(np.fft.fft2(field) * H_transfer)
    return field_prop.astype(np.complex64)


# =========================================================
# Generate objects that are well separated from each other
# =========================================================
def make_random_ellipse_mask(H, W, cy, cx, ay, ax, angle_deg):
    yy, xx = np.mgrid[0:H, 0:W]
    yy = yy - cy
    xx = xx - cx
    th = np.deg2rad(angle_deg)
    xr = np.cos(th) * xx + np.sin(th) * yy
    yr = -np.sin(th) * xx + np.cos(th) * yy
    return ((xr / ax) ** 2 + (yr / ay) ** 2) <= 1.0


def place_separated_objects(
    H,
    W,
    rng,
    n_objects=3,
    min_gap_px=18,
    max_trials=300,
):
    """
    Place objects one by one, requiring:
    1) No overlap
    2) Center distance and bounding scale satisfy a minimum gap
    """
    final_mask = np.zeros((H, W), dtype=bool)
    placed = []
    labels = np.zeros((H, W), dtype=np.int32)

    for obj_id in range(1, n_objects + 1):
        placed_ok = False
        for _ in range(max_trials):
            cy = int(rng.integers(int(0.18 * H), int(0.82 * H)))
            cx = int(rng.integers(int(0.18 * W), int(0.82 * W)))
            ay = int(rng.integers(max(7, H // 38), max(12, H // 11)))
            ax = int(rng.integers(max(7, W // 38), max(12, W // 11)))
            ang = float(rng.uniform(0, 180))

            cand = make_random_ellipse_mask(H, W, cy, cx, ay, ax, ang)

            # Light morphological smoothing on each object only; no global closing/dilation
            cand = ndi.binary_fill_holes(cand)
            cand = ndi.binary_opening(cand, iterations=1)

            if np.any(cand & final_mask):
                continue

            ok = True
            for (py, px, pay, pax) in placed:
                d = np.hypot(cy - py, cx - px)
                safe_dist = 0.65 * (max(ax, ay) + max(pax, pay)) + min_gap_px
                if d < safe_dist:
                    ok = False
                    break

            if not ok:
                continue

            final_mask |= cand
            labels[cand] = obj_id
            placed.append((cy, cx, ay, ax))
            placed_ok = True
            break

        if not placed_ok:
            print(f"[WARN] Failed to place object {obj_id}, skipping.")

    return final_mask.astype(np.uint8), labels.astype(np.int32)


def build_object_field(
    H,
    W,
    rng,
    n_objects=3,
    min_gap_px=18,
):
    obj_mask, obj_label = place_separated_objects(
        H=H,
        W=W,
        rng=rng,
        n_objects=n_objects,
        min_gap_px=min_gap_px,
    )

    # Soft edge via slight blur to avoid merging adjacent object edges
    soft = ndi.gaussian_filter(obj_mask.astype(np.float32), sigma=0.8)
    soft = np.clip(soft, 0.0, 1.0)

    # Amplitude: background near 1, object regions lower than 1
    absorb = rng.uniform(0.16, 0.30)
    A_obj = 1.0 - absorb * soft

    # Phase: assign each object different phase strength to improve distinguishability
    phi_obj = np.zeros((H, W), dtype=np.float32)
    for obj_id in range(1, int(obj_label.max()) + 1):
        one = (obj_label == obj_id).astype(np.float32)
        one_soft = ndi.gaussian_filter(one, sigma=1.0)
        one_soft = np.clip(one_soft, 0.0, 1.0)

        phase_scale = float(rng.uniform(0.45, 1.10))
        phase_noise = ndi.gaussian_filter(rng.normal(size=(H, W)).astype(np.float32), sigma=5.0)
        phase_noise = normalize01(phase_noise) * 2.0 - 1.0
        phi_obj += phase_scale * one_soft * phase_noise

    U_obj = (A_obj * np.exp(1j * phi_obj)).astype(np.complex64)

    return {
        "obj_mask": obj_mask.astype(np.uint8),
        "obj_label": obj_label.astype(np.int32),
        "obj_soft": soft.astype(np.float32),
        "obj_amp": A_obj.astype(np.float32),
        "obj_phase": phi_obj.astype(np.float32),
        "U_obj": U_obj,
    }


def build_smooth_background(H, W, rng):
    y = np.linspace(-1, 1, H, dtype=np.float32)
    x = np.linspace(-1, 1, W, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing='ij')
    R2 = X**2 + Y**2

    amp = (
        0.90
        + rng.uniform(-0.04, 0.04) * X
        + rng.uniform(-0.04, 0.04) * Y
        + rng.uniform(0.00, 0.03) * R2
        + rng.uniform(-0.015, 0.015) * X * Y
        + rng.uniform(-0.01, 0.01) * (X**2 - Y**2)
    )
    amp = np.clip(amp, 0.75, 1.05).astype(np.float32)

    phi = (
        rng.uniform(-0.18, 0.18) * X
        + rng.uniform(-0.18, 0.18) * Y
        + rng.uniform(-0.08, 0.08) * R2
        + rng.uniform(-0.06, 0.06) * X * Y
    ).astype(np.float32)

    U_bg = (amp * np.exp(1j * phi)).astype(np.complex64)
    return {
        "bg_amp": amp,
        "bg_phase": phi,
        "U_bg": U_bg,
    }


def build_parasitic_sensor_field(H, W, rng, n_carriers=2):
    y = np.linspace(-1, 1, H, dtype=np.float32)
    x = np.linspace(-1, 1, W, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing='ij')
    R2 = X ** 2 + Y ** 2

    env = (
        0.10
        + rng.uniform(0.00, 0.06) * X
        + rng.uniform(0.00, 0.06) * Y
        + rng.uniform(0.02, 0.08) * R2
    )
    env = np.clip(env, 0.02, 0.22).astype(np.float32)

    coeff = np.zeros((H, W), dtype=np.complex64)
    for _ in range(n_carriers):
        theta = np.deg2rad(float(rng.uniform(0, 180)))
        f = float(rng.uniform(4.0, 12.0))
        phase0 = float(rng.uniform(0, 2 * np.pi))
        amp_k = float(rng.uniform(0.15, 0.45))
        proj = np.cos(theta) * X + np.sin(theta) * Y
        wave = np.exp(1j * (2.0 * np.pi * f * proj + phase0)).astype(np.complex64)
        coeff += amp_k * env * wave

    real_s = ndi.gaussian_filter(np.real(coeff), sigma=1.0)
    imag_s = ndi.gaussian_filter(np.imag(coeff), sigma=1.0)
    U_para_sensor = (real_s + 1j * imag_s).astype(np.complex64)

    return {
        "para_env": env.astype(np.float32),
        "U_para_sensor": U_para_sensor,
        "para_amp": np.abs(U_para_sensor).astype(np.float32),
        "para_phase": np.angle(U_para_sensor).astype(np.float32),
    }


def add_noise_and_normalize(I_clean, rng, poisson_scale=600.0, gauss_sigma=0.010):
    I = np.clip(I_clean, 0.0, None).astype(np.float32)
    I_norm = I / (I.max() + 1e-8)
    poisson_counts = rng.poisson(I_norm * poisson_scale).astype(np.float32)
    I_poiss = poisson_counts / float(poisson_scale)
    I_noisy = I_poiss + rng.normal(scale=gauss_sigma, size=I.shape).astype(np.float32)
    I_noisy = np.clip(I_noisy, 0.0, None)
    return normalize01(I_noisy).astype(np.float32)


def simulate_one(
    out_dir: Path,
    H=256,
    W=256,
    wavelength=632.8e-9,
    pixel_size=6.9e-6,
    z_distance=0.021,
    seed=42,
    n_objects=3,
    min_gap_px=18,
):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = build_object_field(H, W, rng, n_objects=n_objects, min_gap_px=min_gap_px)
    bg = build_smooth_background(H, W, rng)
    para = build_parasitic_sensor_field(H, W, rng, n_carriers=2)

    U_main_obj = (obj["U_obj"] * bg["U_bg"]).astype(np.complex64)
    U_main_sensor = propagate_asm_np(U_main_obj, z_distance, wavelength, pixel_size)
    U_sensor = (U_main_sensor + para["U_para_sensor"]).astype(np.complex64)

    I_clean = (np.abs(U_sensor) ** 2).astype(np.float32)
    holo = add_noise_and_normalize(I_clean, rng=rng, poisson_scale=650.0, gauss_sigma=0.012)
    A_main_gt = np.abs(U_main_obj).astype(np.float32)

    meta = {
        "seed": int(seed),
        "H": int(H),
        "W": int(W),
        "wavelength": float(wavelength),
        "pixel_size": float(pixel_size),
        "z_distance": float(z_distance),
        "n_objects": int(n_objects),
        "min_gap_px": int(min_gap_px),
        "forward_model": "U_sensor = P(U_obj * U_bg_smooth) + U_parasitic_sensor",
        "notes": "Matched simulation with explicitly separated objects.",
    }

    np.save(out_dir / "sim_hologram.npy", holo)
    np.save(out_dir / "sim_hologram_clean.npy", normalize01(I_clean))
    np.save(out_dir / "gt_obj_mask.npy", obj["obj_mask"])
    np.save(out_dir / "gt_obj_label.npy", obj["obj_label"])
    np.save(out_dir / "gt_obj_soft.npy", obj["obj_soft"])
    np.save(out_dir / "gt_obj_amp.npy", obj["obj_amp"])
    np.save(out_dir / "gt_obj_phase.npy", obj["obj_phase"])
    np.save(out_dir / "gt_bg_amp.npy", bg["bg_amp"])
    np.save(out_dir / "gt_bg_phase.npy", bg["bg_phase"])
    np.save(out_dir / "gt_parasitic_amp.npy", para["para_amp"])
    np.save(out_dir / "gt_parasitic_phase.npy", para["para_phase"])
    np.save(out_dir / "gt_A_main.npy", A_main_gt)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_img(holo, out_dir / "sim_hologram.png", cmap="gray")
    save_img(A_main_gt, out_dir / "gt_A_main.png", cmap="gray")
    save_img(obj["obj_mask"], out_dir / "gt_obj_mask.png", cmap="gray", mode="raw")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(obj["obj_mask"], cmap='gray')
    axs[0].set_title('GT Object Mask')
    axs[1].imshow(A_main_gt, cmap='gray')
    axs[1].set_title('GT $A_{main}$')
    axs[2].imshow(holo, cmap='gray')
    axs[2].set_title('Simulated Hologram')
    for ax in axs.flat:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / "simulation_quicklook.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved simulation to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate matched simulated hologram data with separated objects.")
    parser.add_argument("--out_dir", type=str, default="sim_data/sample_sim_001/patch_0001")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument("--z_distance", type=float, default=0.021)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_objects", type=int, default=3)
    parser.add_argument("--min_gap_px", type=int, default=18)
    args = parser.parse_args()

    simulate_one(
        out_dir=Path(args.out_dir),
        H=args.H,
        W=args.W,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        z_distance=args.z_distance,
        seed=args.seed,
        n_objects=args.n_objects,
        min_gap_px=args.min_gap_px,
    )


if __name__ == "__main__":
    main()