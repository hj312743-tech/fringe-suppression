from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


"""
run_backprop_reconstruct.py
---------------------------
Performs classical angular spectrum back-propagation reconstruction on a single patch.

Design goals:
1. Serves as the first step of the "reconstruct first, then remove fringes" pipeline.
2. Input: single-frame hologram intensity patch (recommended .npy floating-point image).
3. Assumes sensor-plane complex field amplitude is initialized from intensity: A = sqrt(I) (or A = I).
4. Initial phase is set to zero.
5. Uses the angular spectrum method to back-propagate the sensor field to a candidate object plane,
   outputting amplitude and phase.

Note:
- This is a classical physics baseline, not the final paper model.
- Its role is to first map the hologram domain to the reconstruction domain,
  then perform self-supervised background fringe removal in the reconstruction domain.
"""


def load_array(path: str | Path) -> np.ndarray:
    """Load .npy or common image files, uniformly convert to float32 2D array."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float32)

    from PIL import Image

    img = Image.open(path)
    if img.mode not in ("L", "F", "I;16", "I"):
        img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def save_preview(arr: np.ndarray, save_path: Path, cmap: str = "gray", percentile: tuple[float, float] = (1, 99)) -> None:
    """Save a visualization PNG for manual inspection only; does not affect .npy raw data."""
    arr = np.asarray(arr, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        valid = np.array([0.0], dtype=np.float32)
    vmin = float(np.percentile(valid, percentile[0]))
    vmax = float(np.percentile(valid, percentile[1]))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    plt.imsave(save_path, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def angular_spectrum_propagate(field: np.ndarray, z: float, wavelength: float, pixel_size: float) -> np.ndarray:
    """
    NumPy version of the angular spectrum propagation.

    Args:
        field: 2D complex field
        z: propagation distance (meters), positive for forward, negative for backward
        wavelength: wavelength (meters)
        pixel_size: pixel size (meters)
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape={field.shape}")

    h, w = field.shape
    fx = np.fft.fftfreq(w, d=pixel_size)
    fy = np.fft.fftfreq(h, d=pixel_size)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="xy")

    term = 1.0 - (wavelength * fx_grid) ** 2 - (wavelength * fy_grid) ** 2
    # Clipping evanescent components; sufficient as classical baseline.
    term = np.clip(term, a_min=0.0, a_max=None)
    k = 2.0 * np.pi / wavelength
    phase = k * z * np.sqrt(term)
    h_transfer = np.exp(1j * phase).astype(np.complex64)

    field_fft = np.fft.fft2(field)
    field_prop = np.fft.ifft2(field_fft * h_transfer)
    return field_prop.astype(np.complex64)


# backward reconstruction = propagate by -z


def backprop_reconstruct(
    intensity: np.ndarray,
    z: float,
    wavelength: float,
    pixel_size: float,
    input_mode: str = "sqrt",
    normalize_input: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform classical back-propagation reconstruction on an intensity image.

    input_mode:
    - sqrt: initialize sensor field amplitude as sqrt(I)
    - linear: initialize sensor field amplitude as I

    normalize_input:
    - if True, apply min-max normalization first, useful for quick baseline when intensity scales differ across patches
    - if False, keep original floating-point amplitude relationships
    """
    intensity = np.asarray(intensity, dtype=np.float32)
    intensity = np.clip(intensity, a_min=0.0, a_max=None)

    if normalize_input:
        vmin = float(np.min(intensity))
        vmax = float(np.max(intensity))
        if vmax > vmin:
            intensity = (intensity - vmin) / (vmax - vmin)

    if input_mode == "sqrt":
        sensor_amp = np.sqrt(intensity)
    elif input_mode == "linear":
        sensor_amp = intensity.copy()
    else:
        raise ValueError(f"unknown input_mode: {input_mode}")

    # Single-frame classical baseline: initial phase set to 0
    sensor_field = sensor_amp.astype(np.complex64)
    object_field = angular_spectrum_propagate(
        sensor_field,
        z=-float(z),
        wavelength=float(wavelength),
        pixel_size=float(pixel_size),
    )
    amplitude = np.abs(object_field).astype(np.float32)
    phase = np.angle(object_field).astype(np.float32)
    return object_field, amplitude, phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical back-propagation reconstruction on a single patch")
    parser.add_argument("--input", type=str, required=True, help="Input patch, recommended .npy float image")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--wavelength", type=float, required=True, help="Wavelength (meters)")
    parser.add_argument("--pixel_size", type=float, required=True, help="Pixel size (meters)")
    parser.add_argument("--z", type=float, required=True, help="Propagation distance (meters)")
    parser.add_argument("--input_mode", type=str, default="sqrt", choices=["sqrt", "linear"],
                        help="Sensor field amplitude initialization method")
    parser.add_argument("--normalize_input", action="store_true", help="Min-max normalize input first")
    parser.add_argument("--save_complex", action="store_true", help="Save complex_field.npy (complex)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intensity = load_array(args.input)
    if intensity.ndim != 2:
        raise ValueError(f"Input must be 2D, got shape={intensity.shape}")
    intensity = intensity.astype(np.float32)

    obj_field, amp, phase = backprop_reconstruct(
        intensity=intensity,
        z=args.z,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        input_mode=args.input_mode,
        normalize_input=args.normalize_input,
    )

    np.save(out_dir / "input_intensity.npy", intensity)
    np.save(out_dir / "amplitude.npy", amp)
    np.save(out_dir / "phase.npy", phase)
    if args.save_complex:
        np.save(out_dir / "complex_field.npy", obj_field)

    save_preview(intensity, out_dir / "input_intensity.png", cmap="gray")
    save_preview(amp, out_dir / "amplitude.png", cmap="gray")
    # Phase preview fixed to [-pi, pi]
    plt.imsave(out_dir / "phase.png", phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)

    print(f"Done. Saved to: {out_dir}")
    print(f"input shape = {intensity.shape}, z = {args.z:.6f} m")


if __name__ == "__main__":
    main()