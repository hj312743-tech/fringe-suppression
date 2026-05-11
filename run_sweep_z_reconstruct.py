from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_backprop_reconstruct import backprop_reconstruct


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)

    img = Image.open(path)
    if img.mode not in ("L", "F", "I;16", "I"):
        img = img.convert("L")
    return np.asarray(img, dtype=np.float32)



def save_preview(arr: np.ndarray, save_path: Path, cmap: str = "gray", percentile=(1, 99)) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        valid = np.array([0.0], dtype=np.float32)
    vmin = float(np.percentile(valid, percentile[0]))
    vmax = float(np.percentile(valid, percentile[1]))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    plt.imsave(save_path, arr, cmap=cmap, vmin=vmin, vmax=vmax)



def to_uint8_raw_gray(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"仅支持二维数组，当前 shape={arr.shape}")

    if np.issubdtype(arr.dtype, np.uint8):
        return arr.copy()

    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))

    if 0.0 <= vmin and vmax <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0.0, 255.0)
    return np.round(arr).astype(np.uint8)



def save_raw_gray(arr: np.ndarray, save_path: Path) -> None:
    Image.fromarray(to_uint8_raw_gray(arr), mode="L").save(save_path)



def save_input_like_source(input_path: Path, intensity: np.ndarray, save_path: Path) -> None:
    if input_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        shutil.copy2(input_path, save_path)
        return
    save_raw_gray(intensity, save_path)



def tenengrad(img: np.ndarray) -> float:
    img = np.asarray(img, dtype=np.float32)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))



def laplacian_var(img: np.ndarray) -> float:
    img = np.asarray(img, dtype=np.float32)
    lap = np.zeros_like(img)
    lap[1:-1, 1:-1] = (
        img[:-2, 1:-1]
        + img[2:, 1:-1]
        + img[1:-1, :-2]
        + img[1:-1, 2:]
        - 4.0 * img[1:-1, 1:-1]
    )
    return float(np.var(lap))



def gradient_sparsity(img: np.ndarray) -> float:
    img = np.asarray(img, dtype=np.float32)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy)
    num = float(np.mean(mag) ** 2)
    den = float(np.mean(mag * mag) + 1e-8)
    return num / den



def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    if vmax <= vmin:
        return np.zeros_like(x)
    return (x - vmin) / (vmax - vmin)



def make_range(start: float, stop: float, step: float) -> list[float]:
    vals = []
    z = float(start)
    while z <= float(stop) + 1e-12:
        vals.append(round(z, 10))
        z += float(step)
    return vals



def evaluate_focus(amp: np.ndarray) -> dict[str, float]:
    tg = tenengrad(amp)
    lv = laplacian_var(amp)
    gs = gradient_sparsity(amp)
    return {"tenengrad": tg, "laplacian_var": lv, "gradient_sparsity": gs}



def scan_once(
    intensity: np.ndarray,
    z_values: list[float],
    out_dir: Path,
    wavelength: float,
    pixel_size: float,
    input_mode: str,
    normalize_input: bool,
) -> tuple[list[dict[str, float | str]], list[np.ndarray], list[np.ndarray]]:
    records: list[dict[str, float | str]] = []
    amp_maps: list[np.ndarray] = []
    phase_maps: list[np.ndarray] = []

    for z in z_values:
        z_name = f"z_{z:.6f}".replace(".", "p")
        subdir = out_dir / z_name
        subdir.mkdir(parents=True, exist_ok=True)

        _, amp, phase = backprop_reconstruct(
            intensity=intensity,
            z=z,
            wavelength=wavelength,
            pixel_size=pixel_size,
            input_mode=input_mode,
            normalize_input=normalize_input,
        )

        np.save(subdir / "amplitude.npy", amp)
        np.save(subdir / "phase.npy", phase)
        save_preview(amp, subdir / "amplitude.png", cmap="gray")
        plt.imsave(subdir / "phase.png", phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)

        metrics = evaluate_focus(amp)
        rec = {"z_m": z, **metrics, "subdir": str(subdir)}
        records.append(rec)
        amp_maps.append(amp)
        phase_maps.append(phase)

    tg_arr = np.array([r["tenengrad"] for r in records], dtype=np.float32)
    lv_arr = np.array([r["laplacian_var"] for r in records], dtype=np.float32)
    gs_arr = np.array([r["gradient_sparsity"] for r in records], dtype=np.float32)

    tg_n = normalize01(tg_arr)
    lv_n = normalize01(lv_arr)
    gs_n = normalize01(gs_arr)
    combined = 0.45 * tg_n + 0.45 * lv_n - 0.10 * gs_n

    for r, c in zip(records, combined.tolist()):
        r["combined"] = c

    return records, amp_maps, phase_maps



def choose_best(records: list[dict[str, float | str]], metric: str) -> int:
    vals = np.array([r[metric] for r in records], dtype=np.float32)
    return int(np.argmax(vals))



def save_summary_plot(records: list[dict[str, float | str]], out_path: Path, title: str) -> None:
    z = np.array([r["z_m"] for r in records], dtype=np.float32)
    tg = np.array([r["tenengrad"] for r in records], dtype=np.float32)
    lv = np.array([r["laplacian_var"] for r in records], dtype=np.float32)
    cb = np.array([r["combined"] for r in records], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z, normalize01(tg), marker="o", label="tenengrad")
    ax.plot(z, normalize01(lv), marker="s", label="laplacian_var")
    ax.plot(z, normalize01(cb), marker="^", label="combined")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("normalized score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="两阶段 z 扫描：coarse + fine backprop reconstruction (v2)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--wavelength", type=float, required=True)
    parser.add_argument("--pixel_size", type=float, required=True)
    parser.add_argument("--input_mode", type=str, default="sqrt", choices=["sqrt", "linear"])
    parser.add_argument("--normalize_input", action="store_true")
    parser.add_argument("--select_metric", type=str, default="combined", choices=["tenengrad", "laplacian_var", "gradient_sparsity", "combined"])
    parser.add_argument("--coarse_start", type=float, default=0.020)
    parser.add_argument("--coarse_stop", type=float, default=0.030)
    parser.add_argument("--coarse_step", type=float, default=0.001)
    parser.add_argument("--fine_half_window", type=float, default=0.0015)
    parser.add_argument("--fine_step", type=float, default=0.00025)
    parser.add_argument("--skip_fine", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    intensity = load_array(input_path).astype(np.float32)
    if intensity.ndim != 2:
        raise ValueError(f"输入必须为二维，当前 shape={intensity.shape}")

    coarse_dir = out_dir / "coarse"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    coarse_z = make_range(args.coarse_start, args.coarse_stop, args.coarse_step)
    coarse_records, coarse_amps, coarse_phases = scan_once(
        intensity=intensity,
        z_values=coarse_z,
        out_dir=coarse_dir,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        input_mode=args.input_mode,
        normalize_input=args.normalize_input,
    )
    coarse_best_idx = choose_best(coarse_records, args.select_metric)
    coarse_best = coarse_records[coarse_best_idx]
    save_summary_plot(coarse_records, coarse_dir / "coarse_metrics.png", title="Coarse z scan")

    with open(coarse_dir / "z_scan_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["z_m", "tenengrad", "laplacian_var", "gradient_sparsity", "combined", "subdir"])
        writer.writeheader()
        writer.writerows(coarse_records)

    final_records = coarse_records
    final_amps = coarse_amps
    final_phases = coarse_phases
    final_best = coarse_best
    final_best_idx = coarse_best_idx
    best_scope = "coarse"

    if not args.skip_fine:
        fine_dir = out_dir / "fine"
        fine_dir.mkdir(parents=True, exist_ok=True)
        z0 = float(coarse_best["z_m"])
        fine_start = z0 - float(args.fine_half_window)
        fine_stop = z0 + float(args.fine_half_window)
        fine_z = make_range(fine_start, fine_stop, args.fine_step)

        fine_records, fine_amps, fine_phases = scan_once(
            intensity=intensity,
            z_values=fine_z,
            out_dir=fine_dir,
            wavelength=args.wavelength,
            pixel_size=args.pixel_size,
            input_mode=args.input_mode,
            normalize_input=args.normalize_input,
        )
        fine_best_idx = choose_best(fine_records, args.select_metric)
        fine_best = fine_records[fine_best_idx]
        save_summary_plot(fine_records, fine_dir / "fine_metrics.png", title="Fine z scan")

        with open(fine_dir / "z_scan_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["z_m", "tenengrad", "laplacian_var", "gradient_sparsity", "combined", "subdir"])
            writer.writeheader()
            writer.writerows(fine_records)

        final_records = fine_records
        final_amps = fine_amps
        final_phases = fine_phases
        final_best = fine_best
        final_best_idx = fine_best_idx
        best_scope = "fine"

    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_amp = final_amps[final_best_idx]
    best_phase = final_phases[final_best_idx]

    np.save(best_dir / "input_intensity.npy", intensity)
    np.save(best_dir / "amplitude.npy", best_amp)
    np.save(best_dir / "phase.npy", best_phase)

    save_input_like_source(input_path, intensity, best_dir / "input_intensity.png")
    save_preview(best_amp, best_dir / "amplitude.png", cmap="gray")
    plt.imsave(best_dir / "phase.png", best_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)

    with open(out_dir / "best_z.txt", "w", encoding="utf-8") as f:
        f.write(f"best_scope={best_scope}\n")
        f.write(f"best_metric={args.select_metric}\n")
        f.write(f"best_z_m={final_best['z_m']}\n")
        f.write(f"tenengrad={final_best['tenengrad']}\n")
        f.write(f"laplacian_var={final_best['laplacian_var']}\n")
        f.write(f"gradient_sparsity={final_best['gradient_sparsity']}\n")
        f.write(f"combined={final_best['combined']}\n")

    print(f"Done. best z = {final_best['z_m']:.6f} m by {args.select_metric} ({best_scope} scan)")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
