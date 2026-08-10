"""Run a trained amplitude HRNet and write the project's evaluation interface.

For each input patch, the script writes

    outputs/HRNet/<sample_id>/<patch_id>/table1_img.npy

along with ``clean_amp.npy``, the normalized input, a conventional
back-propagation reference, metadata, and optional PNG previews.  HRNet predicts
amplitude only.  Consequently, ``I_pred.npy`` is provided solely as a common
diagnostic: it is the ASM intensity obtained after assigning zero phase to the
predicted amplitude, and is not a native HRNet output or a training constraint.

Single-patch example::

    python HRNet_inference.py --input data/sample_007/patch_0013/patch_0013.npy

Batch example for project-format ``.../<sample>/<patch>/<patch>.npy`` files::

    python HRNet_inference.py --input_root data --skip_existing
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.fft as fft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from HRNet import (
    REFERENCE_CONV_WEIGHT_COUNT,
    HolographicReconstructionNet,
    crop_padding,
    infer_sample_patch_ids,
    load_2d_array,
    normalize01,
    normalize_hologram,
    pad_to_multiple,
    seed_everything,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def propagate_asm(
    field: torch.Tensor,
    z: float,
    wavelength: float,
    pixel_size: float,
) -> torch.Tensor:
    """Angular-spectrum propagation for [B,1,H,W] complex fields."""
    _, _, height, width = field.shape
    fx = fft.fftfreq(width, d=pixel_size, device=field.device)
    fy = fft.fftfreq(height, d=pixel_size, device=field.device)
    fx_grid, fy_grid = torch.meshgrid(fx, fy, indexing="xy")
    wave_number = 2.0 * torch.pi / wavelength
    term = 1.0 - (wavelength * fx_grid) ** 2 - (wavelength * fy_grid) ** 2
    phase = wave_number * z * torch.sqrt(torch.clamp(term, min=0.0))
    transfer = torch.exp(1j * phase)
    return fft.ifft2(fft.fft2(field) * transfer)


def percentile_limits(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    lo = float(np.percentile(array, low))
    hi = float(np.percentile(array, high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def save_image(
    array: np.ndarray,
    path: Path,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if vmin is None or vmax is None:
        vmin, vmax = percentile_limits(array)
    fig, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.axis("off")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def discover_project_inputs(root: str) -> List[str]:
    """Resolve one file or discover real/simulated project holograms.

    Accepted inputs are:
    - a directly supplied ``.npy`` or image file;
    - real-data arrays named ``<patch>/<patch>.npy``;
    - simulated inputs named ``sim_hologram.npy``.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if root_path.is_file():
        if root_path.suffix.lower() not in {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            raise ValueError(f"Unsupported hologram file type: {root_path}")
        return [str(root_path)]

    paths: List[str] = []
    for path in sorted(root_path.rglob("*.npy")):
        if path.stem == path.parent.name or path.name == "sim_hologram.npy":
            paths.append(str(path))
    return paths


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[HolographicReconstructionNet, Dict[str, object]]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"HRNet checkpoint not found: {checkpoint_path}. "
            "Run HRNet_train.py first or pass --checkpoint."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state: {checkpoint_path}")
    model = HolographicReconstructionNet().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.convolutional_weight_count() != REFERENCE_CONV_WEIGHT_COUNT:
        raise RuntimeError("Loaded HRNet architecture does not match the paper's Table 4.")
    model.eval()
    return model, checkpoint


@torch.no_grad()
def infer_one(
    model: HolographicReconstructionNet,
    hologram: np.ndarray,
    input_normalization: str,
    device: torch.device,
    z_distance: float,
    wavelength: float,
    pixel_size: float,
    use_amp: bool,
) -> Dict[str, np.ndarray | float | Tuple[int, int, int, int]]:
    normalized = normalize_hologram(hologram, input_normalization)
    tensor = torch.from_numpy(normalized[None, None]).to(device=device, dtype=torch.float32)
    padded, padding = pad_to_multiple(tensor, multiple=8)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
        predicted_padded = model(padded)
    prediction = crop_padding(predicted_padded.float(), padding)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start

    sensor_field = torch.sqrt(torch.clamp(tensor, min=0.0)).to(torch.complex64)
    bp_field = propagate_asm(sensor_field, -z_distance, wavelength, pixel_size)
    base_amp = torch.abs(bp_field)
    base_phase = torch.angle(bp_field)

    # HRNet has no phase output.  This zero-phase field is used only to preserve
    # the project's optional I_pred diagnostic, never for scoring or training.
    zero_phase_field = prediction.to(torch.complex64)
    diagnostic_sensor = propagate_asm(zero_phase_field, z_distance, wavelength, pixel_size)
    diagnostic_intensity = torch.abs(diagnostic_sensor) ** 2

    predicted_np = prediction.squeeze().cpu().numpy().astype(np.float32)
    i_pred_np = diagnostic_intensity.squeeze().cpu().numpy().astype(np.float32)
    error_map = np.abs(normalize01(i_pred_np) - normalize01(normalized)).astype(np.float32)
    return {
        "table1_img": predicted_np,
        "clean_amp": predicted_np,
        "input_hologram": normalized.astype(np.float32),
        "I_pred": i_pred_np,
        "base_amp": base_amp.squeeze().cpu().numpy().astype(np.float32),
        "base_phase": base_phase.squeeze().cpu().numpy().astype(np.float32),
        "zero_phase_assumption": np.zeros_like(predicted_np, dtype=np.float32),
        "error_map": error_map,
        "runtime_seconds": float(runtime),
        "padding_lrtb": padding,
    }


def save_outputs(
    results: Dict[str, np.ndarray | float | Tuple[int, int, int, int]],
    out_dir: Path,
    checkpoint_path: str,
    checkpoint: Dict[str, object],
    input_path: str,
    sample_id: str,
    patch_id: str,
    input_normalization: str,
    z_distance: float,
    wavelength: float,
    pixel_size: float,
    save_png: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in {
        "table1_img.npy": "table1_img",
        "clean_amp.npy": "clean_amp",
        "input_hologram.npy": "input_hologram",
        "I_pred.npy": "I_pred",
        "base_amp.npy": "base_amp",
        "base_phase.npy": "base_phase",
        "zero_phase_assumption.npy": "zero_phase_assumption",
        "error_map.npy": "error_map",
    }.items():
        np.save(out_dir / filename, np.asarray(results[key], dtype=np.float32))

    meta = {
        "method": "HRNet",
        "full_method_name": "Holographic reconstruction network",
        "paper": "Ren, Xu, and Lam, Advanced Photonics 1, 016004 (2019)",
        "task": "supervised amplitude reconstruction",
        "sample_id": sample_id,
        "patch_id": patch_id,
        "input_path": str(Path(input_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_test_mse": checkpoint.get("test_mse"),
        "input_normalization": input_normalization,
        "table1_image": "predicted amplitude",
        "complex_field_output": False,
        "runtime_seconds": float(results["runtime_seconds"]),
        "padding_left_right_top_bottom": [int(x) for x in results["padding_lrtb"]],
        "wavelength_m": float(wavelength),
        "pixel_size_m": float(pixel_size),
        "z_distance_m": float(z_distance),
        "I_pred_definition": (
            "ASM intensity synthesized from the HRNet amplitude after assigning zero phase; "
            "diagnostic only, not a native HRNet output and not used by eval_metrics.py"
        ),
        "domain_note": (
            "HRNet is supervised; validity on experimental holograms depends on the domain "
            "represented by the paired training set."
        ),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    if save_png:
        prefix = f"HRNet_{sample_id}_{patch_id}"
        save_image(np.asarray(results["table1_img"]), out_dir / f"{prefix}_amplitude.png")
        save_image(np.asarray(results["input_hologram"]), out_dir / f"{prefix}_hologram.png")
        save_image(np.asarray(results["base_amp"]), out_dir / f"{prefix}_BP_amplitude.png")
        save_image(np.asarray(results["I_pred"]), out_dir / f"{prefix}_I_pred_diagnostic.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trained amplitude HRNet and save project-compatible evaluation outputs."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Single hologram (.npy or image). Defaults to sample_007/patch_0013.",
    )
    parser.add_argument(
        "--input_root",
        default=None,
        help=(
            "A directory containing project-format holograms, or a single input file. "
            "Directories are scanned for <patch>/<patch>.npy and sim_hologram.npy."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "HRNet", "hrnet_amplitude_best.pt"),
    )
    parser.add_argument("--out_root", default=os.path.join("outputs", "HRNet"))
    parser.add_argument("--sample_id", default=None, help="Override inferred ID in single-input mode.")
    parser.add_argument("--patch_id", default=None, help="Override inferred ID in single-input mode.")
    parser.add_argument(
        "--input_normalization",
        choices=("checkpoint", "minmax", "standard", "none"),
        default="checkpoint",
    )
    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument("--z_distance", type=float, default=0.02275)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_png", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    args = build_parser().parse_args(argv)
    if args.input and args.input_root:
        raise ValueError("Use either --input or --input_root, not both.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    checkpoint_normalization = str(checkpoint.get("input_normalization", "minmax"))
    input_normalization = (
        checkpoint_normalization
        if args.input_normalization == "checkpoint"
        else args.input_normalization
    )

    if args.input_root:
        input_paths = discover_project_inputs(args.input_root)
        if not input_paths:
            raise FileNotFoundError(
                f"No <patch>/<patch>.npy or sim_hologram.npy inputs were found under "
                f"{args.input_root}"
            )
    else:
        input_paths = [
            args.input
            or os.path.join("data", "sample_007", "patch_0013", "patch_0013.npy")
        ]

    print("[HRNet] supervised amplitude inference")
    print(f"  device              : {device}")
    print(f"  checkpoint          : {args.checkpoint}")
    print(f"  input normalization : {input_normalization}")
    print(f"  patches             : {len(input_paths)}")

    summaries: List[Dict[str, object]] = []
    for index, input_path in enumerate(input_paths, start=1):
        inferred_sample, inferred_patch = infer_sample_patch_ids(input_path)
        sample_id = args.sample_id or inferred_sample if len(input_paths) == 1 else inferred_sample
        patch_id = args.patch_id or inferred_patch if len(input_paths) == 1 else inferred_patch
        out_dir = Path(args.out_root) / sample_id / patch_id
        if args.skip_existing and (out_dir / "table1_img.npy").is_file():
            print(f"[HRNet] skip existing {sample_id}/{patch_id}")
            continue
        hologram = load_2d_array(input_path)
        results = infer_one(
            model=model,
            hologram=hologram,
            input_normalization=input_normalization,
            device=device,
            z_distance=args.z_distance,
            wavelength=args.wavelength,
            pixel_size=args.pixel_size,
            use_amp=args.amp,
        )
        save_outputs(
            results=results,
            out_dir=out_dir,
            checkpoint_path=args.checkpoint,
            checkpoint=checkpoint,
            input_path=input_path,
            sample_id=sample_id,
            patch_id=patch_id,
            input_normalization=input_normalization,
            z_distance=args.z_distance,
            wavelength=args.wavelength,
            pixel_size=args.pixel_size,
            save_png=not args.no_png,
        )
        summary = {
            "sample_id": sample_id,
            "patch_id": patch_id,
            "output_dir": str(out_dir),
            "runtime_seconds": float(results["runtime_seconds"]),
        }
        summaries.append(summary)
        print(
            f"[HRNet] {index}/{len(input_paths)} {sample_id}/{patch_id} | "
            f"{float(results['runtime_seconds']):.4f}s | {out_dir}"
        )
    return summaries


if __name__ == "__main__":
    main()
