"""Train the supervised amplitude HRNet baseline on project-format pairs.

The script follows the settings disclosed by Ren, Xu, and Lam (Advanced
Photonics 1, 016004, 2019): pixelwise MSE, Adam, batch size 10, initial learning
rate 0.01 with exponential decay 0.9, an 80:10:10 split, and 25 epochs (up to
20,000 minibatch updates).  The original experimental labels were manually
cleaned conventional reconstructions and are not publicly available.  For this
project, the corresponding fair supervised labels are ``gt_A_main.npy`` files
from ``run_simulate_hologram.py``; the inputs are ``sim_hologram.npy`` files.

Examples
--------
Generate project-matched training pairs, then train::

    python HRNet_train.py --data_root data/hrnet_training --generate_count 10000

Train from already prepared pairs::

    python HRNet_train.py --data_root data/hrnet_training

Short pipeline test::

    python HRNet_train.py --data_root .codex_tmp/hrnet_data --generate_count 8 \
        --height 64 --width 64 --batch_size 2 --epochs 1 --max_iterations 2 \
        --checkpoint_dir .codex_tmp/hrnet_checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from HRNet import (
    REFERENCE_CONV_WEIGHT_COUNT,
    HolographicReconstructionNet,
    checkpoint_model_config,
    initialize_as_paper,
    load_2d_array,
    normalize_hologram,
    seed_everything,
)


@dataclass(frozen=True)
class PairRecord:
    input_path: str
    target_path: str
    sample_id: str
    patch_id: str


def discover_pairs(data_root: str, input_name: str, target_name: str) -> List[PairRecord]:
    """Recursively discover project-format paired arrays."""
    root = Path(data_root)
    if not root.exists():
        return []
    records: List[PairRecord] = []
    for input_path in sorted(root.rglob(input_name)):
        target_path = input_path.with_name(target_name)
        if not target_path.is_file():
            continue
        patch_id = input_path.parent.name
        sample_id = input_path.parent.parent.name
        records.append(
            PairRecord(
                input_path=str(input_path.resolve()),
                target_path=str(target_path.resolve()),
                sample_id=sample_id,
                patch_id=patch_id,
            )
        )
    return records


def generate_project_pairs(
    data_root: str,
    count: int,
    height: int,
    width: int,
    wavelength: float,
    pixel_size: float,
    z_distance: float,
    seed: int,
    n_objects: int,
    min_gap_px: int,
    overwrite: bool,
) -> None:
    """Generate compact paired files with the existing project simulator.

    Only arrays and metadata needed by HRNet are saved.  The optical/object/
    background/parasitic models are imported directly from
    ``run_simulate_hologram.py`` so the generated distribution remains aligned
    with the manuscript's matched simulation.
    """
    if count <= 0:
        return
    from simulate_hologram import (
        add_noise_and_normalize,
        build_object_field,
        build_parasitic_sensor_field,
        build_smooth_background,
        propagate_asm_np,
    )

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[HRNet:data] generating {count} paired samples in {root}")
    for index in range(count):
        sample_seed = int(seed + index)
        sample_id = f"sample_sim_hrnet_{index + 1:06d}"
        out_dir = root / sample_id / "patch_0001"
        input_path = out_dir / "sim_hologram.npy"
        target_path = out_dir / "gt_A_main.npy"
        if not overwrite and input_path.is_file() and target_path.is_file():
            continue

        rng = np.random.default_rng(sample_seed)
        obj = build_object_field(
            height,
            width,
            rng,
            n_objects=n_objects,
            min_gap_px=min_gap_px,
        )
        bg = build_smooth_background(height, width, rng)
        para = build_parasitic_sensor_field(height, width, rng, n_carriers=2)
        object_background_field = (obj["U_obj"] * bg["U_bg"]).astype(np.complex64)
        main_sensor_field = propagate_asm_np(
            object_background_field,
            z_distance,
            wavelength,
            pixel_size,
        )
        sensor_field = (main_sensor_field + para["U_para_sensor"]).astype(np.complex64)
        clean_intensity = np.abs(sensor_field).astype(np.float32) ** 2
        hologram = add_noise_and_normalize(
            clean_intensity,
            rng=rng,
            poisson_scale=650.0,
            gauss_sigma=0.012,
        )
        target_amplitude = np.abs(object_background_field).astype(np.float32)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(input_path, hologram.astype(np.float32))
        np.save(target_path, target_amplitude.astype(np.float32))
        np.save(out_dir / "gt_obj_mask.npy", obj["obj_mask"].astype(np.uint8))
        with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seed": sample_seed,
                    "H": int(height),
                    "W": int(width),
                    "wavelength": float(wavelength),
                    "pixel_size": float(pixel_size),
                    "z_distance": float(z_distance),
                    "n_objects": int(n_objects),
                    "min_gap_px": int(min_gap_px),
                    "input": "sim_hologram.npy",
                    "supervised_target": "gt_A_main.npy",
                    "forward_model": "U_sensor = P(U_obj * U_bg) + U_parasitic_sensor",
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        if (index + 1) % max(1, min(100, count)) == 0 or index + 1 == count:
            print(f"[HRNet:data] {index + 1}/{count}")


def split_records(
    records: Sequence[PairRecord],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
    """Deterministic train/validation/test split with at least one item per set."""
    n = len(records)
    if n < 3:
        raise ValueError(f"At least three paired samples are required; found {n}.")
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must leave non-empty train, validation, and test sets.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [records[int(i)] for i in order]
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * (1.0 - train_fraction - val_fraction))))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


class PairedHologramDataset(Dataset):
    """Single-channel hologram/amplitude dataset without sample-specific tuning."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        input_normalization: str,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.input_normalization = input_normalization
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        hologram = normalize_hologram(load_2d_array(record.input_path), self.input_normalization)
        target = load_2d_array(record.target_path)
        if hologram.shape != target.shape:
            raise ValueError(
                f"Input/target shape mismatch for {record.sample_id}/{record.patch_id}: "
                f"{hologram.shape} versus {target.shape}"
            )
        h, w = hologram.shape
        if h % 8 or w % 8:
            raise ValueError(
                f"Training dimensions must be divisible by 8; got {(h, w)} for {record.input_path}"
            )
        if self.augment:
            k = int(np.random.randint(0, 4))
            hologram = np.rot90(hologram, k)
            target = np.rot90(target, k)
            if bool(np.random.randint(0, 2)):
                hologram = np.flip(hologram, axis=1)
                target = np.flip(target, axis=1)
        hologram = np.ascontiguousarray(hologram, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        return {
            "hologram": torch.from_numpy(hologram[None]),
            "target": torch.from_numpy(target[None]),
            "sample_id": record.sample_id,
            "patch_id": record.patch_id,
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_loader(
    records: Sequence[PairRecord],
    input_normalization: str,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PairedHologramDataset(records, input_normalization, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    for batch in loader:
        hologram = batch["hologram"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = model(hologram)
        squared_error += float(torch.sum((prediction - target) ** 2).item())
        pixel_count += int(target.numel())
    return squared_error / max(pixel_count, 1)


def save_split_manifest(
    path: Path,
    train_records: Sequence[PairRecord],
    val_records: Sequence[PairRecord],
    test_records: Sequence[PairRecord],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": [asdict(x) for x in train_records],
                "validation": [asdict(x) for x in val_records],
                "test": [asdict(x) for x in test_records],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_val_mse: float,
    args: argparse.Namespace,
    test_mse: Optional[float] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_mse": float(best_val_mse),
        "test_mse": None if test_mse is None else float(test_mse),
        "model_config": checkpoint_model_config(),
        "input_normalization": args.input_normalization,
        "supervised_target": args.target_name,
        "training_args": vars(args),
        "paper_reference": "Ren, Xu, and Lam, Adv. Photonics 1, 016004 (2019)",
    }
    torch.save(payload, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def write_history(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the supervised amplitude HRNet baseline on project-format paired data."
    )
    parser.add_argument("--data_root", default=os.path.join("data", "hrnet_training"))
    parser.add_argument("--input_name", default="sim_hologram.npy")
    parser.add_argument("--target_name", default="gt_A_main.npy")
    parser.add_argument("--checkpoint_dir", default=os.path.join("checkpoints", "HRNet"))
    parser.add_argument("--resume", default=None, help="Checkpoint to resume; omitted starts a new run.")

    parser.add_argument("--generate_count", type=int, default=0)
    parser.add_argument("--overwrite_generated", action="store_true")
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--wavelength", type=float, default=632.8e-9)
    parser.add_argument("--pixel_size", type=float, default=6.9e-6)
    parser.add_argument("--z_distance", type=float, default=0.021)
    parser.add_argument("--n_objects", type=int, default=3)
    parser.add_argument("--min_gap_px", type=int, default=18)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max_iterations", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--input_normalization", choices=("minmax", "standard", "none"), default="minmax")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", help="Optional paired rotations/flips; off matches the paper.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_parser().parse_args(argv)
    if args.height % 8 or args.width % 8:
        raise ValueError("--height and --width must be divisible by 8.")
    seed_everything(args.seed)

    generate_project_pairs(
        data_root=args.data_root,
        count=args.generate_count,
        height=args.height,
        width=args.width,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        z_distance=args.z_distance,
        seed=args.seed,
        n_objects=args.n_objects,
        min_gap_px=args.min_gap_px,
        overwrite=args.overwrite_generated,
    )
    if args.generate_only:
        return {"generated": int(args.generate_count), "data_root": args.data_root}

    records = discover_pairs(args.data_root, args.input_name, args.target_name)
    if not records:
        raise FileNotFoundError(
            f"No {args.input_name}/{args.target_name} pairs were found under {args.data_root}. "
            "Provide paired data or use --generate_count."
        )
    train_records, val_records, test_records = split_records(
        records,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(
        checkpoint_dir / "split_manifest.json",
        train_records,
        val_records,
        test_records,
    )
    with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    device = resolve_device(args.device)
    train_loader = make_loader(
        train_records,
        args.input_normalization,
        args.batch_size,
        shuffle=True,
        augment=args.augment,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    val_loader = make_loader(
        val_records,
        args.input_normalization,
        args.batch_size,
        shuffle=False,
        augment=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        device=device,
    )
    test_loader = make_loader(
        test_records,
        args.input_normalization,
        args.batch_size,
        shuffle=False,
        augment=False,
        num_workers=args.num_workers,
        seed=args.seed + 2,
        device=device,
    )

    model = HolographicReconstructionNet().to(device)
    if model.convolutional_weight_count() != REFERENCE_CONV_WEIGHT_COUNT:
        raise RuntimeError("HRNet convolutional parameter count does not match Table 4.")
    initialize_as_paper(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    criterion = nn.MSELoss(reduction="mean")

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    global_step = 0
    best_val_mse = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val_mse = float(checkpoint["best_val_mse"])
        checkpoint_norm = checkpoint.get("input_normalization", args.input_normalization)
        if checkpoint_norm != args.input_normalization:
            raise ValueError(
                f"Checkpoint uses input_normalization={checkpoint_norm}, "
                f"but command specifies {args.input_normalization}."
            )

    print("[HRNet] supervised amplitude training")
    print(f"  device              : {device}")
    print(f"  pairs               : {len(records)}")
    print(f"  train/val/test      : {len(train_records)}/{len(val_records)}/{len(test_records)}")
    print(f"  convolution weights : {model.convolutional_weight_count():,}")
    print(f"  total trainable     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  checkpoint directory: {checkpoint_dir}")

    history: List[Dict[str, object]] = []
    stop = global_step >= args.max_iterations
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        epoch_start = time.time()
        model.train()
        epoch_sum = 0.0
        epoch_pixels = 0
        for batch in train_loader:
            hologram = batch["hologram"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(hologram)
                loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_sum += float(torch.sum((prediction.detach() - target) ** 2).item())
            epoch_pixels += int(target.numel())
            if global_step % args.log_interval == 0 or global_step == 1:
                print(
                    f"  epoch {epoch + 1:02d}/{args.epochs:02d} | "
                    f"step {global_step:05d}/{args.max_iterations:05d} | "
                    f"MSE {float(loss.item()):.7f} | lr {optimizer.param_groups[0]['lr']:.3e}"
                )
            if global_step >= args.max_iterations:
                stop = True
                break

        train_mse = epoch_sum / max(epoch_pixels, 1)
        val_mse = evaluate(model, val_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "learning_rate": current_lr,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "seconds": elapsed,
        }
        history.append(row)
        write_history(checkpoint_dir / "history.csv", history)
        # Advance once per completed epoch.  Saving after this step makes a
        # resumed run start with the same learning rate as an uninterrupted run.
        scheduler.step()
        save_checkpoint(
            checkpoint_dir / "hrnet_amplitude_last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            min(best_val_mse, val_mse),
            args,
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_checkpoint(
                checkpoint_dir / "hrnet_amplitude_best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_val_mse,
                args,
            )
        print(
            f"[HRNet] epoch {epoch + 1}: train MSE={train_mse:.7f}, "
            f"val MSE={val_mse:.7f}, time={elapsed:.1f}s"
        )

    best_path = checkpoint_dir / "hrnet_amplitude_best.pt"
    if not best_path.is_file():
        raise RuntimeError("Training ended before a best checkpoint could be written.")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_mse = evaluate(model, test_loader, device)
    best_checkpoint["test_mse"] = float(test_mse)
    torch.save(best_checkpoint, best_path)
    with open(checkpoint_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "test_mse": float(test_mse),
                "best_val_mse": float(best_checkpoint["best_val_mse"]),
                "test_samples": len(test_records),
                "checkpoint": str(best_path.resolve()),
            },
            handle,
            indent=2,
        )
    print(f"[HRNet] test MSE: {test_mse:.7f}")
    print(f"[HRNet] best checkpoint: {best_path}")
    return {
        "checkpoint": str(best_path),
        "best_val_mse": float(best_checkpoint["best_val_mse"]),
        "test_mse": float(test_mse),
        "global_step": int(best_checkpoint["global_step"]),
    }


if __name__ == "__main__":
    main()
