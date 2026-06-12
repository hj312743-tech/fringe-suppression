import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


# Basic utilities
def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def robust_normalize01(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Robustly normalize an image to [0, 1] using percentile clipping."""
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        hi = lo + 1e-6

    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def load_array(path: str) -> np.ndarray:
    """Load a 2D grayscale image or NumPy array as float32."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'File does not exist: {path}')

    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path).convert('L'), dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f'Only 2D single-channel images are supported, got shape={arr.shape}, path={path}')

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def make_valid_mask(h: int, w: int, border: int) -> np.ndarray:
    """Create a valid evaluation region by excluding image borders."""
    m = np.zeros((h, w), dtype=bool)
    if border * 2 >= h or border * 2 >= w:
        raise ValueError(f'eval_border is too large for image size {(h, w)}: border={border}')
    m[border:h-border, border:w-border] = True
    return m

# Evaluation-mask lookup
def find_mask_pair(mask_root: str, sample_id: str, patch_id: str):
    """Find the object and background masks for a given sample/patch pair."""
    obj_candidates = [
        os.path.join(mask_root, sample_id, f'{patch_id}_obj.npy'),
        os.path.join(mask_root, sample_id, patch_id, 'obj_mask.npy'),
        os.path.join(mask_root, sample_id, patch_id, f'{patch_id}_obj.npy'),
    ]
    bg_candidates = [
        os.path.join(mask_root, sample_id, f'{patch_id}_bg.npy'),
        os.path.join(mask_root, sample_id, patch_id, 'bg_mask.npy'),
        os.path.join(mask_root, sample_id, patch_id, f'{patch_id}_bg.npy'),
    ]

    obj_path, bg_path = None, None

    for p in obj_candidates:
        if os.path.exists(p):
            obj_path = p
            break

    for p in bg_candidates:
        if os.path.exists(p):
            bg_path = p
            break

    if obj_path is None or bg_path is None:
        raise FileNotFoundError(f'Mask pair not found for {sample_id}/{patch_id}')

    return obj_path, bg_path


# Metrics
def compute_bg_cv(
    table1_img: np.ndarray,
    bg_mask: np.ndarray,
    valid_eval: np.ndarray,
    use_robust_norm: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> float:
    """
    Compute background coefficient of variation (Bg-CV).

    Bg-CV = std / mean over the background evaluation region.
    Robust normalization is applied by default so that methods with
    different output scales can be compared under the same metric.
    """
    x = robust_normalize01(table1_img, p_low=p_low, p_high=p_high) if use_robust_norm else table1_img

    region = (bg_mask > 0.5) & valid_eval
    vals = x[region]
    vals = vals[np.isfinite(vals)]

    if vals.size < 10:
        return np.nan

    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))
    return float(std_v / (mean_v + 1e-8))


def compute_obj_grad(
    table1_img: np.ndarray,
    obj_mask: np.ndarray,
    valid_eval: np.ndarray,
    bg_detrend_sigma: float = 8.0,
    edge_width: int = 1,
    top_percent: float = 30.0,
    use_robust_norm: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> float:
    """
    Compute object-boundary gradient (Obj-Grad).

    Steps:
    1. Robustly normalize the reconstructed amplitude.
    2. Remove the slow background trend with Gaussian smoothing.
    3. Compute the Sobel gradient magnitude.
    4. Average the top-percent gradient responses in the boundary ring.
    """
    x = robust_normalize01(table1_img, p_low=p_low, p_high=p_high) if use_robust_norm else normalize01(table1_img)

    low = ndi.gaussian_filter(x, sigma=bg_detrend_sigma)
    x_det = x - low

    gx = ndi.sobel(x_det, axis=1)
    gy = ndi.sobel(x_det, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    obj = (obj_mask > 0.5) & valid_eval
    if np.sum(obj) == 0:
        return np.nan

    dil = ndi.binary_dilation(obj, iterations=edge_width)
    ero = ndi.binary_erosion(obj, iterations=edge_width)
    boundary_ring = (dil ^ ero) & valid_eval

    vals = grad[boundary_ring]
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return np.nan

    keep_thr = np.percentile(vals, 100.0 - top_percent)
    top_vals = vals[vals >= keep_thr]
    if top_vals.size == 0:
        return np.nan

    return float(np.mean(top_vals))


def evaluate_one(
    table1_img: np.ndarray,
    obj_mask: np.ndarray,
    bg_mask: np.ndarray,
    eval_border: int = 16,
    objgrad_bg_sigma: float = 8.0,
    objgrad_edge_width: int = 1,
    objgrad_top_percent: float = 30.0,
    bgcv_use_robust_norm: bool = True,
    bgcv_p_low: float = 1.0,
    bgcv_p_high: float = 99.0,
    objgrad_use_robust_norm: bool = True,
    objgrad_p_low: float = 1.0,
    objgrad_p_high: float = 99.0,
):
    h, w = table1_img.shape
    valid_eval = make_valid_mask(h, w, border=eval_border)

    return {
        'Bg_CV': compute_bg_cv(
            table1_img=table1_img,
            bg_mask=bg_mask,
            valid_eval=valid_eval,
            use_robust_norm=bgcv_use_robust_norm,
            p_low=bgcv_p_low,
            p_high=bgcv_p_high,
        ),
        'Obj_Grad': compute_obj_grad(
            table1_img=table1_img,
            obj_mask=obj_mask,
            valid_eval=valid_eval,
            bg_detrend_sigma=objgrad_bg_sigma,
            edge_width=objgrad_edge_width,
            top_percent=objgrad_top_percent,
            use_robust_norm=objgrad_use_robust_norm,
            p_low=objgrad_p_low,
            p_high=objgrad_p_high,
        ),
        'Eval_Border': eval_border,
    }


# Method registry
# key: internal method key / output-folder family
# display: method name shown in terminal and CSV files
METHOD_META = {
    'BP': {
        'display': 'BP',
        'category': 'Reconstruction',
        'order': 0,
        'folder_candidates': ['BP'],
    },
    'CS': {
        'display': 'CS',
        'category': 'Reconstruction',
        'order': 1,
        'folder_candidates': ['CS'],
    },
    'BP_FFT_Notch': {
        'display': 'BP+FFT Notch',
        'category': 'Post-processing',
        'order': 2,
        'folder_candidates': ['BP_FFT_Notch'],
    },
    'BM3D': {
        'display': 'BM3D',
        'category': 'Post-processing',
        'order': 3,
        'folder_candidates': ['BM3D'],
    },
    'U-Net-PC': {
        'display': 'U-Net-PC',
        'category': 'Physics-driven baseline',
        'order': 4,
        'folder_candidates': ['U-Net-PC'],
    },
    'DIP-RED-TV': {
        'display': 'DIP-RED-TV',
        'category': 'Physics-driven baseline',
        'order': 5,
        'folder_candidates': ['DIP-RED-TV'],
    },
    'Ours_objectOnly': {
        'display': 'Single-model',
        'category': 'Ablation baseline',
        'order': 6,
        'folder_candidates': ['Ours_objectOnly', 'Single-model'],
    },
    'Ours': {
        'display': 'Ours',
        'category': 'Physics-driven',
        'order': 7,
        'folder_candidates': ['Ours'],
    },
}


DEFAULT_METHOD_KEYS = [
    'BP',
    'CS',
    'BP_FFT_Notch',
    'BM3D',
    'U-Net-PC',
    'DIP-RED-TV',
    'Ours_objectOnly',
    'Ours',
]

# Output-directory scanning
def should_exclude_sample(sample_id: str, exclude_prefixes):
    """Return True if the sample ID matches one of the excluded prefixes."""
    for p in exclude_prefixes:
        if p and sample_id.startswith(p):
            return True
    return False


def resolve_existing_method_root(outputs_root: str, method_key: str):
    """Resolve the existing output directory for a method key."""
    meta = METHOD_META[method_key]
    for folder_name in meta['folder_candidates']:
        candidate = os.path.join(outputs_root, folder_name)
        if os.path.isdir(candidate):
            return candidate, folder_name
    return None, None


def collect_method_patch_dirs(outputs_root: str, method_keys=None, exclude_prefixes=None):
    """Collect all patch directories that contain table1_img.npy."""
    items = []
    method_keys = method_keys or DEFAULT_METHOD_KEYS
    exclude_prefixes = exclude_prefixes or []

    for method_key in method_keys:
        if method_key not in METHOD_META:
            continue

        method_root, folder_name = resolve_existing_method_root(outputs_root, method_key)
        if method_root is None:
            continue

        meta = METHOD_META[method_key]

        for sample_id in sorted(os.listdir(method_root)):
            if should_exclude_sample(sample_id, exclude_prefixes):
                continue

            sample_dir = os.path.join(method_root, sample_id)
            if not os.path.isdir(sample_dir):
                continue

            for patch_id in sorted(os.listdir(sample_dir)):
                patch_dir = os.path.join(sample_dir, patch_id)
                if not os.path.isdir(patch_dir):
                    continue

                table1_path = os.path.join(patch_dir, 'table1_img.npy')
                if os.path.exists(table1_path):
                    items.append({
                        'method_key': method_key,
                        'method': meta['display'],
                        'category': meta['category'],
                        'order': meta['order'],
                        'folder_name': folder_name,
                        'sample_id': sample_id,
                        'patch_id': patch_id,
                        'patch_dir': patch_dir,
                        'table1_path': table1_path,
                    })

    return items


# Terminal output helpers
def fmt_metric(x):
    return 'nan' if np.isnan(x) else f'{x:.5f}'


def print_header(title: str):
    line = '=' * 96
    print(f'\n{line}\n{title}\n{line}')


def print_config(args, num_items):
    print_header('Real-sample evaluation config')
    print(f'outputs_root             : {args.outputs_root}')
    print(f'mask_root                : {args.mask_root}')
    print(f'eval_border              : {args.eval_border}')
    print(f'objgrad_bg_sigma         : {args.objgrad_bg_sigma}')
    print(f'objgrad_edge_width       : {args.objgrad_edge_width}')
    print(f'objgrad_top_percent      : {args.objgrad_top_percent}')
    print(f'bgcv_use_robust_norm     : {args.bgcv_use_robust_norm}')
    print(f'bgcv_p_low / p_high      : {args.bgcv_p_low} / {args.bgcv_p_high}')
    print(f'objgrad_use_robust_norm  : {args.objgrad_use_robust_norm}')
    print(f'objgrad_p_low / p_high   : {args.objgrad_p_low} / {args.objgrad_p_high}')
    print(f'exclude_prefixes         : {", ".join(args.exclude_sample_prefixes) if args.exclude_sample_prefixes else "(none)"}')
    print(f'method_keys              : {", ".join(args.methods)}')
    print(f'num_items                : {num_items}\n')
    print(f'{"Idx":<9} {"Method":<18} {"Sample/Patch":<28} {"Bg_CV":>10} {"Obj_Grad":>12}  Status')
    print('-' * 96)


# Main entry point
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate real-sample metrics using Bg-CV and Obj-Grad within a unified valid region.'
    )

    parser.add_argument('--outputs_root', type=str, default='outputs')
    parser.add_argument('--mask_root', type=str, default=os.path.join('data', 'masks'))
    parser.add_argument('--detail_csv', type=str, default=os.path.join('results', 'table1_real_details.csv'))
    parser.add_argument('--summary_csv', type=str, default=os.path.join('results', 'table1_real_summary.csv'))

    parser.add_argument('--eval_border', type=int, default=8, help='Number of border pixels excluded from evaluation')
    parser.add_argument('--objgrad_bg_sigma', type=float, default=8.0)
    parser.add_argument('--objgrad_edge_width', type=int, default=1)
    parser.add_argument('--objgrad_top_percent', type=float, default=30.0)

    # Bg-CV normalization
    parser.add_argument('--bgcv_use_robust_norm', action='store_true', default=True)
    parser.add_argument('--bgcv_p_low', type=float, default=1.0)
    parser.add_argument('--bgcv_p_high', type=float, default=99.0)

    # Obj-Grad normalization
    parser.add_argument('--objgrad_use_robust_norm', action='store_true', default=True)
    parser.add_argument('--objgrad_p_low', type=float, default=1.0)
    parser.add_argument('--objgrad_p_high', type=float, default=99.0)

    parser.add_argument(
        '--methods',
        nargs='*',
        default=DEFAULT_METHOD_KEYS,
        help='Method keys to evaluate. Available keys: ' + ', '.join(DEFAULT_METHOD_KEYS),
    )
    parser.add_argument(
        '--exclude_sample_prefixes',
        nargs='*',
        default=['sample_sim_'],
        help='Sample-ID prefixes excluded from real-data evaluation, e.g., simulation outputs.',
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.detail_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    items = collect_method_patch_dirs(
        args.outputs_root,
        args.methods,
        exclude_prefixes=args.exclude_sample_prefixes,
    )
    if len(items) == 0:
        print('No real-sample table1_img.npy files were found for evaluation.')
        return

    print_config(args, len(items))

    records = []
    failed = []
    total = len(items)

    for idx, item in enumerate(items, start=1):
        method_key = item['method_key']
        method = item['method']
        category = item['category']
        sample_id = item['sample_id']
        patch_id = item['patch_id']

        try:
            table1_img = load_array(item['table1_path'])
            obj_mask_path, bg_mask_path = find_mask_pair(args.mask_root, sample_id, patch_id)
            obj_mask = load_array(obj_mask_path)
            bg_mask = load_array(bg_mask_path)

            if not (table1_img.shape == obj_mask.shape == bg_mask.shape):
                raise ValueError(
                    f'Shape mismatch: table1={table1_img.shape}, obj_mask={obj_mask.shape}, bg_mask={bg_mask.shape}'
                )

            metrics = evaluate_one(
                table1_img=table1_img,
                obj_mask=obj_mask,
                bg_mask=bg_mask,
                eval_border=args.eval_border,
                objgrad_bg_sigma=args.objgrad_bg_sigma,
                objgrad_edge_width=args.objgrad_edge_width,
                objgrad_top_percent=args.objgrad_top_percent,
                bgcv_use_robust_norm=args.bgcv_use_robust_norm,
                bgcv_p_low=args.bgcv_p_low,
                bgcv_p_high=args.bgcv_p_high,
                objgrad_use_robust_norm=args.objgrad_use_robust_norm,
                objgrad_p_low=args.objgrad_p_low,
                objgrad_p_high=args.objgrad_p_high,
            )

            rec = {
                'method_key': method_key,
                'method': method,
                'category': category,
                'sample_id': sample_id,
                'patch_id': patch_id,
                'eval_border': args.eval_border,
                'bgcv_use_robust_norm': bool(args.bgcv_use_robust_norm),
                'bgcv_p_low': args.bgcv_p_low,
                'bgcv_p_high': args.bgcv_p_high,
                'objgrad_bg_sigma': args.objgrad_bg_sigma,
                'objgrad_edge_width': args.objgrad_edge_width,
                'objgrad_top_percent': args.objgrad_top_percent,
                'objgrad_use_robust_norm': bool(args.objgrad_use_robust_norm),
                'objgrad_p_low': args.objgrad_p_low,
                'objgrad_p_high': args.objgrad_p_high,
            }
            rec.update(metrics)
            records.append(rec)

            print(
                f'[{idx:02d}/{total:02d}]'.ljust(9) +
                f'{method:<18} ' +
                f'{sample_id}/{patch_id:<28} ' +
                f'{fmt_metric(rec["Bg_CV"]):>10} ' +
                f'{fmt_metric(rec["Obj_Grad"]):>12}  OK'
            )

        except Exception as e:
            failed.append({
                'idx': idx,
                'method': method,
                'sample_id': sample_id,
                'patch_id': patch_id,
                'reason': str(e),
            })
            print(
                f'[{idx:02d}/{total:02d}]'.ljust(9) +
                f'{method:<18} ' +
                f'{sample_id}/{patch_id:<28} ' +
                f'{"-":>10} {"-":>12}  SKIP'
            )

    if len(records) == 0:
        print('\nNo records were successfully evaluated.')
        if failed:
            print('\nSkip reasons:')
            for x in failed:
                print(f'  - {x["method"]} | {x["sample_id"]}/{x["patch_id"]} | {x["reason"]}')
        return

    detail_df = pd.DataFrame(records)
    detail_df['method_order'] = detail_df['method_key'].map(lambda x: METHOD_META.get(x, {}).get('order', 999))
    detail_df = detail_df.sort_values(by=['method_order', 'sample_id', 'patch_id']).drop(columns=['method_order'])
    detail_df.to_csv(args.detail_csv, index=False, encoding='utf-8-sig')

    metric_cols = ['Bg_CV', 'Obj_Grad']
    summary_rows = []

    for method_key, subdf in detail_df.groupby('method_key'):
        meta = METHOD_META.get(method_key, {})
        row = {
            'method_key': method_key,
            'method': meta.get('display', method_key),
            'category': meta.get('category', 'Unknown'),
            'n': len(subdf),
            'eval_border': args.eval_border,
            'bgcv_use_robust_norm': bool(args.bgcv_use_robust_norm),
            'bgcv_p_low': args.bgcv_p_low,
            'bgcv_p_high': args.bgcv_p_high,
            'objgrad_bg_sigma': args.objgrad_bg_sigma,
            'objgrad_edge_width': args.objgrad_edge_width,
            'objgrad_top_percent': args.objgrad_top_percent,
            'objgrad_use_robust_norm': bool(args.objgrad_use_robust_norm),
            'objgrad_p_low': args.objgrad_p_low,
            'objgrad_p_high': args.objgrad_p_high,
        }
        for m in metric_cols:
            vals = subdf[m].values.astype(np.float32)
            row[f'{m}_mean'] = float(np.nanmean(vals))
            row[f'{m}_std'] = float(np.nanstd(vals))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df['method_order'] = summary_df['method_key'].map(lambda x: METHOD_META.get(x, {}).get('order', 999))
    summary_df = summary_df.sort_values(by=['method_order']).drop(columns=['method_order'])
    summary_df.to_csv(args.summary_csv, index=False, encoding='utf-8-sig')

    print_header('Real-sample evaluation finished')
    print(f'success / total : {len(records)} / {total}')
    print(f'failed          : {len(failed)}')
    print(f'Detail CSV      : {args.detail_csv}')
    print(f'Summary CSV     : {args.summary_csv}')

    if failed:
        print('\nSkip reasons (compact):')
        for x in failed:
            print(f'  - {x["method"]:<18} {x["sample_id"]}/{x["patch_id"]} | {x["reason"]}')

    print('\nSummary (mean +/- std):')
    for _, row in summary_df.iterrows():
        print(
            f'  {row["method"]:<18} | '
            f'Bg_CV={row["Bg_CV_mean"]:.5f}+/-{row["Bg_CV_std"]:.5f} | '
            f'Obj_Grad={row["Obj_Grad_mean"]:.5f}+/-{row["Obj_Grad_std"]:.5f} | '
            f'n={int(row["n"])}'
        )


if __name__ == '__main__':
    main()
