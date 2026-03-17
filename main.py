#!/usr/bin/env python3
"""
CTF-Constrained PnP-ADMM Denoising for cryo-EM Micrographs

Usage:
    # Basic usage with full/diff pairing
    python main.py --full micrograph_full.mrc --diff micrograph_diff.mrc
    
    # Auto-detect diff file
    python main.py --full micrograph_full.mrc
    
    # Specify CTF explicitly
    python main.py --full full.mrc --diff diff.mrc --ctf ctf_params.json
    
    # Use Topaz denoiser
    python main.py --full full.mrc --diff diff.mrc --denoiser unet --cuda
    
    # Adjust ADMM parameters
    python main.py --full full.mrc --diff diff.mrc --rho 2.0 --alpha 1e-3 -T 10
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from core import pnp_admm_denoise, pnp_admm_denoise_stack
from ctf import (
    load_ctf_for_micrograph,
    pair_full_diff_files,
    parse_micrograph_filename,
)
from noise_psd import load_diff_and_compute_weight, create_uniform_weight
from mrc_io import read_mrc, read_mrc_stack, write_mrc, write_mrc_stack
from denoiser import create_denoiser, create_topaz_denoiser


def parse_args():
    parser = argparse.ArgumentParser(
        description='CTF-Constrained PnP-ADMM Denoising for cryo-EM images (SPEC.md compliant)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm Overview:
  1. Load full (even+odd) and diff (even-odd) micrographs
  2. Compute noise weight W(k) from diff: W(k) = 1 / (S_n(k) + delta)
  3. Load CTF parameters and compute H(k)
  4. Run PnP-ADMM iterations:
     - z-step: weighted fusion + denoiser
     - x-step: CTF-constrained projection
     - u-step: dual update
  5. Output denoised CTF-modulated image z_hat

Examples:
  # Basic usage (auto-detect diff and CTF)
  python main.py --full 000000_empiar_10002_full.mrc
  
  # Explicit paths
  python main.py --full full.mrc --diff diff.mrc --ctf params.json
  
  # Use Topaz UNet denoiser with CUDA
  python main.py --full full.mrc --diff diff.mrc --denoiser unet --cuda
  
  # More iterations with custom ADMM parameters
  python main.py --full full.mrc --diff diff.mrc -T 10 --rho 2.0 --alpha 1e-4
        """
    )
    
    # Input files
    input_group = parser.add_argument_group('Input Files')
    input_group.add_argument('--full', '-f', required=True,
                            help='Full micrograph MRC file (even + odd)')
    input_group.add_argument('--diff', '-d', default=None,
                            help='Diff micrograph MRC file (even - odd). If not provided, auto-detect.')
    input_group.add_argument('--ctf', '-c', default=None,
                            help='CTF parameters JSON file. If not provided, auto-detect.')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', '-o', default=None,
                             help='Output MRC file (default: {input_stem}_z_hat.mrc)')
    output_group.add_argument('--output-x', default=None,
                             help='Optional output for x_hat (CTF-free latent)')
    output_group.add_argument('--save-meta', action='store_true',
                             help='Save sidecar JSON with run settings and metrics')
    
    # Denoiser
    denoiser_group = parser.add_argument_group('Denoiser')
    denoiser_group.add_argument('--denoiser', default='gaussian',
                               choices=['identity', 'gaussian', 'lowpass', 'bm3d', 
                                       'unet', 'unet-small', 'fcnn'],
                               help='Denoiser type (default: gaussian)')
    denoiser_group.add_argument('--cuda', action='store_true',
                               help='Use CUDA for Topaz denoisers')
    denoiser_group.add_argument('--patch-size', type=int, default=-1,
                               help='Patch size for Topaz denoising (-1 for full image)')
    denoiser_group.add_argument('--padding', type=int, default=128,
                               help='Padding for patch-based denoising (default: 128)')
    
    # ADMM parameters
    admm_group = parser.add_argument_group('ADMM Parameters')
    admm_group.add_argument('-T', '--iterations', type=int, default=5,
                           help='Number of ADMM iterations (default: 5)')
    admm_group.add_argument('--rho', type=float, default=1.0,
                           help='ADMM consistency weight (default: 1.0)')
    admm_group.add_argument('--alpha', type=float, default=1e-3,
                           help='Stabilization term for near-CTF-null frequencies (default: 1e-3)')
    
    # Noise weight parameters
    noise_group = parser.add_argument_group('Noise Weight Parameters')
    noise_group.add_argument('--delta', type=float, default=None,
                            help='Floor for noise weight computation (auto-estimate if not set)')
    noise_group.add_argument('--uniform-weight', action='store_true',
                            help='Use uniform weight W(k)=1 (ignore diff, for testing)')
    
    # Precomputed arrays
    precomp_group = parser.add_argument_group('Precomputed Arrays (optional)')
    precomp_group.add_argument('--Hk', default=None,
                              help='Precomputed CTF numpy file (.npy)')
    precomp_group.add_argument('--Wk', default=None,
                              help='Precomputed weight numpy file (.npy)')
    
    return parser.parse_args()


def process_single_image(
    y: np.ndarray,
    Hk: np.ndarray,
    Wk: np.ndarray,
    denoiser,
    T: int,
    rho: float,
    alpha: float,
) -> dict:
    """Process a single 2D image."""
    result = pnp_admm_denoise(
        y=y,
        Hk=Hk,
        Wk=Wk,
        denoiser=denoiser,
        T=T,
        rho=rho,
        alpha=alpha,
    )
    return result


def main():
    args = parse_args()
    
    # Step 1: Pair full and diff files
    full_path, diff_path = pair_full_diff_files(args.full, args.diff)
    print(f"Full micrograph: {full_path}")
    if diff_path:
        print(f"Diff micrograph: {diff_path}")
    else:
        print("Warning: No diff file found. Will use uniform weight.")
    
    # Step 2: Read full image
    print("\n[1/5] Reading full micrograph...")
    y_full, header_info = read_mrc(full_path)
    
    # Handle 3D case
    if y_full.ndim == 3:
        y = y_full[0]  # Take first image
        print(f"  Input is 3D, using first slice. Shape: {y_full.shape} -> {y.shape}")
    else:
        y = y_full
        print(f"  Shape: {y.shape}, dtype: {y.dtype}")
    
    H, W = y.shape
    pixel_size = header_info.get('pixel_size', 1.0)
    
    # Step 3: Compute or load weight W(k)
    print("\n[2/5] Computing noise weight W(k)...")
    if args.Wk:
        print(f"  Loading precomputed weight: {args.Wk}")
        Wk = np.load(args.Wk)
        if Wk.shape != (H, W):
            raise ValueError(f"Weight shape {Wk.shape} doesn't match image shape {(H, W)}")
        weight_info = {'source': 'precomputed', 'path': args.Wk}
    elif args.uniform_weight or diff_path is None:
        print("  Using uniform weight W(k)=1")
        Wk = create_uniform_weight((H, W), value=1.0)
        weight_info = {'source': 'uniform', 'value': 1.0}
    else:
        print(f"  Computing from diff: {diff_path}")
        Wk, weight_info = load_diff_and_compute_weight(
            diff_path,
            delta=args.delta,
            auto_delta_percentile=10.0
        )
        print(f"  Delta: {weight_info['delta']:.2e}")
        print(f"  S_n range: [{weight_info['S_n_min']:.2e}, {weight_info['S_n_max']:.2e}]")
        print(f"  W mean: {weight_info['W_mean']:.2e}")
    
    # Step 4: Compute or load CTF H(k)
    print("\n[3/5] Computing CTF H(k)...")
    if args.Hk:
        print(f"  Loading precomputed CTF: {args.Hk}")
        Hk = np.load(args.Hk)
        if Hk.shape != (H, W):
            raise ValueError(f"CTF shape {Hk.shape} doesn't match image shape {(H, W)}")
        ctf_info = {'source': 'precomputed', 'path': args.Hk}
    else:
        Hk, ctf_info = load_ctf_for_micrograph(full_path, (H, W), args.ctf)
        params = ctf_info['params']
        print(f"  CTF JSON: {ctf_info['ctf_json_path']}")
        print(f"  Dataset: {ctf_info['dataset']}, idx: {ctf_info['mic_idx']}")
        print(f"  Pixel size: {params['pixel_size']:.3f} A")
        print(f"  Defocus U: {params['defocus_u']:.1f} A")
        print(f"  Defocus V: {params['defocus_v']:.1f} A")
        print(f"  kV: {params['kv']:.1f}, Cs: {params['cs']:.2f}")
    
    # Step 5: Create denoiser
    print("\n[4/5] Creating denoiser...")
    print(f"  Type: {args.denoiser}")
    
    if args.denoiser in ['unet', 'unet-small', 'fcnn']:
        # Topaz denoiser
        denoiser = create_topaz_denoiser(
            model=args.denoiser,
            use_cuda=args.cuda,
            patch_size=args.patch_size,
            padding=args.padding
        )
        if args.cuda:
            print("  Using CUDA acceleration")
        if args.patch_size > 0:
            print(f"  Patch size: {args.patch_size}, padding: {args.padding}")
    else:
        # Standard denoisers
        denoiser = create_denoiser(args.denoiser)
    
    # Step 6: Run PnP-ADMM
    print("\n[5/5] Running PnP-ADMM...")
    print(f"  Iterations: {args.iterations}")
    print(f"  rho: {args.rho}")
    print(f"  alpha: {args.alpha}")
    
    result = process_single_image(
        y=y,
        Hk=Hk,
        Wk=Wk,
        denoiser=denoiser,
        T=args.iterations,
        rho=args.rho,
        alpha=args.alpha,
    )
    
    z_hat = result['z_hat']
    x_hat = result['x_hat']
    history = result['history']
    
    # Print convergence info
    print("\n  Convergence:")
    for h in history:
        print(f"    Iter {h['iter']}: residual={h['residual_z']:.4e}, data_dev={h['data_deviation']:.4e}")
    
    # Step 7: Save outputs
    print("\n[Save] Writing outputs...")
    
    # Determine output path
    if args.output is None:
        input_path = Path(full_path)
        output_path = input_path.parent / f"{input_path.stem}_z_hat.mrc"
    else:
        output_path = Path(args.output)
    
    print(f"  z_hat: {output_path}")
    write_mrc(output_path, z_hat, pixel_size=pixel_size)
    
    if args.output_x:
        x_path = Path(args.output_x)
        print(f"  x_hat: {x_path}")
        write_mrc(x_path, x_hat, pixel_size=pixel_size)
    
    # Save metadata
    if args.save_meta:
        meta_path = output_path.with_suffix('.json')
        meta = {
            'input': {
                'full': full_path,
                'diff': diff_path,
            },
            'output': {
                'z_hat': str(output_path),
                'x_hat': str(args.output_x) if args.output_x else None,
            },
            'parameters': {
                'T': args.iterations,
                'rho': args.rho,
                'alpha': args.alpha,
                'denoiser': args.denoiser,
                'cuda': args.cuda,
                'patch_size': args.patch_size,
                'padding': args.padding,
            },
            'weight': weight_info,
            'ctf': {
                'json_path': ctf_info.get('ctf_json_path'),
                'dataset': ctf_info.get('dataset'),
                'mic_idx': ctf_info.get('mic_idx'),
            } if 'ctf_json_path' in ctf_info else {'source': 'precomputed'},
            'convergence': history,
            'image': {
                'shape': [H, W],
                'pixel_size': pixel_size,
                'dtype': str(y.dtype),
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  metadata: {meta_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
