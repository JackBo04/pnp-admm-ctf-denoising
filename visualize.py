#!/usr/bin/env python3
"""
Visualize denoising results: compare original vs denoised images

Usage:
    python visualize.py --original full.mrc --denoised full_z_hat.mrc
    python visualize.py --original full.mrc --denoised full_z_hat.mrc --crop 1024
    python visualize.py --original full.mrc --denoised full_z_hat.mrc --diff diff.mrc --show-weight
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mrc_io import read_mrc
from noise_psd import compute_weight_from_diff, estimate_delta_from_psd, compute_noise_psd_from_diff


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize PnP-ADMM denoising results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python visualize.py -i full.mrc -z full_z_hat.mrc -o comparison.png
  
  # With diff file to show noise weight
  python visualize.py -i full.mrc -z full_z_hat.mrc -d diff.mrc --show-weight
  
  # Crop center region
  python visualize.py -i full.mrc -z full_z_hat.mrc --crop 1024
  
  # With zoom region
  python visualize.py -i full.mrc -z full_z_hat.mrc --crop 1024 --zoom-region 400,400,512
        """
    )
    parser.add_argument('--original', '-i', required=True,
                        help='Original full MRC file')
    parser.add_argument('--denoised', '-z', required=True,
                        help='Denoised z_hat MRC file')
    parser.add_argument('--diff', '-d', default=None,
                        help='Diff MRC file (for showing noise weight)')
    parser.add_argument('--output', '-o', default='denoising_comparison.png',
                        help='Output PNG file (default: denoising_comparison.png)')
    parser.add_argument('--show-weight', action='store_true',
                        help='Show noise weight W(k) if diff is provided')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Min value for colormap (auto if not set)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Max value for colormap (auto if not set)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    parser.add_argument('--crop', type=int, default=None,
                        help='Crop center region to this size (e.g., 1024)')
    parser.add_argument('--zoom-region', type=str, default=None,
                        help='Zoom region as y_start,x_start,size (e.g., 1000,1500,512)')
    
    return parser.parse_args()


def crop_center(data, size):
    """Crop center region."""
    h, w = data.shape[-2:]
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return data[..., y_start:y_start+size, x_start:x_start+size]


def parse_zoom_region(region_str):
    """Parse zoom region string."""
    parts = region_str.split(',')
    if len(parts) != 3:
        raise ValueError("Zoom region must be y_start,x_start,size")
    return int(parts[0]), int(parts[1]), int(parts[2])


def create_comparison_figure(
    original, 
    denoised, 
    vmin=None, 
    vmax=None, 
    zoom_region=None, 
    dpi=150,
    weight=None,
):
    """
    Create comparison figure with multiple views.
    
    Layout:
    - Row 1: Original | Denoised | Difference
    - Row 2 (optional): Noise weight | Zoomed region
    """
    # Ensure 2D
    if original.ndim == 3:
        original = original[0]
    if denoised.ndim == 3:
        denoised = denoised[0]
    
    # Compute difference
    diff = original - denoised
    
    # Determine value range
    if vmin is None:
        vmin = min(original.min(), denoised.min())
    if vmax is None:
        vmax = max(original.max(), denoised.max())
    
    # Determine figure layout
    n_rows = 2 if (weight is not None or zoom_region is not None) else 1
    n_cols = 3
    
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Full view
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Original (y)\nRange: [{original.min():.2f}, {original.max():.2f}]', 
                  fontsize=11)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Denoised
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(denoised, cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Denoised (z_hat)\nRange: [{denoised.min():.2f}, {denoised.max():.2f}]', 
                  fontsize=11)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff_max = max(abs(diff.min()), abs(diff.max()))
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    ax3.set_title(f'Difference (y - z_hat)\nRange: [{diff.min():.2f}, {diff.max():.2f}]', 
                  fontsize=11)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Row 2: Weight visualization and/or zoom
    if n_rows == 2:
        if weight is not None:
            # Show noise weight
            ax4 = fig.add_subplot(gs[1, 0])
            weight_log = np.log10(weight + 1e-10)
            im4 = ax4.imshow(np.fft.fftshift(weight_log), cmap='viridis')
            ax4.set_title(f'Noise Weight log10(W(k)+δ)\nMean: {weight.mean():.2e}', fontsize=11)
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            
            # Show noise PSD
            ax5 = fig.add_subplot(gs[1, 1])
            S_n = 1.0 / weight
            S_n_log = np.log10(S_n + 1e-10)
            im5 = ax5.imshow(np.fft.fftshift(S_n_log), cmap='hot')
            ax5.set_title(f'Noise PSD log10(S_n(k))\nMean: {S_n.mean():.2e}', fontsize=11)
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Zoom region
        if zoom_region:
            y, x, size = zoom_region
            y_end = min(y + size, original.shape[0])
            x_end = min(x + size, original.shape[1])
            
            orig_zoom = original[y:y_end, x:x_end]
            denoised_zoom = denoised[y:y_end, x:x_end]
            diff_zoom = diff[y:y_end, x:x_end]
            
            col_start = 2 if weight is not None else 0
            
            # Original zoom
            ax_z1 = fig.add_subplot(gs[1, col_start])
            im_z1 = ax_z1.imshow(orig_zoom, cmap='gray', vmin=vmin, vmax=vmax)
            ax_z1.set_title(f'Original Zoom ({y},{x})', fontsize=11)
            ax_z1.axis('off')
            plt.colorbar(im_z1, ax=ax_z1, fraction=0.046, pad=0.04)
            
            if weight is None and col_start < 2:
                # Denoised zoom
                ax_z2 = fig.add_subplot(gs[1, col_start + 1])
                im_z2 = ax_z2.imshow(denoised_zoom, cmap='gray', vmin=vmin, vmax=vmax)
                ax_z2.set_title(f'Denoised Zoom ({y},{x})', fontsize=11)
                ax_z2.axis('off')
                plt.colorbar(im_z2, ax=ax_z2, fraction=0.046, pad=0.04)
            
            # Add rectangle on full images to indicate zoom region
            from matplotlib.patches import Rectangle
            rect1 = Rectangle((x, y), size, size, linewidth=2, 
                              edgecolor='red', facecolor='none')
            rect2 = Rectangle((x, y), size, size, linewidth=2,
                              edgecolor='red', facecolor='none')
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)
    
    plt.suptitle(f'CTF-Constrained PnP-ADMM Denoising Results\n'
                 f'Image shape: {original.shape}', 
                 fontsize=13, fontweight='bold')
    
    return fig


def create_side_by_side_comparison(original, denoised, output_path, dpi=150):
    """
    Create a simple side-by-side comparison.
    """
    if original.ndim == 3:
        original = original[0]
    if denoised.ndim == 3:
        denoised = denoised[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    vmin = min(original.min(), denoised.min())
    vmax = max(original.max(), denoised.max())
    
    # Original
    im0 = axes[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Denoised
    im1 = axes[1].imshow(denoised, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised (z_hat)', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Side-by-side comparison saved: {output_path}")


def main():
    args = parse_args()
    
    # Read images
    print(f"Reading original: {args.original}")
    original, _ = read_mrc(args.original)
    print(f"  Shape: {original.shape}")
    
    print(f"Reading denoised: {args.denoised}")
    denoised, _ = read_mrc(args.denoised)
    print(f"  Shape: {denoised.shape}")
    
    # Handle 3D case
    if original.ndim == 3:
        original = original[0]
    if denoised.ndim == 3:
        denoised = denoised[0]
    
    # Crop if specified
    if args.crop:
        print(f"Cropping center to {args.crop}x{args.crop}")
        original = crop_center(original, args.crop)
        denoised = crop_center(denoised, args.crop)
    
    # Parse zoom region
    zoom_region = None
    if args.zoom_region:
        zoom_region = parse_zoom_region(args.zoom_region)
        print(f"Zoom region: y={zoom_region[0]}, x={zoom_region[1]}, size={zoom_region[2]}")
    
    # Compute weight if diff provided and requested
    weight = None
    if args.show_weight and args.diff:
        print(f"Computing noise weight from: {args.diff}")
        diff_img, _ = read_mrc(args.diff)
        if diff_img.ndim == 3:
            diff_img = diff_img[0]
        if args.crop:
            diff_img = crop_center(diff_img, args.crop)
        
        S_n = compute_noise_psd_from_diff(diff_img)
        delta = estimate_delta_from_psd(S_n)
        weight = compute_weight_from_diff(diff_img, delta=delta)
        print(f"  Delta: {delta:.2e}")
        print(f"  W mean: {weight.mean():.2e}")
    
    # Create main comparison figure
    print("Generating comparison figure...")
    fig = create_comparison_figure(
        original, 
        denoised, 
        vmin=args.vmin, 
        vmax=args.vmax,
        zoom_region=zoom_region,
        dpi=args.dpi,
        weight=weight,
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Comparison saved: {output_path}")
    
    # Also create a side-by-side version
    side_by_side_path = output_path.parent / f"{output_path.stem}_side_by_side{output_path.suffix}"
    create_side_by_side_comparison(original, denoised, side_by_side_path, args.dpi)


if __name__ == '__main__':
    main()
