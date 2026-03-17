"""
CTF-Constrained PnP-ADMM Denoising for cryo-EM images

This package implements the CTF-constrained PnP-ADMM denoising algorithm
for cryo-EM micrographs as specified in SPEC.md.

Key features:
- Measurement-domain denoising (output is CTF-modulated z_hat ≈ Hx)
- Noise PSD estimation from odd/even diff files
- Weighted fusion z-step with inverse-noise weighting
- CTF-constrained projection x-step
- Support for Topaz deep learning denoisers

Example usage:
    from pnp_admm import pnp_admm_denoise, load_ctf_for_micrograph
    from pnp_admm.noise_psd import load_diff_and_compute_weight
    from pnp_admm.denoiser import create_denoiser
    
    # Load data
    y, _ = read_mrc('micrograph_full.mrc')
    Hk, _ = load_ctf_for_micrograph('micrograph_full.mrc', y.shape)
    Wk, _ = load_diff_and_compute_weight('micrograph_diff.mrc')
    
    # Create denoiser
    denoiser = create_denoiser('unet', use_cuda=True)
    
    # Run denoising
    result = pnp_admm_denoise(
        y=y,
        Hk=Hk,
        Wk=Wk,
        denoiser=denoiser,
        T=5,
        rho=1.0,
        alpha=1e-3,
    )
    
    z_hat = result['z_hat']  # CTF-modulated denoised output
"""

__version__ = "0.2.0"

# Core algorithm
from .core import (
    pnp_admm_denoise,
    pnp_admm_denoise_stack,
    apply_H_to_x,
    weighted_fusion_zstep,
    solve_x_step,
)

# CTF utilities
from .ctf import (
    compute_ctf_2d,
    load_ctf_params,
    compute_ctf_from_json,
    load_ctf_for_micrograph,
    find_ctf_json_for_micrograph,
    parse_micrograph_filename,
    pair_full_diff_files,
)

# Noise PSD estimation
from .noise_psd import (
    compute_noise_psd_from_diff,
    compute_weight_from_psd,
    compute_weight_from_diff,
    load_diff_and_compute_weight,
    create_uniform_weight,
    estimate_delta_from_psd,
)

# MRC I/O
from .mrc_io import (
    read_mrc,
    write_mrc,
    read_mrc_stack,
    write_mrc_stack,
)

# Denoisers
from .denoiser import (
    Denoiser,
    IdentityDenoiser,
    GaussianDenoiser,
    LowpassDenoiser,
    BM3DDenoiser,
    TopazDenoiser,
    create_denoiser,
    create_topaz_denoiser,
)

__all__ = [
    # Version
    '__version__',
    
    # Core
    'pnp_admm_denoise',
    'pnp_admm_denoise_stack',
    'apply_H_to_x',
    'weighted_fusion_zstep',
    'solve_x_step',
    
    # CTF
    'compute_ctf_2d',
    'load_ctf_params',
    'compute_ctf_from_json',
    'load_ctf_for_micrograph',
    'find_ctf_json_for_micrograph',
    'parse_micrograph_filename',
    'pair_full_diff_files',
    
    # Noise PSD
    'compute_noise_psd_from_diff',
    'compute_weight_from_psd',
    'compute_weight_from_diff',
    'load_diff_and_compute_weight',
    'create_uniform_weight',
    'estimate_delta_from_psd',
    
    # MRC I/O
    'read_mrc',
    'write_mrc',
    'read_mrc_stack',
    'write_mrc_stack',
    
    # Denoisers
    'Denoiser',
    'IdentityDenoiser',
    'GaussianDenoiser',
    'LowpassDenoiser',
    'BM3DDenoiser',
    'TopazDenoiser',
    'create_denoiser',
    'create_topaz_denoiser',
]
