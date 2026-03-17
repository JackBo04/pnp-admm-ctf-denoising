"""
Noise PSD Estimation from Odd/Even Difference

This module implements the noise PSD estimation as specified in SPEC.md Section 1:
- Input: diff = even - odd (provided as *_diff.mrc)
- S_d(k) = |FFT(diff)|^2 (power spectrum of diff)
- S_n(k) = (1/2) * S_d(k) (noise PSD)
- W(k) = 1 / (S_n(k) + delta) (inverse-noise weighting)
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
try:
    from .mrc_io import read_mrc
except ImportError:
    from mrc_io import read_mrc


def compute_noise_psd_from_diff(diff_img: np.ndarray) -> np.ndarray:
    """
    Compute noise power spectral density S_n(k) from diff image.
    
    Principle:
        diff = even - odd = n1 - n2 (signal cancels)
        S_d(k) = E[|D(k)|^2] = E[|N1(k) - N2(k)|^2] = 2 * S_n(k)
        Therefore: S_n(k) = (1/2) * S_d(k)
    
    Args:
        diff_img: diff image (even - odd), shape (H, W)
    
    Returns:
        S_n: noise PSD, shape (H, W)
    """
    # Compute power spectrum of diff
    Diff_k = np.fft.fft2(diff_img)
    S_d = np.abs(Diff_k) ** 2
    
    # Noise PSD is half of diff PSD
    S_n = 0.5 * S_d
    
    return S_n


def compute_weight_from_psd(S_n: np.ndarray, delta: float = 1e-6, normalize: bool = True, method: str = 'exponential') -> np.ndarray:
    """
    Compute inverse-noise weight W(k) from noise PSD.
    
    Methods:
    1. 'inverse': W(k) = 1 / (S_n(k) + delta), then normalize
       - Problem: Wk can be too small in high-noise frequencies (median ~0.1)
    2. 'exponential': W(k) = exp(-S_n / (S_n.mean() + delta))
       - Better: smoother transition, median ~0.8, preserves more signal
    
    Args:
        S_n: noise PSD, shape (H, W)
        delta: small floor to prevent division blow-up
        normalize: whether to normalize (kept for compatibility, not used in exponential)
        method: 'inverse' (old) or 'exponential' (recommended)
    
    Returns:
        Wk: weight, shape (H, W), range [0, 1]
    """
    if method == 'inverse':
        # Old method: 1/(S_n + delta), then normalize
        Wk = 1.0 / (S_n + delta)
        if normalize:
            Wk_max = np.max(Wk)
            if Wk_max > 0:
                Wk = Wk / Wk_max
    elif method == 'exponential':
        # New method: exponential decay based on S_n relative to mean
        # Wk = exp(-S_n / (S_n.mean() + delta))
        # This gives:
        # - Wk ≈ 1 when S_n << mean (clean frequencies)
        # - Wk ≈ 0.37 when S_n = mean
        # - Wk ≈ 0 when S_n >> mean (noisy frequencies)
        scale = S_n.mean() + delta
        Wk = np.exp(-S_n / scale)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return Wk


def compute_weight_from_diff(diff_img: np.ndarray, delta: float = 1e-6, normalize: bool = True) -> np.ndarray:
    """
    Compute noise weight W(k) directly from diff image.
    
    Args:
        diff_img: diff image (even - odd), shape (H, W)
        delta: small floor to prevent division blow-up
        normalize: whether to normalize weights to [0, 1] range
    
    Returns:
        Wk: weight, shape (H, W)
    """
    S_n = compute_noise_psd_from_diff(diff_img)
    Wk = compute_weight_from_psd(S_n, delta, normalize=normalize)
    return Wk


def estimate_delta_from_psd(S_n: np.ndarray, percentile: float = 10.0) -> float:
    """
    Estimate a reasonable delta value from noise PSD.
    
    Uses a percentile of the non-zero S_n values as a reference
    to set a small floor that prevents division blow-up while
    maintaining minimal intrusion.
    
    Args:
        S_n: noise PSD
        percentile: percentile to use for estimation (default 10%)
    
    Returns:
        estimated delta value
    """
    # Exclude zero values
    S_n_nonzero = S_n[S_n > 0]
    if len(S_n_nonzero) == 0:
        return 1e-6
    
    # Use a small fraction of the lower percentile
    delta = np.percentile(S_n_nonzero, percentile) * 0.01
    
    # Clamp to reasonable range
    delta = max(delta, 1e-10)
    delta = min(delta, 1e-3)
    
    return delta


def load_diff_and_compute_weight(
    diff_path: str,
    delta: Optional[float] = None,
    auto_delta_percentile: float = 10.0,
    normalize: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Load diff image and compute noise weight W(k).
    
    Args:
        diff_path: path to diff MRC file
        delta: if None, auto-estimate from PSD
        auto_delta_percentile: percentile for auto delta estimation
        normalize: whether to normalize weights to [0, 1] range
    
    Returns:
        (Wk, info_dict)
    """
    diff_img, header_info = read_mrc(diff_path)
    
    # Handle 3D case (take first image if stack)
    if diff_img.ndim == 3:
        diff_img = diff_img[0]
    
    # Compute noise PSD
    S_n = compute_noise_psd_from_diff(diff_img)
    
    # Auto-estimate delta if not provided
    if delta is None:
        delta = estimate_delta_from_psd(S_n, auto_delta_percentile)
    
    # Compute weight
    Wk = compute_weight_from_psd(S_n, delta, normalize=normalize)
    
    info = {
        'delta': delta,
        'S_n_mean': float(np.mean(S_n)),
        'S_n_max': float(np.max(S_n)),
        'S_n_min': float(np.min(S_n[S_n > 0]) if np.any(S_n > 0) else 0),
        'W_mean': float(np.mean(Wk)),
        'W_max': float(np.max(Wk)),
        'W_min': float(np.min(Wk)),
        'shape': diff_img.shape,
        'pixel_size': header_info.get('pixel_size', 1.0),
    }
    
    return Wk, info


def create_uniform_weight(shape: Tuple[int, int], value: float = 1.0) -> np.ndarray:
    """
    Create a uniform weight (for testing or when diff is not available).
    
    Args:
        shape: (H, W)
        value: uniform weight value
    
    Returns:
        Wk: uniform weight array
    """
    return np.full(shape, value, dtype=np.float32)


def smooth_psd(S_n: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to PSD for more stable weight estimation.
    
    Args:
        S_n: raw noise PSD
        sigma: Gaussian sigma for smoothing (in frequency domain)
    
    Returns:
        smoothed PSD
    """
    from scipy.ndimage import gaussian_filter
    
    # Shift to center for smoothing
    S_n_shifted = np.fft.fftshift(S_n)
    S_n_smooth = gaussian_filter(S_n_shifted, sigma=sigma)
    S_n_out = np.fft.ifftshift(S_n_smooth)
    
    return S_n_out


def compute_weight_from_diff_smoothed(
    diff_img: np.ndarray,
    delta: float = 1e-6,
    smooth_sigma: float = 1.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute noise weight with smoothing for stability.
    
    Args:
        diff_img: diff image
        delta: floor value
        smooth_sigma: smoothing sigma in frequency domain
        normalize: whether to normalize weights to [0, 1] range
    
    Returns:
        Wk: smoothed weight
    """
    S_n = compute_noise_psd_from_diff(diff_img)
    S_n_smooth = smooth_psd(S_n, sigma=smooth_sigma)
    Wk = compute_weight_from_psd(S_n_smooth, delta, normalize=normalize)
    return Wk
