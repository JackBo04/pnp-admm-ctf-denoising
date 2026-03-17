"""
CTF-Constrained PnP-ADMM Core Solver

This module implements the corrected PnP-ADMM algorithm as specified in SPEC.md:
- Goal: CTF-modulated, noise-reduced observation z_hat ≈ Hx (not CTF-free x)
- Uses noise PSD weight W(k) from odd/even diff for weighted fusion
"""
import numpy as np
from typing import Callable, Optional


def fft2(img: np.ndarray) -> np.ndarray:
    """2D FFT wrapper (unshifted)."""
    return np.fft.fft2(img)


def ifft2(F: np.ndarray) -> np.ndarray:
    """2D IFFT wrapper, returns real part."""
    return np.real(np.fft.ifft2(F))


def apply_H_to_x(x: np.ndarray, Hk: np.ndarray) -> np.ndarray:
    """
    Apply CTF operator: z = Hx in Fourier domain.
    
    Args:
        x: real image (H, W)
        Hk: CTF in Fourier domain (H, W)
    
    Returns:
        z: real image (H, W)
    """
    X = fft2(x)
    Z = Hk * X
    z = ifft2(Z)
    return z


def weighted_fusion_zstep(
    Yk: np.ndarray,
    Ck: np.ndarray,
    Wk: np.ndarray,
    rho: float
) -> np.ndarray:
    """
    Step A1: Weighted fusion (closed-form, in Fourier domain).
    
    Computes:
        Z_tilde(k) = (W(k)*Y(k) + rho*C(k)) / (W(k) + rho)
    
    Args:
        Yk: Fourier transform of observed image y
        Ck: Fourier transform of physical center c = Hx - u
        Wk: noise PSD weight W(k) = 1 / (S_n(k) + delta)
        rho: ADMM consistency weight
    
    Returns:
        Z_tilde: Fourier domain fused image
    """
    numerator = Wk * Yk + rho * Ck
    denominator = Wk + rho
    Z_tilde = numerator / denominator
    return Z_tilde


def solve_x_step(
    Zk: np.ndarray,
    Uk: np.ndarray,
    Hk: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Step B: x-step (CTF-Constrained Projection / Inversion).
    
    Solves:
        x^{t+1} = argmin_x ||Hx - (z^{t+1} + u^t)||_2^2 + alpha*||x||_2^2
    
    Fourier closed-form:
        X(k) = H*(k) * (Z(k) + U(k)) / (|H(k)|^2 + alpha)
    
    Args:
        Zk: Fourier transform of z^{t+1}
        Uk: Fourier transform of u^t
        Hk: CTF in Fourier domain
        alpha: stabilization term for near-CTF-null frequencies
    
    Returns:
        x: real image
    """
    H_conj = np.conj(Hk)
    denom = np.abs(Hk) ** 2 + alpha
    Xk = H_conj * (Zk + Uk) / denom
    x = ifft2(Xk)
    return x


def pnp_admm_denoise(
    y: np.ndarray,
    Hk: np.ndarray,
    Wk: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    T: int = 5,
    rho: float = 1.0,
    alpha: float = 1e-3,
) -> dict:
    """
    CTF-Constrained PnP-ADMM Denoising (SPEC.md compliant).
    
    Goal: Denoise CTF-modulated observation y to get z_hat ≈ Hx.
    
    The algorithm iterates:
    1. z-step: Weighted fusion + denoiser
       - A1: Z_tilde(k) = (W(k)*Y(k) + rho*C(k)) / (W(k) + rho)
       - A2: z = D(z_tilde) where D is the plug-and-play denoiser
    
    2. x-step: CTF-constrained projection
       - X(k) = H*(k)*(Z(k) + U(k)) / (|H(k)|^2 + alpha)
    
    3. u-step: Dual update
       - u = u + (z - Hx)
    
    Args:
        y: observed noisy image (H, W), measurement domain
        Hk: CTF in Fourier domain (H, W), unshifted FFT ordering
        Wk: noise PSD weight W(k) = 1 / (S_n(k) + delta), shape (H, W)
        denoiser: callable denoiser function(img) -> img
        T: number of iterations
        rho: ADMM consistency weight
        alpha: stabilization term for near-CTF-null frequencies
    
    Returns:
        dict with keys:
            'z_hat': main output, denoised CTF-modulated image
            'x_hat': internal latent (CTF-free), for diagnostics
            'u': dual variable at convergence
            'history': iteration history
    """
    # Initialization
    z = y.copy()
    x = np.zeros_like(y)
    u = np.zeros_like(y)
    
    # Precompute Y(k)
    Yk = fft2(y)
    
    # Track iteration history
    history = []
    
    for t in range(T):
        # Step A1: Weighted fusion in Fourier domain
        Hx = apply_H_to_x(x, Hk)
        c = Hx - u
        Ck = fft2(c)
        Z_tilde_k = weighted_fusion_zstep(Yk, Ck, Wk, rho)
        z_tilde = ifft2(Z_tilde_k)
        
        # Step A2: Apply Plug-and-Play denoiser
        z = denoiser(z_tilde)
        
        # Step B: x-step (CTF-constrained projection)
        Zk = fft2(z)
        Uk = fft2(u)
        x = solve_x_step(Zk, Uk, Hk, alpha)
        
        # Step C: u-step (dual update)
        Hx_new = apply_H_to_x(x, Hk)
        u = u + (z - Hx_new)
        
        # Track metrics
        residual_z = np.linalg.norm(z - Hx_new)
        data_dev = np.linalg.norm(z - y)
        history.append({
            'iter': t,
            'residual_z': float(residual_z),
            'data_deviation': float(data_dev),
        })
    
    return {
        'z_hat': z,
        'x_hat': x,
        'u': u,
        'history': history,
    }


def pnp_admm_denoise_stack(
    y_stack: np.ndarray,
    Hk: np.ndarray,
    Wk: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    T: int = 5,
    rho: float = 1.0,
    alpha: float = 1e-3,
) -> dict:
    """
    Apply PnP-ADMM denoising to a stack of images.
    
    Args:
        y_stack: stack of observed images (N, H, W)
        Hk: CTF in Fourier domain (H, W)
        Wk: noise PSD weight (H, W)
        denoiser: callable denoiser function
        T: number of iterations
        rho: ADMM consistency weight
        alpha: stabilization term
    
    Returns:
        dict with z_hat_stack, x_hat_stack, etc.
    """
    N = y_stack.shape[0]
    z_hats = []
    x_hats = []
    all_history = []
    
    for i in range(N):
        result = pnp_admm_denoise(
            y=y_stack[i],
            Hk=Hk,
            Wk=Wk,
            denoiser=denoiser,
            T=T,
            rho=rho,
            alpha=alpha,
        )
        z_hats.append(result['z_hat'])
        x_hats.append(result['x_hat'])
        all_history.append(result['history'])
    
    z_stack = np.stack(z_hats, axis=0)
    x_stack = np.stack(x_hats, axis=0)
    
    return {
        'z_hat': z_stack,
        'x_hat': x_stack,
        'history': all_history,
    }
