"""
Plug-and-Play Denoiser Interface

This module provides denoiser interfaces compatible with the PnP-ADMM solver.
It includes simple denoisers for testing and a wrapper for Topaz denoisers.
"""
import numpy as np
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Union


class Denoiser(ABC):
    """
    Abstract base class for plug-and-play denoisers.
    
    The denoiser is a black box that takes a noisy image and returns a denoised version.
    """
    
    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Denoise an image.
        
        Args:
            img: (H, W) real-valued float32/float64
        
        Returns:
            Denoised image with same shape as input
        """
        pass


class IdentityDenoiser(Denoiser):
    """Identity denoiser (no denoising), useful for testing."""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.copy()


class GaussianDenoiser(Denoiser):
    """Simple Gaussian smoothing denoiser (for testing only)."""
    
    def __init__(self, sigma: float = 1.0):
        from scipy.ndimage import gaussian_filter
        self.sigma = sigma
        self._filter = gaussian_filter
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self._filter(img, sigma=self.sigma)


class LowpassDenoiser(Denoiser):
    """Simple Butterworth low-pass filter denoiser."""
    
    def __init__(self, cutoff: float = 0.1, order: int = 2):
        self.cutoff = cutoff
        self.order = order
    
    def _create_butterworth_lp(self, shape: tuple) -> np.ndarray:
        """Create Butterworth low-pass filter mask."""
        h, w = shape
        freq_y = np.fft.fftfreq(h)
        freq_x = np.fft.fftfreq(w)
        ky, kx = np.meshgrid(freq_y, freq_x, indexing='ij')
        k = np.sqrt(kx**2 + ky**2)
        
        kc = self.cutoff
        if kc <= 0:
            return np.ones(shape)
        
        H = 1.0 / (1.0 + (k / kc) ** (2 * self.order))
        return H.astype(np.float32)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        orig_dtype = img.dtype
        img_f = img.astype(np.float32)
        
        F = np.fft.fft2(img_f)
        H = self._create_butterworth_lp(img.shape)
        F_filtered = F * H
        img_filtered = np.real(np.fft.ifft2(F_filtered))
        
        return img_filtered.astype(orig_dtype)


class BM3DDenoiser(Denoiser):
    """BM3D denoiser wrapper (if available)."""
    
    def __init__(self, sigma_psd: float = 0.1):
        try:
            import bm3d
            self.bm3d = bm3d
            self.sigma_psd = sigma_psd
        except ImportError:
            raise ImportError("BM3D denoiser requires 'bm3d' package. Install with: pip install bm3d")
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_f64 = img.astype(np.float64)
        denoised = self.bm3d.bm3d(img_f64, sigma_psd=self.sigma_psd, stage_arg=self.bm3d.BM3DStages.ALL_STAGES)
        return denoised.astype(img.dtype)


class TopazDenoiser(Denoiser):
    """
    Wrapper for Topaz denoising models.
    
    This wraps the topaz.denoise.Denoise class to provide a consistent interface.
    
    Args:
        model: Topaz model name (e.g., 'unet', 'unet-small', 'fcnn') or path to model file
        use_cuda: whether to use GPU acceleration
        patch_size: patch size for denoising (-1 for full image)
        padding: padding for patch-based denoising
    """
    
    # Model name mapping
    MODEL_ALIASES = {
        'unet': 'unet_L2_v0.2.2.sav',
        'unet-small': 'unet_small_L1_v0.2.2.sav',
        'fcnn': 'fcnn_L1_v0.2.2.sav',
        'affine': 'affine_L1_v0.2.2.sav',
        'unet-v0.2.1': 'unet_L2_v0.2.1.sav',
        'unet-3d': 'unet-3d-10a-v0.2.4.sav',
        'unet-3d-10a': 'unet-3d-10a-v0.2.4.sav',
        'unet-3d-20a': 'unet-3d-20a-v0.2.4.sav',
    }
    
    def __init__(
        self,
        model: str = 'unet',
        use_cuda: bool = False,
        patch_size: int = -1,
        padding: int = 128
    ):
        self.model_name = model
        self.use_cuda = use_cuda
        self.patch_size = patch_size
        self.padding = padding
        
        # Try to import topaz
        try:
            import torch
            self.torch = torch
            
            # Add topaz to path
            topaz_path = os.path.join(os.path.dirname(__file__), 'topaz')
            if os.path.exists(topaz_path) and topaz_path not in sys.path:
                sys.path.insert(0, topaz_path)
            
            from topaz.denoising.models import load_model
            self._load_model_fn = load_model
        except Exception as e:
            raise ImportError(
                f"Failed to load Topaz: {e}. "
                "Make sure topaz is installed or the topaz/ directory is present. "
                "If using conda, run: conda activate topaz"
            )
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Topaz model."""
        # Check if it's a known model alias
        if self.model_name in self.MODEL_ALIASES:
            model_file = self.MODEL_ALIASES[self.model_name]
            
            # Find the pretrained model path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pretrained_path = os.path.join(
                script_dir, 'topaz', 'topaz', 'pretrained', 'denoise', model_file
            )
            
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained model not found: {pretrained_path}\n"
                    f"Please ensure topaz pretrained models are available."
                )
            
            # Load using topaz's load_model function
            self.model = self._load_model_fn(self.model_name)
            
            # Move to GPU if requested
            if self.use_cuda:
                self.model = self.model.cuda()
            
            self.model.eval()
        else:
            # Try loading as a custom model path
            if os.path.exists(self.model_name):
                self.model = self._load_model_fn(self.model_name)
                if self.use_cuda:
                    self.model = self.model.cuda()
                self.model.eval()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
    
    def _denoise_tensor(self, x: 'torch.Tensor') -> np.ndarray:
        """Denoise a torch tensor."""
        import torch.nn.functional as F
        
        # Normalize
        mu, std = x.mean(), x.std()
        x_norm = (x - mu) / (std + 1e-8)
        
        # Add batch and channel dimensions
        if len(x_norm.shape) == 2:
            x_norm = x_norm.unsqueeze(0).unsqueeze(0)
        elif len(x_norm.shape) == 3:
            x_norm = x_norm.unsqueeze(1)
        
        # Predict
        with self.torch.no_grad():
            pred = self.model(x_norm)
        
        # Remove extra dimensions
        pred = pred.squeeze()
        
        # Unnormalize
        pred = pred * std + mu
        
        return pred.cpu().numpy()
    
    def _denoise_patches(self, x: np.ndarray) -> np.ndarray:
        """Denoise in patches."""
        h, w = x.shape
        y = np.zeros_like(x)
        
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                # Include padding
                si = max(0, i - self.padding)
                ei = min(h, i + self.patch_size + self.padding)
                sj = max(0, j - self.padding)
                ej = min(w, j + self.patch_size + self.padding)
                
                x_patch = x[si:ei, sj:ej]
                
                # Convert to tensor
                x_tensor = self.torch.from_numpy(x_patch).float()
                if self.use_cuda:
                    x_tensor = x_tensor.cuda()
                
                # Denoise
                y_patch = self._denoise_tensor(x_tensor)
                
                # Place back without padding
                si_out = i - si
                sj_out = j - sj
                y[i:i+self.patch_size, j:j+self.patch_size] = y_patch[
                    si_out:si_out+min(self.patch_size, h-i),
                    sj_out:sj_out+min(self.patch_size, w-j)
                ]
        
        return y
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Denoise an image using Topaz.
        
        Args:
            img: input image (H, W)
        
        Returns:
            denoised image (H, W)
        """
        # Convert to tensor
        x = self.torch.from_numpy(img).float()
        if self.use_cuda:
            x = x.cuda()
        
        # Decide whether to use patches
        use_patch = (self.patch_size > 0) and (
            self.patch_size + self.padding < img.shape[0] or 
            self.patch_size + self.padding < img.shape[1]
        )
        
        if use_patch:
            result = self._denoise_patches(img)
        else:
            result = self._denoise_tensor(x)
        
        return result


def create_denoiser(name: str, **kwargs) -> Denoiser:
    """
    Factory function to create denoiser by name.
    
    Args:
        name: 'identity', 'gaussian', 'lowpass', 'bm3d', or any Topaz model name
              ('unet', 'unet-small', 'fcnn', etc.)
        **kwargs: denoiser-specific parameters
    
    Returns:
        Denoiser instance
    """
    name = name.lower()
    
    # Standard denoisers
    if name == 'identity':
        return IdentityDenoiser()
    elif name == 'gaussian':
        return GaussianDenoiser(**kwargs)
    elif name == 'lowpass':
        return LowpassDenoiser(**kwargs)
    elif name == 'bm3d':
        return BM3DDenoiser(**kwargs)
    
    # Try Topaz models
    try:
        return TopazDenoiser(model=name, **kwargs)
    except (ImportError, ValueError) as e:
        # If it's a known Topaz model name but import failed, raise the error
        if name in TopazDenoiser.MODEL_ALIASES:
            raise ImportError(
                f"Failed to load Topaz denoiser '{name}': {e}\n"
                f"Make sure you have activated the correct conda environment:\n"
                f"  conda activate topaz\n"
                f"Or install dependencies: pip install torch"
            )
    
    raise ValueError(f"Unknown denoiser: {name}")


def create_topaz_denoiser(
    model: str = 'unet',
    use_cuda: bool = False,
    patch_size: int = -1,
    padding: int = 128
) -> TopazDenoiser:
    """
    Create a Topaz denoiser with explicit parameters.
    
    Args:
        model: Topaz model name or path
        use_cuda: use GPU if available
        patch_size: patch size (-1 for full image)
        padding: padding for patches
    
    Returns:
        TopazDenoiser instance
    """
    return TopazDenoiser(
        model=model,
        use_cuda=use_cuda,
        patch_size=patch_size,
        padding=padding
    )
