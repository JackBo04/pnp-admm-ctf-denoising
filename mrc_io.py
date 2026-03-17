"""
MRC file I/O utilities
"""
import mrcfile
import numpy as np
from pathlib import Path
from typing import Tuple


def read_mrc(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read MRC file.
    
    Args:
        path: path to .mrc file
    
    Returns:
        (data, header_info)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRC file not found: {path}")
    
    with mrcfile.open(path, permissive=True) as mrc:
        data = mrc.data.astype(np.float32)
        header_info = {
            'pixel_size': float(mrc.voxel_size.x) if mrc.voxel_size.x else 1.0,
            'shape': data.shape,
            'origin': (float(mrc.header.origin.x), 
                      float(mrc.header.origin.y),
                      float(mrc.header.origin.z)),
        }
    
    return data, header_info


def write_mrc(path: str, data: np.ndarray, pixel_size: float = 1.0):
    """
    Write numpy array to MRC file.
    
    Args:
        path: output path (should end with .mrc)
        data: image data, 2D or 3D
        pixel_size: voxel size in Angstroms
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct dtype
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = pixel_size
    
    return path


def read_mrc_stack(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read MRC stack (3D: N x H x W).
    
    Returns:
        data: (N, H, W) array
        header_info: metadata
    """
    data, header_info = read_mrc(path)
    
    # Ensure 3D format (N, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, H, W)
    elif data.ndim == 3:
        # MRC stacks are typically (N, H, W) or (H, W, N)
        # mrcfile returns (N, H, W) for image stacks
        pass
    
    return data, header_info


def write_mrc_stack(path: str, data: np.ndarray, pixel_size: float = 1.0):
    """
    Write image stack to MRC file.
    
    Args:
        path: output path
        data: (N, H, W) array
        pixel_size: voxel size
    """
    return write_mrc(path, data, pixel_size)
