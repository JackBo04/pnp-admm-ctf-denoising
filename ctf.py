"""
CTF (Contrast Transfer Function) utilities

Supports loading CTF parameters from JSON files and computing 2D CTF.
Compatible with cryocrab directory structure:
    /mnt/data/zouhuangbo/cryo-EM/cryocrab/mic/{dataset}/micrograph/{idx}_{dataset}_full.mrc
    /mnt/data/zouhuangbo/cryo-EM/cryocrab/mic/{dataset}/CTF/{dataset}.json
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def wavelength_kV(kv: float) -> float:
    """
    Electron wavelength in Angstroms for given acceleration voltage (kV).
    Formula: lambda = 12.264 / sqrt(V * (1 + 0.978478e-6 * V))  [Angstrom]
    """
    V = kv * 1000.0  # convert to volts
    lam = 12.2643247 / np.sqrt(V * (1.0 + 0.978476e-6 * V))
    return lam


def compute_ctf_2d(
    shape: tuple,
    pixel_size: float,
    defocus_u: float,
    defocus_v: float,
    defocus_angle: float,  # degrees
    kv: float,
    cs: float,
    amplitude_contrast: float = 0.1,
    phase_flip: bool = True,
) -> np.ndarray:
    """
    Compute 2D CTF H(k) in unshifted FFT ordering.
    
    Args:
        shape: (H, W) image shape
        pixel_size: pixel size in Angstroms
        defocus_u: major axis defocus in Angstroms (underfocus positive)
        defocus_v: minor axis defocus in Angstroms
        defocus_angle: astigmatism angle in degrees
        kv: acceleration voltage in kV
        cs: spherical aberration in mm
        amplitude_contrast: amplitude contrast ratio (0-1)
        phase_flip: if True, apply sign flip (return sign(CTF))
    
    Returns:
        Hk: CTF in Fourier domain (H, W), real-valued
    """
    H, W = shape
    
    # Frequency grids (unshifted FFT ordering)
    freq_y = np.fft.fftfreq(H, d=pixel_size)  # 1/Angstrom
    freq_x = np.fft.fftfreq(W, d=pixel_size)
    ky, kx = np.meshgrid(freq_y, freq_x, indexing='ij')
    
    # Convert to polar coordinates
    k2 = kx**2 + ky**2
    k = np.sqrt(k2)
    phi = np.arctan2(ky, kx)
    
    # Astigmatism: defocus varies with angle
    angle_rad = np.deg2rad(defocus_angle)
    defocus = 0.5 * (defocus_u + defocus_v) + \
              0.5 * (defocus_u - defocus_v) * np.cos(2.0 * (phi - angle_rad))
    
    # Physical parameters
    lam = wavelength_kV(kv)
    cs_m = cs * 1e7  # convert mm to Angstrom
    
    # Phase shift: chi = pi * lambda * defocus * k^2 - pi/2 * Cs * lambda^3 * k^4
    chi = np.pi * lam * defocus * k2 - 0.5 * np.pi * cs_m * lam**3 * k2**2
    
    # CTF = -sqrt(1-ac^2)*sin(chi) - ac*cos(chi)
    ac = amplitude_contrast
    ctf = -np.sqrt(1 - ac**2) * np.sin(chi) - ac * np.cos(chi)
    
    if phase_flip:
        ctf = np.sign(ctf)
    
    return ctf.astype(np.float32)


def load_ctf_params(json_path: str, mic_idx: Optional[int] = None) -> Dict[str, Any]:
    """
    Load CTF parameters from JSON file.
    Supports flexible field names with fallbacks.
    
    Args:
        json_path: path to CTF JSON file
        mic_idx: if JSON is a list, load entry with this idx; if None, take first
    
    Returns:
        CTF parameters dict
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle list format (cryocrab style)
    if isinstance(data, list):
        if mic_idx is not None:
            # Find entry with matching idx
            for entry in data:
                if entry.get('idx') == mic_idx:
                    data = entry
                    break
            else:
                # Fallback to first entry
                data = data[0]
        else:
            data = data[0]  # take first if list
    
    def get(keys, default=None):
        for k in keys:
            if k in data:
                return data[k]
        return default
    
    # Try to extract from cryosparc_metadata if present
    cryosparc = data.get('cryosparc_metadata', {})
    ctf_params = cryosparc.get('ctf_params', {}) if isinstance(cryosparc, dict) else {}
    
    empiar = data.get('empiar_metadata', {})
    optics = empiar.get('optics', {}) if isinstance(empiar, dict) else {}
    
    # Get pixel size from optics (considering binfactor)
    pixel_size = None
    if 'psize_A' in optics:
        psize = optics['psize_A']
        binfactor = optics.get('binfactor', 1)
        pixel_size = psize * binfactor
    
    params = {
        'pixel_size': pixel_size or get(['pixel_size', 'pixelSize', 'apix', 'angpix']),
        'defocus_u': get(['defocus_u', 'defocusU', 'defocus', 'df1', 'dfu_A']),
        'defocus_v': get(['defocus_v', 'defocusV', 'df2', 'dfv_A']),
        'defocus_angle': get(['defocus_angle', 'defocusAngle', 'astigmatism_angle', 'df_angle_rad'], 0.0),
        'kv': get(['kv', 'voltage', 'acceleration_voltage', 'accel_kv'], 300.0),
        'cs': get(['cs', 'spherical_aberration', 'Cs', 'cs_mm'], 2.7),
        'amplitude_contrast': get(['amplitude_contrast', 'amplitudeContrast', 'q0', 'amp_contrast'], 0.1),
        'phase_flip': get(['phase_flip', 'phaseFlip'], True),
    }
    
    # Override with cryosparc CTF params if available
    if 'dfu_A' in ctf_params:
        params['defocus_u'] = ctf_params['dfu_A']
    if 'dfv_A' in ctf_params:
        params['defocus_v'] = ctf_params['dfv_A']
    if 'df_angle_rad' in ctf_params:
        params['defocus_angle'] = np.rad2deg(ctf_params['df_angle_rad'])
    if 'amp_contrast' in ctf_params:
        params['amplitude_contrast'] = ctf_params['amp_contrast']
    
    # Override with optics params if available
    if 'accel_kv' in optics:
        params['kv'] = optics['accel_kv']
    if 'cs_mm' in optics:
        params['cs'] = optics['cs_mm']
    if pixel_size is not None:
        params['pixel_size'] = pixel_size
    
    # Validate required fields
    required = ['pixel_size', 'defocus_u', 'defocus_v']
    missing = [k for k in required if params[k] is None]
    if missing:
        raise ValueError(f"Missing required CTF parameters: {missing}")
    
    return params


def compute_ctf_from_json(
    shape: tuple,
    json_path: str,
    mic_idx: Optional[int] = None
) -> np.ndarray:
    """
    Compute CTF from JSON parameters file.
    
    Args:
        shape: (H, W) image shape
        json_path: path to CTF JSON file
        mic_idx: micrograph index for list-style JSON files
    
    Returns:
        Hk: CTF in Fourier domain
    """
    params = load_ctf_params(json_path, mic_idx=mic_idx)
    return compute_ctf_2d(shape=shape, **params)


def find_ctf_json_for_micrograph(mic_path: str) -> Optional[str]:
    """
    Find the corresponding CTF JSON file for a micrograph.
    
    Uses naming convention:
        {idx}_{dataset}_full.mrc -> {dataset}/CTF/{dataset}.json
    
    Args:
        mic_path: path to micrograph MRC file
    
    Returns:
        path to CTF JSON file, or None if not found
    """
    path = Path(mic_path)
    stem = path.stem  # e.g., "000000_empiar_10002_full"
    
    # Extract dataset name from stem
    # Format: {idx}_{dataset}_full or {idx}_{dataset}_diff
    parts = stem.split('_')
    if len(parts) >= 2:
        # Last part before _full or _diff is dataset name
        if parts[-1] in ['full', 'diff']:
            dataset = '_'.join(parts[1:-1])  # Handle multi-part dataset names
        else:
            dataset = '_'.join(parts[1:])
    else:
        return None
    
    # Look for CTF JSON in standard locations
    # 1. Same directory as micrograph
    same_dir = path.parent / f"{dataset}.json"
    if same_dir.exists():
        return str(same_dir)
    
    # 2. CTF subdirectory (mic/{dataset}/micrograph/ -> mic/CTF/)
    ctf_dir = path.parent.parent.parent / 'CTF' / f"{dataset}.json"
    if ctf_dir.exists():
        return str(ctf_dir)
    
    # 3. CTF subdirectory one level up (alternative structure)
    ctf_dir_alt = path.parent.parent / 'CTF' / f"{dataset}.json"
    if ctf_dir_alt.exists():
        return str(ctf_dir_alt)
    
    # 4. Parent directory (cryocrab structure)
    parent_dir = path.parent.parent / f"{dataset}.json"
    if parent_dir.exists():
        return str(parent_dir)
    
    # 5. Parent of parent directory
    parent_parent_dir = path.parent.parent.parent / f"{dataset}.json"
    if parent_parent_dir.exists():
        return str(parent_parent_dir)

    # 6. CTF subdirectory in parent of parent (cryocrab: mic/CTF/)
    ctf_in_parent = path.parent.parent.parent / 'CTF' / f"{dataset}.json"
    if ctf_in_parent.exists():
        return str(ctf_in_parent)

    return None


def parse_micrograph_filename(mic_path: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Parse micrograph filename to extract metadata.
    
    Args:
        mic_path: path to micrograph
    
    Returns:
        (idx, dataset, variant) where variant is 'full' or 'diff'
    """
    path = Path(mic_path)
    stem = path.stem
    
    parts = stem.split('_')
    if len(parts) < 2:
        return None, None, None
    
    # Try to parse index
    try:
        idx = int(parts[0])
    except ValueError:
        idx = None
    
    # Last part indicates full or diff
    variant = None
    dataset_parts = parts[1:]
    if parts[-1] in ['full', 'diff']:
        variant = parts[-1]
        dataset_parts = parts[1:-1]
    
    dataset = '_'.join(dataset_parts)
    
    return idx, dataset, variant


def pair_full_diff_files(full_path: str, diff_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Ensure full and diff files are correctly paired.
    
    Args:
        full_path: path to full micrograph
        diff_path: path to diff micrograph (if None, auto-detect)
    
    Returns:
        (full_path, diff_path)
    """
    full_path = str(full_path)
    
    if diff_path is not None:
        return full_path, str(diff_path)
    
    # Auto-detect diff file
    path = Path(full_path)
    stem = path.stem
    
    # Try replacing _full with _diff
    if '_full' in stem:
        diff_stem = stem.replace('_full', '_diff')
    else:
        # Try appending _diff
        diff_stem = stem + '_diff'
    
    diff_path_candidate = path.parent / f"{diff_stem}.mrc"
    
    if diff_path_candidate.exists():
        return full_path, str(diff_path_candidate)
    
    # Try alternative extensions
    for ext in ['.mrc', '.mrcs']:
        diff_path_candidate = path.parent / f"{diff_stem}{ext}"
        if diff_path_candidate.exists():
            return full_path, str(diff_path_candidate)
    
    return full_path, None


def load_ctf_for_micrograph(
    mic_path: str,
    shape: tuple,
    ctf_json_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load and compute CTF for a specific micrograph.
    
    Args:
        mic_path: path to micrograph file
        shape: image shape (H, W)
        ctf_json_path: explicit path to CTF JSON (if None, auto-detect)
    
    Returns:
        (Hk, info_dict)
    """
    # Find CTF JSON
    if ctf_json_path is None:
        ctf_json_path = find_ctf_json_for_micrograph(mic_path)
        if ctf_json_path is None:
            raise FileNotFoundError(f"Could not find CTF JSON for {mic_path}")
    
    # Parse micrograph filename to get idx
    idx, dataset, variant = parse_micrograph_filename(mic_path)
    
    # Load CTF parameters
    params = load_ctf_params(ctf_json_path, mic_idx=idx)
    
    # Compute CTF
    Hk = compute_ctf_2d(shape=shape, **params)
    
    info = {
        'ctf_json_path': ctf_json_path,
        'mic_idx': idx,
        'dataset': dataset,
        'params': params,
    }
    
    return Hk, info
